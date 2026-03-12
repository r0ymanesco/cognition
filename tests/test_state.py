"""Tests for the state store — graph operations, associations, memory map, persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from cognition.state import MemoryMap, StateStore
from cognition.types import (
    Association,
    EntryType,
    MemoryMapData,
    RelationshipType,
    StateEntry,
    TopicEntry,
)


@pytest.fixture
def store() -> StateStore:
    return StateStore()


@pytest.fixture
def alice_entry() -> StateEntry:
    return StateEntry(
        content="Alice has 5 apples",
        entry_type=EntryType.FACT,
        confidence=0.9,
        step_created=1,
        tags=["alice", "apples"],
    )


@pytest.fixture
def bob_entry() -> StateEntry:
    return StateEntry(
        content="Bob has 3 oranges",
        entry_type=EntryType.FACT,
        confidence=0.9,
        step_created=2,
        tags=["bob", "oranges"],
    )


class TestEntryOperations:
    def test_write_and_get(self, store: StateStore, alice_entry: StateEntry):
        store.write(alice_entry)
        assert store.get_entry(alice_entry.id) is alice_entry
        assert store.size() == 1

    def test_get_nonexistent(self, store: StateStore):
        assert store.get_entry("nonexistent") is None

    def test_invalidate_entry(self, store: StateStore, alice_entry: StateEntry):
        store.write(alice_entry)
        new_entry = StateEntry(
            content="Alice has 7 apples",
            entry_type=EntryType.FACT,
            step_created=5,
        )
        store.write(new_entry)
        store.invalidate_entry(alice_entry.id, superseded_by=new_entry.id)

        assert not alice_entry.is_active
        assert alice_entry.superseded_by == new_entry.id
        # Invalidated entry doesn't count as active
        assert store.size() == 1

    def test_access_tracking(self, store: StateStore, alice_entry: StateEntry):
        store.write(alice_entry)
        assert alice_entry.access_count == 0

        store.access(alice_entry.id, step=5)
        assert alice_entry.access_count == 1
        assert alice_entry.step_last_accessed == 5

        store.access(alice_entry.id, step=10)
        assert alice_entry.access_count == 2
        assert alice_entry.step_last_accessed == 10

    def test_get_active(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)
        assert len(store.get_active()) == 2

        store.invalidate_entry(alice_entry.id, superseded_by="x")
        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == bob_entry.id

    def test_get_recent(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)
        recent = store.get_recent(1)
        assert len(recent) == 1
        assert recent[0].id == bob_entry.id  # bob was created at step 2

    def test_get_temporal_neighborhood(self, store: StateStore):
        entries = [
            StateEntry(content=f"entry {i}", entry_type=EntryType.OBSERVATION, step_created=i)
            for i in range(10)
        ]
        for e in entries:
            store.write(e)

        neighbors = store.get_temporal_neighborhood(entries[5], window=2)
        neighbor_steps = {e.step_created for e in neighbors}
        assert neighbor_steps == {3, 4, 6, 7}


class TestAssociationOperations:
    def test_add_and_get_associations(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)

        assoc = Association(
            source_id=alice_entry.id,
            target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.5,
            context="fruit inventory",
        )
        store.add_association(assoc)

        # Retrievable from both source and target
        from_alice = store.get_associations(alice_entry.id)
        from_bob = store.get_associations(bob_entry.id)
        assert len(from_alice) == 1
        assert len(from_bob) == 1
        assert from_alice[0].id == assoc.id

    def test_context_scoped_associations(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)

        assoc1 = Association(
            source_id=alice_entry.id, target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.8, context="apple trades",
        )
        assoc2 = Association(
            source_id=alice_entry.id, target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.1, context="family",
        )
        store.add_association(assoc1)
        store.add_association(assoc2)

        # All associations
        all_assocs = store.get_associations(alice_entry.id)
        assert len(all_assocs) == 2

        # Filtered by context
        trade_assocs = store.get_associations_in_context(alice_entry.id, "apple trades")
        assert len(trade_assocs) == 1
        assert trade_assocs[0].weight == 0.8

    def test_strengthen(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)

        assoc = Association(
            source_id=alice_entry.id, target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.3, context="test",
        )
        store.add_association(assoc)

        store.strengthen(assoc.id, 0.2, step=5)
        assert assoc.weight == pytest.approx(0.5)
        assert assoc.step_last_accessed == 5

        # Weight caps at 1.0
        store.strengthen(assoc.id, 0.8, step=6)
        assert assoc.weight == pytest.approx(1.0)

    def test_weaken(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)

        assoc = Association(
            source_id=alice_entry.id, target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.5, context="test",
        )
        store.add_association(assoc)

        store.weaken(assoc.id, 0.3, step=5)
        assert assoc.weight == pytest.approx(0.2)

        # Weight floors at 0.0
        store.weaken(assoc.id, 0.5, step=6)
        assert assoc.weight == pytest.approx(0.0)

    def test_invalidate_association(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)

        assoc = Association(
            source_id=alice_entry.id, target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.8, context="test",
        )
        store.add_association(assoc)

        store.invalidate_association(assoc.id, reason="contradicted by new info", invalidated_by="entry_xyz")

        assert not assoc.valid
        assert assoc.invalidation_reason == "contradicted by new info"
        assert assoc.invalidated_by_entry == "entry_xyz"
        # Weight is preserved (historical record)
        assert assoc.weight == 0.8


class TestGraphNavigation:
    def test_get_neighbors(self, store: StateStore):
        a = StateEntry(content="A", entry_type=EntryType.FACT)
        b = StateEntry(content="B", entry_type=EntryType.FACT)
        c = StateEntry(content="C", entry_type=EntryType.FACT)
        store.write(a)
        store.write(b)
        store.write(c)

        store.add_association(Association(
            source_id=a.id, target_id=b.id,
            relationship=RelationshipType.RELATED_TO, weight=0.5, context="test",
        ))
        store.add_association(Association(
            source_id=b.id, target_id=c.id,
            relationship=RelationshipType.RELATED_TO, weight=0.5, context="test",
        ))

        # 1 hop from A reaches B
        neighbors_1 = store.get_neighbors(a.id, max_hops=1)
        assert len(neighbors_1) == 1
        assert neighbors_1[0].id == b.id

        # 2 hops from A reaches B and C
        neighbors_2 = store.get_neighbors(a.id, max_hops=2)
        assert len(neighbors_2) == 2
        neighbor_ids = {n.id for n in neighbors_2}
        assert neighbor_ids == {b.id, c.id}

    def test_invalid_associations_not_traversed(self, store: StateStore):
        a = StateEntry(content="A", entry_type=EntryType.FACT)
        b = StateEntry(content="B", entry_type=EntryType.FACT)
        store.write(a)
        store.write(b)

        assoc = Association(
            source_id=a.id, target_id=b.id,
            relationship=RelationshipType.RELATED_TO, weight=0.5, context="test",
        )
        store.add_association(assoc)
        store.invalidate_association(assoc.id, reason="wrong")

        neighbors = store.get_neighbors(a.id, max_hops=1)
        assert len(neighbors) == 0


class TestRendering:
    def test_render_neighborhood(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)

        assoc = Association(
            source_id=alice_entry.id, target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.8, context="fruit inventory",
        )
        store.add_association(assoc)

        rendered = store.render_neighborhood([alice_entry.id], depth=1)
        assert "Alice has 5 apples" in rendered
        assert "Bob has 3 oranges" in rendered
        assert "related_to" in rendered
        assert "fruit inventory" in rendered

    def test_render_empty(self, store: StateStore):
        rendered = store.render_neighborhood([], depth=1)
        assert "no entries" in rendered


class TestMemoryMap:
    def test_empty_render(self):
        mm = MemoryMap()
        rendered = mm.render()
        assert "empty" in rendered

    def test_render_with_topics(self):
        mm = MemoryMap(MemoryMapData(
            topics={
                "fruits": TopicEntry(
                    summary="Fruit ownership tracking",
                    entry_points=["e1", "e2"],
                    density="dense",
                ),
            },
            recent_changes=["Added alice entry"],
        ))
        rendered = mm.render()
        assert "fruits" in rendered
        assert "Fruit ownership" in rendered
        assert "e1" in rendered
        assert "dense" in rendered
        assert "Added alice entry" in rendered

    def test_update(self):
        mm = MemoryMap()
        mm.update({
            "new_topics": {
                "fruits": {"summary": "Fruit tracking", "entry_points": ["e1"], "density": "sparse"},
            },
            "recent_changes": ["Added fruits topic"],
        })
        assert "fruits" in mm.topics
        assert mm.topics["fruits"].entry_points == ["e1"]

        # Update existing topic
        mm.update({
            "updated_topics": {
                "fruits": {"add_entry_points": ["e2"], "density": "dense"},
            },
        })
        assert "e2" in mm.topics["fruits"].entry_points
        assert mm.topics["fruits"].density == "dense"

    def test_get_entry_points(self):
        mm = MemoryMap(MemoryMapData(
            topics={
                "a": TopicEntry(summary="A", entry_points=["e1", "e2"]),
                "b": TopicEntry(summary="B", entry_points=["e3"]),
            },
        ))
        points = mm.get_entry_points("anything")
        assert set(points) == {"e1", "e2", "e3"}


class TestPersistence:
    def test_save_and_load(self, store: StateStore, alice_entry: StateEntry, bob_entry: StateEntry):
        store.write(alice_entry)
        store.write(bob_entry)

        assoc = Association(
            source_id=alice_entry.id, target_id=bob_entry.id,
            relationship=RelationshipType.RELATED_TO,
            weight=0.7, context="test",
        )
        store.add_association(assoc)

        store.memory_map.update({
            "new_topics": {
                "fruits": {"summary": "Fruit tracking", "entry_points": [alice_entry.id], "density": "sparse"},
            },
        })

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        store.save(path)

        # Load into a fresh store
        loaded = StateStore()
        loaded.load(path)

        assert len(loaded.entries) == 2
        assert loaded.get_entry(alice_entry.id).content == "Alice has 5 apples"
        assert loaded.get_entry(bob_entry.id).content == "Bob has 3 oranges"

        loaded_assocs = loaded.get_associations(alice_entry.id)
        assert len(loaded_assocs) == 1
        assert loaded_assocs[0].weight == 0.7
        assert loaded_assocs[0].context == "test"

        assert "fruits" in loaded.memory_map.topics

        Path(path).unlink()
