"""State store — a graph of entries (nodes) and associations (edges).

The state store is the externalized memory that grows without bound.
Entries are never deleted, only superseded. Associations are first-class:
typed, weighted, context-scoped, with access tracking.

The MemoryMap sits on top as a compact routing layer — providing entry
points into the graph for a given objective.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime

from cognition.types import (
    Association,
    EntryType,
    MemoryMapData,
    RelationshipType,
    StateEntry,
    TopicEntry,
)

logger = logging.getLogger(__name__)


class MemoryMap:
    """Compact routing layer over the state graph.

    Primary function: given an objective, provide entry points into the graph.
    Also captures topology metadata (cluster density, contested regions,
    weakly-connected entries).

    Small enough to always fit in LLM context. Updated incrementally.
    """

    def __init__(self, data: MemoryMapData | None = None) -> None:
        self.data = data or MemoryMapData()

    def token_size(self) -> int:
        """Approximate token count of the rendered memory map (~4 chars/token)."""
        return len(self.render()) // 4

    @property
    def topics(self) -> dict[str, TopicEntry]:
        return self.data.topics

    @property
    def recent_changes(self) -> list[str]:
        return self.data.recent_changes

    @property
    def contested_regions(self) -> list[str]:
        return self.data.contested_regions

    @property
    def weakly_connected(self) -> list[str]:
        return self.data.weakly_connected

    def render(self) -> str:
        """Render as structured text for LLM context."""
        if not self.data.topics and not self.data.weakly_connected:
            return "Memory Map: (empty — no entries yet)"

        lines = ["Memory Map:"]

        if self.data.topics:
            for name, topic in self.data.topics.items():
                ep = ", ".join(topic.entry_points[:5])
                lines.append(f"  {name}: {topic.summary}")
                lines.append(f"    Entry points: [{ep}]")
                lines.append(f"    Density: {topic.density}")

        if self.data.recent_changes:
            recent = "; ".join(self.data.recent_changes[-5:])
            lines.append(f"  Recent: {recent}")

        if self.data.contested_regions:
            lines.append(f"  Contested: {', '.join(self.data.contested_regions)}")

        if self.data.weakly_connected:
            wc = ", ".join(self.data.weakly_connected[:10])
            lines.append(f"  Weakly connected: [{wc}]")

        return "\n".join(lines)

    def get_entry_points(self, store: "StateStore | None" = None) -> list[str]:
        """Return candidate entry point IDs.

        If a StateStore is provided, filters out stale references
        (entries that no longer exist or are superseded).

        Currently returns all entry points across all topics (the LLM
        selects during orient). Will be optimized later with entity-based
        indexing when the memory map grows too large for context.
        """
        points: list[str] = []
        for topic in self.data.topics.values():
            points.extend(topic.entry_points)
        # Include weakly connected entries as potential entry points
        points.extend(self.data.weakly_connected)
        # Deduplicate
        seen: set[str] = set()
        unique: list[str] = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        # Filter stale references if store provided
        if store is not None:
            unique = [p for p in unique if (e := store.get_entry(p)) is not None and e.is_active]
        return unique

    def update(self, changes: dict) -> None:
        """Incremental update to the memory map.

        Expected keys in changes:
        - new_topics: dict[str, TopicEntry]
        - updated_topics: dict[str, dict] (partial updates to existing topics)
        - recent_changes: list[str]
        - contested_regions: list[str]
        - weakly_connected: list[str]
        - remove_weakly_connected: list[str]
        """
        if "new_topics" in changes:
            for name, topic_data in changes["new_topics"].items():
                if isinstance(topic_data, TopicEntry):
                    self.data.topics[name] = topic_data
                elif isinstance(topic_data, dict):
                    self.data.topics[name] = TopicEntry(**topic_data)

        if "updated_topics" in changes:
            for name, updates in changes["updated_topics"].items():
                if name in self.data.topics:
                    topic = self.data.topics[name]
                    if updates.get("summary") is not None:
                        topic.summary = updates["summary"]
                    if updates.get("entry_points") is not None:
                        topic.entry_points = updates["entry_points"]
                    if updates.get("add_entry_points") is not None:
                        for ep in updates["add_entry_points"]:
                            if ep not in topic.entry_points:
                                topic.entry_points.append(ep)
                    if updates.get("density") is not None:
                        topic.density = updates["density"]

        if "recent_changes" in changes:
            self.data.recent_changes.extend(changes["recent_changes"])
            # Keep only the most recent changes
            self.data.recent_changes = self.data.recent_changes[-20:]

        if "contested_regions" in changes:
            for region in changes["contested_regions"]:
                if region not in self.data.contested_regions:
                    self.data.contested_regions.append(region)

        if "weakly_connected" in changes:
            for entry_id in changes["weakly_connected"]:
                if entry_id not in self.data.weakly_connected:
                    self.data.weakly_connected.append(entry_id)

        if "remove_weakly_connected" in changes:
            self.data.weakly_connected = [
                eid for eid in self.data.weakly_connected
                if eid not in changes["remove_weakly_connected"]
            ]


class StateStore:
    """Graph-based state store with entries (nodes) and associations (edges).

    The state store is the externalized memory that grows without bound.
    The cognitive step navigates it via the memory map and graph traversal
    without needing to fit all of it in context.
    """

    def __init__(self) -> None:
        self.entries: dict[str, StateEntry] = {}
        self.associations: dict[str, Association] = {}
        # Adjacency indexes for fast lookup
        self._assoc_by_source: dict[str, list[str]] = defaultdict(list)
        self._assoc_by_target: dict[str, list[str]] = defaultdict(list)
        self.memory_map = MemoryMap()

    # --- Entry operations ---

    def write(self, entry: StateEntry) -> None:
        self.entries[entry.id] = entry
        # Auto-register in memory map as weakly connected so it's discoverable
        if entry.id not in self.memory_map.data.weakly_connected:
            self.memory_map.data.weakly_connected.append(entry.id)
        logger.debug("state.write entry_id=%s type=%s", entry.id, entry.entry_type)

    def get_entry(self, entry_id: str) -> StateEntry | None:
        return self.entries.get(entry_id)

    def invalidate_entry(self, entry_id: str, superseded_by: str) -> None:
        entry = self.entries.get(entry_id)
        if entry:
            entry.superseded_by = superseded_by
            # Clean up memory map — remove from topic entry_points
            for topic in self.memory_map.data.topics.values():
                if entry_id in topic.entry_points:
                    topic.entry_points.remove(entry_id)
            # Remove from weakly_connected
            if entry_id in self.memory_map.data.weakly_connected:
                self.memory_map.data.weakly_connected.remove(entry_id)
            logger.debug("state.invalidate_entry entry_id=%s superseded_by=%s", entry_id, superseded_by)

    def access(self, entry_id: str, step: int) -> None:
        entry = self.entries.get(entry_id)
        if entry:
            entry.step_last_accessed = step
            entry.access_count += 1

    # --- Association operations ---

    def add_association(self, assoc: Association) -> None:
        self.associations[assoc.id] = assoc
        self._assoc_by_source[assoc.source_id].append(assoc.id)
        self._assoc_by_target[assoc.target_id].append(assoc.id)
        logger.debug(
            "state.add_association %s -> %s (%s, context=%r, weight=%.2f)",
            assoc.source_id, assoc.target_id, assoc.relationship, assoc.context, assoc.weight,
        )

    def get_associations(self, entry_id: str) -> list[Association]:
        """Get all associations where entry_id is source or target."""
        assoc_ids = set(self._assoc_by_source.get(entry_id, []))
        assoc_ids.update(self._assoc_by_target.get(entry_id, []))
        return [self.associations[aid] for aid in assoc_ids if aid in self.associations]

    def get_associations_in_context(self, entry_id: str, context: str) -> list[Association]:
        return [a for a in self.get_associations(entry_id) if a.context == context]

    def strengthen(self, assoc_id: str, delta: float, step: int) -> None:
        assoc = self.associations.get(assoc_id)
        if assoc:
            assoc.weight = min(1.0, assoc.weight + delta)
            assoc.step_last_accessed = step
            logger.debug("state.strengthen assoc_id=%s new_weight=%.2f", assoc_id, assoc.weight)

    def weaken(self, assoc_id: str, delta: float, step: int) -> None:
        assoc = self.associations.get(assoc_id)
        if assoc:
            assoc.weight = max(0.0, assoc.weight - delta)
            assoc.step_last_accessed = step
            logger.debug("state.weaken assoc_id=%s new_weight=%.2f", assoc_id, assoc.weight)

    def invalidate_association(
        self,
        assoc_id: str,
        reason: str,
        invalidated_by: str | None = None,
    ) -> None:
        assoc = self.associations.get(assoc_id)
        if assoc:
            assoc.valid = False
            assoc.invalidation_reason = reason
            assoc.invalidated_by_entry = invalidated_by
            logger.debug("state.invalidate_association assoc_id=%s reason=%r", assoc_id, reason)

    # --- Navigation ---

    def get_neighbors(self, entry_id: str, max_hops: int = 1) -> list[StateEntry]:
        """Get entries reachable within max_hops via valid associations."""
        visited: set[str] = set()
        current: set[str] = {entry_id}

        for _ in range(max_hops):
            next_level: set[str] = set()
            for eid in current:
                for assoc in self.get_associations(eid):
                    if not assoc.valid:
                        continue
                    neighbor = assoc.target_id if assoc.source_id == eid else assoc.source_id
                    if neighbor not in visited and neighbor != entry_id:
                        next_level.add(neighbor)
            visited.update(current)
            current = next_level

        visited.update(current)
        visited.discard(entry_id)
        return [self.entries[eid] for eid in visited if eid in self.entries]

    def get_active(self) -> list[StateEntry]:
        return [e for e in self.entries.values() if e.is_active]

    def get_recent(self, n: int) -> list[StateEntry]:
        active = self.get_active()
        active.sort(key=lambda e: e.step_created, reverse=True)
        return active[:n]

    def get_temporal_neighborhood(self, entry: StateEntry, window: int) -> list[StateEntry]:
        step = entry.step_created
        return [
            e for e in self.entries.values()
            if abs(e.step_created - step) <= window and e.id != entry.id
        ]

    def size(self) -> int:
        return len(self.get_active())

    # --- Rendering for LLM context ---

    def render_neighborhood(self, entry_ids: list[str], depth: int = 1) -> str:
        """Render entries and their associations as a structured document
        for inclusion in LLM context."""
        if not entry_ids:
            return "(no entries to render)"

        lines: list[str] = []
        rendered: set[str] = set()

        for entry_id in entry_ids:
            self._render_entry(entry_id, lines, rendered, depth)

        return "\n".join(lines)

    def _render_entry(
        self,
        entry_id: str,
        lines: list[str],
        rendered: set[str],
        depth: int,
    ) -> None:
        if entry_id in rendered:
            return
        rendered.add(entry_id)

        entry = self.entries.get(entry_id)
        if not entry:
            return

        active = "active" if entry.is_active else f"superseded by {entry.superseded_by}"
        lines.append(
            f'Entry [{entry.id[:8]}]: "{entry.content}" '
            f"({entry.entry_type.value}, confidence {entry.confidence}, "
            f"accessed {entry.access_count}x, {active})"
        )

        associations = self.get_associations(entry_id)
        if associations:
            lines.append("  Associations:")
            # Group by target
            by_target: dict[str, list[Association]] = defaultdict(list)
            for assoc in associations:
                other_id = assoc.target_id if assoc.source_id == entry_id else assoc.source_id
                by_target[other_id].append(assoc)

            for other_id, assocs in by_target.items():
                other = self.entries.get(other_id)
                other_desc = f'"{other.content}"' if other else f"[{other_id[:8]}]"
                for assoc in assocs:
                    valid_str = "valid" if assoc.valid else f"INVALID — {assoc.invalidation_reason}"
                    lines.append(
                        f"    -> {other_desc} ({assoc.relationship.value})"
                    )
                    lines.append(
                        f"        [{assoc.context}] weight: {assoc.weight:.1f}, {valid_str}"
                    )

        # Recurse into neighbors if depth > 0
        if depth > 0:
            for assoc in associations:
                if assoc.valid:
                    other_id = assoc.target_id if assoc.source_id == entry_id else assoc.source_id
                    self._render_entry(other_id, lines, rendered, depth - 1)

    def render_entries(self, entry_ids: list[str]) -> str:
        """Render just the entries (no associations) as structured text."""
        lines: list[str] = []
        for entry_id in entry_ids:
            entry = self.entries.get(entry_id)
            if entry:
                active = "active" if entry.is_active else f"superseded by {entry.superseded_by}"
                lines.append(
                    f'[{entry.id[:8]}] "{entry.content}" '
                    f"({entry.entry_type.value}, conf {entry.confidence}, {active})"
                )
        return "\n".join(lines)

    # --- Persistence ---

    def save(self, path: str) -> None:
        data = {
            "entries": {eid: _entry_to_dict(e) for eid, e in self.entries.items()},
            "associations": {aid: _assoc_to_dict(a) for aid, a in self.associations.items()},
            "memory_map": _memory_map_to_dict(self.memory_map),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("state.save path=%s entries=%d associations=%d", path, len(self.entries), len(self.associations))

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)

        self.entries = {}
        for eid, edata in data.get("entries", {}).items():
            edata["entry_type"] = EntryType(edata["entry_type"])
            if isinstance(edata.get("created_at"), str):
                edata["created_at"] = datetime.fromisoformat(edata["created_at"])
            self.entries[eid] = StateEntry(**edata)

        self.associations = {}
        self._assoc_by_source = defaultdict(list)
        self._assoc_by_target = defaultdict(list)
        for aid, adata in data.get("associations", {}).items():
            adata["relationship"] = RelationshipType(adata["relationship"])
            assoc = Association(**adata)
            self.associations[aid] = assoc
            self._assoc_by_source[assoc.source_id].append(aid)
            self._assoc_by_target[assoc.target_id].append(aid)

        mm_data = data.get("memory_map", {})
        topics = {}
        for name, tdata in mm_data.get("topics", {}).items():
            topics[name] = TopicEntry(**tdata)
        self.memory_map = MemoryMap(MemoryMapData(
            topics=topics,
            recent_changes=mm_data.get("recent_changes", []),
            contested_regions=mm_data.get("contested_regions", []),
            weakly_connected=mm_data.get("weakly_connected", []),
        ))

        logger.info("state.load path=%s entries=%d associations=%d", path, len(self.entries), len(self.associations))


# --- Serialization helpers ---

def _entry_to_dict(entry: StateEntry) -> dict:
    return {
        "id": entry.id,
        "content": entry.content,
        "entry_type": entry.entry_type.value,
        "confidence": entry.confidence,
        "step_created": entry.step_created,
        "step_last_accessed": entry.step_last_accessed,
        "access_count": entry.access_count,
        "superseded_by": entry.superseded_by,
        "tags": entry.tags,
        "created_at": entry.created_at.isoformat(),
    }


def _assoc_to_dict(assoc: Association) -> dict:
    return {
        "id": assoc.id,
        "source_id": assoc.source_id,
        "target_id": assoc.target_id,
        "relationship": assoc.relationship.value,
        "weight": assoc.weight,
        "context": assoc.context,
        "valid": assoc.valid,
        "invalidation_reason": assoc.invalidation_reason,
        "invalidated_by_entry": assoc.invalidated_by_entry,
        "step_created": assoc.step_created,
        "step_last_accessed": assoc.step_last_accessed,
    }


def _memory_map_to_dict(mm: MemoryMap) -> dict:
    topics = {}
    for name, topic in mm.data.topics.items():
        topics[name] = {
            "summary": topic.summary,
            "entry_points": topic.entry_points,
            "density": topic.density,
        }
    return {
        "topics": topics,
        "recent_changes": mm.data.recent_changes,
        "contested_regions": mm.data.contested_regions,
        "weakly_connected": mm.data.weakly_connected,
    }
