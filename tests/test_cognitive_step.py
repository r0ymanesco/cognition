"""Tests for the cognitive step — the recursive primitive.

Uses a mock LLM that returns predictable structured responses,
so we can test the recursion, graph traversal, state mutations,
and termination logic without hitting a real API.
"""

from __future__ import annotations

from typing import Any, Type
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from pydantic import BaseModel

from cognition.cognitive_step import (
    CognitiveStep,
    OrientationResponse,
    SynthesisResponse,
    TraversalStepResponse,
    NewEntrySpec,
    NewAssociationSpec,
    AssociationUpdateSpec,
    MapChangesSpec,
)
from cognition.llm.base import BaseLLM
from cognition.state import StateStore
from cognition.tracing import TraceLogger
from cognition.types import (
    Association,
    MemoryMapData,
    StateEntry,
    TopicEntry,
)


class MockLLM(BaseLLM):
    """Mock LLM that returns pre-configured responses based on response_model type."""

    def __init__(self):
        self.responses: dict[str, Any] = {}
        self.call_log: list[dict] = []

    def set_response(self, model_name: str, response: BaseModel) -> None:
        self.responses[model_name] = response

    def set_response_sequence(self, model_name: str, responses: list[BaseModel]) -> None:
        """Set multiple responses — each call pops the next one."""
        self.responses[model_name] = list(responses)

    async def generate(self, messages: list[dict[str, str]], system: str = "") -> str:
        self.call_log.append({"type": "generate", "messages": messages, "system": system})
        return self.responses.get("generate", "")

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: Type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        self.call_log.append({
            "type": "generate_structured",
            "model": response_model.__name__,
            "messages": messages,
            "system": system,
        })
        key = response_model.__name__
        resp = self.responses.get(key)
        if isinstance(resp, list):
            if resp:
                return resp.pop(0)
            raise ValueError(f"No more responses for {key}")
        if resp is not None:
            return resp
        # Return a default empty instance
        return response_model()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def store() -> StateStore:
    return StateStore()


@pytest.fixture
def tracer() -> TraceLogger:
    return TraceLogger()


@pytest.fixture
def populated_store() -> StateStore:
    """Store with some entries, associations, and a memory map."""
    s = StateStore()

    alice = StateEntry(
        id="alice_1", content="Alice has 5 apples",
        entry_type="fact", step_created=1, tags=["alice", "apples"],
    )
    bob = StateEntry(
        id="bob_1", content="Bob has 3 oranges",
        entry_type="fact", step_created=2, tags=["bob", "oranges"],
    )
    s.write(alice)
    s.write(bob)

    assoc = Association(
        id="assoc_1",
        source_id="alice_1", target_id="bob_1",
        relationship="related_to",
        weight=0.5, context="fruit inventory",
    )
    s.add_association(assoc)

    s.memory_map.update({
        "new_topics": {
            "fruit_inventory": {
                "summary": "Tracking fruit ownership for Alice and Bob",
                "entry_points": ["alice_1", "bob_1"],
                "density": "sparse",
            },
        },
    })

    return s


class TestDirectResolve:
    """Tests for the base case — graph traversal with LLM navigation."""

    @pytest.mark.asyncio
    async def test_single_step_traversal(self, mock_llm: MockLLM, populated_store: StateStore, tracer: TraceLogger):
        """Traversal stops immediately when LLM says objective_satisfied."""
        step = CognitiveStep(mock_llm, max_width=3, max_steps=10)

        # Orient returns entry points, no sub-objectives (triggers direct resolve)
        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1"],
            sub_objectives=[],
            reasoning="Direct lookup",
        ))

        # Traversal: find alice's apples and stop
        mock_llm.set_response("TraversalStepResponse", TraversalStepResponse(
            findings=["Alice has 5 apples"],
            should_stop=True,
            stop_reason="objective_satisfied",
        ))

        # Synthesis produces the final output
        mock_llm.set_response("SynthesisResponse", SynthesisResponse(
            output="Alice has 5 apples",
        ))

        result = await step.execute(
            objective="How many apples does Alice have?",
            state=populated_store,
            context={"input": "test", "step_number": 0},
            tracer=tracer,
        )

        assert "Alice has 5 apples" in result.output
        # Alice should have been accessed
        alice = populated_store.get_entry("alice_1")
        assert alice is not None
        assert alice.access_count == 1

    @pytest.mark.asyncio
    async def test_multi_step_traversal(self, mock_llm: MockLLM, populated_store: StateStore):
        """Traversal follows edges for multiple steps before stopping."""
        step = CognitiveStep(mock_llm, max_width=3, max_steps=10)

        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1"],
            sub_objectives=[],
        ))

        # Step 1: at alice, follow edge to bob
        # Step 2: at bob, objective satisfied
        mock_llm.set_response_sequence("TraversalStepResponse", [
            TraversalStepResponse(
                findings=["Alice has 5 apples"],
                next_nodes=["bob_1"],
                should_stop=False,
            ),
            TraversalStepResponse(
                findings=["Bob has 3 oranges"],
                should_stop=True,
                stop_reason="objective_satisfied",
            ),
        ])

        mock_llm.set_response("SynthesisResponse", SynthesisResponse(
            output="Alice has 5 apples and Bob has 3 oranges",
        ))

        result = await step.execute(
            objective="What fruits do people have?",
            state=populated_store,
            context={"input": "test", "step_number": 0},
        )

        assert "Alice has 5 apples" in result.output
        assert "Bob has 3 oranges" in result.output

    @pytest.mark.asyncio
    async def test_state_mutations_during_traversal(self, mock_llm: MockLLM, populated_store: StateStore):
        """Traversal can create new entries and associations."""
        step = CognitiveStep(mock_llm, max_width=3, max_steps=10)

        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1"],
            sub_objectives=[],
        ))

        mock_llm.set_response("TraversalStepResponse", TraversalStepResponse(
            findings=["Discovered trade between Alice and Bob"],
            should_stop=True,
            stop_reason="objective_satisfied",
            new_entries=[NewEntrySpec(
                content="Alice traded 2 apples to Bob",
                entry_type="observation",
                confidence=0.8,
                tags=["alice", "bob", "trade"],
            )],
            new_associations=[NewAssociationSpec(
                source_id="alice_1",
                target_id="bob_1",
                relationship="related_to",
                weight=0.6,
                context="apple trade",
            )],
            association_updates=[AssociationUpdateSpec(
                assoc_id="assoc_1",
                delta=0.2,
            )],
        ))

        initial_entry_count = len(populated_store.entries)
        initial_assoc_count = len(populated_store.associations)

        await step.execute(
            objective="Find trade relationships",
            state=populated_store,
            context={"input": "test", "step_number": 5},
        )

        # New entry was created
        assert len(populated_store.entries) == initial_entry_count + 1
        # New association was created
        assert len(populated_store.associations) == initial_assoc_count + 1
        # Existing association was strengthened
        assert populated_store.associations["assoc_1"].weight == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_max_steps_safety_cap(self, mock_llm: MockLLM, populated_store: StateStore):
        """Traversal stops at max_steps even if LLM never says stop."""
        step = CognitiveStep(mock_llm, max_width=3, max_steps=3)

        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1"],
            sub_objectives=[],
        ))

        # LLM never stops — always wants to continue
        mock_llm.set_response("TraversalStepResponse", TraversalStepResponse(
            findings=["Still looking..."],
            next_nodes=["alice_1"],
            should_stop=False,
        ))

        await step.execute(
            objective="Infinite search",
            state=populated_store,
            context={"input": "test", "step_number": 0},
        )

        # Should have made exactly max_steps traversal LLM calls
        traversal_calls = [c for c in mock_llm.call_log if c.get("model") == "TraversalStepResponse"]
        assert len(traversal_calls) == 3


class TestDecomposition:
    """Tests for orient → multi-traverse → synthesize."""

    @pytest.mark.asyncio
    async def test_sub_objectives_each_get_traversed(self, mock_llm: MockLLM, populated_store: StateStore, tracer: TraceLogger):
        """Orient produces sub-objectives, each gets its own graph traversal."""
        step = CognitiveStep(mock_llm, max_width=3, max_steps=5)

        # Orient decomposes into 2 sub-objectives
        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1", "bob_1"],
            sub_objectives=["find alice's fruits", "find bob's fruits"],
        ))

        mock_llm.set_response_sequence("TraversalStepResponse", [
            TraversalStepResponse(findings=["Alice has 5 apples"], should_stop=True, stop_reason="objective_satisfied"),
            TraversalStepResponse(findings=["Bob has 3 oranges"], should_stop=True, stop_reason="objective_satisfied"),
        ])

        mock_llm.set_response("SynthesisResponse", SynthesisResponse(
            output="Alice has 5 apples and Bob has 3 oranges",
        ))

        result = await step.execute(
            objective="What fruits does everyone have?",
            state=populated_store,
            context={"input": "test", "step_number": 0},
            tracer=tracer,
        )

        assert "Alice" in result.output or "Bob" in result.output

        # Check trace: root has 2 children (one per sub-objective)
        roots = tracer.get_roots()
        assert len(roots) == 1
        root = roots[0]
        assert len(root.children) == 2

    @pytest.mark.asyncio
    async def test_max_width_limits_sub_objectives(self, mock_llm: MockLLM, populated_store: StateStore):
        """Only max_width sub-objectives are pursued."""
        step = CognitiveStep(mock_llm, max_width=2, max_steps=5)

        # Orient returns 5 sub-objectives but max_width=2
        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1"],
            sub_objectives=["a", "b", "c", "d", "e"],
        ))

        mock_llm.set_response("TraversalStepResponse", TraversalStepResponse(
            findings=["done"], should_stop=True, stop_reason="objective_satisfied",
        ))
        mock_llm.set_response("SynthesisResponse", SynthesisResponse(output="result"))

        await step.execute(
            objective="test",
            state=populated_store,
            context={"input": "test", "step_number": 0},
        )

        # Only 2 traversal calls (one per sub-objective that was pursued)
        traversal_calls = [c for c in mock_llm.call_log if c.get("model") == "TraversalStepResponse"]
        assert len(traversal_calls) == 2

    @pytest.mark.asyncio
    async def test_synthesis_always_runs(self, mock_llm: MockLLM, populated_store: StateStore):
        """Synthesis runs even when orient returns no sub-objectives."""
        step = CognitiveStep(mock_llm, max_width=3, max_steps=5)

        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1"],
            sub_objectives=[],
        ))

        mock_llm.set_response("TraversalStepResponse", TraversalStepResponse(
            findings=["Alice has 5 apples"], should_stop=True, stop_reason="objective_satisfied",
        ))

        mock_llm.set_response("SynthesisResponse", SynthesisResponse(output="done"))

        await step.execute(
            objective="test",
            state=populated_store,
            context={"input": "test", "step_number": 0},
        )

        # Synthesis always runs — it's where memory map gets organized
        synthesis_calls = [c for c in mock_llm.call_log if c.get("model") == "SynthesisResponse"]
        assert len(synthesis_calls) == 1


class TestTracing:
    """Tests for trace capture during cognitive step execution."""

    @pytest.mark.asyncio
    async def test_trace_captures_full_tree(self, mock_llm: MockLLM, populated_store: StateStore, tracer: TraceLogger):
        step = CognitiveStep(mock_llm, max_width=3, max_steps=5)

        mock_llm.set_response("OrientationResponse", OrientationResponse(
            entry_points=["alice_1"], sub_objectives=[],
        ))
        mock_llm.set_response("TraversalStepResponse", TraversalStepResponse(
            findings=["Found alice"], should_stop=True, stop_reason="objective_satisfied",
        ))

        await step.execute(
            objective="test", state=populated_store,
            context={"input": "test", "step_number": 0}, tracer=tracer,
        )

        summary = tracer.summary()
        assert summary["total_cognitive_steps"] >= 1
        assert summary["total_llm_calls"] >= 2  # orient + traversal
        assert summary["direct_resolves"] >= 1
