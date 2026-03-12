"""The Cognitive Step — the single recursive primitive.

Every cognitive act (recall, reason, integrate) is an instance of this
mechanism with a different objective. The structure is:

    orient → recurse on sub-objectives → synthesize

At the base case (max depth), it traverses the graph directly with
LLM-guided navigation: see local neighborhood, decide which edges
to follow, assess termination, read/write state.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from cognition.llm.base import BaseLLM
from cognition.state import StateStore
from cognition.tracing import TraceLogger, TraversalStepTrace
from cognition.types import (
    Association,
    CognitiveResult,
    EntryType,
    RelationshipType,
    StateEntry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class OrientationResponse(BaseModel):
    """LLM output from the orient phase."""

    entry_points: list[str] = Field(
        default_factory=list,
        description="Entry IDs from the memory map to start traversal from",
    )
    sub_objectives: list[str] = Field(
        default_factory=list,
        description="Sub-objectives to pursue — narrower aspects of the main objective",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why these entry points and sub-objectives were chosen",
    )


class NewEntrySpec(BaseModel):
    content: str
    entry_type: str = "observation"
    confidence: float = 1.0
    tags: list[str] = Field(default_factory=list)


class NewAssociationSpec(BaseModel):
    source_id: str
    target_id: str
    relationship: str = "related_to"
    weight: float = 0.3
    context: str = ""


class WeightUpdateSpec(BaseModel):
    assoc_id: str
    delta: float
    reason: str = ""


class AssociationInvalidationSpec(BaseModel):
    assoc_id: str
    reason: str
    invalidated_by: str | None = None


class MapChangesSpec(BaseModel):
    new_topics: dict[str, dict] = Field(default_factory=dict)
    updated_topics: dict[str, dict] = Field(default_factory=dict)
    recent_changes: list[str] = Field(default_factory=list)
    contested_regions: list[str] = Field(default_factory=list)
    weakly_connected: list[str] = Field(default_factory=list)
    remove_weakly_connected: list[str] = Field(default_factory=list)


class TraversalStepResponse(BaseModel):
    """LLM output from a single graph traversal step."""

    findings: list[str] = Field(
        default_factory=list,
        description="Information found at the current nodes relevant to the objective",
    )
    next_nodes: list[str] = Field(
        default_factory=list,
        description="Entry IDs to visit next (follow these edges)",
    )
    should_stop: bool = Field(
        default=False,
        description="Whether to stop traversal",
    )
    stop_reason: str | None = Field(
        default=None,
        description="Why stopping: loop_detected, objective_satisfied, diverging_from_objective, no_relevant_edges",
    )

    # State mutations
    new_entries: list[NewEntrySpec] = Field(default_factory=list)
    new_associations: list[NewAssociationSpec] = Field(default_factory=list)
    weight_updates: list[WeightUpdateSpec] = Field(default_factory=list)
    association_invalidations: list[AssociationInvalidationSpec] = Field(default_factory=list)
    map_changes: MapChangesSpec = Field(default_factory=MapChangesSpec)


class SynthesisResponse(BaseModel):
    """LLM output from the synthesis phase."""

    output: str = Field(
        default="",
        description="Synthesized result for this level's objective",
    )
    map_changes: MapChangesSpec = Field(default_factory=MapChangesSpec)


# ---------------------------------------------------------------------------
# The Cognitive Step
# ---------------------------------------------------------------------------


class CognitiveStep:
    """The single recursive primitive.

    orient → recurse on sub-objectives → synthesize

    At the base case, traverses the graph with LLM-guided navigation.
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_depth: int = 3,
        max_width: int = 5,
        max_steps: int = 20,
    ):
        self.llm = llm
        self.max_depth = max_depth
        self.max_width = max_width
        self.max_steps = max_steps

    async def execute(
        self,
        objective: str,
        state: StateStore,
        context: dict[str, Any],
        depth: int = 0,
        tracer: TraceLogger | None = None,
        parent_trace_id: str | None = None,
    ) -> CognitiveResult:
        trace_id = None
        if tracer:
            trace_id = tracer.begin(objective, depth, parent_trace_id)

        try:
            # Hard safety cap — always go to direct resolve at max depth
            if depth >= self.max_depth:
                result = await self._direct_resolve(
                    objective, state, context, tracer, trace_id
                )
                return result

            # Orient: consult memory map, identify entry points and sub-objectives
            orientation = await self._orient(objective, state, context, tracer, trace_id)

            if not orientation.sub_objectives:
                # No sub-objectives — direct resolve at this level
                result = await self._direct_resolve(
                    objective, state,
                    {**context, "entry_points": orientation.entry_points},
                    tracer, trace_id,
                )
                return result

            # Recurse on each sub-objective
            sub_objectives = orientation.sub_objectives[:self.max_width]
            for sub_objective in sub_objectives:
                await self.execute(
                    objective=sub_objective,
                    state=state,
                    context={**context, "entry_points": orientation.entry_points},
                    depth=depth + 1,
                    tracer=tracer,
                    parent_trace_id=trace_id,
                )

            # Synthesize
            result = await self._synthesize(objective, state, context, tracer, trace_id)
            return result

        finally:
            if tracer and trace_id:
                tracer.end(trace_id)

    async def _orient(
        self,
        objective: str,
        state: StateStore,
        context: dict[str, Any],
        tracer: TraceLogger | None,
        trace_id: str | None,
    ) -> OrientationResponse:
        memory_map = state.memory_map.render()
        all_entry_points = state.memory_map.get_entry_points(objective)

        task_prompt = _get_task_prompt(context)
        system = (
            f"{task_prompt}\n\n" if task_prompt else ""
        ) + (
            "You are a cognitive navigator. Given an objective and a memory map, "
            "identify which entry points to start from and decompose the objective "
            "into sub-objectives.\n\n"
            "If the objective is narrow enough to address directly, return an empty "
            "sub_objectives list — this signals direct resolution.\n\n"
            "Entry points must be valid entry IDs from the memory map."
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Objective: {objective}\n\n"
                    f"Memory Map:\n{memory_map}\n\n"
                    f"Available entry points: {all_entry_points}\n\n"
                    f"Context: {_format_context(context)}\n\n"
                    "Identify entry points and sub-objectives."
                ),
            }
        ]

        if tracer and trace_id:
            tracer.record_llm_call(trace_id)

        result = await self.llm.generate_structured(
            messages=messages,
            response_model=OrientationResponse,
            system=system,
        )

        if tracer and trace_id:
            tracer.record_orient(
                trace_id, memory_map,
                result.entry_points, result.sub_objectives,
            )

        logger.debug(
            "orient objective=%r entry_points=%d sub_objectives=%d",
            objective, len(result.entry_points), len(result.sub_objectives),
        )
        return result

    async def _direct_resolve(
        self,
        objective: str,
        state: StateStore,
        context: dict[str, Any],
        tracer: TraceLogger | None,
        trace_id: str | None,
    ) -> CognitiveResult:
        """Base case: traverse the graph with LLM-guided navigation."""

        entry_points = context.get("entry_points", [])
        if not entry_points:
            # Fallback: use recent entries
            recent = state.get_recent(5)
            entry_points = [e.id for e in recent]

        current_nodes = list(entry_points)
        visited: set[str] = set()
        findings: list[str] = []
        all_entries_written: list[str] = []
        all_assocs_invalidated: list[str] = []
        step_number = context.get("step_number", 0)

        for step in range(self.max_steps):
            # Render the local neighborhood
            neighborhood = state.render_neighborhood(current_nodes, depth=1)

            task_prompt = _get_task_prompt(context)
            system = (
                f"{task_prompt}\n\n" if task_prompt else ""
            ) + (
                "You are a cognitive agent traversing a knowledge graph.\n\n"
                "You see the local neighborhood around your current nodes. "
                "For each step:\n"
                "1. Record findings relevant to the objective\n"
                "2. Decide which edges to follow (next_nodes)\n"
                "3. Create new entries or associations if you discover new information\n"
                "4. Strengthen associations you find useful, weaken misleading ones\n"
                "5. Decide whether to stop, with explicit reasoning:\n"
                "   - loop_detected: you're revisiting nodes you've already seen\n"
                "   - objective_satisfied: you have enough information\n"
                "   - diverging_from_objective: the current path leads away from the goal\n"
                "   - no_relevant_edges: no useful edges to follow from here\n"
            )

            visited_list = sorted(visited)
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Objective: {objective}\n\n"
                        f"Current neighborhood:\n{neighborhood}\n\n"
                        f"Already visited: {visited_list}\n\n"
                        f"Findings so far: {findings}\n\n"
                        f"Context: {_format_context(context)}\n\n"
                        f"Traversal step {step + 1}/{self.max_steps}. "
                        "Decide what to do."
                    ),
                }
            ]

            if tracer and trace_id:
                tracer.record_llm_call(trace_id)

            step_result = await self.llm.generate_structured(
                messages=messages,
                response_model=TraversalStepResponse,
                system=system,
            )

            # Record findings
            findings.extend(step_result.findings)
            visited.update(current_nodes)

            # Execute state mutations
            written_ids = self._apply_mutations(state, step_result, step_number)
            all_entries_written.extend(written_ids)

            inv_ids = [inv.assoc_id for inv in step_result.association_invalidations]
            all_assocs_invalidated.extend(inv_ids)

            # Update access tracking
            for node_id in current_nodes:
                state.access(node_id, step_number)

            # Record traversal step in trace
            if tracer and trace_id:
                tracer.record_traversal_step(
                    trace_id,
                    TraversalStepTrace(
                        step_number=step,
                        current_nodes=list(current_nodes),
                        findings=step_result.findings,
                        entries_written=written_ids,
                        associations_created=[a.source_id + "->" + a.target_id for a in step_result.new_associations],
                        weight_updates=[{"id": w.assoc_id, "delta": w.delta} for w in step_result.weight_updates],
                        associations_invalidated=inv_ids,
                        stop_decision=step_result.stop_reason,
                        next_nodes=step_result.next_nodes,
                    ),
                )

            # Intelligent termination
            if step_result.should_stop:
                logger.debug(
                    "direct_resolve stopping at step %d: %s",
                    step, step_result.stop_reason,
                )
                break

            # Move to next nodes
            current_nodes = step_result.next_nodes
            if not current_nodes:
                logger.debug("direct_resolve stopping: no next nodes")
                break

        # Apply memory map changes from the last step
        if step_result.map_changes:
            state.memory_map.update(step_result.map_changes.model_dump())

        if tracer and trace_id:
            tracer.record_direct_resolve(
                trace_id, visited, findings,
                all_entries_written, all_assocs_invalidated,
            )

        return CognitiveResult(
            output="\n".join(findings) if findings else "",
            entries_written=all_entries_written,
            associations_invalidated=all_assocs_invalidated,
        )

    def _apply_mutations(
        self,
        state: StateStore,
        step_result: TraversalStepResponse,
        step_number: int,
    ) -> list[str]:
        """Apply state mutations from a traversal step. Returns written entry IDs."""
        written_ids: list[str] = []

        for spec in step_result.new_entries:
            entry = StateEntry(
                content=spec.content,
                entry_type=EntryType(spec.entry_type),
                confidence=spec.confidence,
                step_created=step_number,
                tags=spec.tags,
            )
            state.write(entry)
            written_ids.append(entry.id)

        for spec in step_result.new_associations:
            assoc = Association(
                source_id=spec.source_id,
                target_id=spec.target_id,
                relationship=RelationshipType(spec.relationship),
                weight=spec.weight,
                context=spec.context,
                step_created=step_number,
            )
            state.add_association(assoc)

        for spec in step_result.weight_updates:
            if spec.delta > 0:
                state.strengthen(spec.assoc_id, spec.delta, step_number)
            else:
                state.weaken(spec.assoc_id, abs(spec.delta), step_number)

        for spec in step_result.association_invalidations:
            state.invalidate_association(spec.assoc_id, spec.reason, spec.invalidated_by)

        return written_ids

    async def _synthesize(
        self,
        objective: str,
        state: StateStore,
        context: dict[str, Any],
        tracer: TraceLogger | None,
        trace_id: str | None,
    ) -> CognitiveResult:
        """After sub-objectives have executed, synthesize a result."""
        memory_map = state.memory_map.render()

        task_prompt = _get_task_prompt(context)
        system = (
            f"{task_prompt}\n\n" if task_prompt else ""
        ) + (
            "You are synthesizing the results of a cognitive process. "
            "Sub-objectives have been pursued and state has been updated. "
            "Compile a coherent result for the overall objective, and "
            "update the memory map if needed."
        )

        messages = [
            {
                "role": "user",
                "content": (
                    f"Objective: {objective}\n\n"
                    f"Current memory map:\n{memory_map}\n\n"
                    f"Context: {_format_context(context)}\n\n"
                    "Synthesize a result for this objective."
                ),
            }
        ]

        if tracer and trace_id:
            tracer.record_llm_call(trace_id)

        result = await self.llm.generate_structured(
            messages=messages,
            response_model=SynthesisResponse,
            system=system,
        )

        if result.map_changes:
            state.memory_map.update(result.map_changes.model_dump())

        if tracer and trace_id:
            tracer.record_synthesis(trace_id, result.output)

        return CognitiveResult(output=result.output)


def _get_task_prompt(context: dict[str, Any]) -> str:
    """Extract the task-level system prompt from context, if any."""
    return context.get("system_prompt", "")


def _format_context(context: dict[str, Any]) -> str:
    """Format context dict for LLM prompt, excluding large objects and system_prompt."""
    parts: list[str] = []
    for key, val in context.items():
        if key in ("system_prompt",):
            continue  # handled separately
        elif key == "entry_points":
            parts.append(f"entry_points: {val}")
        elif isinstance(val, CognitiveResult):
            parts.append(f"{key}: {val.output[:200]}...")
        elif isinstance(val, str):
            parts.append(f"{key}: {val}")
        elif isinstance(val, (int, float, bool)):
            parts.append(f"{key}: {val}")
    return "\n".join(parts) if parts else "(none)"
