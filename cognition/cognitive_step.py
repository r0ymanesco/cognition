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
    relationship_filter: list[str] = Field(
        default_factory=list,
        description="Relationship types to show during traversal. "
        "Pick from the relationship types listed in the memory map. "
        "Empty list means show all relationships.",
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
    supersedes_entry_id: str | None = Field(
        default=None,
        description="If this entry replaces/corrects an existing entry, "
        "put the old entry's ID here. The old entry will be marked as superseded.",
    )


class NewAssociationSpec(BaseModel):
    source_id: str
    target_id: str
    relationship: str = "related_to"
    weight: float = 0.3
    context: str = ""


class AssociationUpdateSpec(BaseModel):
    """Update an existing association's weight and/or context."""
    assoc_id: str
    delta: float = 0.0
    append_context: str | None = Field(
        default=None,
        description="Text to append to the association's context. "
        "Use this to record new events or reasons related to this association.",
    )


class NamedTopicSpec(BaseModel):
    name: str
    summary: str = ""
    entry_points: list[str] = Field(default_factory=list)
    density: str = "sparse"


class NamedTopicUpdateSpec(BaseModel):
    name: str
    summary: str | None = None
    add_entry_points: list[str] = Field(default_factory=list)
    entry_points: list[str] | None = None
    density: str | None = None


class MapChangesSpec(BaseModel):
    new_topics: list[NamedTopicSpec] = Field(default_factory=list)
    updated_topics: list[NamedTopicUpdateSpec] = Field(default_factory=list)
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
    association_updates: list[AssociationUpdateSpec] = Field(default_factory=list)
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
        max_width: int = 5,
        max_steps: int = 20,
        context_budget: int | None = None,
    ):
        self.llm = llm
        self.max_width = max_width
        self.max_steps = max_steps
        self.context_budget = context_budget

    def _enforce_budget(self, messages: list[dict[str, str]], system: str, label: str) -> None:
        """Raise if total message content exceeds the context budget.

        Checks the full message (system + all user/assistant messages),
        matching what the baseline constrains — total tokens sent to the LLM.
        """
        if self.context_budget is None:
            return
        total = system + " ".join(m.get("content", "") for m in messages)
        estimated = _estimate_tokens(total)
        if estimated > self.context_budget:
            raise RuntimeError(
                f"Context budget exceeded in {label}: ~{estimated} tokens "
                f"(budget: {self.context_budget})."
            )

    def _budget_section(self, total_tokens: int) -> str:
        """Build a context budget awareness section for an LLM prompt.

        Shows the total assembled message size vs budget so the LLM can
        manage context proactively.
        """
        if self.context_budget is None:
            return ""

        remaining = self.context_budget - total_tokens
        return (
            f"CONTEXT BUDGET: ~{total_tokens}/{self.context_budget} tokens used. "
            f"~{remaining} tokens remaining.\n"
            "Stay within budget. To manage context:\n"
            "- Keep memory map topics concise (merge related topics, shorten summaries)\n"
            "- Follow only the most relevant edges during traversal\n"
            "- Create compact entries (essential facts only, not verbose descriptions)\n"
            "- Drop low-value information (filler, one-time observations)\n\n"
        )

    async def execute(
        self,
        objective: str,
        state: StateStore,
        context: dict[str, Any],
        tracer: TraceLogger | None = None,
        parent_trace_id: str | None = None,
    ) -> CognitiveResult:
        """Orient once → traverse per sub-objective → synthesize.

        No recursive decomposition. Orient identifies entry points and
        optionally splits the objective into focused sub-objectives.
        Each sub-objective gets its own graph traversal. Synthesis
        combines results and updates the memory map.
        """
        trace_id = None
        if tracer:
            trace_id = tracer.begin(objective, 0, parent_trace_id)

        try:
            # Orient: consult memory map, find entry points, decompose objective
            orientation = await self._orient(objective, state, context, tracer, trace_id)

            # Collect traversal findings to pass to synthesis
            all_findings: list[str] = []

            traverse_context = {
                **context,
                "entry_points": orientation.entry_points,
                "relationship_filter": orientation.relationship_filter,
            }

            if not orientation.sub_objectives:
                # Single focused objective — traverse directly
                traverse_result = await self._direct_resolve(
                    objective, state, traverse_context, tracer, trace_id,
                )
                if traverse_result.output:
                    all_findings.append(traverse_result.output)
            else:
                # Multiple sub-objectives — traverse each separately
                sub_objectives = orientation.sub_objectives[:self.max_width]
                for sub_objective in sub_objectives:
                    sub_trace_id = None
                    if tracer:
                        sub_trace_id = tracer.begin(sub_objective, 1, trace_id)
                    try:
                        traverse_result = await self._direct_resolve(
                            sub_objective, state, traverse_context,
                            tracer, sub_trace_id,
                        )
                        if traverse_result.output:
                            all_findings.append(traverse_result.output)
                    finally:
                        if tracer and sub_trace_id:
                            tracer.end(sub_trace_id)

            # Always synthesize — compile response and update memory map
            result = await self._synthesize(
                objective, state, context, all_findings, tracer, trace_id,
            )
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
        all_entry_points = state.memory_map.get_entry_points(store=state)

        system = _get_task_prompt(context)

        content_without_budget = (
            "You have a knowledge graph that stores information as entries (nodes) "
            "and associations (edges). You received the following input.\n\n"
            f"Input: {objective}\n\n"
            f"Memory Map:\n{memory_map}\n\n"
            f"Available entry points: {all_entry_points}\n\n"
            "Based on the input and your memory map, identify:\n"
            "1. entry_points: Which existing entries are relevant to this input? "
            "Use entry IDs from the available entry points list. "
            "If the memory is empty or no entries are relevant, return an empty list.\n"
            "2. sub_objectives: If this input involves multiple distinct topics "
            "that should be handled separately, list them. "
            "Otherwise return an empty list.\n"
            "3. reasoning: Briefly explain your choices.\n\n"
            f"Agent context: {_format_context(context)}"
        )
        total_tokens = _estimate_tokens(system + content_without_budget)
        budget_section = self._budget_section(total_tokens)

        messages = [
            {
                "role": "user",
                "content": content_without_budget + budget_section,
            }
        ]

        self._enforce_budget(messages, system, "orient")

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
        relationship_filter = context.get("relationship_filter", [])
        visited: set[str] = set()
        findings: list[str] = []
        all_entries_written: list[str] = []
        step_number = context.get("step_number", 0)

        for step in range(self.max_steps):
            # Render the local neighborhood
            neighborhood = state.render_neighborhood(
                current_nodes, relationship_filter=relationship_filter,
            )

            system = _get_task_prompt(context)

            visited_list = sorted(visited)
            content_without_budget = (
                "You are processing an input using a knowledge graph.\n\n"
                "THE GRAPH:\n"
                "- Entries are nodes (facts, observations, decisions, etc.)\n"
                "- Associations are edges with free-form relationship types "
                "(e.g., 'traded_with', 'received_from', 'supersedes', 'part_of')\n"
                "- Each association has a weight (0-1) and a context string that "
                "records WHY the association exists and any subsequent events\n"
                "- Two entries can have multiple associations with different relationships\n"
                "- REUSE existing relationship types from the memory map when possible "
                "— avoid creating near-duplicates like 'traded_with' and 'gave_items_to'\n"
                "- When creating associations, provide meaningful context explaining "
                "what led to this association (e.g., 'Trade at step 15: Alice gave "
                "3 apples to Bob')\n"
                "- You see the current entries fully + a compact edge list showing "
                "available connections. Use next_nodes to follow edges.\n\n"
                "YOUR JOB:\n"
                "- STORE: create new entries for new information from the input\n"
                "- CONNECT: create associations between related entries. Name the "
                "relationship descriptively (not just 'related_to')\n"
                "- RETRIEVE: follow edges to find information relevant to the input\n"
                "- ANSWER: if the input is a question, put the answer in findings\n"
                "- CORRECT: if the input corrects a previous fact, create a new entry "
                "with supersedes_entry_id set to the old entry's ID\n"
                "- If the graph is empty, create entries from the input — that IS the work\n\n"
                f"Input: {objective}\n\n"
                f"Current neighborhood:\n{neighborhood}\n\n"
                f"Already visited: {visited_list}\n\n"
                f"Findings so far: {findings}\n\n"
                f"Agent context: {_format_context(context)}\n\n"
                f"Traversal step {step + 1}/{self.max_steps}.\n\n"
                "Decide what to do. Set should_stop=true when done processing "
                "this input. Use next_nodes to follow edges to related entries."
            )
            total_tokens = _estimate_tokens(system + content_without_budget)
            budget_section = self._budget_section(total_tokens)

            messages = [
                {
                    "role": "user",
                    "content": content_without_budget + budget_section,
                }
            ]

            self._enforce_budget(messages, system, "traverse")

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
                        weight_updates=[{"id": u.assoc_id, "delta": u.delta} for u in step_result.association_updates],
                        associations_invalidated=[],
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

        # Memory map updates happen in synthesis, not during traversal

        if tracer and trace_id:
            tracer.record_direct_resolve(
                trace_id, visited, findings,
                all_entries_written, [],
            )

        return CognitiveResult(
            output="\n".join(findings) if findings else "",
            entries_written=all_entries_written,
            associations_invalidated=[],
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
                entry_type=spec.entry_type,
                confidence=spec.confidence,
                step_created=step_number,
                tags=spec.tags,
            )
            state.write(entry)
            written_ids.append(entry.id)

            # If this entry supersedes an old one, create a supersedes association
            if spec.supersedes_entry_id:
                state.supersede_entry(spec.supersedes_entry_id, entry.id, step_number)

        for spec in step_result.new_associations:
            assoc = Association(
                source_id=spec.source_id,
                target_id=spec.target_id,
                relationship=spec.relationship,
                weight=spec.weight,
                context=spec.context,
                step_created=step_number,
            )
            state.add_association(assoc)

        for spec in step_result.association_updates:
            if spec.delta > 0:
                state.strengthen(spec.assoc_id, spec.delta, step_number)
            elif spec.delta < 0:
                state.weaken(spec.assoc_id, abs(spec.delta), step_number)
            if spec.append_context:
                state.append_association_context(spec.assoc_id, spec.append_context)

        return written_ids

    async def _synthesize(
        self,
        objective: str,
        state: StateStore,
        context: dict[str, Any],
        findings: list[str],
        tracer: TraceLogger | None,
        trace_id: str | None,
    ) -> CognitiveResult:
        """After traversal, synthesize a response and organize the memory map."""
        memory_map = state.memory_map.render()

        # Show traversal findings so the LLM can base its response on them
        if findings:
            findings_section = (
                "Findings from graph traversal:\n"
                + "\n".join(f"- {f}" for f in findings)
                + "\n\n"
            )
        else:
            findings_section = "No findings from traversal.\n\n"

        # Show the actual content of weakly_connected entries so the LLM
        # can reason about how to organize them into topics
        weakly_connected = state.memory_map.data.weakly_connected
        if weakly_connected:
            wc_entries = state.render_entries(weakly_connected)
            wc_section = (
                f"Entries not yet organized into topics ({len(weakly_connected)}):\n"
                f"{wc_entries}\n\n"
                "IMPORTANT: Organize these entries into topics using map_changes. "
                "For each group of related entries, create a new_topics entry with:\n"
                "- A descriptive topic name\n"
                "- A brief summary of what the topic covers\n"
                "- The entry IDs that belong to this topic\n"
                "Then list those entry IDs in remove_weakly_connected so they "
                "are no longer unorganized.\n"
            )
        else:
            wc_section = ""

        system = _get_task_prompt(context)

        content_without_budget = (
            "The knowledge graph has been traversed. During traversal, "
            "new entries and associations may have been created, and old "
            "entries may have been superseded.\n\n"
            "Now:\n"
            "1. Compile a response to the input based on the findings below. "
            "If it was a statement, acknowledge briefly. "
            "If it was a question, answer it using the findings.\n"
            "2. Organize the memory map — group related entries into topics. "
            "Topics should reflect the graph's structure: entities, "
            "relationships between entities, trade histories, corrections, etc. "
            "Keep topic summaries concise and include relevant entry point IDs.\n\n"
            f"Input: {objective}\n\n"
            f"{findings_section}"
            f"Current memory map:\n{memory_map}\n\n"
            f"{wc_section}"
            f"Agent context: {_format_context(context)}"
        )
        total_tokens = _estimate_tokens(system + content_without_budget)
        budget_section = self._budget_section(total_tokens)

        messages = [
            {
                "role": "user",
                "content": content_without_budget + budget_section,
            }
        ]

        self._enforce_budget(messages, system, "synthesis")

        if tracer and trace_id:
            tracer.record_llm_call(trace_id)

        result = await self.llm.generate_structured(
            messages=messages,
            response_model=SynthesisResponse,
            system=system,
        )

        if result.map_changes:
            state.memory_map.update(_map_changes_to_dict(result.map_changes))

        if tracer and trace_id:
            tracer.record_synthesis(trace_id, result.output)

        return CognitiveResult(output=result.output)


def _map_changes_to_dict(spec: MapChangesSpec) -> dict[str, Any]:
    """Convert list-based MapChangesSpec to the dict format MemoryMap.update() expects."""
    result: dict[str, Any] = {}
    if spec.new_topics:
        result["new_topics"] = {
            t.name: {"summary": t.summary, "entry_points": t.entry_points, "density": t.density}
            for t in spec.new_topics
        }
    if spec.updated_topics:
        updates: dict[str, dict[str, Any]] = {}
        for t in spec.updated_topics:
            u: dict[str, Any] = {}
            if t.summary is not None:
                u["summary"] = t.summary
            if t.entry_points is not None:
                u["entry_points"] = t.entry_points
            if t.add_entry_points:
                u["add_entry_points"] = t.add_entry_points
            if t.density is not None:
                u["density"] = t.density
            updates[t.name] = u
        result["updated_topics"] = updates
    if spec.recent_changes:
        result["recent_changes"] = spec.recent_changes
    if spec.contested_regions:
        result["contested_regions"] = spec.contested_regions
    if spec.weakly_connected:
        result["weakly_connected"] = spec.weakly_connected
    if spec.remove_weakly_connected:
        result["remove_weakly_connected"] = spec.remove_weakly_connected
    return result


def _estimate_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token)."""
    return len(text) // 4


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
