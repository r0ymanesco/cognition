"""Tracing and observability for the cognitive process.

Every cognitive step produces a trace — a structured record of the full
recursion and traversal tree. This captures: how objectives were decomposed,
how the graph was navigated, what state mutations occurred, why traversal
stopped, and how many LLM calls were made.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class TraversalStepTrace:
    """Record of a single graph traversal step at the base case."""

    step_number: int
    current_nodes: list[str]
    findings: list[str] = field(default_factory=list)
    edges_followed: list[str] = field(default_factory=list)
    entries_written: list[str] = field(default_factory=list)
    associations_created: list[str] = field(default_factory=list)
    weight_updates: list[dict] = field(default_factory=list)
    associations_invalidated: list[str] = field(default_factory=list)
    stop_decision: str | None = None
    next_nodes: list[str] = field(default_factory=list)


@dataclass
class CognitiveTrace:
    """A single node in the recursion tree."""

    trace_id: str
    parent_trace_id: str | None
    depth: int
    objective: str
    resolved_directly: bool = False

    # Orient phase
    memory_map_snapshot: str = ""
    entry_points: list[str] = field(default_factory=list)
    sub_objectives: list[str] = field(default_factory=list)

    # Traversal (if resolved directly)
    traversal_steps: list[TraversalStepTrace] = field(default_factory=list)
    total_nodes_visited: int = 0

    # Synthesis
    synthesis_output: str = ""

    # Metadata
    llm_calls: int = 0
    start_time: float = 0.0
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Children in the recursion tree
    children: list[CognitiveTrace] = field(default_factory=list)


class TraceLogger:
    """Collects traces during cognitive step execution.

    Usage:
        tracer = TraceLogger()
        trace_id = tracer.begin("recall", depth=0)
        tracer.record_orient(trace_id, memory_map, sub_objectives)
        # ... recursion and traversal happen ...
        tracer.end(trace_id)

        tracer.print_tree()
        tracer.export_json("trace.json")
    """

    def __init__(self) -> None:
        self._traces: dict[str, CognitiveTrace] = {}
        self._root_ids: list[str] = []
        self._llm_call_counts: dict[str, int] = {}

    def begin(
        self,
        objective: str,
        depth: int,
        parent_trace_id: str | None = None,
    ) -> str:
        trace_id = str(uuid4())
        trace = CognitiveTrace(
            trace_id=trace_id,
            parent_trace_id=parent_trace_id,
            depth=depth,
            objective=objective,
            start_time=time.monotonic(),
        )
        self._traces[trace_id] = trace
        self._llm_call_counts[trace_id] = 0

        if parent_trace_id and parent_trace_id in self._traces:
            self._traces[parent_trace_id].children.append(trace)
        else:
            self._root_ids.append(trace_id)

        logger.debug("trace.begin depth=%d objective=%r trace_id=%s", depth, objective, trace_id)
        return trace_id

    def record_orient(
        self,
        trace_id: str,
        memory_map: str,
        entry_points: list[str],
        sub_objectives: list[str],
    ) -> None:
        trace = self._traces[trace_id]
        trace.memory_map_snapshot = memory_map
        trace.entry_points = entry_points
        trace.sub_objectives = sub_objectives
        logger.debug(
            "trace.orient trace_id=%s entry_points=%d sub_objectives=%d",
            trace_id, len(entry_points), len(sub_objectives),
        )

    def record_traversal_step(
        self,
        trace_id: str,
        step: TraversalStepTrace,
    ) -> None:
        trace = self._traces[trace_id]
        trace.resolved_directly = True
        trace.traversal_steps.append(step)
        trace.total_nodes_visited += len(step.current_nodes)
        logger.debug(
            "trace.traversal trace_id=%s step=%d nodes=%s stop=%s",
            trace_id, step.step_number, step.current_nodes, step.stop_decision,
        )

    def record_direct_resolve(
        self,
        trace_id: str,
        visited: set[str],
        findings: list[str],
        entries_written: list[str],
        associations_invalidated: list[str],
    ) -> None:
        trace = self._traces[trace_id]
        trace.resolved_directly = True
        trace.total_nodes_visited = len(visited)
        logger.debug(
            "trace.direct_resolve trace_id=%s visited=%d findings=%d written=%d",
            trace_id, len(visited), len(findings), len(entries_written),
        )

    def record_synthesis(self, trace_id: str, output: str) -> None:
        trace = self._traces[trace_id]
        trace.synthesis_output = output
        logger.debug("trace.synthesis trace_id=%s output_len=%d", trace_id, len(output))

    def record_llm_call(self, trace_id: str) -> None:
        """Increment LLM call counter for a trace node."""
        self._llm_call_counts[trace_id] = self._llm_call_counts.get(trace_id, 0) + 1
        self._traces[trace_id].llm_calls += 1

    def end(self, trace_id: str) -> None:
        trace = self._traces[trace_id]
        trace.duration_ms = (time.monotonic() - trace.start_time) * 1000
        logger.debug(
            "trace.end trace_id=%s duration_ms=%.1f llm_calls=%d",
            trace_id, trace.duration_ms, trace.llm_calls,
        )

    # --- Analysis ---

    def get_roots(self) -> list[CognitiveTrace]:
        return [self._traces[tid] for tid in self._root_ids if tid in self._traces]

    def get_all_traces(self) -> list[CognitiveTrace]:
        """All trace nodes in depth-first order."""
        result: list[CognitiveTrace] = []
        for root in self.get_roots():
            self._collect_dfs(root, result)
        return result

    def _collect_dfs(self, node: CognitiveTrace, result: list[CognitiveTrace]) -> None:
        result.append(node)
        for child in node.children:
            self._collect_dfs(child, result)

    def summary(self) -> dict:
        all_traces = self.get_all_traces()
        total_llm_calls = sum(t.llm_calls for t in all_traces)
        total_duration = sum(t.duration_ms for t in all_traces if t.parent_trace_id is None)
        max_depth = max((t.depth for t in all_traces), default=0)
        total_traversal_steps = sum(len(t.traversal_steps) for t in all_traces)
        direct_resolves = sum(1 for t in all_traces if t.resolved_directly)

        return {
            "total_cognitive_steps": len(all_traces),
            "total_llm_calls": total_llm_calls,
            "total_duration_ms": total_duration,
            "max_recursion_depth": max_depth,
            "total_traversal_steps": total_traversal_steps,
            "direct_resolves": direct_resolves,
        }

    def print_tree(self, indent: int = 2) -> None:
        for root in self.get_roots():
            self._print_node(root, level=0, indent=indent)

    def _print_node(self, node: CognitiveTrace, level: int, indent: int) -> None:
        prefix = " " * (level * indent)
        resolved = "DIRECT" if node.resolved_directly else f"{len(node.children)} children"
        print(
            f"{prefix}[d{node.depth}] {node.objective} "
            f"({resolved}, {node.llm_calls} llm calls, {node.duration_ms:.0f}ms)"
        )
        if node.traversal_steps:
            for step in node.traversal_steps:
                step_prefix = " " * ((level + 1) * indent)
                stop = f" STOP: {step.stop_decision}" if step.stop_decision else ""
                print(f"{step_prefix}step {step.step_number}: nodes={step.current_nodes}{stop}")
        for child in node.children:
            self._print_node(child, level + 1, indent)

    def export_json(self, path: str) -> None:
        import dataclasses

        def _serialize(obj: object) -> object:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                d = {}
                for f in dataclasses.fields(obj):
                    val = getattr(obj, f.name)
                    d[f.name] = _serialize(val)
                return d
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, set):
                return sorted(str(_serialize(v)) for v in obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        roots = self.get_roots()
        data = [_serialize(r) for r in roots]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def reset(self) -> None:
        self._traces.clear()
        self._root_ids.clear()
        self._llm_call_counts.clear()
