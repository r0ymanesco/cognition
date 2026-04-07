"""Core types for the cognition scaffold.

These are the fundamental data structures: entries (graph nodes),
associations (graph edges), memory map (routing layer), and
result types for the cognitive step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


# ---------------------------------------------------------------------------
# State graph: entries (nodes) and associations (edges)
# ---------------------------------------------------------------------------


@dataclass
class StateEntry:
    """A node in the state graph. Represents a piece of information."""

    content: str
    entry_type: str = "observation"
    confidence: float = 1.0
    step_created: int = 0
    step_last_accessed: int = 0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Association:
    """An edge in the state graph. Represents a relationship between entries.

    Relationships are free-form strings — the LLM creates whatever
    relationship types make sense for the domain (e.g., "traded_with",
    "supersedes", "corrected_by", "received_from", "part_of").

    Multiple associations can exist between the same pair of entries,
    each with a different relationship or context.
    """

    source_id: str
    target_id: str
    relationship: str
    weight: float = 0.5
    context: str = ""
    step_created: int = 0
    step_last_accessed: int = 0
    id: str = field(default_factory=lambda: str(uuid4()))


# ---------------------------------------------------------------------------
# Memory map: compact routing layer over the state graph
# ---------------------------------------------------------------------------


@dataclass
class TopicEntry:
    """A topic in the memory map with entry points into the graph."""

    summary: str
    entry_points: list[str] = field(default_factory=list)
    density: str = "sparse"  # "dense" | "sparse"


@dataclass
class MemoryMapData:
    """Raw data for the memory map. The MemoryMap class in state.py
    wraps this with render/update logic."""

    topics: dict[str, TopicEntry] = field(default_factory=dict)
    recent_changes: list[str] = field(default_factory=list)
    contested_regions: list[str] = field(default_factory=list)
    weakly_connected: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cognitive step result types
# ---------------------------------------------------------------------------


@dataclass
class CognitiveResult:
    """Output of a cognitive step execution."""

    output: str = ""
    entries_written: list[str] = field(default_factory=list)
    entries_invalidated: list[str] = field(default_factory=list)
    associations_created: list[str] = field(default_factory=list)
    associations_invalidated: list[str] = field(default_factory=list)
