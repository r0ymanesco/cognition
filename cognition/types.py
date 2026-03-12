"""Core types for the cognition scaffold.

These are the fundamental data structures: entries (graph nodes),
associations (graph edges), memory map (routing layer), and
result types for the cognitive step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class EntryType(str, Enum):
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    DECISION = "decision"
    OBSERVATION = "observation"


class RelationshipType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    RELATED_TO = "related_to"
    PART_OF = "part_of"


# ---------------------------------------------------------------------------
# State graph: entries (nodes) and associations (edges)
# ---------------------------------------------------------------------------


@dataclass
class StateEntry:
    """A node in the state graph. Represents a piece of information."""

    content: str
    entry_type: EntryType
    confidence: float = 1.0
    step_created: int = 0
    step_last_accessed: int = 0
    access_count: int = 0
    superseded_by: str | None = None
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_active(self) -> bool:
        return self.superseded_by is None


@dataclass
class Association:
    """An edge in the state graph. Represents a relationship between entries.

    Associations are first-class, context-scoped, and have a lifecycle:
    - Created with initial weight
    - Strengthened when found useful during cognitive traversal
    - Weakened when found misleading
    - Invalidated when contradicted by new evidence

    Multiple associations can exist between the same pair of entries,
    each in a different context (e.g., "Alice-Bob" can be linked in
    "apple trades" context and separately in "family" context).
    """

    source_id: str
    target_id: str
    relationship: RelationshipType
    weight: float
    context: str
    valid: bool = True
    invalidation_reason: str | None = None
    invalidated_by_entry: str | None = None
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


@dataclass
class OrientationResult:
    """Output of the orient phase — entry points and sub-objectives."""

    entry_points: list[str] = field(default_factory=list)
    sub_objectives: list[str] = field(default_factory=list)


@dataclass
class TraversalStepResult:
    """Output of a single graph traversal step during direct resolve."""

    findings: list[str] = field(default_factory=list)
    next_nodes: list[str] = field(default_factory=list)
    should_stop: bool = False
    stop_reason: str | None = None

    # State mutations
    new_entries: list[StateEntry] = field(default_factory=list)
    new_associations: list[Association] = field(default_factory=list)
    weight_updates: list[WeightUpdate] = field(default_factory=list)
    association_invalidations: list[AssociationInvalidation] = field(default_factory=list)
    map_changes: dict = field(default_factory=dict)


@dataclass
class WeightUpdate:
    """A weight adjustment on an existing association."""

    assoc_id: str
    delta: float  # positive = strengthen, negative = weaken
    reason: str = ""


@dataclass
class AssociationInvalidation:
    """An invalidation of an existing association."""

    assoc_id: str
    reason: str
    invalidated_by: str | None = None
