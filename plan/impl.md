# Cognition Scaffold — Implementation Plan

## Context

We're building an inference-time scaffold for LLMs that provides theoretically unbounded context through externalized, structured memory. Instead of extending the context window, all working state is externalized into a queryable graph store. The LLM's context window only ever holds: current objective + memory map (compact routing layer) + locally retrieved entries.

The key differentiator from existing approaches (like RLM) is that this is **stateful and task-global** — it maintains, retrieves, and revises beliefs across arbitrarily many steps. RLM is stateless and task-local; it processes more tokens per query but has no persistent memory. This scaffold targets tasks that require belief revision, temporal dependencies, and evolving understanding over long horizons — tasks where RLM structurally fails.

## Phase 1: Core Scaffold

### Architecture

#### Core Principle: One Primitive

The scaffold has one fundamental primitive: the **Cognitive Step**. It orients via a memory map, traverses the state graph following associations, reads and writes entries, and synthesizes results. The recursion is structural — every cognitive step always recurses, terminating based on an intelligent condition (LLM-assessed, with hard caps as safety nets).

Everything is composed from this primitive. Recalling memories, reasoning about input, integrating new information — these are all instances of the same Cognitive Step with different **objectives**. The architecture is self-similar: the same mechanism operates at every level.

Reading and writing are not separate operations — they are the same cognitive process. Every cognitive step can both read from and write to state. The distinction is in the objective, not the mechanism.

#### Components

1. **State Store** — a graph of entries (nodes) and associations (edges). Associations are first-class: typed, weighted, context-scoped, with access tracking. Weights evolve through use — strengthened when useful, weakened when misleading, invalidated when contradicted. The graph co-evolves with the cognitive process (Hebbian-like: entries accessed together become linked together).

2. **Memory Map** — a compact routing layer over the state graph. Organized by entities and relationships. Provides entry points into the graph for a given objective. Also captures topology metadata (cluster density, contested regions, weakly-connected entries). Small enough to always fit in context. Updated incrementally by cognitive steps.

3. **Cognitive Step** — the single recursive primitive. Orients via memory map to find entry points, traverses the graph with LLM-guided navigation, reads/writes entries and associations at every level. Terminates based on LLM assessment of traversal state (loop detection, information sufficiency, divergence from objective) with configurable hard caps on depth and width as safety nets.

4. **LLM Interface** — provider-agnostic abstraction (Anthropic, OpenAI, OpenRouter).

5. **Agent Loop** — sequences three phases of the Cognitive Step: recall → reason → integrate. Same mechanism, different objectives.

6. **Tracing** — structured traces of the full recursion/traversal tree for observability, debugging, and analysis.

### Project Structure

```
cognition/
├── plan/
│   ├── plan.md                # Research motivation (conversation transcript)
│   └── impl.md                # This file
├── readme.md
├── pyproject.toml
├── cognition/
│   ├── __init__.py
│   ├── cognitive_step.py      # The single recursive primitive
│   ├── agent.py               # Agent loop (three phases of CognitiveStep)
│   ├── state.py               # StateStore (graph), StateEntry, Association, MemoryMap
│   ├── tracing.py             # CognitiveTrace + TraceLogger
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract LLM interface
│   │   ├── anthropic.py       # Claude implementation
│   │   ├── openai.py          # OpenAI implementation
│   │   └── openrouter.py      # OpenRouter implementation
│   └── types.py               # Shared types/dataclasses
├── experiments/
│   └── toy_task.py            # Toy task to validate scaffold
└── tests/
    ├── test_state.py
    ├── test_cognitive_step.py
    └── test_agent.py
```

### Component Details

#### 1. State Store (`cognition/state.py`)

The state store is a **graph**. Entries are nodes. Associations are edges. Both are first-class objects with rich metadata.

##### Entries (nodes)

```python
@dataclass
class StateEntry:
    id: str                           # UUID
    content: str                      # The actual information
    entry_type: str                   # "fact" | "hypothesis" | "decision" | "observation"
    confidence: float                 # 0.0 - 1.0
    step_created: int                 # when this was created
    step_last_accessed: int           # when last touched by a cognitive step
    access_count: int                 # how often this has been accessed
    superseded_by: Optional[str]      # not deleted — old entries persist
    tags: list[str]                   # semantic tags
    created_at: datetime
```

Old entries are never deleted. When new information contradicts an old entry, the old entry is marked as superseded — but preserved. The history of what was believed and when is itself valuable state. The cognitive step decides whether to use an old or new entry based on the current context.

Access tracking (`step_last_accessed`, `access_count`) gives the LLM signal about what's well-trodden vs. rarely visited — analogous to how frequently-accessed human memories are easier to recall.

##### Associations (edges)

```python
@dataclass
class Association:
    id: str
    source_id: str
    target_id: str
    relationship: str                 # "supports" | "contradicts" | "supersedes" | "related_to" | "part_of"
    weight: float                     # strength within this context — evolves through use
    context: str                      # WHY this link exists — the objective when it was created
    valid: bool                       # current belief — does this still hold in this context?
    invalidation_reason: Optional[str]  # if invalid, why
    invalidated_by_entry: Optional[str] # entry ID that disproved this
    step_created: int
    step_last_accessed: int
```

**Associations are first-class.** They have their own metadata, lifecycle, and context-scoping. Key design decisions:

- **Context-scoped**: Multiple associations can exist between the same pair of entries, each in a different context. "Alice → Bob" might be strong in "apple trades" context but invalid in "family" context. Each is a separate Association object with its own weight and validity.

- **Weight evolves through cognitive use, not automatically.** Strengthening is not automatic on access — it's a cognitive decision. When the LLM follows an association during traversal, it evaluates:
  - Was this association useful for the current objective? → strengthen
  - Was it irrelevant? → no change
  - Was it actively misleading? → weaken
  - Is it contradicted by new information? → invalidate

- **Both `weight` and `valid` are tracked.** Weight is historical (how frequently used and found useful). Validity is current belief (does this still hold?). An association can be high-weight but invalid: "this was a well-trodden path that turned out to be wrong." This is useful signal — prevents re-discovering false associations.

- **Invalidation preserves history.** Invalid associations aren't deleted — they record what was once believed and why it was disproved. The LLM can see "this connection was once believed and later contradicted by entry X."

##### Association lifecycle

```
created (weak, weight ~0.2)
  → accessed and found useful → strengthened (0.4, 0.6, 0.8...)
  → accessed and found irrelevant → unchanged
  → accessed and found misleading → weakened (0.5, 0.3, 0.1...)
  → contradicted by new evidence → invalidated (valid=false, with reason and evidence)
```

##### State Store interface

```python
class StateStore:
    entries: dict[str, StateEntry]            # nodes
    associations: dict[str, list[Association]] # adjacency list (keyed by source_id)
    memory_map: MemoryMap

    # Entry operations
    def write(self, entry: StateEntry) -> None
    def invalidate(self, entry_id: str, superseded_by: str) -> None
    def access(self, entry_id: str, step: int) -> None  # update access tracking

    # Association operations
    def add_association(self, assoc: Association) -> None
    def strengthen(self, assoc_id: str, delta: float, step: int) -> None
    def weaken(self, assoc_id: str, delta: float, step: int) -> None
    def invalidate_association(self, assoc_id: str, reason: str,
                               invalidated_by: Optional[str] = None) -> None
    def get_associations(self, entry_id: str) -> list[Association]
    def get_associations_in_context(self, entry_id: str, context: str) -> list[Association]

    # Navigation
    def get_neighbors(self, entry_id: str, max_hops: int = 1) -> list[StateEntry]
    def get_active(self) -> list[StateEntry]       # non-superseded entries
    def get_temporal_neighborhood(self, entry: StateEntry, window: int) -> list[StateEntry]
    def size(self) -> int                           # number of active entries

    # Rendering for LLM context
    def render_neighborhood(self, entry_ids: list[str], depth: int = 1) -> str
    def render_entries(self, entry_ids: list[str]) -> str

    # Persistence
    def save(self, path: str) -> None               # JSON dump (entries + associations + memory map)
    def load(self, path: str) -> None
```

`render_neighborhood` is the bridge between the graph and the LLM's context. It produces a structured document showing entries and their associations:

```
Entry: "Alice has 7 apples" (fact, confidence 0.9, accessed 12 times)
  Associations:
    → "Bob received 3 apples from Alice" (related_to)
        [apple trades] weight: 0.8, valid
        [family]       weight: 0.1, INVALID — "not related by family"
    → "Alice has 5 apples" (supersedes)
        [apple tracking] weight: 0.9, valid
```

#### 2. Memory Map

The memory map is a **compact routing layer** over the state graph. Its primary function is to provide **entry points** into the graph for a given objective. Without it, the cognitive step has no way into the graph.

```python
@dataclass
class TopicEntry:
    summary: str                      # what this topic covers
    entry_points: list[str]           # entry IDs — where to start traversal
    density: str                      # "dense" | "sparse" — how well-connected this region is

class MemoryMap:
    topics: dict[str, TopicEntry]     # topic_name → summary + entry points
    recent_changes: list[str]         # what was recently added/invalidated/strengthened
    contested_regions: list[str]      # topics with many invalidated associations
    weakly_connected: list[str]       # entry IDs not yet linked to any topic

    def render(self) -> str:
        """Renders as structured document for LLM context."""
        ...

    def update(self, changes: dict) -> None:
        """Incremental update — new topics, revised entry points, etc."""
        ...

    def get_entry_points(self, objective: str) -> list[str]:
        """Given an objective, return candidate entry point IDs.
        Currently returns all entry points (LLM selects during orient).
        Will be optimized later with entity-based indexing when the
        memory map grows too large for context."""
        ...
```

Rendered for the LLM during orient:

```
Memory Map:
  apple_trades: Exchanges of apples between Alice, Bob, Charlie.
    Entry points: [entry_12, entry_45, entry_67]
    Density: dense (well-consolidated)

  family_relationships: Known family connections.
    Entry points: [entry_3, entry_89]
    Density: sparse (few associations)

  Recent: Alice's apple count updated (step 30), new Bob-Charlie trade (step 42)
  Contested: Charlie's reliability (3 invalidated associations)
  Weakly connected: [entry_91, entry_92] (not yet linked to any topic)
```

**Scaling strategy (for later):** When the memory map grows too large for context, `get_entry_points` will use an entity index — extract entities from the objective, look up which topic clusters contain those entities, and only show relevant clusters to the LLM. This is cheap (set intersection), uses structure (entities, not keywords), and doesn't require embeddings.

#### 3. Cognitive Step (`cognition/cognitive_step.py`)

**The single recursive primitive.** Every cognitive act is an instance of this mechanism with a different objective.

The cognitive step traverses the state graph with LLM-guided navigation. At each level, it orients (finds entry points via memory map), traverses (follows associations, reads/writes entries), and synthesizes results. The LLM plays an intelligent role in both navigation and termination.

```python
class CognitiveStep:
    def __init__(self, llm: BaseLLM, max_depth: int = 3,
                 max_width: int = 5, max_steps: int = 20):
        self.llm = llm
        self.max_depth = max_depth     # hard cap on recursion depth
        self.max_width = max_width     # hard cap on sub-objectives per level
        self.max_steps = max_steps     # hard cap on traversal steps per level

    def execute(self, objective: str, state: StateStore,
                context: dict, depth: int = 0,
                tracer: TraceLogger = None) -> CognitiveResult:
        """The single recursive primitive."""

        trace_id = tracer.begin(objective, depth) if tracer else None

        # --- HARD SAFETY CAP ---
        if depth >= self.max_depth:
            return self._direct_resolve(objective, state, context, tracer, trace_id)

        # --- ORIENT ---
        # Consult memory map. LLM identifies entry points and
        # decomposes objective into sub-objectives.
        orientation = self.llm.generate_structured(
            messages=build_orient_prompt(objective, state.memory_map, context),
            schema=OrientationSchema
        )
        entry_points = orientation.entry_points
        sub_objectives = orientation.sub_objectives[:self.max_width]

        if tracer:
            tracer.record_orient(trace_id, state.memory_map.render(), sub_objectives)

        # --- RECURSE ON SUB-OBJECTIVES ---
        # Each sub-call traverses the graph starting from entry points,
        # reading and writing to state. Later sub-calls benefit from
        # state modifications made by earlier ones.
        for sub_objective in sub_objectives:
            self.execute(
                objective=sub_objective,
                state=state,
                context={**context, "entry_points": entry_points},
                depth=depth + 1,
                tracer=tracer
            )

        # --- SYNTHESIZE ---
        result = self._synthesize(objective, state, context)

        if tracer:
            tracer.record_synthesis(trace_id, result.output)
            tracer.end(trace_id)

        return result

    def _direct_resolve(self, objective: str, state: StateStore,
                        context: dict, tracer: TraceLogger = None,
                        trace_id: str = None) -> CognitiveResult:
        """Base case. Traverses the graph directly from entry points.

        The LLM navigates the graph step by step:
        1. See current nodes + their associations
        2. Decide which edges to follow, what to record, what to write
        3. Assess termination: loop detected? objective satisfied? diverging?
        4. Continue or stop (with explicit reasoning)

        This is where actual state mutations happen: new entries, new
        associations, weight updates, invalidations, memory map updates."""

        entry_points = context.get("entry_points", [])
        current_nodes = entry_points if entry_points else self._get_default_entry_points(state)
        visited = set()
        findings = []

        for step in range(self.max_steps):
            # Render the local neighborhood for the LLM
            neighborhood = state.render_neighborhood(current_nodes, depth=1)

            # LLM sees: current neighborhood, objective, traversal history
            step_result = self.llm.generate_structured(
                messages=build_traverse_prompt(
                    objective=objective,
                    neighborhood=neighborhood,
                    visited=visited,
                    findings=findings,
                    context=context
                ),
                schema=TraversalStepSchema
            )

            # Record findings
            findings.extend(step_result.findings)
            visited.update(current_nodes)

            # Execute state mutations
            for entry in step_result.new_entries:
                state.write(entry)
            for assoc in step_result.new_associations:
                state.add_association(assoc)
            for update in step_result.weight_updates:
                if update.delta > 0:
                    state.strengthen(update.assoc_id, update.delta, context.get("step_number", 0))
                else:
                    state.weaken(update.assoc_id, abs(update.delta), context.get("step_number", 0))
            for inv in step_result.association_invalidations:
                state.invalidate_association(inv.assoc_id, inv.reason, inv.invalidated_by)

            # Update access tracking on touched nodes
            for node_id in current_nodes:
                state.access(node_id, context.get("step_number", 0))

            if tracer:
                tracer.record_traversal_step(trace_id, step, current_nodes,
                                             step_result, visited, findings)

            # Intelligent termination — LLM assesses with explicit reasoning
            # Reasons: "loop_detected" | "objective_satisfied" |
            #          "diverging_from_objective" | "no_relevant_edges"
            if step_result.should_stop:
                break

            current_nodes = step_result.next_nodes

        # Update memory map with what was learned during traversal
        if step_result.map_changes:
            state.memory_map.update(step_result.map_changes)

        if tracer:
            tracer.record_direct_resolve(trace_id, visited, findings,
                                         step_result.new_entries,
                                         step_result.association_invalidations)
            tracer.end(trace_id)

        return CognitiveResult(output=findings, entries_written=step_result.new_entries)

    def _synthesize(self, objective: str, state: StateStore,
                    context: dict) -> CognitiveResult:
        """After all sub-objectives have executed, synthesize a result."""
        return self.llm.generate_structured(
            messages=build_synthesize_prompt(objective, state.memory_map, context),
            schema=SynthesisSchema
        )

    def _get_default_entry_points(self, state: StateStore) -> list[str]:
        """Fallback when no entry points provided — use recent entries."""
        recent = state.get_recent(5)
        return [e.id for e in recent]
```

**Key properties:**

- **Self-similar** — the same mechanism at every level. No special-cased read or write operations.
- **Graph traversal with LLM navigation** — at the base case, the LLM walks the graph step by step, seeing the local neighborhood and deciding which edges to follow. The LLM is the traversal intelligence.
- **Intelligent termination** — the LLM assesses whether to stop at each traversal step, with explicit reasoning (loop detection, information sufficiency, divergence). Hard caps (`max_depth`, `max_steps`) are safety nets.
- **State mutations during traversal** — entries and associations are created, weights are updated, associations are invalidated. The graph co-evolves with the cognitive process.
- **Access tracking** — every node touched during traversal has its access metadata updated, informing future traversals about what's well-trodden.

**How the three agent phases map to this primitive:**

| Agent Phase | Objective | What traversal does |
|---|---|---|
| Recall | "What do I need to know for this input?" | Traverses from relevant entry points, follows associations, records findings. May discover and write new associations. |
| Reason | "Given this input and recalled context, what should I do/respond?" | Traverses related entries, reasons about them, may write interim conclusions or hypotheses. |
| Integrate | "What new information should I remember from this output?" | Writes new entries with initial associations to existing entries, invalidates contradicted entries/associations, updates memory map. |

#### 4. LLM Interface (`cognition/llm/`)

```python
class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], system: str = "") -> str: ...

    @abstractmethod
    async def generate_structured(self, messages: list[dict], schema: dict,
                                  system: str = "") -> dict: ...

class AnthropicLLM(BaseLLM): ...    # Uses anthropic SDK
class OpenAILLM(BaseLLM): ...       # Uses openai SDK
class OpenRouterLLM(BaseLLM): ...   # Uses OpenRouter API (OpenAI-compatible)
```

OpenRouter uses an OpenAI-compatible API (`https://openrouter.ai/api/v1`) so `OpenRouterLLM` can extend `OpenAILLM` with a different base URL and auth header.

`generate_structured` is key — the cognitive step needs the LLM to return structured JSON (sub-objectives, traversal decisions, state mutations), not free-form text.

#### 5. Agent Loop (`cognition/agent.py`)

The agent loop sequences three phases of the Cognitive Step. Each phase is the same `execute` call with a different objective.

```python
class Agent:
    def __init__(self, llm: BaseLLM, state: StateStore,
                 max_depth: int = 3, max_width: int = 5, max_steps: int = 20):
        self.step_count = 0
        self.state = state
        self.cognitive_step = CognitiveStep(llm, max_depth, max_width, max_steps)
        self.tracer = TraceLogger()

    def step(self, input: str) -> str:
        context = {"input": input, "step_number": self.step_count}

        # Phase 1: RECALL — "what do I need to know?"
        recall_result = self.cognitive_step.execute(
            objective="recall", state=self.state,
            context=context, tracer=self.tracer
        )

        # Phase 2: REASON — "what should I do/respond?"
        context["recalled"] = recall_result
        reason_result = self.cognitive_step.execute(
            objective="respond", state=self.state,
            context=context, tracer=self.tracer
        )

        # Phase 3: INTEGRATE — "what should I remember?"
        context["output"] = reason_result
        self.cognitive_step.execute(
            objective="integrate", state=self.state,
            context=context, tracer=self.tracer
        )

        self.step_count += 1
        return reason_result.output

    async def run(self, inputs: list[str]) -> list[str]:
        """Process a sequence of inputs, maintaining state across all steps."""
        outputs = []
        for input in inputs:
            output = self.step(input)
            outputs.append(output)
        return outputs
```

#### 6. Tracing and Logging (`cognition/tracing.py`)

Every cognitive step execution produces a **trace** — a structured record of the full recursion and traversal tree.

```python
@dataclass
class TraversalStep:
    """Record of a single graph traversal step."""
    step_number: int
    current_nodes: list[str]           # entry IDs being examined
    edges_followed: list[str]          # association IDs followed
    findings: list[str]                # what was found/recorded
    state_mutations: dict              # entries written, associations added/updated/invalidated
    stop_decision: Optional[str]       # if stopping, the reason
    next_nodes: list[str]              # where to go next (if continuing)

@dataclass
class CognitiveTrace:
    """A single node in the recursion tree."""
    trace_id: str
    parent_trace_id: Optional[str]
    depth: int
    objective: str
    resolved_directly: bool            # True if hit base case (traversal)

    # Orient phase
    memory_map_snapshot: str
    sub_objectives: list[str]
    entry_points: list[str]

    # Traversal (if resolved_directly)
    traversal_steps: list[TraversalStep]
    total_nodes_visited: int
    total_state_mutations: int

    # Synthesis output
    synthesis_output: str

    # Metadata
    llm_calls: int
    timestamp: datetime
    duration_ms: float
    children: list["CognitiveTrace"]

class TraceLogger:
    """Collects traces during execution and provides analysis."""

    def begin(self, objective: str, depth: int,
              parent_trace_id: str = None) -> str: ...
    def record_orient(self, trace_id: str, memory_map: str,
                      sub_objectives: list[str]) -> None: ...
    def record_traversal_step(self, trace_id: str, step: int,
                              current_nodes: list[str],
                              step_result: dict,
                              visited: set, findings: list) -> None: ...
    def record_direct_resolve(self, trace_id: str, visited: set,
                              findings: list, entries_written: list,
                              invalidations: list) -> None: ...
    def record_synthesis(self, trace_id: str, output: str) -> None: ...
    def end(self, trace_id: str) -> None: ...

    def get_tree(self) -> CognitiveTrace: ...
    def get_flat(self) -> list[CognitiveTrace]: ...
    def summary(self) -> dict: ...
    def export_json(self, path: str) -> None: ...
    def print_tree(self) -> None: ...
```

This gives us:
- **Full recursion + traversal tree** — see how objectives were decomposed and how the graph was navigated at each leaf
- **State mutation audit** — every write, association change, and invalidation traced to the specific traversal step that caused it
- **Termination reasoning** — why traversal stopped at each base case (loop, satisfied, diverging)
- **LLM call accounting** — total calls per step, per phase, per depth level
- **Exportable JSON traces** — for post-hoc analysis and comparison across ablation runs

Standard Python `logging` is used alongside tracing for operational logs (debug/info/warning).

### Toy Task for Validation (`experiments/toy_task.py`)

**Entity tracking with updates and contradictions.** Simple to evaluate (exact match) but clearly requires statefulness:

1. Feed the agent a stream of ~50 messages, each introducing or updating facts about named entities
2. Intersperse recall questions ("How many apples does Alice have?")
3. Include updates that contradict earlier facts ("Alice actually has 7 apples now")
4. Include questions after updates (should reflect the updated value)

Example stream:
```
Step 1:  "Alice has 5 apples."
Step 2:  "Bob has 3 oranges."
...
Step 20: "Question: How many apples does Alice have?"  → expect "5"
...
Step 30: "Correction: Alice now has 7 apples."
...
Step 35: "Question: How many apples does Alice have?"  → expect "7"
Step 40: "Question: How many oranges does Bob have?"   → expect "3"
```

Metrics: recall accuracy, post-update accuracy (tests invalidation), and via tracing: traversal depth, associations formed, association weight evolution.

Baseline comparison: same task against a plain LLM with full history in context.

## Build Order

1. **`types.py`** — shared dataclasses (`StateEntry`, `Association`, `MemoryMap`, `CognitiveResult`, schema types)
2. **`llm/base.py` + `llm/anthropic.py` + `llm/openai.py` + `llm/openrouter.py`** — LLM interface + providers
3. **`tracing.py`** — CognitiveTrace, TraversalStep, TraceLogger (built early so all subsequent components use it)
4. **`state.py`** — StateStore (graph: entries + associations + memory map), rendering, persistence
5. **`cognitive_step.py`** — the single recursive primitive with graph traversal at base case, instrumented with TraceLogger
6. **`agent.py`** — agent loop: three phases (recall, reason, integrate) of CognitiveStep
7. **`experiments/toy_task.py`** — toy task to validate end-to-end
8. **Tests** for state store, cognitive step, and agent loop

## Verification

- Unit tests for StateStore (entry CRUD, association CRUD with context-scoping, weight updates, invalidation, access tracking, memory map update, persistence)
- Unit tests for CognitiveStep with mock LLM (test recursion, graph traversal, intelligent termination, state mutations)
- Tracing verification: confirm traces capture full recursion tree and all state mutations
- End-to-end: run toy task, check recall accuracy and post-update correctness
- Ablation: `max_depth=0` (flat baseline) vs increasing depths; vary `max_steps` and `max_width`
- Compare against naive baseline (plain LLM with full history in context) on same task
