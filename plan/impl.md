# Cognition Scaffold — Implementation Plan

## Context

We're building an inference-time scaffold around LLMs that acts as a stateful cognitive architecture — like an RNN's persistent state but using external structured memory instead of a fixed hidden state. The goal is to give LLM agents theoretically unbounded context by externalizing all working state into a queryable store, so the context window only ever holds: current input + retrieved relevant state + scaffolding for the next decision.

The key differentiator from existing approaches (like RLM) is that this is **stateful and task-global** — it maintains, retrieves, and revises beliefs across arbitrarily many steps.

## Phase 1: Core Scaffold (what we're building now)

### Architecture

#### Core Principle: One Primitive

The scaffold has one fundamental primitive: the **Cognitive Step**. It is a recursive process that orients via a memory map, decomposes its objective into sub-objectives, recurses on each, and synthesizes the results — reading from and writing to an external state store at every level.

Everything is composed from this primitive. Recalling memories, reasoning about input, integrating new information — these are all instances of the same Cognitive Step, differing only in their **objective**. The architecture is self-similar: the same mechanism operates at every level of recursion.

**The infinite context mechanism**: The state store is the externalized memory that grows without bound. The Cognitive Step is the mechanism that navigates it without needing to fit it all in context. At every level, the LLM's context window holds only: the current objective + the memory map (compact, fixed-capacity) + locally retrieved entries. The memory map is a lossy but dynamic summary — updated at every step, always biased toward what's recent and relevant.

#### Components

1. **State Store** — structured external memory (in-memory dict + JSON persistence) + a **Memory Map** (compact topical summary of what's in state — the agent's "fuzzy sense" of what it knows)
2. **Cognitive Step** — the single recursive primitive. Orients via memory map, decomposes into sub-objectives, recurses on each (reading and writing to state at every level), synthesizes results. Termination is structural: based on state size and a configurable hard depth cap. The LLM decides **what** to explore (sub-objectives), never **whether** to continue.
3. **LLM Interface** — provider-agnostic abstraction over language models
4. **Agent Loop** — sequences three phases of the Cognitive Step: recall (objective: "what do I need to know?"), reason (objective: "what should I do/respond?"), integrate (objective: "what should I remember?"). Same mechanism, different objectives.

### Project Structure

```
cognition/
├── plan.md
├── readme.md
├── pyproject.toml
├── cognition/
│   ├── __init__.py
│   ├── cognitive_step.py      # The single recursive primitive
│   ├── agent.py               # Agent loop (three phases of CognitiveStep)
│   ├── state.py               # StateStore + StateEntry + MemoryMap
│   ├── tracing.py             # CognitiveTrace + TraceLogger
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract LLM interface
│   │   ├── anthropic.py       # Claude implementation
│   │   ├── openai.py          # OpenAI implementation
│   │   └── openrouter.py      # OpenRouter implementation (access to many models)
│   └── types.py               # Shared types/dataclasses
├── experiments/
│   └── toy_task.py            # Toy task to validate scaffold
└── tests/
    ├── test_state.py
    ├── test_cognitive_step.py
    └── test_agent.py
```

#### Tracing and Logging

Every cognitive step execution produces a **trace** — a structured record of the full recursion tree. This is essential for understanding, debugging, and analyzing the cognitive process.

```python
@dataclass
class CognitiveTrace:
    """A single node in the recursion tree."""
    trace_id: str                          # unique ID for this trace node
    parent_trace_id: Optional[str]         # parent in the recursion tree (None for root)
    depth: int
    objective: str
    resolved_directly: bool                # True if hit base case

    # Orient phase
    memory_map_snapshot: str               # memory map as seen at orient time
    sub_objectives: list[str]              # what orient produced

    # Base case (if resolved_directly)
    state_before: list[str]                # entry IDs in active state before resolve
    entries_written: list[str]             # entry IDs written
    entries_invalidated: list[str]         # entry IDs invalidated
    map_changes: dict                      # memory map mutations

    # Synthesis output
    synthesis_output: str

    # Metadata
    llm_calls: int                         # number of LLM calls in this node
    timestamp: datetime
    duration_ms: float

    children: list["CognitiveTrace"]       # sub-objective traces

class TraceLogger:
    """Collects traces during execution and provides analysis."""

    def begin(self, objective, depth, parent_trace_id) -> str:  # returns trace_id
    def record_orient(self, trace_id, memory_map, sub_objectives): ...
    def record_direct_resolve(self, trace_id, state_before, writes, invalidations, map_changes): ...
    def record_synthesis(self, trace_id, output): ...
    def end(self, trace_id): ...

    def get_tree(self) -> CognitiveTrace:          # full recursion tree
    def get_flat(self) -> list[CognitiveTrace]:    # all nodes depth-first
    def summary(self) -> dict:                      # stats: total depth, total LLM calls,
                                                    # entries written/invalidated, etc.
    def export_json(self, path: str): ...           # dump full trace for analysis
    def print_tree(self): ...                       # human-readable tree to stdout
```

The `TraceLogger` is passed into `CognitiveStep` and the agent loop. Every `execute` call wraps its work in `begin`/`end`. This gives us:

- **Full recursion tree** — see exactly how the objective was decomposed, how deep it went, what happened at each leaf
- **State mutation audit** — every write and invalidation traced back to the specific base-case resolve that caused it
- **LLM call accounting** — total calls per step, per phase, per depth level
- **Performance profiling** — duration at each node, identifying bottlenecks
- **Exportable JSON traces** — for post-hoc analysis, visualization, and comparison across ablation runs

Standard Python `logging` is used alongside tracing for operational logs (debug/info/warning). Tracing captures the structured cognitive process; logging captures operational events.


### Component Details

#### 1. State Store (`cognition/state.py`)

```python
@dataclass
class StateEntry:
    id: str                          # UUID
    content: str                     # The actual information
    entry_type: str                  # "fact" | "hypothesis" | "decision" | "observation"
    confidence: float                # 0.0 - 1.0
    step_number: int                 # When this was created
    superseded_by: Optional[str]     # ID of entry that invalidated this
    tags: list[str]                  # Semantic tags for filtering
    created_at: datetime

class MemoryMap:
    """Compact topical summary of what's in the state store.
    The agent's 'fuzzy sense' of what it knows — like a hippocampal
    index routing queries to the right neighborhood of detailed memories.

    Small enough to always fit in context. Tells the agent WHERE to look,
    not WHAT's there. Updated incrementally by write operations."""

    topics: dict[str, str]           # topic_name → brief summary of what's known
    open_questions: list[str]        # unresolved threads
    recent_changes: list[str]        # what was recently added/invalidated

    def render(self) -> str:         # format for inclusion in LLM context
    def update(self, changes: dict): # incremental update (not full rebuild)

class StateStore:
    entries: dict[str, StateEntry]
    memory_map: MemoryMap

    # Core operations
    write(entry: StateEntry) -> None
    read(query: str, filters: dict) -> list[StateEntry]   # keyword/tag match for now
    invalidate(entry_id: str, superseded_by: str) -> None

    # Navigation
    get_recent(n: int) -> list[StateEntry]
    get_active() -> list[StateEntry]       # non-superseded only
    get_by_type(entry_type: str) -> list[StateEntry]
    get_temporal_neighborhood(entry: StateEntry, window: int) -> list[StateEntry]
    size() -> int                          # number of active entries

    # Persistence
    save(path: str) -> None                # JSON dump
    load(path: str) -> None                # JSON load
```

The read method starts as simple keyword + tag matching. We can swap in embedding-based retrieval later without changing the interface.

`get_temporal_neighborhood` supports memory reconstruction — walking temporally adjacent entries to "relive" a sequence of events rather than retrieving isolated facts.

#### 2. Cognitive Step (`cognition/cognitive_step.py`)

**The single recursive primitive.** Every cognitive act — recalling memories, reasoning about input, integrating new information — is an instance of this same mechanism with a different objective.

The structure is: **orient → recurse on sub-objectives → synthesize**. The recursion is structural (always happens). Termination is determined by state size and a hard depth cap — never by the LLM.

```python
DIRECT_THRESHOLD = 20   # configurable: below this, direct resolve

class CognitiveStep:
    def __init__(self, llm: BaseLLM, max_depth: int = 3, max_width: int = 5):
        self.llm = llm
        self.max_depth = max_depth   # hard safety cap on recursion depth
        self.max_width = max_width   # max sub-objectives per level

    def execute(self, objective: str, state: StateStore,
                context: dict, depth: int = 0) -> CognitiveResult:
        """The single recursive primitive.

        At every level:
        1. Check termination (state size or depth cap)
        2. Orient — consult memory map, decompose into sub-objectives
        3. Recurse — execute each sub-objective (each can read/write state)
        4. Synthesize — compile results for this level's objective
        """

        # --- TERMINATION (structural, not LLM-decided) ---
        if state.size() <= DIRECT_THRESHOLD or depth >= self.max_depth:
            return self._direct_resolve(objective, state, context)

        # --- ORIENT ---
        # Consult the memory map (compact, always fits in context).
        # LLM determines WHAT to explore (sub-objectives), never
        # WHETHER to continue — that's decided by termination above.
        orientation = self.llm.generate_structured(
            messages=build_orient_prompt(objective, state.memory_map, context),
            schema=OrientationSchema
        )
        sub_objectives = orientation.sub_objectives[:self.max_width]

        # --- RECURSE ---
        # Always recurse on every sub-objective. Each sub-call can
        # read from and write to state. Later sub-calls benefit from
        # state modifications made by earlier ones.
        for sub_objective in sub_objectives:
            self.execute(
                objective=sub_objective,
                state=state,
                context=context,
                depth=depth + 1
            )

        # --- SYNTHESIZE ---
        # State has been modified by all sub-calls. Compile a result
        # for this level's objective given the current state.
        return self.synthesize(objective, state, context)

    def _direct_resolve(self, objective: str, state: StateStore,
                        context: dict) -> CognitiveResult:
        """Base case. State is small enough to work with directly.

        The LLM sees the objective + full active state (it fits in
        context because state.size() <= DIRECT_THRESHOLD) and
        performs the cognitive work in a single step:
        - Reads relevant entries
        - Reasons about them
        - Writes new entries / invalidates old ones / updates memory map
        - Returns result

        This is where actual state mutations happen. The higher levels
        of recursion are purely about decomposition and navigation."""

        result = self.llm.generate_structured(
            messages=build_direct_resolve_prompt(
                objective=objective,
                active_state=state.get_active(),
                memory_map=state.memory_map,
                context=context
            ),
            schema=DirectResolveSchema
        )

        # Execute state mutations
        for entry in result.new_entries:
            state.write(entry)
        for invalidation in result.invalidations:
            state.invalidate(invalidation.entry_id, invalidation.superseded_by)
        state.memory_map.update(result.map_changes)

        return CognitiveResult(
            output=result.output,
            entries_written=result.new_entries,
            entries_invalidated=result.invalidations
        )

    def synthesize(self, objective: str, state: StateStore,
                   context: dict) -> CognitiveResult:
        """After all sub-objectives have recursed (modifying state),
        synthesize a result for this level's objective."""

        return self.llm.generate_structured(
            messages=build_synthesize_prompt(objective, state.memory_map, context),
            schema=SynthesisSchema
        )
```

**Key properties:**

- **Self-similar** — the same mechanism at every level. No special-cased read or write operations. Every cognitive act is orient → recurse → synthesize.
- **Structurally recursive** — the recursion always happens. Termination is based on `state.size()` (intelligent) and `max_depth` (safety cap). The LLM decides **what** to explore, never **whether** to continue.
- **State mutations happen at the base case** — `_direct_resolve` is where entries are actually read, written, and invalidated. Higher levels decompose and navigate; the leaves do the work. This means every recursive call can modify state, and later calls (at any level) see the updated state.
- **Configurable width and depth** — `max_width` caps how many sub-objectives per level. `max_depth` caps recursion depth. Both are ablation knobs: `max_depth=0` degrades to flat single-step processing (baseline). Increasing depth/width tests whether recursive cognition helps.

**How the three agent phases map to this primitive:**

| Agent Phase | Objective | What the base case does |
|---|---|---|
| Recall | "What do I need to know for this input?" | Retrieves relevant entries, discovers associations, writes association links back to state |
| Reason | "Given this input and recalled context, what should I do/respond?" | Reasons about the input, may write interim conclusions or hypotheses to state |
| Integrate | "What new information should I remember from this output?" | Writes new entries, invalidates contradicted entries, updates memory map |

All three use `CognitiveStep.execute` — same code path, different objective string and context.

#### 3. LLM Interface (`cognition/llm/`)

```python
class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict], system: str = "") -> str: ...

    @abstractmethod
    async def generate_structured(self, messages: list[dict], schema: dict,
                                  system: str = "") -> dict: ...

class AnthropicLLM(BaseLLM): ...    # Uses anthropic SDK
class OpenAILLM(BaseLLM): ...       # Uses openai SDK
class OpenRouterLLM(BaseLLM): ...   # Uses OpenRouter API (OpenAI-compatible, access to many models)
```

OpenRouter uses an OpenAI-compatible API (`https://openrouter.ai/api/v1`) so `OpenRouterLLM` can extend `OpenAILLM` with a different base URL and auth header. This gives us access to a wide range of models (Llama, Mistral, Gemini, etc.) through a single provider.

`generate_structured` is key — the cognitive step needs the LLM to return structured JSON (sub-objectives, state mutations, synthesis), not free-form text.

#### 4. Agent Loop (`cognition/agent.py`)

The agent loop sequences three phases of the Cognitive Step primitive. Each phase is the same `execute` call with a different objective. Between phases, state has been modified by the previous phase's recursive execution.

```python
class Agent:
    def __init__(self, llm: BaseLLM, state: StateStore,
                 max_depth: int = 3, max_width: int = 5):
        self.step_count = 0
        self.state = state
        self.cognitive_step = CognitiveStep(llm, max_depth, max_width)

    def step(self, input: str) -> str:
        context = {"input": input, "step_number": self.step_count}

        # Phase 1: RECALL — "what do I need to know?"
        # Navigates state store, retrieves relevant entries,
        # may write associations discovered during recall.
        recall_result = self.cognitive_step.execute(
            objective="recall",
            state=self.state,
            context=context
        )

        # Phase 2: REASON — "what should I do/respond?"
        # State now contains everything recall surfaced.
        # May write interim conclusions or hypotheses.
        context["recalled"] = recall_result
        reason_result = self.cognitive_step.execute(
            objective="respond",
            state=self.state,
            context=context
        )

        # Phase 3: INTEGRATE — "what should I remember?"
        # Writes new entries, invalidates contradicted entries,
        # updates memory map.
        context["output"] = reason_result
        self.cognitive_step.execute(
            objective="integrate",
            state=self.state,
            context=context
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

### Toy Task for Validation (`experiments/toy_task.py`)

**Entity tracking with updates and contradictions.** This is simple to evaluate (exact match) but clearly requires statefulness:

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

We evaluate: accuracy on recall questions, especially post-update accuracy (tests invalidation).

We also run the same task against a **baseline** — a plain LLM with all history stuffed into context (no scaffold) — to verify the scaffold at least matches it on short sequences, then show it handles sequences that overflow the baseline's context window.

## Build Order

1. **`types.py`** — shared dataclasses (`StateEntry`, `MemoryMap`, `CognitiveResult`, prompt/schema types)
2. **`llm/base.py` + `llm/anthropic.py` + `llm/openai.py` + `llm/openrouter.py`** — LLM interface + providers (OpenRouter extends OpenAI with different base URL)
3. **`tracing.py`** — CognitiveTrace, TraceLogger (built early so all subsequent components can use it)
4. **`state.py`** — StateStore with read/write/invalidate, MemoryMap with render/update, JSON persistence
5. **`cognitive_step.py`** — the single recursive primitive (orient → recurse → synthesize + direct_resolve base case), instrumented with TraceLogger
6. **`agent.py`** — agent loop: three phases (recall, reason, integrate) of CognitiveStep
6. **`experiments/toy_task.py`** — toy task to validate end-to-end
7. **Tests** for state store, cognitive step, and agent loop

## Verification

- Unit tests for StateStore (write, read, invalidate, memory map update, persistence)
- Unit tests for CognitiveStep with mock LLM (test recursion, termination conditions, state mutations at base case)
- End-to-end: run toy task, check recall accuracy and post-update correctness
- Ablation: run toy task with `max_depth=0` (flat baseline) vs increasing depths to measure whether recursion helps
- Compare against naive baseline (plain LLM with full history in context) on same task

