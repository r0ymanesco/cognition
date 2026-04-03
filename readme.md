# Cognition

An inference-time scaffold for LLMs that provides theoretically unbounded context through externalized, structured memory.

## What is this?

Current LLM architectures are bounded by their context window. Existing approaches to extending context (like RLM) are stateless and task-local — they can process more tokens per query but have no persistent memory across steps. This means they can't handle tasks that require belief revision, temporal dependencies, or evolving understanding over long horizons.

Cognition takes a different approach: instead of extending the context window, it externalizes all working state into a graph-based store. The LLM's context window only ever holds the current objective, a compact memory map, and locally retrieved entries. The total state can grow without bound while the active context stays fixed.

## Core idea

The scaffold is built around a single primitive: the **Cognitive Step**. Given an input, it orients via a memory map (a compact routing layer over the knowledge graph), then traverses the graph — reading entries, following associations, writing new information, and revising beliefs — before synthesizing a response.

This is similar to how LLM-based agents explore file systems, but with two key differences:

1. **The store is a graph, not a file tree.** Information is stored as entries (nodes) connected by weighted, typed, context-scoped associations (edges). This captures relationships between facts, supports belief revision (invalidation), and allows associative navigation — following connections between related information rather than searching by keyword or path.

2. **The LLM reasons through the graph, not just over retrieved results.** In typical RAG architectures, retrieval and generation are separate: an algorithm queries the store, then the results are stuffed into the LLM's context for response generation. Here, the LLM is the traversal algorithm. It sees the local neighborhood of the graph, decides which edges to follow, creates new entries and associations, strengthens or invalidates connections, and determines when it has enough information to respond — all within the same cognitive process. Retrieval, reasoning, and memory update are not separate phases; they emerge naturally from the graph traversal.

## Key properties

- **Stateful and task-global** — maintains, retrieves, and revises beliefs across arbitrarily many steps
- **LLM-navigated graph** — the LLM is the traversal intelligence, not a consumer of pre-retrieved results
- **Associative memory** — connections strengthen through use, weaken when misleading, and are scoped to context
- **Observable** — full tracing of the traversal tree, state mutations, and LLM calls for analysis and debugging

## Setup

Requires Python 3.11+ and [pyenv](https://github.com/pyenv/pyenv).

```bash
# Create and activate virtualenv
pyenv virtualenv 3.12.9 cognition
pyenv local cognition

# Install package and dev dependencies
pip install -e ".[dev]"
```

Set your API keys in `.env` (auto-loaded by experiments):

```bash
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
```

## Running experiments

### Toy task

The toy task validates the scaffold end-to-end: it feeds the agent a stream of facts, filler, corrections, and recall questions, then checks whether answers reflect the latest state.

```bash
# Run with Anthropic (default)
python -m experiments.toy_task --provider anthropic --model claude-sonnet-4-6

# Run with OpenAI
python -m experiments.toy_task --provider openai --model gpt-4o

# Run with OpenRouter
python -m experiments.toy_task --provider openrouter --model anthropic/claude-sonnet-4

# Tune parameters
python -m experiments.toy_task --max-width 3 --max-steps 10

# Extended thinking (Anthropic only)
python -m experiments.toy_task --provider anthropic --model claude-sonnet-4-6 --llm-kwargs budget_tokens=4096

# Export full trace for analysis
python -m experiments.toy_task --trace-file trace.json

# Verbose debug logging
python -m experiments.toy_task --verbose
```

Or use the experiment script for named runs with automatic trace/log/state output:

```bash
./scripts/toy_experiment.sh --name sonnet46 --provider anthropic --model claude-sonnet-4-6
./scripts/toy_experiment.sh --run-ablation anthropic claude-sonnet-4-6
```

### Tests

```bash
python -m pytest tests/ -v
pyright
```

## Project structure

```
cognition/
├── cognition/
│   ├── cognitive_step.py      # Orient → traverse graph → synthesize
│   ├── agent.py               # Thin wrapper: input → cognitive step → response
│   ├── state.py               # Graph store (entries + associations + memory map)
│   ├── tracing.py             # Structured traces of the cognitive process
│   ├── types.py               # Core data types
│   └── llm/                   # Provider-agnostic LLM interface
│       ├── base.py            # Abstract interface
│       ├── anthropic.py       # Claude (with extended thinking support)
│       ├── openai.py          # GPT-4o, GPT-5 etc.
│       └── openrouter.py      # OpenRouter (many models)
├── experiments/
│   └── toy_task.py            # Entity tracking with corrections
├── scripts/
│   └── toy_experiment.sh      # Experiment runner with trace output
├── tests/
├── plan/                      # Research motivation and implementation plan
└── readme.md
```

## Status

Early experimental stage. See `plan/` for the research motivation and implementation plan.
