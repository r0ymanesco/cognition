# Cognition

An inference-time scaffold for LLMs that provides theoretically unbounded context through externalized, structured memory.

## What is this?

Current LLM architectures are bounded by their context window. Existing approaches to extending context (like RLM) are stateless and task-local — they can process more tokens per query but have no persistent memory across steps. This means they can't handle tasks that require belief revision, temporal dependencies, or evolving understanding over long horizons.

Cognition takes a different approach: instead of extending the context window, it externalizes all working state into a queryable store. The LLM's context window only ever holds the current objective, a compact memory map, and locally retrieved entries. The total state can grow without bound while the active context stays fixed.

## Core idea

The scaffold is built on a single recursive primitive: the **Cognitive Step**. It orients via a memory map (a compact, lossy summary of everything the agent knows), decomposes its objective into sub-objectives, recurses on each — reading from and writing to external state at every level — and synthesizes the results.

Everything is composed from this primitive. Recalling memories, reasoning about input, integrating new information — these are all instances of the same Cognitive Step with different objectives. The architecture is self-similar: the same mechanism operates at every level of recursion, and the recursion is structural (always happens, terminates based on state size and a configurable depth cap).

This is analogous to an RNN's persistent hidden state, but using external structured memory instead of a fixed-size vector — avoiding the information bottleneck and trainability issues that limited RNNs, while preserving the theoretical benefit of unbounded state.

## Key properties

- **Stateful and task-global** — maintains, retrieves, and revises beliefs across arbitrarily many steps
- **Self-similar** — one recursive primitive composes into all cognitive acts (recall, reasoning, integration)
- **Structurally recursive** — recursion always happens; termination is based on state size and depth, not LLM judgment
- **Observable** — full tracing of the recursion tree, state mutations, and LLM calls for analysis and debugging

## Setup

Requires Python 3.11+ and [pyenv](https://github.com/pyenv/pyenv).

```bash
# Create and activate virtualenv
pyenv virtualenv 3.12.9 cognition
pyenv local cognition

# Install package and dev dependencies
pip install -e ".[dev]"
```

Set your API key for the LLM provider you want to use:

```bash
export ANTHROPIC_API_KEY="sk-..."     # for Anthropic
export OPENAI_API_KEY="sk-..."        # for OpenAI
export OPENROUTER_API_KEY="sk-..."    # for OpenRouter
```

## Running experiments

### Toy task

The toy task validates the scaffold end-to-end: it feeds the agent a stream of facts, filler, corrections, and recall questions, then checks whether answers reflect the latest state.

```bash
# Run with Anthropic (default)
python -m experiments.toy_task --provider anthropic

# Run with OpenAI
python -m experiments.toy_task --provider openai --model gpt-4o

# Run with OpenRouter
python -m experiments.toy_task --provider openrouter --model anthropic/claude-sonnet-4

# Export full trace for analysis
python -m experiments.toy_task --provider anthropic --trace-file trace.json

# Flat baseline (no recursion — max_depth=0)
python -m experiments.toy_task --provider anthropic --max-depth 0

# Tune recursion parameters
python -m experiments.toy_task --max-depth 2 --max-width 3 --max-steps 10

# Verbose debug logging
python -m experiments.toy_task --verbose
```

Output includes: per-question pass/fail, overall accuracy, pre/post-correction recall accuracy, LLM call count, recursion depth, and traversal step count. State is saved to `toy_task_state.json` for inspection.

### Tests

```bash
python -m pytest tests/ -v
```

## Project structure

```
cognition/
├── cognition/
│   ├── cognitive_step.py      # The single recursive primitive
│   ├── agent.py               # Agent loop (recall → reason → integrate)
│   ├── state.py               # Graph-based state store + memory map
│   ├── tracing.py             # Structured traces of the cognitive process
│   ├── types.py               # Core data types
│   └── llm/                   # Provider-agnostic LLM interface
│       ├── base.py            # Abstract interface
│       ├── anthropic.py       # Claude
│       ├── openai.py          # GPT-4o etc.
│       └── openrouter.py      # OpenRouter (many models)
├── experiments/
│   └── toy_task.py            # Entity tracking with corrections
├── tests/
├── plan/                      # Research motivation and implementation plan
└── readme.md
```

## Status

Early experimental stage. See `plan/` for the research motivation and implementation plan.
