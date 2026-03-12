# Project Setup

## Environment

- Python 3.12 via pyenv
- Virtualenv: `pyenv virtualenv 3.12.9 cognition` (already created, set via `.python-version`)
- Install: `pip install -e ".[dev]"`
- Do NOT use system python or create venvs with `python -m venv`. Always use `pyenv virtualenv`.

## Running

- Tests: `python -m pytest tests/ -v`
- Type checking: `pyright`
- Linting: `flake8`
- Toy task experiment: `python -m experiments.toy_task --provider anthropic`

## Code Quality

- pyright is the language server and type checker (configured in `pyproject.toml` under `[tool.pyright]`)
- flake8 for linting
- All code must pass `pyright` with 0 errors before committing
- Use pyright/LSP for code navigation rather than grepping

## Architecture

Single recursive primitive (`CognitiveStep`) composes all cognitive acts. See `plan/impl.md` for the full design.

Key files:
- `cognition/cognitive_step.py` — the recursive primitive (orient → recurse → synthesize)
- `cognition/agent.py` — three-phase loop (recall → reason → integrate)
- `cognition/state.py` — graph-based state store (entries + associations + memory map)
- `cognition/tracing.py` — structured observability for the cognitive process
- `cognition/llm/` — provider-agnostic LLM interface (Anthropic, OpenAI, OpenRouter)

## Conventions

- Async throughout — the agent and cognitive step are async
- Pydantic models for all structured LLM output
- `BaseLLM.generate_structured` uses TypeVar for type-safe returns
- State mutations only happen at the base case of the cognitive step (`_direct_resolve`)
- Tracing is opt-in via `TraceLogger` passed to `Agent` and `CognitiveStep`
