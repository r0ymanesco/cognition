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

## Status

Early experimental stage. See `plan/` for the research motivation and implementation plan.
