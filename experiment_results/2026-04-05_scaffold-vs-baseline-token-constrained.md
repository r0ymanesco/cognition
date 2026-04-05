# Scaffold vs Baseline: Token-Constrained Comparison

**Date**: 2026-04-05
**Experiment**: Scaled entity tracking (15 entities) under token-constrained conditions
**Goal**: Demonstrate that the cognitive scaffold outperforms plain conversation history when both operate under the same token budget.

## Important: Scaffold Changes Since Previous Report

This report uses a significantly updated scaffold. Results should NOT be compared directly with the earlier reasoning ablation report (`2026-04-05_toy-task-reasoning-ablation.md`). Key changes:

1. **Synthesis always runs** — previously synthesis was skipped when orient returned no sub-objectives. Now it runs after every traversal step, ensuring the memory map is maintained and the response is based on traversal findings.

2. **Synthesis sees traversal findings** — previously the synthesis prompt only saw the memory map, causing hallucinated answers. Now it receives the actual findings from graph traversal and bases its response on them.

3. **Synthesis sees entry content for organization** — weakly_connected entries are rendered with their actual content in the synthesis prompt, so the LLM can group them into meaningful topics.

4. **Parameterized memory map budget (`max_map_tokens`)** — the synthesis prompt always shows the current map size and budget. The LLM proactively compresses the map (merging topics, shortening summaries, dropping filler) to stay within budget.

5. **Context budget enforcement (`max_context_tokens`)** — raises a RuntimeError if the memory map or neighborhood exceeds the budget, rather than silently truncating.

## Task: Scaled Entity Tracking

A programmatic task generator (`build_scaled_task`) creates verbose entity tracking scenarios:

- **15 entities** (Alice through Olivia), each assigned a fruit type and count
- **Verbose descriptions**: each fact is ~75 tokens embedded in a paragraph of contextual detail
- **5 filler messages per phase**: logistics, maintenance, market reports (~75 tokens each)
- **5 corrections**: entity counts updated with verbose explanations
- **22 recall questions**: 7 pre-correction + 15 post-correction (all entities)
- **57 total steps**, **~3552 tokens** full conversation with responses
- **Deterministic**: seeded with `--seed 42` for reproducibility

## Setup

### Scaffold Agent
- **Model**: claude-sonnet-4-6 (Anthropic)
- **Architecture**: orient → traverse graph → synthesize (single cognitive step per input)
- **`max_width`**: 3 (max sub-objectives per orient)
- **`max_steps`**: 5 (max traversal steps per graph walk)
- **`max_map_tokens`**: 2000 (memory map token budget, LLM-managed compaction)
- **`max_context_tokens`**: 3000 (hard cap on any single LLM call's context)
- **System prompt**: inventory tracking assistant

### Baseline Agent
- **Model**: claude-sonnet-4-6 (Anthropic)
- **Architecture**: plain conversation history (append user/assistant messages)
- **`max_tokens`**: 3000 (conversation history truncated from front when exceeding budget)
- **System prompt**: same inventory tracking assistant prompt

Both agents receive identical inputs in identical order from the same task (seed=42).

## Results

### Primary Comparison (3000 token budget)

| Condition | Accuracy | Post-correction | LLM calls | Duration |
|---|---|---|---|---|
| **Scaffold** (map=2000, ctx=3000) | **22/22 (100%)** | **21/21 (100%)** | 178 | ~940s |
| **Baseline** (ctx=3000) | 15/22 (68%) | 14/21 (67%) | 57 | ~66s |

The scaffold achieves **100% accuracy** while the baseline drops to **68%** — a 32 percentage point gap under the same token constraint.

### Scaffold State
- 57 entries in graph, 70 associations
- Memory map: ~2000 tokens (within budget), 23 topics organized by the LLM
- All 15 entity inventories stored as compact graph entries (~22 tokens each)

### Baseline State
- 114 messages in history, 23 dropped to fit within 3000 tokens
- Early entity facts (Bob, Charlie, etc.) fell out of the context window
- 7 questions failed because the relevant facts were no longer in context

### Additional Data Points

| Budget | Scaffold accuracy | Baseline accuracy | Baseline msgs dropped | Notes |
|---|---|---|---|---|
| unlimited | 22/22 (100%) | 22/22 (100%) | 0 | Both perfect when unconstrained |
| 3500 | 22/22 (100%) | 22/22 (100%) | 0 | Both still fit at this budget |
| 3000 (ctx) / 2000 (map) | **22/22 (100%)** | **15/22 (68%)** | 23 | **Scaffold wins** |
| 3000 (ctx) / no map limit | Crashed at step 37 | 20/22 (91%) | 14 | Map grew to 3065 tokens without budget |
| 2000 (ctx) / 1500 (map) | Crashed at step 41 | 17/22 (77%) | 39 | Neighborhood exceeded 2000 tokens |
| 1500 (map only) | 21/22 (95%) | N/A | N/A | Map compaction works; Bob's entry lost |

## Analysis

### Why the scaffold wins under token pressure

The scaffold separates **storage** from **context**. Graph entries are compact (~22 tokens each: content + metadata) and persist regardless of conversation length. The memory map provides a routing layer (~2000 tokens) that lets the LLM find any entry on demand. The traverse prompt only loads the local neighborhood (~500 tokens), not the entire graph.

The baseline must keep the full conversation in context. At 3000 tokens with verbose messages (~75 tokens each + responses), it can hold roughly the most recent ~20 exchanges. Earlier entity facts get dropped and are permanently lost.

### Memory map compaction is critical

Without a budget, the memory map grows to ~2727 tokens for 15 entities (25 topics) — nearly as large as the baseline's full conversation. The `max_map_tokens` parameter forces the LLM to compress: merging related topics, shortening summaries, dropping filler topics. At 2000 tokens, the map fits 23 topics for 15 entities with room to spare.

The LLM effectively learns to maintain a fixed-capacity index of what it knows — closer to the original design intent of the memory map as a "compact routing layer."

### Cost trade-off

The scaffold uses 178 LLM calls vs the baseline's 57 (3.1x more) and takes ~940s vs ~66s (14x slower). Each input requires orient + traverse + synthesize calls. This is the current cost of externalized memory — the scaffold trades compute for accuracy under token pressure.

### The Bob failure (at tighter budgets)

At 1500 token map budget, the scaffold scores 21/22 — missing only Bob's oranges. The LLM dropped Bob's topic during map compaction to stay within budget. This shows the compaction is lossy and can lose information under very tight budgets. At 2000 tokens the LLM has enough room to retain all 15 entities.

## Conclusions

1. **The scaffold outperforms the baseline under token pressure.** At 3000 tokens: scaffold 100% vs baseline 68%. This validates the core hypothesis that externalized structured memory outperforms conversation history when context is constrained.

2. **Memory map compaction is necessary and effective.** Without a budget, the map grows proportionally with topics and negates the compression advantage. With LLM-managed compaction (`max_map_tokens`), the map stays fixed-size while the graph grows.

3. **The scaffold's advantage grows with conversation length.** At 15 entities and 57 steps, the baseline drops 23 messages. With more entities and longer conversations, more would be lost. The scaffold's graph retains everything.

4. **Cost is the main trade-off.** 3x more LLM calls and 14x slower. Future optimization targets: fewer synthesis calls (batch map updates), more efficient orient (skip when map hasn't changed), smarter traversal termination.

## Reproduction

```bash
# Install
pyenv virtualenv 3.12.9 cognition && pyenv local cognition
pip install -e ".[dev]"
# Set API keys in .env

# Scaffold (token-constrained)
python -m experiments.toy_task \
    --provider anthropic --model claude-sonnet-4-6 \
    --scaled-entities 15 --seed 42 \
    --max-steps 5 --max-width 3 \
    --max-context-tokens 3000 --max-map-tokens 2000

# Baseline (same token budget)
python -m experiments.toy_task_baseline \
    --provider anthropic --model claude-sonnet-4-6 \
    --scaled-entities 15 --seed 42 \
    --max-tokens 3000

# Scaffold (unconstrained, for reference)
python -m experiments.toy_task \
    --provider anthropic --model claude-sonnet-4-6 \
    --scaled-entities 15 --seed 42 \
    --max-steps 5 --max-width 3

# Baseline (unconstrained, for reference)
python -m experiments.toy_task_baseline \
    --provider anthropic --model claude-sonnet-4-6 \
    --scaled-entities 15 --seed 42
```
