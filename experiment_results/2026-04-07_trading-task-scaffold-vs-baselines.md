# Trading Task: Scaffold vs Baselines

**Date**: 2026-04-07
**Experiment**: Trading network with relationships, history, and multi-hop reasoning
**Goal**: Test the scaffold on a task with richer structure than independent entity tracking — trades create relationships, corrections cascade, and questions require following chains.

## Scaffold Changes Since Previous Report

This report uses the scaffold with several changes since the token-constrained comparison:

1. **Flexible schema** — `RelationshipType` and `EntryType` enums removed. Associations use free-form relationship strings (e.g., "traded_with", "supersedes", "co_located"). Entry types are free-form strings. The LLM creates whatever relationship types make sense for the domain.

2. **Supersession via associations** — `superseded_by` field removed from `StateEntry`. Supersession is now a "supersedes" association. Historical entries remain in the graph and are reachable by following supersedes associations backwards.

3. **Compact edge rendering** — `render_neighborhood` shows current entries fully + a compact edge list (relationship + target summary). The LLM picks which edges to follow via `next_nodes`. This keeps prompts small regardless of graph connectivity.

4. **Relationship vocabulary tracking** — the memory map tracks all relationship types that exist in the graph. Orient sees this vocabulary and outputs a `relationship_filter` to limit which edge types are shown during traversal.

5. **Association context updates** — `AssociationUpdateSpec` allows the LLM to append context to existing associations, recording new events on a living association record.

6. **Traverse prompt shows vocabulary** — existing relationship types are displayed directly in the traversal prompt with an instruction to reuse them rather than creating near-duplicates.

## Task: Trading Network

A programmatic task with 20 entities, 15 trades, 3 corrections, and filler:

- **20 entities** with initial inventory (verbose descriptions, ~75 tokens each)
- **15 trades** between entities — each modifies both parties' counts and creates a relationship
- **3 corrections** to initial counts (cascading effect on current values)
- **30 questions** across 4 types:
  - **Current state** (21 questions): "How many X does Y currently have?"
  - **Historical** (8 questions): "What was Y's count before the trade/correction?" and "What was Y's ORIGINAL count?"
  - **Multi-hop** (3 questions): "A gave to B, B later gave to C — how many does C have?"
  - **Relationship** (1 question): "How many different people did X trade with?"
- **80 total steps**, **~3919 tokens** full conversation
- Seeded with `--seed 42`

### Why this task should differentiate

- **Multi-hop**: requires following trade associations across entities — graph navigation vs flat summary scanning
- **Historical**: requires accessing superseded entries — the scaffold preserves them, compaction may discard them
- **Cascading corrections**: a correction changes the initial count, affecting all subsequent trade calculations
- **Relationships**: requires counting distinct trade partners — graph has explicit associations, summaries flatten this

## Setup

All runs use Sonnet 4.6 (claude-sonnet-4-6) with a 3000 token budget.

### Scaffold
- `--context-budget 3000` (unified budget, enforced on full message, LLM-managed)
- `--max-width 3`, `--max-steps 5`

### Truncation Baseline
- `--max-tokens 3000` (drops oldest messages from conversation history)
- `--strategy truncation`

### Compaction Baseline
- `--max-tokens 3000` (LLM summarizes history when exceeded)
- `--strategy compaction`

## Results

| Condition | Overall | Current (21) | Historical (8) | Multi-hop (3) | Relationship (1) | LLM calls | Duration |
|---|---|---|---|---|---|---|---|
| **Scaffold** | **19/30 (63%)** | 13/21 | **6/8** | 0/3 | 0/1 | 273 | ~1253s |
| **Truncation** | **20/30 (67%)** | 12/21 | **5/5** | 0/3 | **1/1** | 80 | ~117s |
| **Compaction** | 16/30 (53%) | 9/21 | 1/3 | 0/3 | 0/1 | 81 | ~122s |

### Scaffold failures (11)

| Step | Type | Question | Expected | Got |
|---|---|---|---|---|
| 48 | current | Rachel's dates | 21 | 19 (pre-trade value) |
| 50 | current | Quinn's figs | 18 | 13 (pre-trade value) |
| 52 | historical | Paul's papayas before trade 1 | 13 | wrong |
| 53 | historical | Paul's papayas before trade 2 | 12 | wrong |
| 56 | multi-hop | Paul's grapes (via Eve chain) | 15 | wrong |
| 57 | multi-hop | Paul's grapes (via Eve chain) | 15 | wrong |
| 58 | multi-hop | Iris's mangoes (via Frank chain) | 17 | wrong |
| 68 | current | Paul's papayas post-correction | 15 | wrong |
| 72 | current | Rachel's dates post-correction | 21 | 19 |
| 74 | current | Quinn's figs post-correction | 18 | 13 |
| 79 | relationship | Quinn's trade partners | 2 | wrong |

### Truncation failures (10)

Similar pattern: trade details drop out of context. Multi-hop all fail. Historical questions pass because recent messages contain the context. Truncation beats scaffold slightly because it retains the most recent messages including corrections.

### Compaction failures (14)

Worst performance. The summary loses trade details, historical counts, and relationship structure. Historical questions mostly fail (1/3 on ORIGINAL counts) because the summary discards old values.

## Analysis

### Multi-hop: 0/3 across all approaches

The hardest question type failed universally. For the scaffold, the graph has 107 associations including trade relationships, but the traversal doesn't follow the chain correctly. The multi-hop question asks "A gave to B, B later gave to C — how many does C have?" This requires:
1. Finding C's current entry
2. Or: following the trade chain A→B→C through associations

The scaffold has the associations but the 5-step traversal limit and the compact edge rendering may not provide enough depth to follow a 2-hop chain. Investigation needed.

### Historical: scaffold 6/8, truncation 5/5

The scaffold correctly answered 6/8 historical questions, including ORIGINAL pre-correction counts (3/3). The 2 failures were pre-trade historical questions (Paul's papayas at specific trade points) — these require precise temporal state reconstruction which is harder than just finding the original entry.

Truncation scored 5/5 on its subset because the historical questions it saw were answerable from recent context. It only had 5 historical questions in its window (not 8).

### Compaction: worst overall (53%)

Compaction's summary is lossy in exactly the ways predicted:
- Lost trade details (who traded what with whom)
- Lost historical counts (original values discarded as "outdated")
- Lost relationship structure (trade partner lists compressed away)

This validates that compaction struggles with structured, relational information — but the scaffold didn't capitalize on this advantage.

### Why the scaffold didn't win

The scaffold has the graph structure (107 associations, 31 active entries) but scored lower than truncation. Possible reasons:
1. **Traversal depth insufficient** — 5 steps may not be enough to follow trade chains
2. **Compact edge rendering loses detail** — the LLM sees edge summaries (60 chars) which may not contain enough information for precise calculations
3. **Budget pressure** — at 3000 tokens, the scaffold's prompts are large (orient + traverse + synthesis), leaving less room for the actual information
4. **Trade math** — computing current counts from a chain of trades requires arithmetic, which all approaches struggle with

### Cost comparison

| Condition | LLM calls | Duration | Cost ratio |
|---|---|---|---|
| Truncation | 80 | ~117s | 1x |
| Compaction | 81 | ~122s | 1x |
| Scaffold | 273 | ~1253s | 3.4x calls, 10.7x time |

The scaffold is significantly more expensive for this task.

## Conclusions

1. **The scaffold doesn't yet beat baselines on the trading task.** 63% vs truncation's 67%. The graph structure exists but isn't being navigated effectively for complex queries.

2. **Compaction is worst (53%)** — validates that flat summarization loses relational structure. But the scaffold doesn't fully capitalize on this.

3. **Multi-hop reasoning is unsolved** — 0/3 across all approaches. This is the key capability the scaffold needs to demonstrate. The graph has the associations; the traversal needs to follow them.

4. **Historical queries are a scaffold strength** — 6/8 including all ORIGINAL-count questions. Superseded entries preserved in the graph and accessible via associations.

5. **Next steps**: Investigate why multi-hop traversal fails despite the graph having the right associations. Possible fixes: increase max_steps, show more edge detail, or add explicit chain-following guidance to the traversal prompt.

## Reproduction

```bash
# Scaffold
python -m experiments.toy_task \
    --provider anthropic --model claude-sonnet-4-6 \
    --trading --seed 42 \
    --max-steps 5 --max-width 3 \
    --context-budget 3000

# Truncation baseline
python -m experiments.toy_task_baseline \
    --provider anthropic --model claude-sonnet-4-6 \
    --trading --seed 42 \
    --max-tokens 3000 --strategy truncation

# Compaction baseline
python -m experiments.toy_task_baseline \
    --provider anthropic --model claude-sonnet-4-6 \
    --trading --seed 42 \
    --max-tokens 3000 --strategy compaction
```
