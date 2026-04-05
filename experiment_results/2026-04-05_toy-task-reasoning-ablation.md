# Toy Task: Model & Reasoning Ablation Study

**Date**: 2026-04-05
**Experiment**: Entity tracking with corrections (toy task)
**Goal**: Validate the cognition scaffold end-to-end and measure the effect of model choice and reasoning effort on accuracy.

## Experiment Setup

### Task

The toy task feeds the agent a stream of 26 inputs:
- **Steps 0-4**: Establish facts (Alice has 5 apples, Bob has 3 oranges, Charlie has 10 bananas, Diana has 2 pears, Eve has 7 grapes)
- **Steps 5-9**: Filler statements (weather, market hours, prices, shipments, temperature)
- **Steps 10-12**: Pre-correction recall questions (3 questions)
- **Steps 13-16**: More filler + context (transport, inventory system, Bob's trade mention, Alice's market visit)
- **Steps 17-18**: Corrections (Alice now has 7 apples, Charlie now has 4 bananas)
- **Steps 19-20**: More filler
- **Steps 21-25**: Post-correction recall questions (5 questions)

Evaluation: exact match of expected value in the agent's response. Post-correction questions should reflect updated values.

### Scaffold Configuration

All runs used the same scaffold settings:
- **max_width**: 3 (max sub-objectives per orient)
- **max_steps**: 5 (max graph traversal steps per direct resolve)
- **System prompt**: Inventory tracking assistant — store facts, recall on questions, use most recent values after corrections

### Cognitive Step Flow

Each input goes through: orient (consult memory map, find entry points) -> traverse graph (read/write entries, follow associations, decide when to stop) -> synthesize (compile response, update memory map).

The LLM sees the local neighborhood of the graph at each traversal step and decides whether to store new entries, create associations, follow edges, or stop. There are no separate recall/reason/integrate phases — these emerge naturally from the traversal.

## Results

### Model Comparison (No Reasoning)

| Model | Provider | Accuracy | Pre-correction | Post-correction | Entries | Associations | LLM Calls | Duration |
|---|---|---|---|---|---|---|---|---|
| gpt-5-nano | OpenAI | 1/8 (12%) | 0/3 (0%) | 1/5 (20%) | 7 | 1 | 88 | ~1614s |
| gpt-4o | OpenAI | 2/8 (25%) | 2/3 (67%) | 0/5 (0%) | 23 | 14 | 57 | ~124s |
| gpt-4o (old prompts) | OpenAI | 3/8 (38%) | 2/3 (67%) | 1/5 (20%) | 5 | 0 | 52 | ~85s |
| **claude-sonnet-4.6** | **Anthropic** | **7/8 (88%)** | **3/3 (100%)** | **4/5 (80%)** | **18** | **13** | **53** | **~248s** |

**Notes**:
- gpt-4o "old prompts" was run before the prompt rewrite that explicitly tells the LLM to store facts and create entries.
- gpt-4o with new prompts stores more entries (23 vs 5) and associations (14 vs 0) but accuracy dropped because it can't navigate the graph to find answers — many responses are empty strings.
- gpt-5-nano stores facts but can't retrieve them; uses 88 LLM calls (many traversal steps that find nothing).
- Sonnet 4.6 is the only model to achieve >50% accuracy without reasoning, with 100% pre-correction and 80% post-correction recall.

### GPT-5.4 Reasoning Effort Ablation

All runs via OpenAI API with `reasoning_effort` parameter.

| Reasoning Effort | Accuracy | Pre-correction | Post-correction | Entries | Associations | LLM Calls | Duration |
|---|---|---|---|---|---|---|---|
| low | 4/8 (50%) | 2/3 (67%) | 2/5 (40%) | 18 | 0 | 52 | ~190s |
| medium | 2/8 (25%) | 0/3 (0%) | 2/5 (40%) | 18 | 1 | 52 | ~310s |
| **high** | **5/8 (62%)** | **2/3 (67%)** | **3/5 (60%)** | **19** | **3** | **53** | **~576s** |

**Key finding**: Non-monotonic relationship between reasoning effort and accuracy. Medium (25%) is worse than low (50%). High (62%) is the best GPT-5.4 result but still below Sonnet 4.6 without reasoning (88%).

The "valley" at medium suggests the model starts reasoning about graph decisions but doesn't have enough depth to complete the thought — leading to worse outcomes than quick intuitive responses (low) or thorough deliberation (high).

### Sonnet 4.6 Extended Thinking Ablation

All runs via Anthropic API with `thinking.budget_tokens` parameter. Note: extended thinking requires `tool_choice: auto` instead of forced tool use, which means the model may occasionally return text instead of tool calls.

| Thinking Budget | Accuracy | Pre-correction | Post-correction | Entries | Associations | LLM Calls | Duration |
|---|---|---|---|---|---|---|---|
| **none (baseline)** | **7/8 (88%)** | **3/3 (100%)** | **4/5 (80%)** | **18** | **13** | **53** | **~248s** |
| 1024 tokens | 4/8 (50%) | 2/3 (67%) | 2/5 (40%) | 19 | 38 | 58 | ~449s |
| **4096 tokens** | **7/8 (88%)** | **3/3 (100%)** | **4/5 (80%)** | **20** | **30** | **58** | **~591s** |

**Key finding**: U-shaped curve — same pattern as GPT-5.4. Insufficient thinking budget (1024) is worse than no thinking at all.

- **No thinking (88%)**: Strong baseline instruction-following.
- **1024 tokens (50%)**: Model starts reasoning but can't complete the thought. Creates many more associations (38) suggesting it's overthinking graph structure without following through. The `tool_choice: auto` constraint may also cause some calls to not use the tool.
- **4096 tokens (88%)**: Matches baseline accuracy. Enough budget to complete reasoning. Creates more entries (20) and associations (30) — deeper thinking enriches the graph without hurting retrieval.

## Shared Failure Analysis

The one consistent failure across the best runs (Sonnet 4.6 no-thinking and 4096-thinking): **Diana's pears** in post-correction recall (step 24). Diana's fact is established at step 3, never corrected, and never mentioned in filler. By step 24, the graph has 18-20 entries and the traversal can't find Diana's entry from the available entry points.

Alice's correction (step 17: "Alice now has 7 apples") is another weak point. The old entry ("Alice has 5 apples") is not reliably invalidated across models. Step 21 asks for Alice's current apple count — models that haven't invalidated the old entry may return 5 instead of 7.

## Conclusions

1. **Model quality matters more than reasoning for this scaffold.** Sonnet 4.6 at 88% without reasoning beats GPT-5.4 at 62% with high reasoning effort. The scaffold relies on the LLM to make correct graph traversal decisions — stronger instruction-following translates directly to better storage and retrieval.

2. **Insufficient reasoning is worse than no reasoning.** Both Sonnet 4.6 and GPT-5.4 show a "valley" at medium reasoning — the model starts deliberating but can't finish, leading to worse decisions than quick intuitive responses.

3. **The prompt rewrite was critical.** The original prompts produced 0-38% accuracy across all models. The rewrite (explicitly telling the LLM to store facts, create entries from empty graphs, and answer questions from findings) brought Sonnet 4.6 to 88%.

4. **Graph population is necessary but not sufficient.** GPT-4o with new prompts creates 23 entries and 14 associations but only gets 25% accuracy — it can't navigate the graph it built. Sonnet 4.6 creates 18 entries and 13 associations and gets 88% — it builds less but navigates better.

5. **Belief revision (invalidation) remains the hardest capability.** Post-correction accuracy is lower than pre-correction across all models and settings. The scaffold supports invalidation mechanically (superseded_by field, association invalidation), but the LLM doesn't consistently use it.

## Baseline Comparison: Scaffold vs Plain Conversation History

To determine whether the scaffold adds value, we ran the same task with a plain LLM baseline — no graph, no memory map, no cognitive step. The LLM simply accumulates the full conversation history in its context window and responds directly. Same system prompt, same models.

### Setup

- **Baseline agent**: appends each input as a user message and each response as an assistant message. The LLM sees the entire conversation history at every step. One LLM call per step.
- **Scaffold agent**: orient → graph traversal → synthesize. Two or more LLM calls per step (orient + traversal steps).
- **Same system prompt** for both: inventory tracking assistant.
- **Same models**: claude-sonnet-4-6, gpt-5.4 (no reasoning/thinking for either).

### Results

| Model | Baseline (no scaffold) | With scaffold | Baseline calls | Scaffold calls | Baseline time | Scaffold time |
|---|---|---|---|---|---|---|
| claude-sonnet-4.6 | **8/8 (100%)** | 7/8 (88%) | 26 | 53 | ~30s | ~248s |
| gpt-5.4 | **8/8 (100%)** | 5/8 (62%) | 26 | 53 | ~29s | ~576s |

### Breakdown

| Model | Condition | Pre-correction | Post-correction |
|---|---|---|---|
| claude-sonnet-4.6 | baseline | 3/3 (100%) | 5/5 (100%) |
| claude-sonnet-4.6 | scaffold | 3/3 (100%) | 4/5 (80%) |
| gpt-5.4 | baseline | 3/3 (100%) | 5/5 (100%) |
| gpt-5.4 (high reasoning) | scaffold | 2/3 (67%) | 3/5 (60%) |

### Analysis

The scaffold **hurts performance** on this task. Both models achieve perfect accuracy with plain conversation history — faster, cheaper, and more accurate.

This is the expected result. The toy task has 26 steps producing ~52 messages, which fits easily within any model's context window. The baseline keeps the full history in context and answers perfectly. The scaffold adds overhead (graph traversal, structured output parsing, memory map management) without benefit because the problem doesn't require externalized memory.

**The scaffold's value proposition is not "better at tasks that fit in context."** It is "still works when the history exceeds the context window." This task doesn't test that boundary.

To demonstrate the scaffold's value, the next experiment needs:

1. **History that overflows the context window** — hundreds or thousands of steps so the baseline is forced to truncate.
2. **Long-range recall** — questions about facts from early in the history, after many intervening steps have pushed them out of the baseline's context.
3. **Belief revision at distance** — corrections that arrive far from the original fact, requiring the scaffold to find and invalidate old entries that the baseline has already forgotten.

At that scale, the baseline must either truncate (losing old facts) or summarize (lossy compression). The scaffold stores everything in the graph and navigates it on demand — that's where it should win.

## Reproduction

```bash
# Install
pyenv virtualenv 3.12.9 cognition && pyenv local cognition
pip install -e ".[dev]"
# Set API keys in .env

# Model comparison
python -m experiments.toy_task --provider openai --model gpt-5-nano --max-steps 5 --max-width 3
python -m experiments.toy_task --provider openai --model gpt-4o --max-steps 5 --max-width 3
python -m experiments.toy_task --provider anthropic --model claude-sonnet-4-6 --max-steps 5 --max-width 3

# GPT-5.4 reasoning ablation
python -m experiments.toy_task --provider openai --model gpt-5.4 --max-steps 5 --max-width 3 --llm-kwargs reasoning_effort=low
python -m experiments.toy_task --provider openai --model gpt-5.4 --max-steps 5 --max-width 3 --llm-kwargs reasoning_effort=medium
python -m experiments.toy_task --provider openai --model gpt-5.4 --max-steps 5 --max-width 3 --llm-kwargs reasoning_effort=high

# Sonnet 4.6 thinking ablation
python -m experiments.toy_task --provider anthropic --model claude-sonnet-4-6 --max-steps 5 --max-width 3
python -m experiments.toy_task --provider anthropic --model claude-sonnet-4-6 --max-steps 5 --max-width 3 --llm-kwargs budget_tokens=1024
python -m experiments.toy_task --provider anthropic --model claude-sonnet-4-6 --max-steps 5 --max-width 3 --llm-kwargs budget_tokens=4096

# Baseline (no scaffold)
python -m experiments.toy_task_baseline --provider anthropic --model claude-sonnet-4-6
python -m experiments.toy_task_baseline --provider openai --model gpt-5.4
```
