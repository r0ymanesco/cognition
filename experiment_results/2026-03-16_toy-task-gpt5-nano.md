# Experiment: Toy Task Entity Tracking with gpt-5-nano

**Date**: 2026-03-16
**Requested by**: PI
**Run by**: Experiment Runner Agent

## Objective

**PI's request**: Run the toy task experiment using gpt-5-nano via the OpenAI provider with constrained parameters (max-steps=5, max-width=3). Report per-question pass/fail, overall accuracy, pre/post-correction accuracy, LLM call count, traversal steps, errors, and trace file location.

**Operationalized**: Execute `./scripts/toy_experiment.sh --name gpt5-nano --provider openai --model gpt-5-nano --max-steps 5 --max-width 3` and capture all output metrics from the 26-step entity tracking task (5 entity introductions, fillers, 8 recall questions split across pre- and post-correction phases).

## Methodology

### Command Run

```bash
./scripts/toy_experiment.sh --name gpt5-nano --provider openai --model gpt-5-nano --max-steps 5 --max-width 3
```

### Environment

| Component | Version |
|-----------|---------|
| Python | 3.12.9 (pyenv) |
| virtualenv | `cognition` |
| openai SDK | 2.26.0 |
| pydantic | 2.12.5 |
| OS | Linux 6.19.7-100.fc42.x86_64 (Fedora 42) |

### Parameters

| Parameter | Value | Default |
|-----------|-------|---------|
| provider | openai | anthropic |
| model | gpt-5-nano | gpt-4o |
| max-steps | 5 | 20 |
| max-width | 3 | 5 |

### Number of Trials

1 (experiment failed on first LLM call within the traversal phase; no point re-running since the failure is deterministic and schema-related, not transient).

### Pre-registered Expectations

1. Expected the 26-step task to run end-to-end, with gpt-5-nano potentially struggling on post-correction recall due to the constrained max-steps and max-width.
2. Expected timing to be fast for a nano-class model.
3. Did **not** anticipate a schema compatibility failure -- this was discovered upon execution.

## Raw Results

### Execution Output

The experiment **failed on the very first task step** ("Alice has 5 apples.") during the second LLM call (the `_direct_resolve` traversal phase, after the `_orient` phase succeeded).

```
agent.step 0: Alice has 5 apples.
HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 400 Bad Request"
```

**Error**:
```
openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid schema for
response_format 'TraversalStepResponse': In context=(), 'required' is required to be
supplied and to be an array including every key in properties. Extra required key
'new_topics' supplied.", 'type': 'invalid_request_error', 'param': 'response_format',
'code': None}}
```

### Exit Code

1 (failure)

### Files Created

| File | Status |
|------|--------|
| `traces/toy_task/20260316_175016/gpt5-nano/run.log` | Created, contains the error traceback |
| `traces/toy_task/20260316_175016/gpt5-nano/trace.json` | **Not created** (experiment aborted before export) |
| `traces/toy_task/20260316_175016/gpt5-nano/state.json` | **Not created** |

### Per-Question Results

No questions were reached. The experiment failed on step 0 (the first statement input, "Alice has 5 apples.").

| Metric | Value |
|--------|-------|
| Overall accuracy | N/A (0 questions evaluated) |
| Pre-correction accuracy | N/A |
| Post-correction accuracy | N/A |
| LLM calls completed | 1 (orient phase only) |
| LLM calls failed | 1 (traversal phase) |
| Total traversal steps | 0 |
| Trace file | Not generated |

## Analysis

### Root Cause

The failure is a **schema incompatibility** between the `TraversalStepResponse` Pydantic model and OpenAI's strict structured output API.

**Specifics**: The `TraversalStepResponse` model contains a nested `MapChangesSpec` field, which in turn has `dict[str, TopicSpec]` typed fields (`new_topics`, `updated_topics`). These serialize to JSON Schema with `additionalProperties: { "$ref": "#/$defs/TopicSpec" }`.

OpenAI's structured output API operates in "strict mode" where:
1. Every property listed in `properties` must also be in `required`.
2. The SDK (via `beta.chat.completions.parse`) transforms the schema to enforce this.

The transformation appears to fail on the `MapChangesSpec` schema -- it adds `new_topics` to the `required` array at the wrong nesting level (the root `TraversalStepResponse` rather than the nested `MapChangesSpec`), producing the error "Extra required key 'new_topics' supplied."

This is likely a bug or limitation in the openai SDK v2.26.0's schema transformation logic when handling models with `additionalProperties` containing `$ref` types inside nested objects.

**Key observation**: The first LLM call (`OrientationResponse`) succeeded (HTTP 200), confirming that the API key is valid, the model name `gpt-5-nano` is valid, and the provider is reachable. Only the `TraversalStepResponse` schema triggers the error.

### Why This is Not a gpt-5-nano-Specific Issue

This failure would occur with **any** OpenAI model (gpt-4o, gpt-4o-mini, etc.) because the error is in schema validation, which happens server-side before the model even processes the prompt. The `OrientationResponse` schema (which is simpler, with no `dict[str, Model]` fields) works fine.

### Confounds Considered

| Confound | Assessment |
|----------|------------|
| API key issue | Ruled out -- first call succeeded (HTTP 200) |
| Model name invalid | Ruled out -- first call succeeded |
| Rate limiting | Ruled out -- error is 400 not 429 |
| Network issue | Ruled out -- deterministic schema validation error |
| Transient server error | Ruled out -- 400 is a client error, not server error; the schema is deterministically invalid |
| SDK version issue | Possible -- openai SDK 2.26.0 may have a bug in schema transformation; newer versions may handle this |
| Pydantic version issue | Possible -- pydantic 2.12.5 JSON schema output may differ from what the OpenAI SDK expects |

## Key Findings

1. **The toy task experiment cannot run with the OpenAI provider due to a schema incompatibility in `TraversalStepResponse`.** The `dict[str, TopicSpec]` and `dict[str, TopicUpdateSpec]` fields in `MapChangesSpec` produce a JSON schema that OpenAI's structured output API rejects. (HIGH CONFIDENCE -- deterministic, reproducible failure)

2. **The `OrientationResponse` schema works fine with OpenAI, confirming the issue is specific to complex nested schemas with `additionalProperties` containing `$ref` types.** (HIGH CONFIDENCE -- the first HTTP call returned 200)

3. **No experiment metrics (accuracy, LLM calls, traversal steps) can be reported** because the experiment did not progress past the first task step. (HIGH CONFIDENCE -- zero data collected)

## Limitations & Caveats

- This was a single run, but additional runs would produce the identical failure since the error is deterministic.
- We cannot assess gpt-5-nano's actual performance on entity tracking because we never reached the question/answer phase.
- The `--max-steps 5 --max-width 3` parameters were never exercised, so we cannot report on their impact.
- We did not test whether newer versions of the openai SDK resolve this schema transformation issue.
- We did not test the Anthropic provider as a control to confirm the experiment code itself works.

## Recommendations

1. **Fix the OpenAI schema compatibility**: The `MapChangesSpec` model needs to be restructured to avoid `dict[str, Model]` types, which generate `additionalProperties` with `$ref` -- a pattern that OpenAI's strict structured output mode cannot handle. Options:
   - Replace `dict[str, TopicSpec]` with `list[NamedTopicSpec]` where `NamedTopicSpec` has a `name: str` field.
   - Add a custom `model_json_schema()` override that produces an OpenAI-compatible schema.
   - Use a different structured output approach for OpenAI (e.g., function calling instead of `response_format`).

2. **Upgrade the openai SDK** to the latest version and retest -- newer versions may handle this schema pattern correctly.

3. **Run the same experiment with `--provider anthropic`** as a control to confirm the experiment code itself functions correctly, then come back to gpt-5-nano after fixing the schema issue.

4. **Add a schema validation pre-check** to the experiment runner that catches these issues before making API calls, to fail fast with a clear error message.
