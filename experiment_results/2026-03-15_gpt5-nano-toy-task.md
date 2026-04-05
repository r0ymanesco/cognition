# Experiment: Toy Task Entity Tracking with gpt-5-nano

**Date**: 2026-03-15
**Requested by**: PI
**Run by**: Experiment Runner Agent

## Objective

Run the toy task experiment (entity tracking with belief revision) using OpenAI's `gpt-5-nano` model with constrained cognitive parameters (`max-steps=5`, `max-width=3`).

The experiment tests whether the cognition scaffold can:
1. Store facts about 5 entities
2. Recall facts across filler steps
3. Handle corrections (belief revision)
4. Return updated values after corrections

Metrics requested: per-question accuracy, overall accuracy, pre/post-correction accuracy, LLM call count, traversal steps, any errors, and trace file location.

## Methodology

### Command executed

```bash
./scripts/toy_experiment.sh --name gpt5-nano --provider openai --model gpt-5-nano --max-steps 5 --max-width 3
```

This translates to:
```bash
python -m experiments.toy_task \
    --trace-file /home/tzeyang/workspace/cognition/traces/toy_task/20260316_174133/gpt5-nano/trace.json \
    --provider openai --model gpt-5-nano --max-steps 5 --max-width 3
```

### Environment

- Python 3.12.9 (pyenv virtualenv `cognition`)
- OpenAI Python SDK (installed in virtualenv)
- Platform: Linux 6.19.7-100.fc42.x86_64
- Working directory: `/home/tzeyang/workspace/cognition`
- Date/time of run: 2026-03-16 17:41:33 (local time)

### Parameters

| Parameter   | Value     | Default |
|-------------|-----------|---------|
| provider    | openai    | anthropic |
| model       | gpt-5-nano | gpt-4o |
| max-width   | 3         | 5       |
| max-steps   | 5         | 20      |

### Task structure

The toy task has 26 steps total:
- 5 fact-establishing statements (Phase 1: Alice/5 apples, Bob/3 oranges, Charlie/10 bananas, Diana/2 pears, Eve/7 grapes)
- 5 filler statements (Phase 2)
- 3 pre-correction recall questions (Phase 3: Alice apples, Bob oranges, Charlie bananas)
- 4 filler statements (Phase 4)
- 2 corrections (Phase 5: Alice 5->7 apples, Charlie 10->4 bananas)
- 2 filler statements (Phase 6)
- 5 post-correction recall questions (Phase 7: Alice apples, Charlie bananas, Bob oranges, Diana pears, Eve grapes)

Total questions: 8 (3 pre-correction + 5 post-correction)

### Pre-registered expectations

1. The experiment should complete all 26 steps and produce accuracy metrics.
2. With `gpt-5-nano` (a smaller model), post-correction recall might be weaker than pre-correction.
3. `max-steps 5` and `max-width 3` are relatively constrained, which may limit graph traversal.
4. Potential risk: OpenAI's structured output mode has strict schema requirements that Pydantic's default schema generation may not satisfy.

## Raw Results

**The experiment FAILED on step 0 (the very first input: "Alice has 5 apples.") with an HTTP 400 error from OpenAI.**

**No task steps completed. No accuracy data was collected.**

### Exit code

1 (error)

### Complete stdout/stderr output

```
═══════════════════════════════════════════════════════
  Experiment: toy_task / gpt5-nano
  Args: --provider openai --model gpt-5-nano --max-steps 5 --max-width 3
  Output: /home/tzeyang/workspace/cognition/traces/toy_task/20260316_174133/gpt5-nano
═══════════════════════════════════════════════════════
Running toy task: 26 steps
Provider: openai, Model: gpt-5-nano
Config: max_width=3, max_steps=5
------------------------------------------------------------
agent.step 0: Alice has 5 apples.
HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"
HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 400 Bad Request"
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/tzeyang/workspace/cognition/experiments/toy_task.py", line 350, in <module>
    main()
  File "/home/tzeyang/workspace/cognition/experiments/toy_task.py", line 338, in main
    asyncio.run(run_experiment(
  File "/home/tzeyang/.pyenv/versions/3.12.9/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
  File "/home/tzeyang/.pyenv/versions/3.12.9/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
  File "/home/tzeyang/.pyenv/versions/3.12.9/lib/python3.12/asyncio/base_events.py", line 691, in run_until_complete
    return future.result()
  File "/home/tzeyang/workspace/cognition/experiments/toy_task.py", line 248, in run_experiment
    output = await agent.step(step.input_text)
  File "/home/tzeyang/workspace/cognition/cognition/agent.py", line 57, in step
    result = await self.cognitive_step.execute(
  File "/home/tzeyang/workspace/cognition/cognition/cognitive_step.py", line 178, in execute
    result = await self._direct_resolve(
  File "/home/tzeyang/workspace/cognition/cognition/cognitive_step.py", line 321, in _direct_resolve
    step_result = await self.llm.generate_structured(
  File "/home/tzeyang/workspace/cognition/cognition/llm/openai.py", line 76, in generate_structured
    response = await self.client.beta.chat.completions.parse(
  File "/home/tzeyang/.pyenv/versions/cognition/lib/python3.12/site-packages/openai/resources/chat/completions/completions.py", line 1694, in parse
    return await self._post(
  File "/home/tzeyang/.pyenv/versions/cognition/lib/python3.12/site-packages/openai/_base_client.py", line 1884, in post
    return await self.request(cast_to, opts, stream=stream, stream_cls=stream_cls)
  File "/home/tzeyang/.pyenv/versions/cognition/lib/python3.12/site-packages/openai/_base_client.py", line 1669, in request
    raise self._make_status_error_from_response(err.response) from None
openai.BadRequestError: Error code: 400 - {'error': {'message': "Invalid schema for
response_format 'TraversalStepResponse': In context=('properties', 'new_topics',
'additionalProperties'), 'additionalProperties' is required to be supplied and to be
false.", 'type': 'invalid_request_error', 'param': 'response_format', 'code': None}}
```

### Call flow that succeeded and failed

| # | HTTP Status | Endpoint | Phase | Response Model | Result |
|---|-------------|----------|-------|----------------|--------|
| 1 | 200 OK | `POST /v1/chat/completions` | `_orient()` | `OrientationResponse` | Success |
| 2 | 400 Bad Request | `POST /v1/chat/completions` | `_direct_resolve()` | `TraversalStepResponse` | **Schema rejected** |

### Artifacts produced

| File | Status |
|------|--------|
| `traces/toy_task/20260316_174133/gpt5-nano/run.log` | Created (contains error output) |
| `traces/toy_task/20260316_174133/gpt5-nano/trace.json` | **NOT created** (crash before export) |
| `traces/toy_task/20260316_174133/gpt5-nano/state.json` | **NOT created** (crash before state dump) |

### Requested metrics

| Metric | Value |
|--------|-------|
| Per-question accuracy | N/A -- no questions reached |
| Overall accuracy | N/A |
| Pre-correction accuracy | N/A |
| Post-correction accuracy | N/A |
| LLM call count | 2 attempted (1 succeeded, 1 failed with 400) |
| Traversal steps | 0 completed |
| Errors | 1 fatal -- `openai.BadRequestError` (schema validation) |
| Trace file location | Not generated |

## Analysis

### Root cause

The failure is a **schema incompatibility** between Pydantic's JSON schema generation and OpenAI's structured output API requirements.

OpenAI's structured output mode (used by `client.beta.chat.completions.parse()`) requires that **every object type** in the JSON schema includes `"additionalProperties": false`. Pydantic's `model_json_schema()` does not emit this property by default.

The specific failing model is `TraversalStepResponse`, which contains a nested field `map_changes: MapChangesSpec`. The `MapChangesSpec` class (in `cognition/cognitive_step.py`, lines 83-89) declares:

```python
class MapChangesSpec(BaseModel):
    new_topics: dict[str, dict] = Field(default_factory=dict)
    updated_topics: dict[str, dict] = Field(default_factory=dict)
    ...
```

The `dict[str, dict]` type generates a JSON schema using `additionalProperties` with an open object schema (no `additionalProperties: false` inside), which OpenAI rejects.

The error message explicitly calls out the path: `context=('properties', 'new_topics', 'additionalProperties')`.

### Notable observation: _orient() now succeeds

A prior run (logged in a previous version of this file, timestamp `20260316_173715`) failed at the `_orient()` phase with `OrientationResponse` being rejected. In this run, `_orient()` succeeded (HTTP 200 OK) and the failure moved downstream to `_direct_resolve()` with `TraversalStepResponse`. This indicates that `OrientationResponse`'s schema was fixed (or the SDK updated) between the two runs, but `TraversalStepResponse`'s schema remains incompatible.

### Is this a gpt-5-nano-specific issue?

**No.** The error is in schema validation, which happens before any model inference. The first call's success (HTTP 200 OK) confirms that `gpt-5-nano` is a valid, accessible model. This same failure would occur with `gpt-4o`, `o1`, or any other OpenAI model when using `TraversalStepResponse` as the response model.

### Confounds considered

| Confound | Assessment |
|----------|------------|
| API key invalid | Ruled out -- first call returned 200 OK |
| `gpt-5-nano` not a real model | Ruled out -- first call succeeded, meaning the model accepted the request |
| Network issues | Ruled out -- both requests reached the server and returned responses |
| Rate limiting | Ruled out -- error is 400 (client error), not 429 |
| SDK version mismatch | Possible contributor -- the SDK may handle schema differently across versions, but the root cause is the Pydantic model definition |

## Key Findings

1. **The experiment could not produce any results due to a schema compatibility bug.** The `TraversalStepResponse` Pydantic model generates a JSON schema that OpenAI's structured output API rejects. Zero out of 26 task steps completed. Zero out of 8 questions answered. **HIGH CONFIDENCE**

2. **The bug is in `MapChangesSpec`, specifically the `dict[str, dict]` fields (`new_topics`, `updated_topics`).** OpenAI requires `additionalProperties: false` on all nested objects, but `dict[str, dict]` produces an open-ended schema. **HIGH CONFIDENCE**

3. **This is a systemic issue affecting ALL OpenAI models, not gpt-5-nano-specific.** Any experiment using `--provider openai` will fail identically when the agent reaches `_direct_resolve()`. **HIGH CONFIDENCE**

4. **The `_orient()` phase works with OpenAI** -- the `OrientationResponse` schema is compatible, and `gpt-5-nano` is accessible. **HIGH CONFIDENCE**

## Limitations & Caveats

1. No experimental data whatsoever was collected. All accuracy, performance, and cognitive metrics remain unknown for gpt-5-nano.
2. This report documents an infrastructure failure, not a model evaluation. The PI's original research question (gpt-5-nano performance on entity tracking) is entirely unanswered.
3. I did not test with `--provider anthropic` as a control, since the PI specifically requested the OpenAI/gpt-5-nano configuration.
4. I did not verify whether the OpenRouter provider would exhibit the same issue when routing to OpenAI-hosted models.
5. Only a single run was attempted because the failure is deterministic (schema validation), not stochastic.

## Recommendations

1. **Fix `MapChangesSpec` before any OpenAI experiments.** Replace `dict[str, dict]` with explicitly typed Pydantic models so that `additionalProperties: false` is generated. For example:
   ```python
   class TopicSpec(BaseModel):
       summary: str = ""
       entry_ids: list[str] = Field(default_factory=list)

   class MapChangesSpec(BaseModel):
       new_topics: dict[str, TopicSpec] = Field(default_factory=dict)
       updated_topics: dict[str, TopicSpec] = Field(default_factory=dict)
       ...
   ```

2. **Audit ALL Pydantic response models** for OpenAI compatibility. Any use of bare `dict`, `dict[str, dict]`, `Any`, or `Union` types may trigger similar failures. Models to check: `OrientationResponse`, `TraversalStepResponse`, `SynthesisResponse`, `NewEntrySpec`, `NewAssociationSpec`, `WeightUpdateSpec`, `AssociationInvalidationSpec`, `MapChangesSpec`.

3. **Add a provider smoke test** that calls `generate_structured` with each response model on a trivial prompt before running a full experiment. This would catch schema issues in seconds rather than after experiment setup.

4. **Re-run this exact experiment** (`--name gpt5-nano --provider openai --model gpt-5-nano --max-steps 5 --max-width 3`) after the fix to collect the originally requested metrics.

5. **Establish a working baseline first** by running with `--provider anthropic` to confirm the experiment logic itself is sound, independent of the OpenAI schema issue.
