---
name: OpenAI structured output schema bug
description: OpenAI provider generate_structured fails with 400 because Pydantic schemas lack additionalProperties:false required by strict mode
type: project
---

The OpenAI LLM provider (`cognition/llm/openai.py`) cannot complete structured output calls for `TraversalStepResponse`. The `generate_structured` method uses `client.beta.chat.completions.parse()` which enforces strict schema validation requiring `"additionalProperties": false` on every object node.

As of 2026-03-15: `OrientationResponse` works (HTTP 200 OK with gpt-5-nano). `TraversalStepResponse` fails because `MapChangesSpec.new_topics` and `MapChangesSpec.updated_topics` use `dict[str, dict]` which generates an open-ended schema. The error specifically references `context=('properties', 'new_topics', 'additionalProperties')`.

**Why:** OpenAI's Structured Output API mandates `additionalProperties: false`. Pydantic does not include it by default for `dict` types. This blocks ALL experiments using `--provider openai` (crashes at `_direct_resolve()`).

**How to apply:** Any experiment using the OpenAI provider will fail on the first agent step at `_direct_resolve()`. Must fix `MapChangesSpec` (replace `dict[str, dict]` with typed Pydantic models) before running OpenAI experiments. The Anthropic provider is unaffected.
