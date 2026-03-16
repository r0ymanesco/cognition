---
name: OpenAI structured output schema incompatibility
description: TraversalStepResponse schema fails with OpenAI provider due to dict[str, Model] fields in MapChangesSpec -- blocks all OpenAI experiments
type: project
---

The `TraversalStepResponse` Pydantic model (in `cognition/cognitive_step.py`) is incompatible with OpenAI's structured output API. The `MapChangesSpec` nested model uses `dict[str, TopicSpec]` and `dict[str, TopicUpdateSpec]` fields, which produce JSON Schema with `additionalProperties: { "$ref": ... }`. OpenAI's strict mode rejects this.

**Why:** OpenAI's `response_format` strict mode requires all properties in `required` and cannot handle `additionalProperties` with `$ref` types. The openai SDK v2.26.0 + pydantic 2.12.5 produce schemas that trigger this. Error: "Extra required key 'new_topics' supplied."

**How to apply:** When running experiments with `--provider openai`, expect this failure until the schema is fixed. The `OrientationResponse` and `SynthesisResponse` schemas work fine individually -- only `TraversalStepResponse` fails. Recommend restructuring `dict[str, Model]` to `list[NamedModel]` or upgrading the SDK.
