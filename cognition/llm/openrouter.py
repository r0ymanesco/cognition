"""OpenRouter LLM provider.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1
with access to many models (Llama, Mistral, Gemini, Claude, etc.).

Extended thinking is supported via the `reasoning` extra param in the API.
Pass budget_tokens in extra_kwargs to enable it.
"""

from __future__ import annotations

from typing import Any

from cognition.llm.openai import OpenAILLM


class OpenRouterLLM(OpenAILLM):
    """LLM provider using OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        extra_kwargs = dict(extra_kwargs) if extra_kwargs else {}

        # Transform reasoning params into OpenRouter's unified format.
        # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
        #
        # Accepts:
        #   budget_tokens=N   → reasoning.max_tokens (direct token budget, for Anthropic models)
        #   reasoning_effort=X → reasoning.effort (effort level, for OpenAI o-series models)
        reasoning_config: dict[str, Any] = {}
        if "budget_tokens" in extra_kwargs:
            budget = max(1024, int(extra_kwargs.pop("budget_tokens")))
            reasoning_config["max_tokens"] = budget
        if "reasoning_effort" in extra_kwargs:
            reasoning_config["effort"] = extra_kwargs.pop("reasoning_effort")
        if reasoning_config:
            extra_kwargs["reasoning"] = reasoning_config

        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            extra_kwargs=extra_kwargs,
        )
