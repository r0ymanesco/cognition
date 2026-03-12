"""OpenRouter LLM provider.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1
with access to many models (Llama, Mistral, Gemini, Claude, etc.).
"""

from __future__ import annotations

from cognition.llm.openai import OpenAILLM


class OpenRouterLLM(OpenAILLM):
    """LLM provider using OpenRouter's OpenAI-compatible API."""

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str | None = None,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
