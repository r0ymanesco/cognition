"""OpenAI LLM provider."""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from cognition.llm.base import BaseLLM

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAILLM(BaseLLM):
    """LLM provider using the OpenAI API.

    Uses response_format with json_schema for structured output.
    Subclassed by OpenRouterLLM with a different base URL.

    extra_kwargs are merged into every API call. Useful for:
    - temperature, max_tokens
    - reasoning_effort (for o-series models)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(extra_kwargs=extra_kwargs)
        import openai

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self.model = model
        self.client = openai.AsyncOpenAI(**client_kwargs)

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        msgs: list[dict[str, Any]] = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, msgs),
            **self.extra_kwargs,
        )
        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        system: str = "",
    ) -> T:
        msgs: list[dict[str, Any]] = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=cast(Any, msgs),
            response_format=response_model,
            **self.extra_kwargs,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"Failed to parse response into {response_model.__name__}")
        return parsed
