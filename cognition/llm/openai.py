"""OpenAI LLM provider."""

from __future__ import annotations

import json
import logging
from typing import Any, Type

from pydantic import BaseModel

from cognition.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """LLM provider using the OpenAI API.

    Uses response_format with json_schema for structured output.
    Subclassed by OpenRouterLLM with a different base URL.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
    ):
        import openai

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        if default_headers:
            kwargs["default_headers"] = default_headers

        self.model = model
        self.client = openai.AsyncOpenAI(**kwargs)

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
        )
        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: Type[BaseModel],
        system: str = "",
    ) -> BaseModel:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                },
            },
        )
        raw = response.choices[0].message.content or "{}"
        return response_model.model_validate_json(raw)
