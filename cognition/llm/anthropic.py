"""Anthropic Claude LLM provider."""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from cognition.llm.base import BaseLLM

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """LLM provider using the Anthropic API.

    Uses tool_use for structured output — Claude is instructed to call
    a tool whose input schema matches the desired Pydantic model.

    extra_kwargs are merged into every API call. Useful for:
    - temperature, max_tokens
    - thinking (extended thinking config)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(extra_kwargs=extra_kwargs)
        import anthropic

        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
            **self.extra_kwargs,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        system: str = "",
    ) -> T:
        schema = response_model.model_json_schema()
        tool_name = response_model.__name__

        tool = {
            "name": tool_name,
            "description": f"Return a {tool_name} object.",
            "input_schema": schema,
        }

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": tool_name},
            **self.extra_kwargs,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return response_model.model_validate(block.input)

        raise ValueError(f"No tool_use block found for {tool_name} in response")
