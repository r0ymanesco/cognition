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

        # If budget_tokens is set, configure extended thinking
        self._thinking_budget: int | None = None
        if "budget_tokens" in self.extra_kwargs:
            self._thinking_budget = int(self.extra_kwargs.pop("budget_tokens"))

    def _build_kwargs(self, messages: list[dict[str, str]], system: str) -> dict[str, Any]:
        """Build API kwargs, handling extended thinking config."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.extra_kwargs,
        }
        if self._thinking_budget:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }
            # With thinking enabled, max_tokens must be > budget_tokens
            kwargs["max_tokens"] = self._thinking_budget + 4096
        else:
            kwargs["max_tokens"] = kwargs.get("max_tokens", 4096)
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        kwargs = self._build_kwargs(messages, system)
        response = await self.client.messages.create(**kwargs)
        # With thinking enabled, the text block may not be first
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

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

        kwargs = self._build_kwargs(messages, system)
        kwargs["tools"] = [tool]
        # Extended thinking doesn't support forced tool_choice
        if self._thinking_budget:
            kwargs["tool_choice"] = {"type": "auto"}
        else:
            kwargs["tool_choice"] = {"type": "tool", "name": tool_name}

        response = await self.client.messages.create(**kwargs)

        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return response_model.model_validate(block.input)

        raise ValueError(f"No tool_use block found for {tool_name} in response")
