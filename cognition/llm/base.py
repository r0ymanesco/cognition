"""Abstract LLM interface.

Provider-agnostic abstraction. The cognitive step needs two capabilities:
- generate: free-form text response
- generate_structured: JSON response conforming to a Pydantic model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        system: str = "",
    ) -> str:
        """Generate a free-form text response."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        system: str = "",
    ) -> T:
        """Generate a response conforming to a Pydantic model.

        The provider is responsible for ensuring the response parses
        into the given model — via tool_use, response_format, or
        prompt-based JSON extraction as appropriate.
        """
        ...
