"""Agent — thin wrapper around the Cognitive Step.

The agent receives inputs, passes them through a single cognitive step
execution, and returns the result. The system prompt defines the agent's
role; the cognitive step handles recall, reasoning, and integration
naturally during graph traversal.
"""

from __future__ import annotations

import logging
from typing import Any

from cognition.cognitive_step import CognitiveStep
from cognition.llm.base import BaseLLM
from cognition.state import StateStore
from cognition.tracing import TraceLogger

logger = logging.getLogger(__name__)


class Agent:
    """Stateful agent backed by a cognitive scaffold.

    Processes inputs one at a time, maintaining state across all steps.
    Each step is a single cognitive process — recall, reasoning, and
    integration emerge naturally from the graph traversal rather than
    being separate phases.
    """

    def __init__(
        self,
        llm: BaseLLM,
        state: StateStore | None = None,
        system_prompt: str = "",
        max_width: int = 5,
        max_steps: int = 20,
        max_context_tokens: int | None = None,
        max_map_tokens: int | None = None,
        tracer: TraceLogger | None = None,
    ):
        self.llm = llm
        self.state = state or StateStore()
        self.system_prompt = system_prompt
        self.cognitive_step = CognitiveStep(
            llm, max_width, max_steps, max_context_tokens, max_map_tokens,
        )
        self.tracer = tracer or TraceLogger()
        self.step_count = 0

    async def step(self, input_text: str) -> str:
        """Process a single input through the cognitive step."""
        context: dict[str, Any] = {
            "input": input_text,
            "step_number": self.step_count,
            "system_prompt": self.system_prompt,
        }

        logger.info("agent.step %d: %s", self.step_count, input_text[:100])

        result = await self.cognitive_step.execute(
            objective=input_text,
            state=self.state,
            context=context,
            tracer=self.tracer,
        )

        self.step_count += 1
        return result.output

    async def run(self, inputs: list[str]) -> list[str]:
        """Process a sequence of inputs, maintaining state across all steps."""
        outputs: list[str] = []
        for input_text in inputs:
            output = await self.step(input_text)
            outputs.append(output)
        return outputs
