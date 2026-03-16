"""Agent loop — sequences three phases of the Cognitive Step.

Each phase is the same execute() call with a different objective:
1. Recall — "what do I need to know for this input?"
2. Reason — "what should I do/respond?"
3. Integrate — "what should I remember from this?"

Same mechanism, different objectives.
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
    Each step runs three phases of the cognitive primitive.
    """

    def __init__(
        self,
        llm: BaseLLM,
        state: StateStore | None = None,
        system_prompt: str = "",
        max_width: int = 5,
        max_steps: int = 20,
        tracer: TraceLogger | None = None,
    ):
        self.llm = llm
        self.state = state or StateStore()
        self.system_prompt = system_prompt
        self.cognitive_step = CognitiveStep(llm, max_width, max_steps)
        self.tracer = tracer or TraceLogger()
        self.step_count = 0

    async def step(self, input_text: str) -> str:
        """Process a single input through the three cognitive phases."""
        context: dict[str, Any] = {
            "input": input_text,
            "step_number": self.step_count,
            "system_prompt": self.system_prompt,
        }

        logger.info("agent.step %d: %s", self.step_count, input_text[:100])

        # Phase 1: RECALL — "what do I need to know?"
        recall_result = await self.cognitive_step.execute(
            objective=f"Recall relevant information for: {input_text}",
            state=self.state,
            context=context,
            tracer=self.tracer,
        )
        logger.debug("agent.recall result=%s", recall_result.output[:200] if recall_result.output else "(empty)")

        # Phase 2: REASON — "what should I do/respond?"
        context["recalled"] = recall_result
        reason_result = await self.cognitive_step.execute(
            objective=f"Given the recalled context, respond to: {input_text}",
            state=self.state,
            context=context,
            tracer=self.tracer,
        )
        logger.debug("agent.reason result=%s", reason_result.output[:200] if reason_result.output else "(empty)")

        # Phase 3: INTEGRATE — "what should I remember?"
        context["output"] = reason_result
        await self.cognitive_step.execute(
            objective=f"Integrate new information from this interaction into memory. Input was: {input_text}. Response was: {reason_result.output}",
            state=self.state,
            context=context,
            tracer=self.tracer,
        )
        logger.debug("agent.integrate complete")

        self.step_count += 1
        return reason_result.output

    async def run(self, inputs: list[str]) -> list[str]:
        """Process a sequence of inputs, maintaining state across all steps."""
        outputs: list[str] = []
        for input_text in inputs:
            output = await self.step(input_text)
            outputs.append(output)
        return outputs
