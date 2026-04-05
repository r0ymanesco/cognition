"""Baseline: plain LLM with conversation history (no cognitive scaffold).

Same task as toy_task.py but without the graph store, memory map, or
cognitive step. The LLM sees the full conversation history in its
context window and responds directly.

This is the control experiment — if the scaffold doesn't beat this,
it's not adding value.

Usage:
    python -m experiments.toy_task_baseline --provider anthropic --model claude-sonnet-4-6
    python -m experiments.toy_task_baseline --provider openai --model gpt-5.4
    python -m experiments.toy_task_baseline --provider openai --model gpt-5.4 --llm-kwargs reasoning_effort=high
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from cognition.llm.base import BaseLLM
from experiments.tasks import build_scaled_task, build_toy_task


# Task definition imported from experiments.tasks


# ---------------------------------------------------------------------------
# Evaluation (identical to toy_task.py)
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question: str
    expected: str
    actual: str
    correct: bool
    step_index: int


def evaluate_answer(expected: str, actual: str) -> bool:
    return expected.lower() in actual.lower()


# ---------------------------------------------------------------------------
# LLM factory (identical to toy_task.py)
# ---------------------------------------------------------------------------


def parse_llm_kwargs(raw: list[str] | None) -> dict[str, Any]:
    if not raw:
        return {}
    kwargs: dict[str, Any] = {}
    for item in raw:
        key, _, value = item.partition("=")
        if not value:
            raise ValueError(f"Invalid kwarg format: {item!r} (expected key=value)")
        if value.lower() in ("true", "false"):
            kwargs[key] = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).isdigit():
            kwargs[key] = float(value) if "." in value else int(value)
        else:
            kwargs[key] = value
    return kwargs


def create_llm(
    provider: str,
    model: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> BaseLLM:
    if provider == "anthropic":
        from cognition.llm.anthropic import AnthropicLLM
        return AnthropicLLM(model=model or "claude-sonnet-4-20250514", extra_kwargs=extra_kwargs)
    elif provider == "openai":
        from cognition.llm.openai import OpenAILLM
        return OpenAILLM(model=model or "gpt-4o", extra_kwargs=extra_kwargs)
    elif provider == "openrouter":
        from cognition.llm.openrouter import OpenRouterLLM
        api_key = os.environ.get("OPENROUTER_API_KEY")
        return OpenRouterLLM(model=model or "anthropic/claude-sonnet-4", api_key=api_key, extra_kwargs=extra_kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Baseline agent: plain conversation history
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token)."""
    return len(text) // 4


class BaselineAgent:
    """No scaffold — just conversation history in the LLM context.

    If max_tokens is set, the conversation history is truncated from
    the front (oldest messages dropped first) to fit within the token
    budget. This simulates a constrained context window.

    The system prompt is NOT counted against the budget — it's always
    sent. This matches the scaffold, where the system prompt is always
    present regardless of graph size.
    """

    def __init__(self, llm: BaseLLM, system_prompt: str = "",
                 max_tokens: int | None = None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history: list[dict[str, str]] = []
        self.max_tokens = max_tokens
        self.messages_dropped = 0

    def _truncate_to_budget(self) -> list[dict[str, str]]:
        """Return history truncated to fit within max_tokens."""
        if self.max_tokens is None:
            return list(self.history)

        # Walk backwards, accumulating tokens until we hit the budget
        budget = self.max_tokens
        included: list[dict[str, str]] = []
        for msg in reversed(self.history):
            msg_tokens = estimate_tokens(msg["content"])
            if budget - msg_tokens < 0 and included:
                break
            budget -= msg_tokens
            included.append(msg)

        dropped = len(self.history) - len(included)
        if dropped > self.messages_dropped:
            self.messages_dropped = dropped

        included.reverse()
        return included

    async def step(self, input_text: str) -> str:
        self.history.append({"role": "user", "content": input_text})

        messages = self._truncate_to_budget()

        response = await self.llm.generate(
            messages=messages,
            system=self.system_prompt,
        )

        self.history.append({"role": "assistant", "content": response})
        return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_experiment(
    provider: str,
    model: str | None,
    max_tokens: int | None,
    verbose: bool,
    verbose_facts: bool = False,
    scaled_entities: int | None = None,
    seed: int = 42,
    llm_kwargs: dict[str, Any] | None = None,
) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    llm = create_llm(provider, model, extra_kwargs=llm_kwargs)

    system_prompt = (
        "You are an inventory tracking assistant. People will tell you facts "
        "about how many items various people have. Remember these facts accurately. "
        "When asked a question, answer with the correct current value based on "
        "everything you've been told, including any corrections or updates.\n\n"
        "If someone tells you a corrected value, that replaces the old value — "
        "always use the most recent information.\n\n"
        "Keep your answers concise — just state the number and item."
    )

    agent = BaselineAgent(llm=llm, system_prompt=system_prompt, max_tokens=max_tokens)

    if scaled_entities:
        task = build_scaled_task(num_entities=scaled_entities, seed=seed)
    else:
        task = build_toy_task(verbose=verbose_facts)
    results: list[QuestionResult] = []
    total_start = time.monotonic()

    print(f"Running toy task BASELINE (no scaffold): {len(task)} steps")
    print(f"Provider: {provider}, Model: {model or 'default'}")
    print(f"Context budget: {f'{max_tokens} tokens' if max_tokens else 'unlimited'}")
    print(f"LLM kwargs: {llm_kwargs or 'none'}")
    print("-" * 60)

    for i, step in enumerate(task):
        step_start = time.monotonic()
        output = await agent.step(step.input_text)
        step_duration = (time.monotonic() - step_start) * 1000

        if step.expected_answer is not None:
            correct = evaluate_answer(step.expected_answer, output)
            results.append(QuestionResult(
                question=step.input_text,
                expected=step.expected_answer,
                actual=output,
                correct=correct,
                step_index=i,
            ))
            marker = "PASS" if correct else "FAIL"
            print(f"  [{i:2d}] [{marker}] {step.input_text}")
            print(f"        Expected: {step.expected_answer} | Got: {output[:100]}")
        else:
            print(f"  [{i:2d}] [STMT] {step.input_text} ({step_duration:.0f}ms)")

    total_duration = (time.monotonic() - total_start) * 1000

    # Results summary
    print()
    print("=" * 60)
    print("RESULTS (BASELINE — no scaffold)")
    print("=" * 60)

    total_questions = len(results)
    correct_count = sum(1 for r in results if r.correct)
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0

    pre_correction = [r for r in results if r.step_index < 17]
    post_correction = [r for r in results if r.step_index >= 21]

    pre_correct = sum(1 for r in pre_correction if r.correct)
    post_correct = sum(1 for r in post_correction if r.correct)

    print(f"Overall accuracy: {correct_count}/{total_questions} ({accuracy:.0%})")
    if pre_correction:
        print(f"Pre-correction recall: {pre_correct}/{len(pre_correction)} ({pre_correct / len(pre_correction):.0%})")
    if post_correction:
        print(f"Post-correction recall: {post_correct}/{len(post_correction)} ({post_correct / len(post_correction):.0%})")

    print(f"\nTotal duration: {total_duration:.0f}ms")
    print(f"LLM calls: {len(task)} (1 per step)")
    print(f"Conversation history: {len(agent.history)} messages")
    if max_tokens:
        print(f"Context budget: {max_tokens} tokens")
        print(f"Max messages dropped: {agent.messages_dropped}")

    # Failed questions
    failed = [r for r in results if not r.correct]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed:
            print(f"  Step {r.step_index}: {r.question}")
            print(f"    Expected: {r.expected} | Got: {r.actual[:100]}")

    if not all(r.correct for r in results):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognition toy task BASELINE (no scaffold)")
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "openrouter"],
        default="anthropic", help="LLM provider",
    )
    parser.add_argument("--model", default=None, help="Model name (provider-specific)")
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Max tokens for conversation history (truncates oldest messages). "
             "Simulates a constrained context window.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose-facts", action="store_true",
                        help="Use verbose fact descriptions (higher token count per message)")
    parser.add_argument("--scaled-entities", type=int, default=None,
                        help="Use scaled task with N entities (overrides --verbose-facts)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for scaled task generation (default: 42)")
    parser.add_argument(
        "--llm-kwargs", nargs="*", metavar="KEY=VALUE",
        help="Extra kwargs passed to every LLM call (e.g. temperature=0.5 reasoning_effort=high)",
    )
    args = parser.parse_args()

    asyncio.run(run_experiment(
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        verbose_facts=args.verbose_facts,
        scaled_entities=args.scaled_entities,
        seed=args.seed,
        llm_kwargs=parse_llm_kwargs(args.llm_kwargs),
    ))


if __name__ == "__main__":
    main()
