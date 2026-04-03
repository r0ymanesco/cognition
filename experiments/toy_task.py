"""Toy task: entity tracking with updates and contradictions.

Validates the scaffold end-to-end by feeding the agent a stream of
messages that introduce facts about entities, update them, and ask
recall questions. Tests whether the agent:

1. Stores facts correctly
2. Recalls facts across many steps
3. Handles corrections (belief revision / invalidation)
4. Returns the updated value after a correction

Usage:
    python -m experiments.toy_task --provider anthropic --model claude-sonnet-4-20250514
    python -m experiments.toy_task --provider openai --model gpt-4o
    python -m experiments.toy_task --provider openrouter --model anthropic/claude-sonnet-4

    # With tracing output
    python -m experiments.toy_task --provider anthropic --trace-file trace.json

    # Flat baseline (no recursion)
    python -m experiments.toy_task --provider anthropic --max-depth 0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from cognition.agent import Agent
from cognition.llm.base import BaseLLM
from cognition.state import StateStore
from cognition.tracing import TraceLogger


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


@dataclass
class TaskStep:
    input_text: str
    expected_answer: str | None = None  # None for statements, string for questions


def build_task() -> list[TaskStep]:
    """Build the entity tracking task.

    Structure:
    - Introduce facts about multiple entities
    - Intersperse with filler statements to push facts out of easy reach
    - Ask recall questions
    - Introduce corrections (contradictions)
    - Ask recall questions again (should reflect corrected values)
    """
    steps: list[TaskStep] = []

    # --- Phase 1: Establish facts ---
    steps.append(TaskStep("Alice has 5 apples."))
    steps.append(TaskStep("Bob has 3 oranges."))
    steps.append(TaskStep("Charlie has 10 bananas."))
    steps.append(TaskStep("Diana has 2 pears."))
    steps.append(TaskStep("Eve has 7 grapes."))

    # --- Phase 2: Filler to push facts further back ---
    steps.append(TaskStep("The weather today is sunny."))
    steps.append(TaskStep("The market opens at 9am."))
    steps.append(TaskStep("Fruit prices have been stable this week."))
    steps.append(TaskStep("A new shipment of mangoes arrived yesterday."))
    steps.append(TaskStep("The warehouse temperature is 4 degrees Celsius."))

    # --- Phase 3: First recall questions ---
    steps.append(TaskStep(
        "Question: How many apples does Alice have?",
        expected_answer="5",
    ))
    steps.append(TaskStep(
        "Question: How many oranges does Bob have?",
        expected_answer="3",
    ))
    steps.append(TaskStep(
        "Question: How many bananas does Charlie have?",
        expected_answer="10",
    ))

    # --- Phase 4: More filler ---
    steps.append(TaskStep("Transport trucks arrive on Tuesdays and Fridays."))
    steps.append(TaskStep("The inventory system was updated last month."))
    steps.append(TaskStep("Bob mentioned he might trade some oranges."))
    steps.append(TaskStep("Alice visited the market this morning."))

    # --- Phase 5: Corrections (belief revision) ---
    steps.append(TaskStep("Correction: Alice now has 7 apples."))
    steps.append(TaskStep("Correction: Charlie now has 4 bananas."))

    # --- Phase 6: More filler ---
    steps.append(TaskStep("The store will close early on Friday."))
    steps.append(TaskStep("New pricing will take effect next week."))

    # --- Phase 7: Post-correction recall ---
    steps.append(TaskStep(
        "Question: How many apples does Alice have?",
        expected_answer="7",  # Should be updated
    ))
    steps.append(TaskStep(
        "Question: How many bananas does Charlie have?",
        expected_answer="4",  # Should be updated
    ))
    steps.append(TaskStep(
        "Question: How many oranges does Bob have?",
        expected_answer="3",  # Unchanged
    ))
    steps.append(TaskStep(
        "Question: How many pears does Diana have?",
        expected_answer="2",  # Unchanged
    ))
    steps.append(TaskStep(
        "Question: How many grapes does Eve have?",
        expected_answer="7",  # Unchanged
    ))

    return steps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question: str
    expected: str
    actual: str
    correct: bool
    step_index: int


def evaluate_answer(expected: str, actual: str) -> bool:
    """Check if the expected value appears in the actual response."""
    return expected.strip().lower() in actual.strip().lower()


# ---------------------------------------------------------------------------
# LLM provider factory
# ---------------------------------------------------------------------------


def parse_llm_kwargs(raw: list[str] | None) -> dict[str, Any]:
    """Parse key=value pairs into a dict, coercing types."""
    if not raw:
        return {}
    kwargs: dict[str, Any] = {}
    for item in raw:
        key, _, value = item.partition("=")
        if not value:
            raise ValueError(f"Invalid kwarg format: {item!r} (expected key=value)")
        # Coerce types
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
        import os
        from cognition.llm.openrouter import OpenRouterLLM
        api_key = os.environ.get("OPENROUTER_API_KEY")
        return OpenRouterLLM(model=model or "anthropic/claude-sonnet-4", api_key=api_key, extra_kwargs=extra_kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_experiment(
    provider: str,
    model: str | None,
    max_width: int,
    max_steps: int,
    trace_file: str | None,
    verbose: bool,
    llm_kwargs: dict[str, Any] | None = None,
) -> None:
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    llm = create_llm(provider, model, extra_kwargs=llm_kwargs)
    state = StateStore()
    tracer = TraceLogger()

    system_prompt = (
        "You are an inventory tracking assistant. People will tell you facts "
        "about how many items various people have. Remember these facts accurately. "
        "When asked a question, answer with the correct current value based on "
        "everything you've been told, including any corrections or updates.\n\n"
        "If someone tells you a corrected value, that replaces the old value — "
        "always use the most recent information.\n\n"
        "Keep your answers concise — just state the number."
    )

    agent = Agent(
        llm=llm,
        state=state,
        system_prompt=system_prompt,
        max_width=max_width,
        max_steps=max_steps,
        tracer=tracer,
    )

    task = build_task()
    results: list[QuestionResult] = []
    total_start = time.monotonic()

    print(f"Running toy task: {len(task)} steps")
    print(f"Provider: {provider}, Model: {model or 'default'}")
    print(f"Config: max_width={max_width}, max_steps={max_steps}")
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

    # --- Results summary ---
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    total_questions = len(results)
    correct_count = sum(1 for r in results if r.correct)
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0

    # Split pre/post correction
    pre_correction = [r for r in results if r.step_index < 17]  # before corrections
    post_correction = [r for r in results if r.step_index >= 21]  # after corrections

    pre_correct = sum(1 for r in pre_correction if r.correct)
    post_correct = sum(1 for r in post_correction if r.correct)

    print(f"Overall accuracy: {correct_count}/{total_questions} ({accuracy:.0%})")
    if pre_correction:
        print(f"Pre-correction recall: {pre_correct}/{len(pre_correction)} ({pre_correct / len(pre_correction):.0%})")
    if post_correction:
        print(f"Post-correction recall: {post_correct}/{len(post_correction)} ({post_correct / len(post_correction):.0%})")

    print(f"\nTotal duration: {total_duration:.0f}ms")
    print(f"State: {state.size()} active entries, {len(state.associations)} associations")

    # Trace summary
    trace_summary = tracer.summary()
    print(f"LLM calls: {trace_summary['total_llm_calls']}")
    print(f"Max recursion depth: {trace_summary['max_recursion_depth']}")
    print(f"Total traversal steps: {trace_summary['total_traversal_steps']}")

    if trace_file:
        tracer.export_json(trace_file)
        print(f"\nTrace exported to: {trace_file}")

    # --- Failed questions ---
    failed = [r for r in results if not r.correct]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed:
            print(f"  Step {r.step_index}: {r.question}")
            print(f"    Expected: {r.expected} | Got: {r.actual[:100]}")

    # --- Dump state for inspection ---
    state.save("toy_task_state.json")
    print(f"\nState saved to: toy_task_state.json")

    # Exit code: 0 if all correct, 1 otherwise
    if not all(r.correct for r in results):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognition scaffold toy task")
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "openrouter"],
        default="anthropic", help="LLM provider",
    )
    parser.add_argument("--model", default=None, help="Model name (provider-specific)")
    parser.add_argument("--max-width", type=int, default=5, help="Max sub-objectives per orient")
    parser.add_argument("--max-steps", type=int, default=20, help="Max traversal steps per graph walk")
    parser.add_argument("--trace-file", default=None, help="Path to export JSON trace")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--llm-kwargs", nargs="*", metavar="KEY=VALUE",
        help="Extra kwargs passed to every LLM call (e.g. temperature=0.5 max_tokens=8192)",
    )
    args = parser.parse_args()

    asyncio.run(run_experiment(
        provider=args.provider,
        model=args.model,
        max_width=args.max_width,
        max_steps=args.max_steps,
        trace_file=args.trace_file,
        verbose=args.verbose,
        llm_kwargs=parse_llm_kwargs(args.llm_kwargs),
    ))


if __name__ == "__main__":
    main()
