#!/usr/bin/env bash
#
# Run the toy task experiment with different configurations.
#
# Traces and logs go to: traces/toy_task/<timestamp>/<run_name>/
#
# Usage:
#   ./scripts/toy_experiment.sh                          # default (anthropic, depth=3)
#   ./scripts/toy_experiment.sh --provider openrouter --model meta-llama/llama-3-70b
#   ./scripts/toy_experiment.sh --max-depth 0            # flat baseline
#   ./scripts/toy_experiment.sh --run-ablation            # run ablation suite
#
# All arguments are passed through to `python -m experiments.toy_task`
# unless --run-ablation is specified.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT="toy_task"
TRACES_ROOT="$PROJECT_DIR/traces"

# Shared timestamp for all runs in a single invocation
SESSION_TS="$(date +"%Y%m%d_%H%M%S")"

run_single() {
    local name="$1"
    shift

    local run_dir="$TRACES_ROOT/$EXPERIMENT/$SESSION_TS/$name"
    mkdir -p "$run_dir"

    local trace_file="$run_dir/trace.json"
    local log_file="$run_dir/run.log"
    local state_file="$run_dir/state.json"

    echo "═══════════════════════════════════════════════════════"
    echo "  Experiment: $EXPERIMENT / $name"
    echo "  Args: $*"
    echo "  Output: $run_dir"
    echo "═══════════════════════════════════════════════════════"

    python -m experiments.toy_task \
        --trace-file "$trace_file" \
        "$@" 2>&1 | tee "$log_file"

    # Move the state dump into the run dir if it was created
    if [[ -f "$PROJECT_DIR/toy_task_state.json" ]]; then
        mv "$PROJECT_DIR/toy_task_state.json" "$state_file"
    fi

    echo ""
    echo "  Output: $run_dir"
    echo ""
}

run_ablation() {
    local provider="${1:-anthropic}"
    local model="${2:-}"
    local model_args=()

    if [[ -n "$model" ]]; then
        model_args=(--model "$model")
    fi

    echo "Running ablation suite with provider=$provider model=${model:-default}"
    echo "Session: $TRACES_ROOT/$EXPERIMENT/$SESSION_TS"
    echo ""

    # Width ablation: 1, 3, 5
    for width in 1 3 5; do
        run_single "width_${width}" \
            --provider "$provider" "${model_args[@]}" \
            --max-width "$width"
    done

    # Steps ablation: 5, 10, 20
    for steps in 5 10 20; do
        run_single "steps_${steps}" \
            --provider "$provider" "${model_args[@]}" \
            --max-steps "$steps"
    done

    echo "═══════════════════════════════════════════════════════"
    echo "  Ablation complete."
    echo "  Results: $TRACES_ROOT/$EXPERIMENT/$SESSION_TS"
    echo "═══════════════════════════════════════════════════════"
}

# --- Main ---

# Parse --name before dispatching
RUN_NAME=""
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            REMAINING_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${REMAINING_ARGS[@]+"${REMAINING_ARGS[@]}"}"

if [[ "${1:-}" == "--run-ablation" ]]; then
    shift
    run_ablation "$@"
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage:"
    echo "  ./scripts/toy_experiment.sh [--name <name>] [args]"
    echo "  ./scripts/toy_experiment.sh --run-ablation [provider] [model]"
    echo ""
    echo "Options:"
    echo "  --name <name>    Name for this run (used as subfolder name)"
    echo ""
    echo "Traces go to: traces/toy_task/<timestamp>/<run_name>/"
    echo "  Each run dir contains: trace.json, run.log, state.json"
    echo ""
    echo "Examples:"
    echo "  ./scripts/toy_experiment.sh --name baseline --max-depth 0"
    echo "  ./scripts/toy_experiment.sh --name sonnet --provider anthropic"
    echo "  ./scripts/toy_experiment.sh --name llama70b --provider openrouter --model meta-llama/llama-3-70b"
    echo "  ./scripts/toy_experiment.sh --run-ablation anthropic"
    echo ""
    python -m experiments.toy_task --help
else
    run_single "${RUN_NAME:-default}" "$@"
fi
