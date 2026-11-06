#!/usr/bin/env bash
# Memory ablations on the multi-order prequential suite built by build_datasets.sh:
# the 11-method core run, nine query orders, the retrieval-budget sweep (k = 2..12),
# group-first/group-later rescoring, the specialized-model cohort and the report.

set -euo pipefail
source "$(dirname "$0")/common.sh"

SUITE=${PROCESSED}/memory_ablation
RUN=${RESULTS}/memory_ablation
CORE_METHODS=(
  llm_only static_multiview_rag react_static_memory rolling_context_icl dynamic_case_rag
  react_case_only react_prediction_feedback react_trajectory_feedback full_score_only
  full_delayed_feedback matrecall
)
ORDER_METHODS=(react_static_memory react_case_only matrecall)
COMMON=(
  "${SAMPLING[@]}" --max-tokens 2048 --planner-max-tokens 768 --evidence-budget 8 --history-k 4
  --feedback-delay-steps 4 --max-attempts 2 --task-concurrency 16
)
require "${SUITE}/manifest.json"

# ablate RUN_DIR STREAM EXPERIENCE_K PROMPT_SET METHOD...
ablate() {
  local run=$1 stream=$2 k=$3 prompt_set=$4
  shift 4
  matrecall run_memory_ablation --release "${SUITE}/streams/${stream}" "${COMMON[@]}" \
    --experience-k "${k}" --prompt-set "${prompt_set}" --methods "$@" --output "${run}/evaluation"
}

ablate "${RUN}/core/run_core_0" iid/seed-0 6 original "${CORE_METHODS[@]}"
for index in 0 1 2 3 4 5 6 7 8; do
  if (( index < 4 )); then stream=iid/seed-$((index + 1)); else stream=group_blocked/seed-$((index - 4)); fi
  ablate "${RUN}/order/run_order_${index}" "${stream}" 6 original "${ORDER_METHODS[@]}"
done
BUDGETS=(2 4 6 8 10 12)
for index in "${!BUDGETS[@]}"; do
  ablate "${RUN}/budget/run_budget_${index}" iid/seed-0 "${BUDGETS[${index}]}" revised matrecall
done
if [[ "${DRY_RUN}" == 1 ]]; then
  exit 0
fi

# Native metrics on all queries and on the first query of each group.
score_run() {
  local run=$1
  shift
  score "${run}/evaluation/combined" "${run}/metrics" "$@"
  matrecall prepare_ablation_metric_subsets --run "${run}/evaluation" \
    --output "${run}/metric_subsets/group_first" --methods "$@" --subset group_first
  score "${run}/metric_subsets/group_first" "${run}/metrics_group_first" "$@"
}
score_run "${RUN}/core/run_core_0" "${CORE_METHODS[@]}"
for index in 0 1 2 3 4 5 6 7 8; do
  score_run "${RUN}/order/run_order_${index}" "${ORDER_METHODS[@]}"
done
for index in "${!BUDGETS[@]}"; do
  score_run "${RUN}/budget/run_budget_${index}" matrecall
done

# Queries whose group was already seen earlier in the stream (core and order runs).
for run in "${RUN}/core/run_core_0" "${RUN}"/order/run_order_*; do
  matrecall prepare_ablation_metric_subsets --run "${run}/evaluation" \
    --output "${run}/metric_subsets/group_later" --methods react_static_memory matrecall \
    --subset group_later
  score "${run}/metric_subsets/group_later" "${run}/metrics_group_later" react_static_memory matrecall
done

SPECIALIZED_ARGS=()
if [[ -f "${BASELINES}/process/test_predictions.jsonl" ]]; then
  SPECIALIZED=${RESULTS}/memory_ablation_specialized
  matrecall prepare_specialized_rescore --release "${SUITE}/streams/iid/seed-0" \
    "${SPECIALIZED_INPUTS[@]}" --output "${SPECIALIZED}/matched"
  score_specialized "${SPECIALIZED}"
  matrecall report_specialized_rescore --verification "${SPECIALIZED}/matched/manifest.json" \
    --unified-metrics "${SPECIALIZED}/metrics/unified.json" \
    --process-metrics "${SPECIALIZED}/metrics/process.json" --output "${SPECIALIZED}/summary.json"
  SPECIALIZED_ARGS=(--specialized-summary "${SPECIALIZED}/summary.json")
else
  echo "skipping the specialized-model cohort: run build_datasets.sh and run_baselines.sh first" >&2
fi

matrecall report_memory_ablation --suite "${SUITE}" --core-run "${RUN}/core/run_core_0" \
  --order-root "${RUN}/order" --order-array-job-id order \
  --budget-root "${RUN}/budget" --budget-array-job-ids budget \
  ${SPECIALIZED_ARGS[@]+"${SPECIALIZED_ARGS[@]}"} --output "${RESULTS}/reports/memory_ablation"
