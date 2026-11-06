#!/usr/bin/env bash
# Memory retention: six methods answer group-disjoint queries (463 cases, 16 tasks) from a
# frozen online-adaptation memory, plus the episodic-only ablation. Scoring and reports need
# the specialized-model baselines (build_datasets.sh, run_baselines.sh).

set -euo pipefail
source "$(dirname "$0")/common.sh"

RELEASE=${DATA}/memory_retention
RUN=${RESULTS}/memory_retention
EPISODIC=${RESULTS}/retention_episodic_only
SPECIALIZED=${RESULTS}/memory_retention_specialized
METHODS=(llm_only static_multiview_rag react_static_memory rolling_context_icl dynamic_case_rag matrecall)
EPISODIC_METHODS=(case_episodic_only full_episodic_only)
COMMON=(
  --release "${RELEASE}" "${SAMPLING[@]}" --max-tokens 2048 --planner-max-tokens 768
  --evidence-budget 8 --history-k 4 --experience-k 6 --max-attempts 2 --task-concurrency 16
)
require "${RELEASE}/manifest.json"

matrecall run_frozen_memory "${COMMON[@]}" --prompt-set original --methods "${METHODS[@]}" \
  --output "${RUN}/evaluation"
matrecall run_frozen_memory "${COMMON[@]}" --prompt-set revised --methods "${EPISODIC_METHODS[@]}" \
  --exclude-task meltpool_shape --output "${EPISODIC}/evaluation"
if [[ "${DRY_RUN}" == 1 ]]; then
  exit 0
fi

matrecall prepare_retention_metrics --release "${RELEASE}" --run "${RUN}/evaluation" \
  --output "${RUN}/metric_inputs" --methods "${METHODS[@]}"
score "${RUN}/metric_inputs" "${RUN}/metrics" "${METHODS[@]}"
matrecall prepare_retention_metrics --release "${RELEASE}" --run "${EPISODIC}/evaluation" \
  --output "${EPISODIC}/metric_inputs" --methods "${EPISODIC_METHODS[@]}" --exclude-task meltpool_shape
score "${EPISODIC}/metric_inputs" "${EPISODIC}/metrics" "${EPISODIC_METHODS[@]}"

require "${BASELINES}/process/test_predictions.jsonl"
matrecall prepare_retention_specialized --release "${RELEASE}" "${SPECIALIZED_INPUTS[@]}" \
  --output "${SPECIALIZED}/matched"
score_specialized "${SPECIALIZED}"

matrecall report_memory_retention --release "${RELEASE}" --run "${RUN}" \
  --specialized-run "${SPECIALIZED}" --output "${RESULTS}/reports/memory_retention"
matrecall report_retention_episodic_ablation --release "${RELEASE}" --base-run "${RUN}" \
  --episodic-run "${EPISODIC}" --specialized-run "${SPECIALIZED}" --exclude-task meltpool_shape \
  --output "${RESULTS}/reports/retention_episodic_ablation"
