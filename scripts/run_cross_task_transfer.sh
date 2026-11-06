#!/usr/bin/env bash
# Cross-task transfer: 8 target tasks answered from the frozen memory of a related and of an
# unrelated source task (16 pairs, 235 target queries). The metric join reuses the
# memory-retention run and its specialized cohort (run_memory_retention.sh) as controls.

set -euo pipefail
source "$(dirname "$0")/common.sh"

RELEASE=${DATA}/cross_task_transfer
RUN=${RESULTS}/cross_task_transfer
METHODS=(case_episodic_only full_episodic_only dynamic_case_rag matrecall)
require "${RELEASE}/manifest.json"

matrecall run_frozen_memory --release "${RELEASE}" "${SAMPLING[@]}" --prompt-set revised \
  --max-tokens 2048 --planner-max-tokens 768 --evidence-budget 8 --history-k 4 --experience-k 6 \
  --max-attempts 2 --task-concurrency 16 --methods "${METHODS[@]}" --output "${RUN}/evaluation"
if [[ "${DRY_RUN}" == 1 ]]; then
  exit 0
fi

require "${RESULTS}/memory_retention/metric_inputs" "${RESULTS}/memory_retention_specialized/matched"
matrecall prepare_cross_task_metrics --release "${RELEASE}" --run "${RUN}/evaluation" \
  --base-run "${RESULTS}/memory_retention" --specialized-run "${RESULTS}/memory_retention_specialized" \
  --output "${RUN}/metric_inputs"
ALL_METHODS=$("${PYTHON}" -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["all_methods"]))' \
  "${RUN}/metric_inputs/manifest.json")
for pair in "${RUN}"/metric_inputs/*/; do
  pair=$(basename "${pair}")
  mkdir -p "${RUN}/metrics/${pair}"
  for method in ${ALL_METHODS}; do
    matrecall evaluate_unified_predictions \
      --predictions "${RUN}/metric_inputs/${pair}/${method}/predictions.jsonl" \
      --gold "${RUN}/metric_inputs/${pair}/${method}/gold.jsonl" \
      --output "${RUN}/metrics/${pair}/${method}.json" ${UNIFIED_RELEASE[@]+"${UNIFIED_RELEASE[@]}"}
  done
done
matrecall report_cross_task_transfer --release "${RELEASE}" --metric-inputs "${RUN}/metric_inputs" \
  --metrics "${RUN}/metrics" --output "${RESULTS}/reports/cross_task_transfer"
