#!/usr/bin/env bash
# Online adaptation: six methods over 22 prequential task streams (942 queries), the
# episodic-only ablation on six tasks, native metrics and reports. The specialized-model
# comparison runs when the outputs of build_datasets.sh and run_baselines.sh exist.

set -euo pipefail
source "$(dirname "$0")/common.sh"

RELEASE=${DATA}/online_adaptation
RUN=${RESULTS}/online_adaptation
EPISODIC=${RESULTS}/online_episodic_only
METHODS=(llm_only static_multiview_rag react_static_memory rolling_context_icl dynamic_case_rag matrecall)
EPISODIC_METHODS=(case_episodic_only full_episodic_only)
EPISODIC_TASKS=(
  bulk_modulus_kv composition_to_phase_assemblage mbj_bandgap
  target_and_precursors_to_heating_conditions target_composition_to_precursor_set
  raccuglia_crystallization_success
)
COMMON=(
  --release "${RELEASE}" "${SAMPLING[@]}" --max-tokens 4096 --planner-max-tokens 1024
  --evidence-budget 8 --history-k 4 --experience-k 6 --task-concurrency 16
)
require "${RELEASE}/manifest.json"

matrecall run_online_adaptation "${COMMON[@]}" --prompt-set original --methods "${METHODS[@]}" \
  --output "${RUN}/evaluation"
matrecall run_online_adaptation "${COMMON[@]}" --prompt-set revised --methods "${EPISODIC_METHODS[@]}" \
  --tasks "${EPISODIC_TASKS[@]}" --output "${EPISODIC}/evaluation"
if [[ "${DRY_RUN}" == 1 ]]; then
  exit 0
fi

score "${RUN}/evaluation/combined" "${RUN}/metrics" "${METHODS[@]}"
score "${EPISODIC}/evaluation/combined" "${EPISODIC}/metrics" "${EPISODIC_METHODS[@]}"
matrecall report_online_adaptation --release "${RELEASE}" --run "${RUN}" \
  --output "${RESULTS}/reports/online_adaptation"
matrecall report_online_episodic_ablation --release "${RELEASE}" --base-run "${RUN}" \
  --episodic-run "${EPISODIC}" --output "${RESULTS}/reports/online_episodic_ablation"

if [[ -f "${BASELINES}/process/test_predictions.jsonl" ]]; then
  SPECIALIZED=${RESULTS}/online_specialized
  matrecall prepare_specialized_rescore --release "${RELEASE}" "${SPECIALIZED_INPUTS[@]}" \
    --output "${SPECIALIZED}/matched"
  score_specialized "${SPECIALIZED}"
  matrecall report_specialized_rescore --verification "${SPECIALIZED}/matched/manifest.json" \
    --unified-metrics "${SPECIALIZED}/metrics/unified.json" \
    --process-metrics "${SPECIALIZED}/metrics/process.json" \
    --comparison-scope "frozen specialized predictions on the exact online-adaptation cohort (942 cases)" \
    --output "${SPECIALIZED}/summary.json"
else
  echo "skipping the specialized-model comparison: run build_datasets.sh and run_baselines.sh first" >&2
fi
