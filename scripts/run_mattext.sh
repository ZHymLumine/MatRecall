#!/usr/bin/env bash
# MatRecall x MatText: three MatBench tasks in five text representations, plus the
# coordinate--category control (seven composition weights). Each task runs an exposure stage
# (memory is written) and a frozen stage (memory is read-only, including cross-representation,
# shuffled and unrelated memory). The ExtraTrees baselines need the raw MatText download.
#
#   JOBS   CPU workers for the baselines (default: 8)

set -euo pipefail
source "$(dirname "$0")/common.sh"

RUN=${RESULTS}/mattext
SPECIALIZED=${RESULTS}/mattext_baselines
REAL_TASKS=(mattext_bulk_modulus_log10_gpa mattext_shear_modulus_log10_gpa mattext_perovskite_formation_energy)
CONTROL_TASKS=(
  mattext_coordinate_category_alpha_0p0 mattext_coordinate_category_alpha_0p2
  mattext_coordinate_category_alpha_0p4 mattext_coordinate_category_alpha_0p5
  mattext_coordinate_category_alpha_0p6 mattext_coordinate_category_alpha_0p8
  mattext_coordinate_category_alpha_1p0
)
REAL_ARGS=(--release "${DATA}/mattext"
  --representations composition atom_sequences_plusplus slices local_env crystal_text_llm)
CONTROL_ARGS=(--release "${DATA}/mattext_control" --representations composition crystal_text_llm
  --methods llm_only static_multiview_rag dynamic_case_rag matrecall)
COMMON=(
  "${SAMPLING[@]}" --max-tokens 4096 --planner-max-tokens 1024 --evidence-budget 6
  --history-k 4 --experience-k 6 --cell-concurrency 12
)
require "${DATA}/mattext/manifest.json" "${DATA}/mattext_control/manifest.json"

# run_stage KIND STAGE TASK ARGS...
run_stage() {
  local kind=$1 stage=$2 task=$3
  shift 3
  local frozen=()
  if [[ "${stage}" == frozen ]]; then
    frozen=(--exposure-root "${RUN}/${kind}" --source-representations composition crystal_text_llm)
  fi
  matrecall run_mattext "$@" --task "${task}" "${COMMON[@]}" --stage "${stage}" \
    ${frozen[@]+"${frozen[@]}"} --output "${RUN}/${kind}/${task}"
}

for task in "${REAL_TASKS[@]}"; do run_stage real exposure "${task}" "${REAL_ARGS[@]}"; done
for task in "${CONTROL_TASKS[@]}"; do run_stage control exposure "${task}" "${CONTROL_ARGS[@]}"; done
if [[ "${DRY_RUN}" == 1 ]]; then
  exit 0  # the frozen stage reads the memory written by a real exposure run
fi
for task in "${REAL_TASKS[@]}"; do run_stage real frozen "${task}" "${REAL_ARGS[@]}"; done
for task in "${CONTROL_TASKS[@]}"; do run_stage control frozen "${task}" "${CONTROL_ARGS[@]}"; done

require "${WORK}/raw/mattext"
for task in "${REAL_TASKS[@]}"; do
  matrecall run_mattext_baselines --raw-root "${WORK}/raw/mattext" --release "${DATA}/mattext" \
    --output "${SPECIALIZED}/real/${task}" --task "${task}" --jobs "${JOBS:-8}"
done
for task in "${CONTROL_TASKS[@]}"; do
  matrecall run_mattext_control_baselines --release "${DATA}/mattext_control" \
    --output "${SPECIALIZED}/control/${task}" --task "${task}" --jobs "${JOBS:-8}"
done

matrecall report_mattext --release "${DATA}/mattext" --llm-root "${RUN}/real" \
  --specialized-root "${SPECIALIZED}/real" --output "${RESULTS}/reports/mattext/real"
matrecall report_mattext --release "${DATA}/mattext_control" --llm-root "${RUN}/control" \
  --specialized-root "${SPECIALIZED}/control" --output "${RESULTS}/reports/mattext/control"
