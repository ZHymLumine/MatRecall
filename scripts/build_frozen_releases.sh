#!/usr/bin/env bash
# Build the memory-retention and cross-task transfer releases. Both freeze the verified
# experience memory of a completed online-adaptation MatRecall run (run_online_adaptation.sh)
# and need the static memory from build_datasets.sh. Released copies are in data/.

set -euo pipefail
source "$(dirname "$0")/common.sh"

EXPOSURE_RUN=${RESULTS}/online_adaptation/evaluation
require "${PROCESSED}/static_memory" "${EXPOSURE_RUN}/completion.json"

matrecall prepare_memory_retention --source "${PROCESSED}/static_memory" \
  --exposure-release "${DATA}/online_adaptation" --exposure-run "${EXPOSURE_RUN}" \
  --output "${PROCESSED}/memory_retention" --max-cases-per-task 32 --seed 0
matrecall prepare_cross_task_transfer --target-release "${PROCESSED}/memory_retention" \
  --exposure-release "${DATA}/online_adaptation" --exposure-run "${EXPOSURE_RUN}" \
  --output "${PROCESSED}/cross_task_transfer"
