#!/usr/bin/env bash
# Train and score the specialized-model baselines on the frozen test cohorts
# (build_datasets.sh). The ALIGNN baseline needs ALIGNN installed and a GPU by default.
#
#   JOBS     CPU workers for the simple baselines (default: 16)
#   DEVICE   ALIGNN device, cuda or cpu (default: cuda)

set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT=${PROCESSED}
require "${OUT}/process_tasks" "${OUT}/process_evaluation" "${OUT}/unified_tasks" "${OUT}/unified_evaluation"

matrecall run_process_baselines --release "${OUT}/process_tasks" \
  --evaluation-release "${OUT}/process_evaluation" --output "${BASELINES}/process" \
  --seed 0 --jobs "${JOBS:-16}"
matrecall evaluate_process_predictions --predictions "${BASELINES}/process/test_predictions.jsonl" \
  --gold "${OUT}/process_evaluation/test/gold.jsonl" \
  --output "${BASELINES}/process/test_metrics.json" --bootstrap-samples 1000

matrecall run_unified_baselines --release "${OUT}/unified_tasks" \
  --evaluation-release "${OUT}/unified_evaluation" --output "${BASELINES}/unified" \
  --seed 0 --jobs "${JOBS:-16}"
matrecall evaluate_unified_predictions --predictions "${BASELINES}/unified/test_predictions.jsonl" \
  --gold "${OUT}/unified_evaluation/test/gold.jsonl" \
  --output "${BASELINES}/unified/test_metrics.json" --release "${OUT}/unified_tasks"

matrecall run_alignn_baselines --release "${OUT}/unified_tasks" \
  --evaluation-release "${OUT}/unified_evaluation" --output "${BASELINES}/alignn" \
  --model-revision 7b3111906ca61f6c1659cdf3b4f19f87b9a57692 --device "${DEVICE:-cuda}"
matrecall evaluate_unified_predictions --predictions "${BASELINES}/alignn/test_predictions.jsonl" \
  --gold "${OUT}/unified_evaluation/test/gold.jsonl" \
  --output "${BASELINES}/alignn/test_metrics.json" --release "${OUT}/unified_tasks"
