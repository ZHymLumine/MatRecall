# Shared settings for the example scripts. Source this file; do not run it directly.
#
#   WORK            working directory for downloads, built datasets and results (required)
#   DATA            released evaluation data (default: <repository>/data)
#   PYTHON          Python interpreter with MatRecall installed (default: python)
#   BASE_URL        OpenAI-compatible endpoint serving the model (default: http://127.0.0.1:8000/v1)
#   MODEL           served model name (default: Qwen/Qwen3.8-27B)
#   MODEL_REVISION  model revision recorded with every prediction
#   DRY_RUN=1       build and save prompts without calling the model; skip scoring and reports

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${WORK:?set WORK to a working directory}
DATA=${DATA:-${REPO}/data}
RESULTS=${RESULTS:-${WORK}/results}
PROCESSED=${PROCESSED:-${WORK}/processed}
PYTHON=${PYTHON:-python}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
MODEL_REVISION=${MODEL_REVISION:-1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
DRY_RUN=${DRY_RUN:-0}
export PYTHONPATH=${REPO}${PYTHONPATH:+:${PYTHONPATH}}

# Decoding settings used for every reported LLM run.
SAMPLING=(
  --model "${MODEL}" --model-revision "${MODEL_REVISION}" --base-url "${BASE_URL}"
  --seed 0 --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5
  --disable-thinking --allow-unescaped-control-chars --timeout 1800
)
if [[ "${DRY_RUN}" == 1 ]]; then
  SAMPLING+=(--dry-run)
fi

# The unified task release supplies the forward surrogate used to score inverse formulation.
UNIFIED_TASKS=${UNIFIED_TASKS:-${PROCESSED}/unified_tasks}
UNIFIED_RELEASE=()
if [[ -d "${UNIFIED_TASKS}" ]]; then
  UNIFIED_RELEASE=(--release "${UNIFIED_TASKS}")
fi

matrecall() {
  "${PYTHON}" -m "matrecall.$1" "${@:2}"
}

require() {
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "missing ${path}" >&2
      exit 1
    fi
  done
}

# score INPUT_ROOT METRIC_ROOT METHOD...: native task metrics for each method's
# INPUT_ROOT/<method>/{unified,process}/{predictions,gold}.jsonl.
score() {
  local inputs=$1 metrics=$2 method
  shift 2
  if [[ ${#UNIFIED_RELEASE[@]} == 0 ]]; then
    echo "note: ${UNIFIED_TASKS} not found; forward-surrogate metrics for" \
      "target_property_to_formulation_composition are not computed" >&2
  fi
  for method in "$@"; do
    mkdir -p "${metrics}/${method}"
    matrecall evaluate_unified_predictions \
      --predictions "${inputs}/${method}/unified/predictions.jsonl" \
      --gold "${inputs}/${method}/unified/gold.jsonl" \
      --output "${metrics}/${method}/unified.json" ${UNIFIED_RELEASE[@]+"${UNIFIED_RELEASE[@]}"}
    matrecall evaluate_process_predictions \
      --predictions "${inputs}/${method}/process/predictions.jsonl" \
      --gold "${inputs}/${method}/process/gold.jsonl" \
      --output "${metrics}/${method}/process.json" --bootstrap-samples 1000
  done
}

# Specialized-model predictions from run_baselines.sh, rebound to an LLM cohort.
BASELINES=${RESULTS}/baselines
SPECIALIZED_INPUTS=(
  --original-unified "${PROCESSED}/unified_evaluation/test"
  --original-process "${PROCESSED}/process_evaluation/test"
  --predictions "${BASELINES}/alignn/test_predictions.jsonl"
  "${BASELINES}/unified/test_predictions.jsonl" "${BASELINES}/process/test_predictions.jsonl"
)

# score_specialized MATCHED_ROOT: metrics for a rebound specialized cohort.
score_specialized() {
  local root=$1
  mkdir -p "${root}/metrics"
  matrecall evaluate_unified_predictions \
    --predictions "${root}/matched/unified/predictions.jsonl" \
    --gold "${root}/matched/unified/gold.jsonl" \
    --output "${root}/metrics/unified.json" ${UNIFIED_RELEASE[@]+"${UNIFIED_RELEASE[@]}"}
  matrecall evaluate_process_predictions \
    --predictions "${root}/matched/process/predictions.jsonl" \
    --gold "${root}/matched/process/gold.jsonl" \
    --output "${root}/metrics/process.json" --bootstrap-samples 1000
}
