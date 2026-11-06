#!/usr/bin/env bash
# Build the task releases, the static memory and the prequential/MatText releases from the
# downloaded sources (download_sources.sh). Outputs go to ${WORK}/processed. The releases that
# need a completed online-adaptation run are built by build_frozen_releases.sh.
#
#   ALIGNN_MODELS   directory holding the ALIGNN checkpoints (default: ${WORK}/models/alignn)
#   JARVIS_JSON     JARVIS-DFT 3D snapshot (default: ${WORK}/raw/jarvis_dft_3d/jdft_3d-9-24-2025.json)
#   WORKERS         processes for the structure-descriptor index (default: 16)

set -euo pipefail
source "$(dirname "$0")/common.sh"

SOURCES=${WORK}/sources
RAW=${WORK}/raw
ALIGNN_MODELS=${ALIGNN_MODELS:-${WORK}/models/alignn}
JARVIS_JSON=${JARVIS_JSON:-${RAW}/jarvis_dft_3d/jdft_3d-9-24-2025.json}
OUT=${PROCESSED}
require "${JARVIS_JSON}" "${ALIGNN_MODELS}" "${SOURCES}" "${RAW}/mattext"

# Crystal-property tasks (JARVIS-DFT, split by the ALIGNN checkpoints)
matrecall build_crystal_reference --jarvis-json "${JARVIS_JSON}" --output "${OUT}/crystal_reference"
matrecall build_crystal_tasks --reference-dir "${OUT}/crystal_reference" \
  --alignn-model-root "${ALIGNN_MODELS}" --output "${OUT}/crystal_tasks"

# Process-outcome tasks
matrecall build_process_tasks \
  --raccuglia-historical "${RAW}/raccuglia_2016/historical_reactions.csv" \
  --raccuglia-prospective "${RAW}/raccuglia_2016/prospective_reactions.csv" \
  --inp-csv "${SOURCES}/Indium-Phosphide/dataset/dataset_InP_raw.csv" \
  --meltpool-csv "${SOURCES}/MeltpoolNet/Data/meltpoolnet_regression.csv" \
  --mgt-json "${RAW}/mgt_2021/solid-state_dataset_2019-06-27.json" \
  --mgt-exclude-py "${SOURCES}/inorg-synth-graph/data/exclude.py" \
  --iron-oxide-paper "${RAW}/iron_oxide_2023/paper.pdf" \
  --output "${OUT}/process_tasks"

# Unified task release: crystal and process tasks plus the synthesis task families
matrecall build_unified_tasks_base \
  --crystal-release "${OUT}/crystal_tasks" --process-release "${OUT}/process_tasks" \
  --synthesis-similarity-jsonl "${SOURCES}/SynthesisSimilarity/SynthesisSimilarity/rsc/reactions_v20_20210820_ss.jsonl" \
  --synthesis-similarity-revision 21f013ff6a1fe1f5eeb8c48e32bf20d601d2fb86 \
  --syntmte-year-split "${SOURCES}/SyntMTE/data/conditions/year_split" \
  --syntmte-revision fb9cd5c2b05e43080ab054584c856dfe0f84b2ee \
  --spende-json "${SOURCES}/spende/dataset/tmsr.json" \
  --spende-revision baaa5f3da856713840e81d5288331f2be1c75dd8 \
  --diffsyn-workbook "${SOURCES}/zeosyn_gen/data/ZEOSYN-2.xlsx" \
  --diffsyn-system-list "${SOURCES}/zeosyn_gen/data/zeo_osda_systems.json" \
  --diffsyn-osda-descriptors "${SOURCES}/zeosyn_gen/data/osda_descriptors.csv" \
  --diffsyn-zeolite-descriptors "${SOURCES}/zeosyn_gen/data/zeolite_descriptors.csv" \
  --diffsyn-cif-dir "${SOURCES}/zeosyn_gen/data/cifs" \
  --diffsyn-revision 74b0b97563c794eebc0db4951d5d4be1b7ae76b9 \
  --diffsyn-split-code-revision 3bd03b524b87a2ea0c6bbffa0c49decc3472a9d9 \
  --output "${OUT}/unified_tasks_base"
matrecall build_unified_tasks --base-release "${OUT}/unified_tasks_base" \
  --sg-predict-csv "${SOURCES}/SG_predict/ML/cubic.csv" \
  --sg-predict-revision bd5ace46f49a97d6fe0976a284ef8048cddb0f96 \
  --pcmsp-original "${SOURCES}/PcMSP/original" \
  --pcmsp-revision cf37dcd7848281b8878101c8d02fb79129f54faf \
  --feconi-imputed-csv "${SOURCES}/FeCoNi_inverse_design/2-Imputation/FeNiCo_comp-prop_imp.csv" \
  --feconi-raw-csv "${SOURCES}/FeCoNi_inverse_design/FeNiCo_comp-prop.csv" \
  --feconi-revision 0ab1b994338cbd6e1d58464e2835cddfc36f64cc \
  --hea-csv "${SOURCES}/ML_HEAs_Phase_Dataset/Dataset/dataset10387_70.csv" \
  --hea-revision e823c1721254adb0719ea8b8a994317811d83bf1 \
  --output "${OUT}/unified_tasks"

# Frozen test cohorts (with train-only retrieval baselines selected on validation)
matrecall prepare_process_evaluation --release "${OUT}/process_tasks" --output "${OUT}/process_evaluation"
matrecall prepare_unified_evaluation --release "${OUT}/unified_tasks" --output "${OUT}/unified_evaluation"

# Static multiview memory
matrecall build_structure_index \
  --structures "${OUT}/unified_tasks/crystal_to_property/alignn_jarvis/structures.parquet" \
  --output "${OUT}/structure_index" --workers "${WORKERS:-16}"
matrecall build_static_memory --unified-release "${OUT}/unified_tasks" \
  --process-release "${OUT}/process_tasks" --jarvis-indexes "${OUT}/structure_index" \
  --output "${OUT}/static_memory"

# Prequential releases (online adaptation is also shipped in data/)
matrecall prepare_online_adaptation --source "${OUT}/static_memory" \
  --output "${OUT}/online_adaptation" --max-cases-per-task 48 --seed 0
matrecall prepare_memory_ablation --source "${OUT}/static_memory" \
  --output "${OUT}/memory_ablation" --max-cases-per-task 32 --sample-seed 0 --order-seeds 0 1 2 3 4

# MatText releases (also shipped in data/)
matrecall prepare_mattext --raw-root "${RAW}/mattext" --output "${OUT}/mattext" \
  --static-cases 256 --exposure-cases 16 --frozen-cases 8 --seed 0
matrecall prepare_mattext_control --real-release "${OUT}/mattext" --output "${OUT}/mattext_control" \
  --exposure-cases 16 --frozen-cases 8
