#!/usr/bin/env bash
# Download the public source datasets at the revisions used for the paper into ${WORK}.
#
#   ${WORK}/sources/<repository>   Git checkouts at fixed commits
#   ${WORK}/raw/...                files downloaded over HTTPS, checked against SHA-256
#
# Two inputs are not downloaded automatically; see "Source data" in README.md:
# the JARVIS-DFT 3D snapshot and the ALIGNN pretrained checkpoints.

set -euo pipefail
source "$(dirname "$0")/common.sh"

SOURCES=${WORK}/sources
RAW=${WORK}/raw
mkdir -p "${SOURCES}" "${RAW}/raccuglia_2016" "${RAW}/mgt_2021" "${RAW}/iron_oxide_2023" "${RAW}/mattext"

if command -v sha256sum >/dev/null 2>&1; then
  CHECKSUM=(sha256sum)
else
  CHECKSUM=(shasum -a 256)
fi

clone_at_revision() {
  local url=$1 destination=$2 revision=$3
  if [[ -d "${destination}/.git" ]]; then
    if [[ "$(git -C "${destination}" rev-parse HEAD)" != "${revision}" ]]; then
      echo "existing checkout has an unexpected revision: ${destination}" >&2
      exit 1
    fi
    return
  fi
  if [[ -e "${destination}" ]]; then
    echo "refusing to replace a non-Git path: ${destination}" >&2
    exit 1
  fi
  git clone "${url}" "${destination}"
  git -C "${destination}" checkout --detach "${revision}"
}

fetch() {
  local url=$1 target=$2 sha256=$3
  if [[ ! -f "${target}" ]]; then
    curl -fL --retry 3 "${url}" -o "${target}"
  fi
  if [[ "$("${CHECKSUM[@]}" "${target}" | cut -d' ' -f1)" != "${sha256}" ]]; then
    echo "SHA-256 mismatch: ${target}" >&2
    exit 1
  fi
}

# Process-outcome tasks
clone_at_revision https://github.com/Cossairt-Lab/Indium-Phosphide.git \
  "${SOURCES}/Indium-Phosphide" 23a685dc383dca60bec8ffa370e5a38eeeab4255
clone_at_revision https://github.com/BaratiLab/MeltpoolNet.git \
  "${SOURCES}/MeltpoolNet" 6d68e2acaad8d074134a7f3d843278649774d661
clone_at_revision https://github.com/s-a-malik/inorg-synth-graph.git \
  "${SOURCES}/inorg-synth-graph" f270a5b609c80859aa59582482d507e2d7a68abd
ESM='https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fnature17439/MediaObjects'
fetch "${ESM}/41586_2016_BFnature17439_MOESM231_ESM.csv" "${RAW}/raccuglia_2016/historical_reactions.csv" \
  5d484611593a08f535ce2937189d8d9359295ea0cc830544600c1341170f6deb
fetch "${ESM}/41586_2016_BFnature17439_MOESM232_ESM.csv" "${RAW}/raccuglia_2016/prospective_reactions.csv" \
  cc5b8fa1f8b74f96035e429c8786b78204a282cfb5a0dc337eac5780edb16a2b
fetch https://ndownloader.figshare.com/files/17412674 "${RAW}/mgt_2021/solid-state_dataset_2019-06-27.json" \
  4be37c9ab9aa23d7ceaaec692e6900f391e628b175d56b89995fa501a5130ab0
fetch https://arxiv.org/pdf/2303.11244 "${RAW}/iron_oxide_2023/paper.pdf" \
  4cb97a1df7204c00ee3d0ca0c64850fc7d4ff6441f84a3103dc7913a77351fc4

# Synthesis tasks
clone_at_revision https://github.com/CederGroupHub/SynthesisSimilarity.git \
  "${SOURCES}/SynthesisSimilarity" 21f013ff6a1fe1f5eeb8c48e32bf20d601d2fb86
clone_at_revision https://github.com/Thorben010/SyntMTE.git \
  "${SOURCES}/SyntMTE" fb9cd5c2b05e43080ab054584c856dfe0f84b2ee
clone_at_revision https://github.com/ngs00/spende.git \
  "${SOURCES}/spende" baaa5f3da856713840e81d5288331f2be1c75dd8
clone_at_revision https://github.com/eltonpan/zeosyn_gen.git \
  "${SOURCES}/zeosyn_gen" 74b0b97563c794eebc0db4951d5d4be1b7ae76b9
clone_at_revision https://github.com/Yuxinya/SG_predict.git \
  "${SOURCES}/SG_predict" bd5ace46f49a97d6fe0976a284ef8048cddb0f96
clone_at_revision https://github.com/Xianjun-Yang/PcMSP.git \
  "${SOURCES}/PcMSP" cf37dcd7848281b8878101c8d02fb79129f54faf
clone_at_revision https://github.com/Shakti-95/Multi-propertyML_Multi-objectiveBO_FeCoNi_Alloy_design.git \
  "${SOURCES}/FeCoNi_inverse_design" 0ab1b994338cbd6e1d58464e2835cddfc36f64cc
clone_at_revision https://github.com/Iman-Peivaste/ML_HEAs_Phase_Dataset.git \
  "${SOURCES}/ML_HEAs_Phase_Dataset" e823c1721254adb0719ea8b8a994317811d83bf1

# MatText: the three MatBench-labelled configurations, all five folds (30 parquet files).
tail -n +2 "${REPO}/scripts/mattext_files.tsv" | while IFS=$'\t' read -r _ _ path _ sha256 url; do
  mkdir -p "$(dirname "${RAW}/mattext/${path}")"
  fetch "${url}" "${RAW}/mattext/${path}" "${sha256}"
done

echo "downloads complete; place the JARVIS-DFT snapshot and ALIGNN checkpoints as described in README.md"
