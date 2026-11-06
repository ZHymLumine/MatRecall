# MatRecall

**Experience-grounded materials prediction with verified episodic memory.**

MatRecall is a language-model agent for materials prediction. For each query it plans
retrieval over two memories: a static multiview memory of training records and an
episodic memory of its own earlier predictions, each stored once the true outcome has
been revealed and scored. This repository contains the code and evaluation data for the
paper _MatRecall: Experience-Grounded Materials Prediction with Verified Episodic Memory_.

## Installation

```bash
git clone <repository-url>
cd matrecall
pip install -e .
```

Requires Python 3.12+. Install `.[test]` to run the tests with `pytest`, and `.[alignn]`
for the ALIGNN baseline.

## Data

`data/` contains the evaluation sets used in the paper.

| Folder                | Experiment              | Size                                                   |
| --------------------- | ----------------------- | ------------------------------------------------------ |
| `online_adaptation`   | Online adaptation       | 22 tasks, 942 queries                                  |
| `memory_retention`    | Memory retention        | 16 tasks, 463 queries                                  |
| `cross_task_transfer` | Cross-task transfer     | 8 target tasks × related/unrelated source, 235 queries |
| `mattext`             | MatText representations | 3 tasks × 5 folds × 5 representations                  |
| `mattext_control`     | MatText control         | 7 composition weights × 5 folds                        |

Each task directory holds `cases.jsonl` (model inputs) and `gold.jsonl` (answers for
scoring), plus `order.jsonl`, `frozen_exposure_memory.jsonl`, `transfer_index.jsonl` or
`static_library.jsonl` depending on the experiment.

## Usage

Serve the model behind an OpenAI-compatible endpoint, for example with
[vLLM](https://github.com/vllm-project/vllm):

```bash
vllm serve Qwen/Qwen3.8-27B --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs 16 --enable-prefix-caching \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --reasoning-parser qwen3 --port 8000
```

Then run an experiment:

```bash
export WORK=/path/to/workdir
export BASE_URL=http://127.0.0.1:8000/v1
bash scripts/run_online_adaptation.sh
```

| Script                       | Experiment                                    |
| ---------------------------- | --------------------------------------------- |
| `run_online_adaptation.sh`   | Prequential prediction with verified feedback |
| `run_memory_ablation.sh`     | Ablations of the memory components            |
| `run_memory_retention.sh`    | Frozen memory on new queries                  |
| `run_cross_task_transfer.sh` | Memory from related and unrelated tasks       |
| `run_mattext.sh`             | Memory across crystal text representations    |

Results and reports are written to `$WORK/results`. Set `DRY_RUN=1` to generate the
prompts without querying the model. See `scripts/common.sh` for all settings.

## Building the datasets

```bash
bash scripts/download_sources.sh
bash scripts/build_datasets.sh
bash scripts/run_baselines.sh
bash scripts/build_frozen_releases.sh   # after run_online_adaptation.sh
```

`build_datasets.sh` also expects the JARVIS-DFT 3D snapshot `jdft_3d-9-24-2025.json` in
`$WORK/raw/jarvis_dft_3d/` and the pretrained ALIGNN checkpoints in `$WORK/models/alignn/`.
The memory ablation and the specialized-model comparisons use these outputs.

## Repository structure

```text
matrecall/   Python package
scripts/     data download, dataset construction and experiment scripts
tests/       unit tests
data/        evaluation data
```

## Citation

If you use MatRecall, please cite _MatRecall: Experience-Grounded Materials Prediction with Verified Episodic Memory_.

## License

MIT, see [LICENSE](LICENSE). The datasets in `data/` are derived from public sources
listed in `scripts/download_sources.sh`, which retain their original licenses.
