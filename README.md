# PAIR: Perception And Imagery Recall

EEG recordings and benchmarks for **Replaying the Movie in Your Mind: Decoding Dynamic Visual Perception and Recall from EEG**, accepted at ICONIP 2026.

PAIR pairs naturalistic video viewing with immediate recall in 20 participants, with three recordings each and 250 video stimuli. This repository provides a `uv`-managed Python package, archived paper scores, and explicit paper/corrected experiment modes.

## Quick start: reproduce paper statistics and figures

```bash
git clone https://github.com/freesky3/PAIR.git
cd PAIR
uv sync --locked
uv run pair-eeg reproduce-paper
```

Outputs go to `outputs/paper/`. This command reads archived scores and does **not** train models or require a GPU. It reproduces participant-level means/SDs and exploratory paired comparisons, together with the semantic/meta-feature bar chart and common-scale sensor maps.

## Dataset status

Anonymous EEG, PSD/DE features and metadata have been prepared for the Hugging Face dataset `skywalker-p/PAIR`. The data release uses **CC BY-NC 4.0**: attribution is required and noncommercial use/adaptation is permitted; commercial use requires separate permission. The repository code uses **MIT** for original PAIR code. Historical and third-party files retain their original terms; see `THIRD_PARTY_NOTICES.md`. The download command will work once the `v1.0.0` dataset tag is published. Eye tracking and source videos are not included.

```bash
uv run pair-eeg download --data-dir data
uv run pair-eeg validate-data --data-dir data
```

The release is about 7.59 GB. See [data layout and limitations](docs/data.md) for dimensions, label definitions, provenance, and missing acquisition details.

## Run an explicit new experiment

Install one training extra:

```bash
uv sync --locked --extra train   # CPU
# or: uv sync --locked --extra cuda   # CUDA 12.1; requires a compatible driver
```

Inspect a configuration without training:

```bash
uv run --extra train pair-eeg train --config configs/content/eegnet.toml --output outputs/example --dry-run
```

Start a run only when intended:

```bash
uv run --extra train pair-eeg train --config configs/content/eegnet.toml --data-dir data --output outputs/eegnet-watch --device cpu
uv run --extra train pair-eeg train --config configs/content/svm_psd.toml --mode corrected --data-dir data --output outputs/svm-psd-corrected
```

For CUDA, use `--extra cuda` and `--device cuda:0`. The extras are mutually exclusive. Existing output directories are never overwritten. A list of configs can be run with `uv run --extra train scripts/run_suite.py ... --output outputs/suite`; each config starts a new process and RNG stream.

**Paper mode is the default.** It preserves the supported historical operations and exposes known incompatible configurations instead of silently repairing them. **Corrected mode** fixes feature routing and derives dimensions from the selected task/input. Neither mode changes the paper's 80/20 training/validation protocol into independent testing. New training scores must not be presented as the archived paper scores.

The same seed alone does not guarantee the same historical accuracy. [Reproducibility details](docs/reproducibility.md) explain the preserved RNG sequence, missing historical splits, SVM seed omission, GLFNet spectral-width mismatch, and CPU/GPU limitations.

## Project structure

```text
src/pair_eeg/
  cli.py, config.py
  data/               # release download, validation, epochs, partitions
  models/             # CNN, Conformer, spectral MLP, GCN-LSTM
  training/           # shared loop, checkpoint/prediction/RNG provenance
  analysis/           # archived-score statistics and plotting
configs/              # explicit example experiments
scripts/              # dataset preparation and batch/reproduction entry points
tests/                # numerical equivalence and protocol regression checks
results/paper/        # immutable anonymous paper scores and expected summaries
metadata/             # stimulus labels and electrode graph
archived_code/        # historical source retained for provenance and comparisons
docs/                 # data and reproducibility contracts
```

Core interfaces have type hints and Google-style docstrings. Runtime paths and devices are explicit. Importing the package does not load datasets or initialize training.

## Verification

```bash
uv sync --locked --extra train
uv run --extra train ruff check src scripts tests
uv run --extra train ruff format --check src scripts tests
uv run --extra train pytest -q
```

Regression checks compare old/new initialization, forward results, an Adam step, normalization, augmentation, splits, and short synthetic runs. Archived statistics are checked against the frozen CSVs. No complete 20-participant retraining or GPU equivalence run has been performed.

## Scientific scope

Scores are within-participant/session **validation** scores. Neural networks report the best validation epoch; SVM uses fixed settings. There is no independent test set, and selection bias is not removed by paired statistics. The four mislabeled spectral-SVM rows were removed from the revised content tables. Full/ablated state scores remain archived, including configurations whose original source settings cannot be fully recovered.

Eye-tracking controls and cross-subject evaluation were not performed. Sensor maps are descriptive and do not localize cortical sources. Numerical similarity and nonsignificant differences do not establish equivalent performance or shared neural representations.

## Citation and rights

Cite the paper title and final ICONIP proceedings entry once available. Original PAIR code is MIT-licensed; dataset permissions are specified separately on the Hugging Face release as CC BY-NC 4.0. Historical and third-party rights remain with their holders. Please use repository issues for project questions.
