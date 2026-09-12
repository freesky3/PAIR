# PAIR: Perception And Imagery Recall

Data and code project for **Replaying the Movie in Your Mind: Decoding Dynamic Visual Perception and Recall from EEG**, accepted at ICONIP 2026.

PAIR pairs naturalistic video viewing and immediate visual recall in 20 participants, with three recording sessions per participant and 250 two-second video stimuli. Tasks cover hierarchical semantics, five visual meta-features, and perception-versus-recall state discrimination.

## Release status

- **Available here:** archived experimental scripts, stimulus labels and metadata, and anonymous recorded validation scores with statistical-analysis scripts.
- **EEG recordings and PSD/DE features:** a Hugging Face release is being prepared for availability before the camera-ready submission. The dataset link will be added here after upload and verification.
- **Eye tracking:** recorded during acquisition but not used in the reported analyses; it is not included in this release.

The EEG dataset is **not yet downloadable from this project page**. This status will be updated when the Hugging Face release is available.

## Repository contents

| Path | Contents |
| --- | --- |
| `metadata/stimulus_labels.csv` | Human-readable labels for all 250 stimuli; row order matches the flattened clip dimension |
| `metadata/` | Original label/feature metadata arrays, color mapping, and electrode adjacency matrix |
| `archived_code/benchmark/` | Original content-decoding and sensor-analysis scripts |
| `archived_code/biclassification/` | Original state-decoding scripts |
| `paper_analysis/` | Anonymous session scores, participant summaries, paired comparisons, and plotting scripts |

The archived algorithms are preserved. Machine-specific data paths were replaced by `data/`; run these scripts from the repository root after placing the published dataset and metadata in that directory. The archived scripts contain historical configuration and feature-routing limitations described below; they are not a newly validated training pipeline.

## Recompute the reported statistics without training

Install NumPy, SciPy, Matplotlib, and MNE. The analysis was checked with Python 3.11, NumPy 2.1.2, SciPy 1.16.1, Matplotlib 3.10.6, and MNE 1.11.0. These are analysis-environment versions, not a claim about the historical training environment.

```bash
python paper_analysis/summarize_existing_results.py
python paper_analysis/plot_sensor_maps.py
```

These commands consume the anonymous archived scores, regenerate CSV summaries and figures in `images/`, and do not train models. The optional import argument in the scripts is for the authors' original working-directory layout; it is unnecessary for the included scores.

## Evaluation scope and known limitations

Each participant/session was modeled separately. Content decoding used a random 200/50 training/validation split of 250 epochs. State decoding used the first 200 clips per condition for training and the last 50 for validation, keeping paired conditions together. Neural-network scores are the **best validation accuracies** selected during early stopping; SVM scores use a fixed configuration. There was no independent test set. Statistical comparisons of these scores are exploratory and do not remove selection bias.

The camera-ready summaries average the three sessions within each participant before computing the mean and sample standard deviation across 20 participants. Perception–recall paired tests use Holm correction across all 84 retained content comparisons. A separate exploratory family contains the 14 alpha-ablation comparisons. Nonsignificance does not establish equivalence.

The archived content SVM script reads time-domain EEG even when its feature argument is set to PSD/DE. Consequently, the mislabeled spectral SVM rows are excluded from the revised content tables and accompanying statistics. This issue does not apply to the separate state-SVM feature-loading branch. Some duplicate training entry points also calculate task dimensions from their initial configuration; inspect the configuration and documented provenance before using them. Historical exact split indices were not archived, and reproducing training bit-for-bit is not claimed.

Content decoding uses two-second perception and three-second recall epochs. Raw state decoding crops recall to 0.5–2.5 seconds. Raw alpha ablation removes 8–13 Hz; feature ablation omits the alpha feature described as 8–14 Hz. Eye-tracking control and cross-subject evaluation were not performed. Sensor maps are descriptive and are not cortical source localization.

## Labels and data layout

At 200 Hz, each session has the following arrays:

| Folder | Shape | Axes |
| --- | --- | --- |
| `watch_cleaned` | `(5, 50, 62, 400)` | block, clip, channel, time |
| `recall_cleaned` | `(5, 50, 62, 600)` | block, clip, channel, time |
| `watch_PSD_DE` / `recall_PSD_DE` | `(2, 5, 50, 62, 5)` | feature type (PSD then DE), block, clip, channel, frequency band |

Cleaned arrays are preprocessed time-domain EEG, not unprocessed acquisition files. There are 250 unique stimuli; repeated presentations should not be counted as additional unique videos. The frequency bands in the manuscript are 1–4, 4–8, 8–14, 14–31, and 31–99 Hz. Detailed acquisition metadata will accompany the EEG release.

`stimulus_index_0based` is the flattened block/clip index. Semantic labels are 1–20 in the original array; four-class labels are `(label - 1) // 5`. Fast/slow uses optical-flow threshold 0.6427. Number classes are 0 for counts up to one, 1 for two through four, and 2 for more than four objects. Face and human labels indicate presence. Global majority-class proportions are descriptive dataset statistics, not historical validation-set baseline scores.

The source videos came from public video platforms. This repository currently distributes labels and derived metadata, not the original video files.

## Citation and contact

Use the paper title and authors in the final ICONIP proceedings once the publication metadata is available. Questions about the dataset and planned release can be opened as repository issues.
