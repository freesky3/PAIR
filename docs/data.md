# Dataset release

The Hugging Face repository is `skywalker-p/PAIR`, with a planned `v1.0.0` release tag. The dataset license is CC BY-NC 4.0. Publication status and terms are documented on that dataset page. Code and data revisions are independent; use the documented tag or commit for reproducible downloads.

## Files

The release contains 20 anonymous participants and three recordings per participant. It preserves 240 numeric EEG/feature files byte-for-byte while replacing identifying filenames with `sub-001_recording-01.npy` through `sub-020_recording-03.npy`.

| Folder | Per-recording shape | Axes |
| --- | --- | --- |
| `watch_cleaned` | `(5, 50, 62, 400)` | block, trial, electrode, time |
| `recall_cleaned` | `(5, 50, 62, 600)` | block, trial, electrode, time |
| `watch_PSD_DE`, `recall_PSD_DE` | `(2, 5, 50, 62, 5)` | PSD/DE, block, trial, electrode, band |

The first feature axis is PSD then DE. The stored arrays are preprocessed time-domain EEG and derived features, not original acquisition files. Sampling rate is 200 Hz after downsampling from 1,000 Hz. The manuscript reports 0.1–100 Hz filtering and ICA artifact attenuation. Reference-electrode details, stored physical amplitude units, and the full raw-to-feature preprocessing script were not available in the source release and are not inferred or rescaled here. The acquisition documentation should be supplemented when those records are recovered.

`recordings.csv` provides stable anonymous identity, source index, and file order. Recording numbers denote sorted recording order, not a claim that all original sessions were named consistently. `metadata/stimulus_labels.csv` provides all 250 clip labels without pickle loading. The private original-name mapping is not distributed.

The frequency bands described in the manuscript are 1–4, 4–8, 8–14, 14–31, and 31–99 Hz. Raw state ablation used 8–13 Hz, whereas spectral ablation omitted the third feature. These operations are documented separately rather than silently unified.

## Verify and load

```bash
uv run pair-eeg download --data-dir data
uv run pair-eeg validate-data --data-dir data
```

Validation checks sizes, SHA-256 digests, shapes, dtypes and finite values. `--headers-only` skips hashes/value scans. `--include 'watch_PSD_DE/*'` supports partial downloads; a complete-manifest validation requires every listed file.

```python
import numpy as np

watch = np.load("data/watch_cleaned/sub-001_recording-01.npy", allow_pickle=False)
epochs = watch.reshape(250, 62, 400)
```

There are 250 unique stimuli repeated over participants and sessions, rather than 15,000 unique videos. Across complete sessions there are 15,000 paired trials and 30,000 condition epochs. Original video files, eye tracking and personal identifiers are not included. Class imbalance is described in the metadata; full-stimulus majority-class proportions are not historical validation-set baseline scores.

The dataset uses CC BY-NC 4.0, permitting attributed noncommercial use and adaptation. Original PAIR code uses MIT; third-party implementations retain their original terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md` in the code repository.
