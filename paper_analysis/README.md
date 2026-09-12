# Archived validation-score analysis

These scripts recompute statistics and figures from saved scores. They do not fit models, change historical splits, or use new EEG/eye-tracking data.

`session_scores.csv` contains anonymous scores imported from the authors' three result JSON files. The source index follows the sorted session-file order used by the final experimental scripts. The four data directories were checked to contain the same 60 recording filenames, corresponding to 20 participants with three sessions each. Anonymous recording identifiers denote the within-participant sorted order. Personal filenames and their mapping are not distributed here.

Content-score provenance is `benchmark/benchmark_results.json`; state scores come from `biclassification/benchmark_results.json` and `biclassification/benchmark_results_alpha_removed.json`. `source_file`, `source_key`, and `source_index` preserve the link to each original score. Content PSD/DE SVM entries are excluded because their final entry script loaded raw EEG regardless of those feature labels. Extra archived model variants that did not appear in the paper are not added to the comparison set.

Each participant contributes the average of three session scores. Tables use the resulting mean and sample SD (`ddof=1`) over 20 participants. Every retained table mean was checked against the original LaTeX values within rounding tolerance; changes to SD reflect the corrected aggregation unit.

Exploratory two-sided paired t-tests compare perception minus recall for 12 retained model/feature combinations across seven tasks (84 tests). Holm correction uses that entire family, including raw SVM. Confidence intervals are unadjusted 95% t intervals for paired differences. A separate exploratory family comprises 14 full-minus-alpha-ablated state comparisons; it is included for transparency but the main manuscript describes the state ablations without significance claims. The code asserts that paired differences have nonzero variance rather than silently assigning p-values to degenerate inputs.

Two content comparisons survive Holm correction: raw 4-class decoding with DeepNet (4.40 percentage points, 95% CI 2.65–6.15; adjusted p=0.00380) and GLFNet (5.87 points, 95% CI 4.22–7.51; adjusted p=0.0000383). These are comparisons of validation-selected scores, not unbiased generalization estimates. Nonsignificance is not evidence of equivalence.

The original sensor-score arrays for `Human` are 60 × 62 values for a shared single-channel Conformer. `sensor_means.csv` preserves their means and the original display coordinates; `plot_sensor_maps.py` applies one color scale to both conditions. No spatial significance test or source localization is implied.

From this repository root:

```bash
python paper_analysis/summarize_existing_results.py
python paper_analysis/plot_sensor_maps.py
```

Checked analysis environment: Python 3.11.13, NumPy 2.1.2, SciPy 1.16.1, Matplotlib 3.10.6, MNE 1.11.0. The optional `--import-workspace` argument targets the authors' original directory layout and is unnecessary with the included anonymous scores. These versions are not asserted to be the historical training environment.
