# Real-data implementation comparison

`real-data-comparison.json` records three selected experiments on `sub-001/recording-01` on 2026-09-13. Both implementations used seed 42, four loader workers, the same GPU, and the same input values; only source paths and the requested state ablation were bound to the archived functions. Maximum epochs were 400, with patience 40.

| Model / Task | Epochs | Legacy (%) | Refactored (%) | Archived session (%) |
| --- | ---: | ---: | ---: | ---: |
| EEGNet, raw EEG, 20-class perception | 46 | 8.0 | 8.0 | 12.0 |
| MLP, PSD, 4-class perception | 67 | 28.0 | 28.0 | 36.0 |
| MLP, DE, perception/recall state | 84 | 98.0 | 98.0 | 98.0 |

All train/validation indices, epoch metrics and best checkpoint tensors matched exactly between implementations. Two historical session scores were not reproduced. Their cause is unresolved: historical configurations, RNG sequence, split indices and software environment were not fully archived. These comparisons do not replace the paper's aggregate scores or demonstrate independent test performance.

The archived training/evaluation functions were extracted from their source to avoid triggering unrelated experiment loops. The comparison uses newly seeded single-configuration runs; it does not reconstruct an unknown historical sequence of configurations.

To repeat the comparison with the downloaded dataset:

```bash
uv run --extra cuda scripts/compare_real_experiments.py \
  --data-dir data --output outputs/real-parity \
  --device cuda:0 --recordings 1 --epochs 400 --num-workers 4
```

This command explicitly starts training. For CPU, use `--extra train` and `--device cpu`.
The report records the environment used; CPU/GPU and dependency differences can
change training trajectories. The reported run used the already-installed CUDA
environment listed in the JSON, including torch-geometric 2.5.1 rather than the
new package lock's 2.6.1. None of the three selected models use graph convolution.
