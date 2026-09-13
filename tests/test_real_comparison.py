"""Check real-data comparison plumbing on synthetic data without a training run."""

import importlib.util
import json

import numpy as np
import torch

from pair_eeg.config import Experiment


def comparison_script(repo_root):
    """Import the opt-in comparison script without invoking its experiment entry point."""
    spec = importlib.util.spec_from_file_location(
        "real_comparison", repo_root / "scripts/compare_real_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_components_keep_requested_state_variant(tiny_release, repo_root):
    root, recording = tiny_release
    module = comparison_script(repo_root)
    config = Experiment(task="state", model="mlpnet", feature="de", num_workers=0)
    _, loader, functions = module.legacy_components(config, root)
    train, valid = loader.get_dataloaders("de", str(root / "watch_PSD_DE" / recording.filename))
    assert train.dataset.dataset.watch_data.shape == (250, 62, 5)
    assert len(train.dataset) == 400 and len(valid.dataset) == 100
    assert set(functions) >= {"train_one_epoch", "evaluate"}


def test_comparison_separates_parity_from_historical_accuracy(tiny_release, repo_root, tmp_path):
    root, record = tiny_release
    module = comparison_script(repo_root)
    config = Experiment(model="mlpnet", task="4-c", feature="psd")
    output = tmp_path / "comparison"
    for implementation in ["legacy", "refactored"]:
        folder = output / implementation / f"{record.subject}_{record.recording}"
        folder.mkdir(parents=True)
        (folder / "split.json").write_text(json.dumps({"train": [0], "validation": [1]}))
        (folder / "epochs.csv").write_text(
            "epoch,training_loss,training_accuracy,validation_loss,validation_accuracy\n"
            "0,1.2,30,1.3,32\n1,1.1,35,1.2,34\n"
        )
        torch.save({"weights": torch.tensor([1.0, 2.0])}, folder / "best_model.pt")
    result = module.compare(config, root, output, 1)[0]
    assert result["trace_exact"] and result["checkpoint_exact"] and result["split_equal"]
    assert result["legacy_best_validation"] == result["refactored_best_validation"] == 34
    assert result["archived_validation"] is not None
    assert result["refactored_minus_archived_pp"] == 34 - result["archived_validation"]
    assert np.isfinite(result["max_checkpoint_absolute_difference"])
