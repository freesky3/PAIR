"""Exercise public datasets against the historical loader on identical arrays."""

import importlib.util
import json
import random
import sys
import types

import numpy as np
import pytest
import torch

from pair_eeg.config import Experiment
from pair_eeg.data.epochs import ContentEpochs, StateEpochs, make_loaders


def legacy_loader(root, repo_root, monkeypatch, state=False):
    config = types.SimpleNamespace(
        GT_label=str(root / "metadata/GT_label.npy"),
        optical_flow_score=str(root / "metadata/optical_flow_score.npy"),
        video_features=str(root / "metadata/video_analysis_results.npy"),
        map_color=json.loads((root / "metadata/color.json").read_text()),
        color2num={
            "Neutral Light": 0,
            "Earth & Dark": 1,
            "Cool Tones": 2,
            "Green Nature": 3,
            "Warm Vibrant": 4,
        },
        optical_flow_threshold=0.6427,
        train_ratio=0.8,
        batch_size=16,
        num_workers=0,
    )
    fake = types.ModuleType("Config")
    fake.benchmark_config = lambda: config
    monkeypatch.setitem(sys.modules, "Config", fake)
    folder = "biclassification" if state else "benchmark"
    spec = importlib.util.spec_from_file_location(
        "old_loader", repo_root / f"archived_code/{folder}/dataloader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("feature", ["raw", "psd", "de"])
def test_content_labels_normalization_augmentation_and_splits(
    tiny_release, repo_root, monkeypatch, feature
):
    root, record = tiny_release
    old = legacy_loader(root, repo_root, monkeypatch)
    folder = "watch_cleaned" if feature == "raw" else "watch_PSD_DE"
    path = str(root / folder / record.filename)
    for task in ["20-c", "4-c", "fast_slow", "color", "number", "face", "human"]:
        config = Experiment(
            task=task,
            feature=feature,
            model="eegnet" if feature == "raw" else "mlpnet",
            num_workers=0,
        )
        before = old.UnifiedDataset(path, feature, task, "train")
        after = ContentEpochs(root, record, config, True)
        for index in [0, 1, 2, 4, 17, 249]:
            torch.manual_seed(42)
            a, ya = before[index]
            torch.manual_seed(42)
            b, yb = after[index]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert ya.item() == yb.item()
    random.seed(42)
    a, b = old.get_dataloaders(path, feature, "20-c")
    state = random.getstate()
    random.seed(42)
    c, d, split = make_loaders(root, record, config)
    assert split["train"] == a.dataset.indices
    assert split["validation"] == b.dataset.indices
    assert random.getstate() == state


@pytest.mark.parametrize("feature", ["raw", "psd", "de"])
@pytest.mark.parametrize("ablation", [False, True])
def test_state_normalization_and_alpha_match(
    tiny_release, repo_root, monkeypatch, feature, ablation
):
    root, record = tiny_release
    old = legacy_loader(root, repo_root, monkeypatch, state=True)
    folder = "watch_cleaned" if feature == "raw" else "watch_PSD_DE"
    before = old.UnifiedDataset(
        feature, str(root / folder / record.filename), remove_alpha=ablation
    )
    config = Experiment(
        task="state",
        model="eegnet" if feature == "raw" else "mlpnet",
        feature=feature,
        remove_alpha=ablation,
        num_workers=0,
    )
    after = StateEpochs(root, record, config)
    for index in [0, 199, 249, 250, 499]:
        a, ya = before[index]
        b, yb = after[index]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert ya == yb
    random.seed(42)
    _, _, split = make_loaders(root, record, config)
    train = set(split["train"])
    valid = set(split["validation"])
    assert not train & valid and len(train) == 400 and len(valid) == 100
    assert all((i in train) == (i + 250 in train) for i in range(250))


def test_svm_feature_fix_changes_only_explicit_mode(tiny_release):
    root, record = tiny_release
    paper = Experiment(task="4-c", model="svm", feature="psd")
    corrected = Experiment(task="4-c", model="svm", feature="psd", mode="corrected")
    assert ContentEpochs(root, record, paper, False).data.shape == (250, 62, 400)
    assert ContentEpochs(root, record, corrected, False).data.shape == (250, 62, 5)
