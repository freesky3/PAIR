"""Verify the user-facing nontraining commands and frozen paper statistics."""

import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from pair_eeg.analysis.statistics import summarize
from pair_eeg.cli import main
from pair_eeg.config import Experiment


def test_archived_statistics_are_unchanged(repo_root, tmp_path):
    summarize(repo_root / "results/paper", tmp_path)
    for filename in ["summary.csv", "paired_comparisons.csv", "subject_scores.csv"]:
        old = list(csv.DictReader((repo_root / "results/paper" / filename).open()))
        new = list(csv.DictReader((tmp_path / filename).open()))
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert a.keys() == b.keys()
            for key in a:
                try:
                    np.testing.assert_allclose(float(a[key]), float(b[key]), rtol=1e-13, atol=1e-13)
                except ValueError:
                    assert a[key] == b[key]


def test_dry_run_is_explicit_and_lightweight(repo_root, capsys):
    main(
        [
            "train",
            "--config",
            str(repo_root / "configs/content/svm_psd.toml"),
            "--output",
            "unused",
            "--dry-run",
        ]
    )
    assert json.loads(capsys.readouterr().out)["actual_feature"] == "raw"
    main(
        [
            "train",
            "--config",
            str(repo_root / "configs/content/svm_psd.toml"),
            "--output",
            "unused",
            "--mode",
            "corrected",
            "--dry-run",
        ]
    )
    assert json.loads(capsys.readouterr().out)["actual_feature"] == "psd"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pair_eeg.cli; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_config_rejects_silent_protocol_changes():
    with pytest.raises(ValueError):
        Experiment(task="4-c", remove_alpha=True)
    with pytest.raises(ValueError):
        Experiment(mode="corrected", legacy_output_classes=20)
