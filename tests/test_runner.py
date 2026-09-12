"""Check that refactoring the full loop preserves RNG progression and recorded scores."""

import json
import random

import numpy as np
import pytest
import torch

from pair_eeg.config import Experiment
from pair_eeg.data.epochs import make_loaders
from pair_eeg.models import build_model
from pair_eeg.training.runner import run, seed_run


@pytest.mark.parametrize("workers", [0, 2])
def test_two_epoch_run_and_saved_provenance(tiny_release, tmp_path, workers):
    """A synthetic two-epoch trace must match the original loop, including worker seeding."""
    root, record = tiny_release
    torch.set_num_threads(1)
    config = Experiment(
        task="4-c", model="mlpnet", feature="psd", epochs=2, patience=2, num_workers=workers
    )
    seed_run(config)
    train, valid, split = make_loaders(root, record, config)
    old = build_model(config, 62, 5, root / "metadata/adj_matrix.npy")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(old.parameters(), lr=0.001, weight_decay=0.0001)
    best = 0.0
    for _ in range(2):
        old.train()
        for x, y in train:
            optimizer.zero_grad()
            loss = criterion(old(x), y)
            loss.backward()
            optimizer.step()
        old.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in valid:
                output = old(x)
                criterion(output, y)
                predicted = output.max(1)[1]
                correct += predicted.eq(y).sum().item()
                total += y.size(0)
        best = max(best, 100 * correct / total)
    expected_torch = torch.get_rng_state().clone()
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    output = tmp_path / "new-run"
    result = run(config, root, output)
    assert result[0]["accuracy_percent"] == best
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert random.getstate() == expected_python
    assert np.array_equal(np.random.get_state()[1], expected_numpy[1])
    directory = output / "sub-001_recording-01"
    assert json.loads((directory / "split.json").read_text()) == split
    assert (directory / "best_model.pt").is_file()
    assert (directory / "validation_predictions.csv").is_file()
    assert json.loads((output / "run.json").read_text())["config"]["mode"] == "paper"
    with pytest.raises(FileExistsError):
        run(config, root, output)
