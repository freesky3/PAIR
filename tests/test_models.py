"""Check the public model constructor against frozen historical implementations."""

import importlib.util
import random

import numpy as np
import pytest
import torch

from pair_eeg.config import Experiment
from pair_eeg.models import build_model


def seed():
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def legacy_module(path):
    spec = importlib.util.spec_from_file_location("legacy_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,feature,samples,task",
    [
        ("shallownet", "raw", 400, "20-c"),
        ("deepnet", "raw", 400, "20-c"),
        ("eegnet", "raw", 400, "20-c"),
        ("conformer", "raw", 400, "20-c"),
        ("tsconv", "raw", 400, "20-c"),
        ("glfnet", "raw", 400, "20-c"),
        ("mlpnet", "psd", 5, "20-c"),
        ("glfnet_mlp", "de", 5, "20-c"),
        ("glfnet_mlp", "de", 4, "state"),
        ("GCN_LSTM", "raw", 400, "20-c"),
    ],
)
def test_initialization_forward_and_update_match(name, feature, samples, task, tmp_path, repo_root):
    """Same initialization, logits, loss and one Adam update must agree on CPU."""
    folder = "biclassification" if task == "state" else "benchmark"
    legacy = legacy_module(repo_root / f"archived_code/{folder}/model.py")
    adjacency = tmp_path / "adj.npy"
    np.save(adjacency, np.eye(62, dtype=np.float32))
    config = Experiment(task=task, model=name, feature=feature, remove_alpha=(samples == 4))
    kwargs = dict(out_dim=config.classes)
    if name in {"mlpnet", "glfnet_mlp"}:
        kwargs["input_dim"] = 62 * samples
        if name == "glfnet_mlp":
            kwargs["emb_dim"] = 128
    else:
        kwargs.update(C=62, T=samples)
        if name == "GCN_LSTM":
            kwargs["adj_path"] = str(adjacency)
    seed()
    old = getattr(legacy, name)(**kwargs)
    old_rng = torch.get_rng_state().clone()
    seed()
    new = build_model(config, 62, samples, adjacency)
    assert torch.equal(torch.get_rng_state(), old_rng)
    assert old.state_dict().keys() == new.state_dict().keys()
    for key in old.state_dict():
        torch.testing.assert_close(new.state_dict()[key], old.state_dict()[key], rtol=0, atol=0)
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(2, 62, samples, generator=generator)
    y = torch.tensor([0, 1])
    outputs, losses, updated = [], [], []
    for model in [old, new]:
        seed()
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        opt.zero_grad()
        logits = model(x)
        loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(logits, y)
        loss.backward()
        opt.step()
        outputs.append(logits.detach())
        losses.append(loss.detach())
        updated.append(model.state_dict())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for key in updated[0]:
        torch.testing.assert_close(updated[0][key], updated[1][key], rtol=0, atol=0)


def test_corrected_spectral_width_is_explicit(tmp_path):
    seed()
    legacy = Experiment(task="state", model="glfnet_mlp", feature="de")
    with pytest.raises(ValueError, match="Archived GLFNet"):
        build_model(legacy, 62, 5, tmp_path / "adj.npy")
    corrected = Experiment(task="state", model="glfnet_mlp", feature="de", mode="corrected")
    assert build_model(corrected, 62, 5, tmp_path / "adj.npy")(torch.zeros(2, 62, 5)).shape == (
        2,
        2,
    )
