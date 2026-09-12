"""Shared training loop preserving the historical numerical operations."""

import csv
import importlib.metadata
import json
import os
import platform
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pair_eeg.config import Experiment
from pair_eeg.data.epochs import make_loaders, recordings
from pair_eeg.models import build_model


def seed_run(config: Experiment) -> None:
    """Initialize RNGs once per run, preserving the archived SVM Python-RNG omission.

    Args:
        config: Protocol, seed, and optional strict deterministic settings.
    """
    np.random.seed(config.seed)
    if config.mode == "corrected" or config.model != "svm":
        random.seed(config.seed)
    else:
        warnings.warn(
            "Archived SVM did not seed Python random; exact historical splits "
            "cannot be reconstructed. Corrected mode seeds all RNGs.",
            stacklevel=2,
        )
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config.strict_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(config.strict_determinism)


def train_epoch(model, loader, criterion, optimizer, device, cast_float: bool = False):
    """Run the original Adam update sequence once over the supplied loader.

    Args:
        model: Neural baseline in its current parameter state.
        loader: Training batches in the selected historical order.
        criterion: Cross entropy with the configured label smoothing.
        optimizer: Adam optimizer associated with the model.
        device: CPU or CUDA device string.
        cast_float: Convert ablated state arrays back to float32 as the legacy loop does.

    Returns:
        Mean batch loss and sample accuracy in percent.
    """
    model.train()
    loss_sum = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if cast_float:
            x = x.float()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        predicted = output.max(1)[1]
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device, cast_float: bool = False):
    """Evaluate once and collect predictions during that same loader iteration.

    Returns:
        Loss, accuracy, labels, and predictions. No extra iterator consumes RNG state.
    """
    model.eval()
    loss_sum = correct = total = 0
    labels, predictions = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if cast_float:
                x = x.float()
            output = model(x)
            loss = criterion(output, y)
            loss_sum += loss.item()
            predicted = output.max(1)[1]
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            labels.extend(y.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
    return loss_sum / len(loader), 100.0 * correct / total, labels, predictions


def _features(loader):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.view(x.size(0), -1).numpy())
        ys.append(y.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _svm(config, train, valid):
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    x_train, y_train = _features(train)
    x_valid, y_valid = _features(valid)
    if config.task == "state":
        model = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000, random_state=config.seed)
    else:
        model = make_pipeline(
            PCA(n_components=None, random_state=config.seed),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=config.seed,
                class_weight="balanced",
            ),
        )
    model.fit(x_train, y_train)
    prediction = model.predict(x_valid)
    return float(accuracy_score(y_valid, prediction) * 100), y_valid.tolist(), prediction.tolist()


def environment() -> dict:
    """Return nonsecret versions and device details for run provenance."""
    packages = {}
    for name in ["torch", "numpy", "scipy", "scikit-learn", "torch-geometric", "einops"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def run(
    config: Experiment, data: Path, output: Path, device: str = "cpu", limit: int | None = None
) -> list[dict]:
    """Fit an explicit experiment, preserving recording/RNG order and saving provenance.

    Args:
        config: Validated protocol and optimizer settings.
        data: Dataset root with a recording manifest and safe label CSV.
        output: A new directory; existing directories are rejected to protect prior results.
        device: Explicit CPU/CUDA device, never silently changed.
        limit: Optional first-N recording limit for debugging, not a paper-reproduction claim.

    Returns:
        Per-recording validation scores and effective score definitions.
    """
    records = recordings(data)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        records = records[:limit]
    if not records:
        raise ValueError("No recordings selected.")
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(
        json.dumps(
            {
                "config": config.as_dict(),
                "environment": environment(),
                "recording_limit": limit,
                "note": "Validation reporting; no independent test set. Historical equality is not guaranteed.",
            },
            indent=2,
        )
        + "\n"
    )
    seed_run(config)
    results = []
    for record in records:
        directory = output / f"{record.subject}_{record.recording}"
        directory.mkdir()
        train, valid, split = make_loaders(data, record, config)
        (directory / "split.json").write_text(json.dumps(split) + "\n")
        # Snapshot without drawing randomness; exact future replays can use this state.
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            directory / "rng_before_model.pt",
        )
        if config.model == "svm":
            score, labels, predictions = _svm(config, train, valid)
            epoch = None
        else:
            dataset = train.dataset.dataset
            sample_shape = (
                dataset.conditions[0].shape if config.task == "state" else dataset.data.shape
            )
            model = build_model(
                config, sample_shape[-2], sample_shape[-1], data / "metadata/adj_matrix.npy"
            ).to(device)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
            score, epoch, patience = 0.0, 0, 0
            labels = predictions = None
            with (directory / "epochs.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "epoch",
                        "training_loss",
                        "training_accuracy",
                        "validation_loss",
                        "validation_accuracy",
                    ]
                )
                for current in range(config.epochs):
                    loss, accuracy = train_epoch(
                        model, train, criterion, optimizer, device, config.task == "state"
                    )
                    val_loss, val_accuracy, ys, ps = evaluate(
                        model, valid, criterion, device, config.task == "state"
                    )
                    writer.writerow([current, loss, accuracy, val_loss, val_accuracy])
                    handle.flush()
                    if val_accuracy > score:
                        score, epoch, patience = val_accuracy, current, 0
                        labels, predictions = ys, ps
                        torch.save(model.state_dict(), directory / "best_model.pt")
                    else:
                        patience += 1
                    if patience >= config.patience:
                        break
        if labels is not None:
            with (directory / "validation_predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["epoch_index", "label", "prediction"])
                writer.writerows(zip(split["validation"], labels, predictions))
        result = {
            "subject_id": record.subject,
            "recording_id": record.recording,
            "source_index": record.index,
            "accuracy_percent": score,
            "best_epoch": epoch,
            "score_type": "validation" if config.model == "svm" else "best_validation",
        }
        results.append(result)
        (output / "scores.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{record.subject}/{record.recording}: {score:.2f}%", flush=True)
    return results
