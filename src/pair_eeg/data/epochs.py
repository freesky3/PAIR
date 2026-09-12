"""EEG loading and historical normalization, with explicit dataset partitions."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, Dataset, Subset

from pair_eeg.config import Experiment

COLOR_NAMES = ["Neutral Light", "Earth & Dark", "Cool Tones", "Green Nature", "Warm Vibrant"]


@dataclass(frozen=True)
class Recording:
    """An anonymized recording identity in the original processing order."""

    index: int
    subject: str
    recording: str
    filename: str


def recordings(root: Path) -> list[Recording]:
    """Read the released manifest in its historical source order.

    Args:
        root: Dataset directory containing ``recordings.csv``.

    Returns:
        Records ordered by source_index, not by a user's filesystem enumeration.
    """
    with (root / "recordings.csv").open() as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["source_index"]))
    indices = [int(row["source_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate recording source indices.")
    return [
        Recording(int(r["source_index"]), r["subject_id"], r["recording_id"], r["filename"])
        for r in rows
    ]


def _path(root: Path, record: Recording, condition: str, feature: str) -> Path:
    folder = f"{condition}_cleaned" if feature == "raw" else f"{condition}_PSD_DE"
    return root / folder / record.filename


def _labels(root: Path, task: str) -> torch.Tensor:
    """Read safe CSV labels, preserving the stored float32 optical-flow comparison."""
    with (root / "metadata/stimulus_labels.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 250 or [int(r["stimulus_index_0based"]) for r in rows] != list(range(250)):
        raise ValueError("Stimulus labels must contain 250 ordered clip indices.")
    columns = {
        "20-c": "semantic_20class_1based",
        "4-c": "semantic_4class_0based",
        "number": "number_class_0based",
        "face": "face",
        "human": "human",
    }
    if task == "fast_slow":
        return (
            torch.tensor([float(r["optical_flow_score"]) for r in rows], dtype=torch.float32)
            > 0.6427
        ).long()
    if task == "color":
        values = [COLOR_NAMES.index(r["dominant_color_group"]) for r in rows]
    else:
        values = [int(r[columns[task]]) - (1 if task == "20-c" else 0) for r in rows]
    return torch.tensor(values, dtype=torch.long)


class ContentEpochs(Dataset):
    """Content trials using Torch sample-SD normalization and optional training noise.

    Args:
        root: Released dataset directory.
        record: Recording selected from its manifest.
        config: Experiment defining condition, actual feature, and task labels.
        augment: Add Gaussian noise after normalization, as in the archived training loader.
    """

    def __init__(self, root: Path, record: Recording, config: Experiment, augment: bool):
        data = torch.from_numpy(
            np.load(
                _path(root, record, config.condition, config.actual_feature), allow_pickle=False
            )
        ).float()
        if config.actual_feature != "raw":
            data = data[0 if config.actual_feature == "psd" else 1].reshape(250, 62, 5)
        else:
            data = data.reshape(250, 62, -1)
        self.data = data
        self.labels = _labels(root, config.task)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        x = (x - x.mean()) / (x.std() + 1e-6)
        if self.augment:
            x = x + torch.randn_like(x) * 0.05
        return x, self.labels[index]


class StateEpochs(Dataset):
    """Paired state inputs preserving raw crop, filter, and NumPy/Torch SD behavior.

    Ablation in the archived loader converts arrays to NumPy, whose population SD
    differs from Torch's sample SD. Paper mode preserves this numerical detail.
    Corrected mode also preserves it: correcting routing does not silently redefine
    normalization or the experiment's scientific protocol.
    """

    def __init__(self, root: Path, record: Recording, config: Experiment):
        self.conditions = []
        for condition in ["watch", "recall"]:
            data = torch.from_numpy(
                np.load(_path(root, record, condition, config.feature), allow_pickle=False)
            ).float()
            if config.feature == "raw":
                data = data.reshape(250, 62, -1)
                if condition == "recall":
                    data = data[..., 100:500].clone()
                if config.remove_alpha:
                    b, a = butter(4, [8 / 100, 13 / 100], btype="bandstop")
                    data = filtfilt(b, a, data, axis=-1)
            else:
                data = data.permute(1, 2, 3, 0, 4).reshape(250, 62, 10)
                data = (data[..., :5] if config.feature == "psd" else data[..., 5:]).clone()
                if config.remove_alpha:
                    data = np.delete(data, 2, axis=-1)
            self.conditions.append(data)

    def __len__(self) -> int:
        return 500

    def __getitem__(self, index: int):
        condition = int(index >= 250)
        x = self.conditions[condition][index % 250]
        return (x - x.mean()) / (x.std() + 1e-6), condition


def make_loaders(root: Path, record: Recording, config: Experiment):
    """Create loaders and expose the exact indices without drawing additional random values.

    Returns:
        Training loader, validation loader, and serializable split indices.
        Iteration order and RNG consumption follow the original DataLoader setup.
    """
    if config.task == "state":
        train = valid = StateEpochs(root, record, config)
        train_indices = list(range(200)) + list(range(250, 450))
        valid_indices = list(range(200, 250)) + list(range(450, 500))
        random.shuffle(train_indices)
    else:
        train = ContentEpochs(root, record, config, augment=True)
        valid = ContentEpochs(root, record, config, augment=False)
        indices = list(range(len(train)))
        random.shuffle(indices)
        train_indices, valid_indices = indices[:200], indices[200:]
    train_loader = DataLoader(
        Subset(train, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(
        Subset(valid, valid_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, valid_loader, {"train": train_indices, "validation": valid_indices}
