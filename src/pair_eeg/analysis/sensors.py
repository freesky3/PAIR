"""Redraw archived sensor means using one shared color scale; no model fitting."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


def plot(input_path: Path, output_path: Path):
    """Render frozen sensor means on a common scale.

    Args:
        input_path: CSV containing channel positions and condition means.
        output_path: Destination PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as handle:
        rows = list(csv.DictReader(handle))
    positions = np.array([[float(row["x"]), float(row["y"])] for row in rows])
    values = [
        np.array([float(row[condition]) for row in rows]) for condition in ["perception", "recall"]
    ]
    bounds = (min(v.min() for v in values), max(v.max() for v in values))
    radius = np.linalg.norm(positions, axis=1).max()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), layout="constrained")
    for ax, scores, label in zip(axes, values, ["Perception", "Recall"]):
        im, _ = mne.viz.plot_topomap(
            scores,
            positions,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            contours=6,
            sensors=True,
            sphere=(0, 0, 0, radius),
            vlim=bounds,
        )
        ax.set_title(label)
    fig.colorbar(im, ax=axes, shrink=0.78, label="Validation accuracy (%)")
    fig.suptitle("Human meta-feature: sensor-level validation scores")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Shared color scale:", bounds)
