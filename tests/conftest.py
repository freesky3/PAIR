"""Small synthetic recordings for protocol checks; real participants are never trained here."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tiny_release(tmp_path):
    """Build one 62-channel recording and matching legacy/release metadata."""
    from pair_eeg.data.epochs import Recording

    rng = np.random.default_rng(19)
    for folder, shape in [
        ("watch_cleaned", (5, 50, 62, 400)),
        ("recall_cleaned", (5, 50, 62, 600)),
        ("watch_PSD_DE", (2, 5, 50, 62, 5)),
        ("recall_PSD_DE", (2, 5, 50, 62, 5)),
    ]:
        (tmp_path / folder).mkdir()
        values = rng.normal(size=shape).astype(np.float32)
        np.save(tmp_path / folder / "sub-001_recording-01.npy", values)
    (tmp_path / "metadata").mkdir()
    categories = np.arange(250) % 20 + 1
    flow = rng.random(250).astype(np.float32)
    colors = ["Neutral Light", "Earth & Dark", "Cool Tones", "Green Nature", "Warm Vibrant"]
    video = [
        {
            "main_color": colors[i % 5],
            "object_count": int(i % 7),
            "has_face": int(i % 2),
            "has_human": int((i // 2) % 2),
        }
        for i in range(250)
    ]
    np.save(tmp_path / "metadata/GT_label.npy", categories)
    np.save(tmp_path / "metadata/optical_flow_score.npy", flow)
    np.save(tmp_path / "metadata/video_analysis_results.npy", np.array(video, dtype=object))
    np.save(tmp_path / "metadata/adj_matrix.npy", np.eye(62, dtype=np.float32))
    (tmp_path / "metadata/color.json").write_text(json.dumps(dict(zip(colors, colors))))
    with (tmp_path / "metadata/stimulus_labels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stimulus_index_0based",
                "semantic_20class_1based",
                "semantic_4class_0based",
                "optical_flow_score",
                "fast_label",
                "dominant_color_group",
                "object_count",
                "number_class_0based",
                "face",
                "human",
            ]
        )
        for i, row in enumerate(video):
            count = row["object_count"]
            writer.writerow(
                [
                    i,
                    categories[i],
                    (categories[i] - 1) // 5,
                    flow[i],
                    int(flow[i] > 0.6427),
                    row["main_color"],
                    count,
                    int(count > 1) + int(count > 4),
                    row["has_face"],
                    row["has_human"],
                ]
            )
    (tmp_path / "recordings.csv").write_text(
        "source_index,subject_id,recording_id,filename\n"
        "0,sub-001,recording-01,sub-001_recording-01.npy\n"
    )
    return tmp_path, Recording(0, "sub-001", "recording-01", "sub-001_recording-01.npy")


@pytest.fixture(scope="session")
def repo_root():
    """Locate the project independently of the test working directory."""
    return Path(__file__).resolve().parents[1]
