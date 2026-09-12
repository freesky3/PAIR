"""Prepare an anonymous byte-preserving EEG release; no network uploads occur here."""

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from pair_eeg.data.release import sha256, validate


def prepare(source: Path, metadata: Path, output: Path) -> dict:
    """Copy numeric arrays to anonymous filenames without modifying signal values.

    Args:
        source: Authors' data directory containing four condition/feature folders.
        metadata: Public metadata directory containing safe CSV labels and adjacency.
        output: New release directory. Existing directories are rejected.

    Returns:
        Validated manifest statistics. Personal-name mappings are never written to output.
    """
    names = sorted(p.name for p in (source / "watch_cleaned").glob("*.npy"))
    if len(names) != 60:
        raise ValueError("Expected 60 source recordings.")
    people = sorted({name.split("_")[0] for name in names})
    if len(people) != 20:
        raise ValueError("Expected 20 participants.")
    folders = {
        "watch_cleaned": (5, 50, 62, 400),
        "recall_cleaned": (5, 50, 62, 600),
        "watch_PSD_DE": (2, 5, 50, 62, 5),
        "recall_PSD_DE": (2, 5, 50, 62, 5),
    }
    for folder in folders:
        found = sorted(p.name.replace("_PSD_DE", "") for p in (source / folder).glob("*.npy"))
        if found != names:
            raise ValueError(f"Recording mismatch in {folder}.")
    output.mkdir(parents=True, exist_ok=False)
    for folder in [*folders, "metadata"]:
        (output / folder).mkdir()
    rows, entries, counts = [], [], Counter()
    for index, name in enumerate(names):
        person = name.split("_")[0]
        counts[person] += 1
        subject = f"sub-{people.index(person) + 1:03d}"
        recording = f"recording-{counts[person]:02d}"
        filename = f"{subject}_{recording}.npy"
        rows.append(
            {
                "source_index": index,
                "subject_id": subject,
                "recording_id": recording,
                "filename": filename,
            }
        )
        for folder, expected in folders.items():
            original = name.replace(".npy", "_PSD_DE.npy") if folder.endswith("PSD_DE") else name
            origin = source / folder / original
            array = np.load(origin, mmap_mode="r", allow_pickle=False)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"Invalid source array at recording index {index} in {folder}.")
            target = output / folder / filename
            shutil.copyfile(origin, target)
            entries.append(
                {
                    "path": str(target.relative_to(output)),
                    "bytes": target.stat().st_size,
                    "sha256": sha256(target),
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                }
            )
        print(f"Prepared {subject}/{recording}", flush=True)
    if set(counts.values()) != {3}:
        raise ValueError("Each participant must have three recordings.")
    with (output / "recordings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in [
        "stimulus_labels.csv",
        "adj_matrix.npy",
        "color.json",
        "GT_label.npy",
        "optical_flow_score.npy",
        "channels.tsv",
    ]:
        shutil.copyfile(metadata / name, output / "metadata" / name)
    # Object/pickle metadata are intentionally represented by the safe CSV, not redistributed here.
    for path in [output / "recordings.csv", *(output / "metadata").iterdir()]:
        entries.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "version": "1.0.0",
        "participants": 20,
        "recordings": 60,
        "stimuli": 250,
        "sampling_rate_hz": 200,
        "files": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return validate(output, full=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("metadata"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.metadata, args.output), indent=2))
