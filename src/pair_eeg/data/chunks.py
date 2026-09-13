"""Byte-preserving transport chunks for unreliable large-file connections."""

import fnmatch
import json
import os
from pathlib import Path

from pair_eeg.data.release import sha256


def relative_path(value: str) -> Path:
    """Validate a dataset-relative path before reading or assembling a file."""
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid dataset-relative path: {value!r}")
    return path


def assemble(root: Path, selected: list[str] | None = None) -> dict:
    """Restore original NPY files from ordered chunks and validate every digest.

    Args:
        root: Download directory containing transport.json, manifest.json and chunks.
        selected: Original-file glob patterns for a partial download, or None for all.

    Returns:
        Counts of restored and already-valid files. Chunks remain available for resume.

    Raises:
        ValueError: A chunk or restored file differs from its expected hash/size.
    """
    transport = json.loads((root / "transport.json").read_text())
    restored = existing = 0
    for entry in transport["files"]:
        name = entry["path"]
        if selected and not any(fnmatch.fnmatchcase(name, pattern) for pattern in selected):
            continue
        target = root / relative_path(name)
        if (
            target.is_file()
            and target.stat().st_size == entry["bytes"]
            and sha256(target) == entry["sha256"]
        ):
            existing += 1
            continue
        if not entry.get("parts"):
            raise ValueError(f"Original file missing or invalid: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".assembling")
        with temporary.open("wb") as handle:
            for part in entry["parts"]:
                source = root / relative_path(part["path"])
                if source.stat().st_size != part["bytes"] or sha256(source) != part["sha256"]:
                    raise ValueError(f"Chunk checksum mismatch: {part['path']}")
                with source.open("rb") as chunk:
                    for block in iter(lambda: chunk.read(1024 * 1024), b""):
                        handle.write(block)
        if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
            raise ValueError(f"Restored checksum mismatch: {name}")
        os.replace(temporary, target)
        restored += 1
    return {"restored": restored, "already_valid": existing}


def download_paths(transport: dict, selected: list[str] | None = None) -> list[str]:
    """Resolve logical file patterns to actual stored chunks plus required metadata."""
    names = {"transport.json", "manifest.json", "README.md", "LICENSE", "recordings.csv"}
    for entry in transport["files"]:
        if (
            selected
            and not entry["path"].startswith("metadata/")
            and not any(fnmatch.fnmatchcase(entry["path"], pattern) for pattern in selected)
        ):
            continue
        if entry.get("parts"):
            names.update(part["path"] for part in entry["parts"])
        else:
            names.add(entry["path"])
    return sorted(names)
