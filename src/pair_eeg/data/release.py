"""Validate and download a versioned release without importing training modules."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    """Hash a file in bounded memory.

    Args:
        path: File whose exact bytes identify the released artifact.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate(root: Path, full: bool = True) -> dict:
    """Check all release files, numeric shapes, and optional complete checksums.

    Args:
        root: Directory containing manifest.json and the dataset files.
        full: Verify file hashes and all numeric values rather than headers only.

    Returns:
        File and byte counts after successful validation.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in seen:
            raise ValueError("Unsafe or duplicate manifest path.")
        seen.add(entry["path"])
        path = root / relative
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        if full and sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {relative}")
        if "shape" in entry:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                raise ValueError(f"Array header mismatch: {relative}")
            if full and not np.isfinite(array).all():
                raise ValueError(f"Nonfinite data: {relative}")
    return {
        "files": len(seen),
        "bytes": sum(e["bytes"] for e in manifest["files"]),
        "full_checksums": full,
        "version": manifest["version"],
    }


def download(
    root: Path,
    repo: str = "skywalker-p/PAIR",
    revision: str = "v1.0.0",
    patterns: list[str] | None = None,
) -> Path:
    """Download a pinned dataset release, honoring the user's HF proxy settings.

    Args:
        root: Local download directory.
        repo: Hugging Face dataset repository identifier.
        revision: Immutable release tag or commit; defaults to the documented data release.
        patterns: Optional Hub file patterns for a partial download.

    Returns:
        The resolved local dataset directory. Partial downloads require targeted checks.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=root,
        allow_patterns=patterns,
    )
    return root.resolve()
