"""Verify lossless transport restoration and failure behavior."""

import hashlib
import json

import pytest

from pair_eeg.data.chunks import assemble, download_paths


def chunked(root, name, content):
    """Create a tiny transport entry with reproducible part hashes."""
    parts = []
    for index, offset in enumerate(range(0, len(content), 7)):
        block = content[offset : offset + 7]
        path = root / f"{name}.part-{index:04d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(block)
        parts.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(block),
                "sha256": hashlib.sha256(block).hexdigest(),
            }
        )
    return {
        "path": name,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "parts": parts,
    }


def test_lossless_assembly_and_resume(tmp_path):
    data = bytes(range(256)) + b"NPY-test"
    entry = chunked(tmp_path, "watch_cleaned/sub-001.npy", data)
    (tmp_path / "transport.json").write_text(json.dumps({"files": [entry]}))
    assert assemble(tmp_path) == {"restored": 1, "already_valid": 0}
    assert (tmp_path / entry["path"]).read_bytes() == data
    assert assemble(tmp_path) == {"restored": 0, "already_valid": 1}


def test_corruption_does_not_replace_original(tmp_path):
    entry = chunked(tmp_path, "watch_cleaned/sub-001.npy", b"original-array-bytes")
    (tmp_path / "transport.json").write_text(json.dumps({"files": [entry]}))
    original = tmp_path / entry["path"]
    original.write_bytes(b"existing-not-overwritten-on-failure")
    (tmp_path / entry["parts"][0]["path"]).write_bytes(b"broken!")
    with pytest.raises(ValueError, match="Chunk checksum"):
        assemble(tmp_path)
    assert original.read_bytes() == b"existing-not-overwritten-on-failure"


def test_partial_pattern_resolves_chunks_and_restores_only_selected(tmp_path):
    watch = chunked(tmp_path, "watch_cleaned/sub-001.npy", b"watch-bytes")
    recall = chunked(tmp_path, "recall_cleaned/sub-001.npy", b"recall-bytes")
    payload = {"files": [watch, recall]}
    (tmp_path / "transport.json").write_text(json.dumps(payload))
    paths = download_paths(payload, ["watch_cleaned/*"])
    assert all(p["path"] in paths for p in watch["parts"])
    assert all(p["path"] not in paths for p in recall["parts"])
    assert assemble(tmp_path, ["watch_cleaned/*"])["restored"] == 1
    assert not (tmp_path / recall["path"]).exists()


def test_path_escape_rejected(tmp_path):
    (tmp_path / "transport.json").write_text(
        json.dumps(
            {"files": [{"path": "../outside.npy", "bytes": 1, "sha256": "invalid", "parts": []}]}
        )
    )
    with pytest.raises(ValueError, match="dataset-relative"):
        assemble(tmp_path)
