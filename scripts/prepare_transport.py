"""Split pending EEG files into small transport parts without changing original bytes."""

import argparse
import json
import shutil
from pathlib import Path

from pair_eeg.data.release import sha256


def prepare(source: Path, target: Path, chunk_bytes: int, completed: set[str]) -> dict:
    """Write a new transport directory retaining already-uploaded files in original form.

    Args:
        source: Original verified anonymous dataset.
        target: New transport directory; originals are never modified.
        chunk_bytes: Maximum part size before upload.
        completed: Original waveform paths that already exist remotely with matching hashes.

    Returns:
        Transport file and byte counts.
    """
    target.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    entries = []
    for original in manifest["files"]:
        entry = {k: original[k] for k in ["path", "bytes", "sha256"]}
        path = source / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Source checksum mismatch: {entry['path']}")
        waveform = entry["path"].split("/")[0] in {"watch_cleaned", "recall_cleaned"}
        if waveform and entry["path"] not in completed:
            entry["parts"] = []
            with path.open("rb") as handle:
                index = 0
                while block := handle.read(chunk_bytes):
                    part = Path(entry["path"] + f".part-{index:04d}")
                    output = target / part
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(block)
                    entry["parts"].append(
                        {"path": str(part), "bytes": len(block), "sha256": sha256(output)}
                    )
                    index += 1
        else:
            output = target / entry["path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
        entries.append(entry)
    for name in ["README.md", "LICENSE", "manifest.json"]:
        shutil.copyfile(source / name, target / name)
    (target / ".gitattributes").write_text(
        "*.npy filter=lfs diff=lfs merge=lfs -text\n*.part-* filter=lfs diff=lfs merge=lfs -text\n"
    )
    transport = {
        "version": 1,
        "chunk_bytes": chunk_bytes,
        "files": entries,
        "original_manifest_sha256": sha256(source / "manifest.json"),
    }
    (target / "transport.json").write_text(json.dumps(transport, indent=2) + "\n")
    return {"originals": len(entries), "chunks": sum(len(e.get("parts", [])) for e in entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument(
        "--completed", type=Path, help="JSON list of verified remote waveform paths."
    )
    args = parser.parse_args()
    if args.chunk_mib < 1:
        parser.error("chunk-mib must be positive")
    completed = set(json.loads(args.completed.read_text())) if args.completed else set()
    print(json.dumps(prepare(args.source, args.target, args.chunk_mib * 1024**2, completed)))
