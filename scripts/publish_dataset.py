"""Upload the approved anonymous dataset privately with per-file resume and verification."""

import argparse
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from pair_eeg.data.chunks import relative_path
from pair_eeg.data.release import download, sha256, validate


def publish(root: Path, repo: str, workers: int, public: bool, status: Path) -> None:
    """Resume files by digest, verify every file, and optionally publish the release.

    Args:
        root: Prepared anonymous dataset directory.
        repo: Owned Hugging Face dataset repository.
        workers: Concurrent file uploads, each committed independently.
        public: Make the verified repository public and create its version tag.
        status: JSON-lines progress log outside the uploaded directory.
    """
    status.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    stopped = threading.Event()
    # Transport warnings may include temporary signed storage URLs.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    def log(event: str, **details) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        line = json.dumps(record)
        with lock:
            with status.open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    manifest = json.loads((root / "manifest.json").read_text())
    transport = None
    if (root / "transport.json").is_file():
        transport = json.loads((root / "transport.json").read_text())
        if sha256(root / "manifest.json") != transport["original_manifest_sha256"]:
            raise ValueError("Transport references a different source manifest.")
        originals = {e["path"]: e for e in manifest["files"]}
        if set(originals) != {e["path"] for e in transport["files"]}:
            raise ValueError("Transport does not cover every original file.")
        files = {}
        for entry in transport["files"]:
            if any(entry[k] != originals[entry["path"]][k] for k in ["bytes", "sha256"]):
                raise ValueError("Transport original digest mismatch.")
            digest = hashlib.sha256()
            total = 0
            for part in entry.get("parts") or [entry]:
                path = root / relative_path(part["path"])
                if path.stat().st_size != part["bytes"] or sha256(path) != part["sha256"]:
                    raise ValueError("Local transport hash mismatch.")
                if part["path"] in files:
                    raise ValueError("Duplicate transport path.")
                files[part["path"]] = part
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024**2), b""):
                        digest.update(block)
                        total += len(block)
            if total != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                raise ValueError("Transport cannot reconstruct the original file.")
        log(
            "local_validation",
            originals=len(originals),
            transport_files=len(files),
            bytes=sum(e["bytes"] for e in originals.values()),
            transport="lossless-chunks",
        )
    else:
        log("local_validation", **validate(root, full=True))
        files = {entry["path"]: entry for entry in manifest["files"]}
    for name in ["README.md", "LICENSE", ".gitattributes", "manifest.json"] + (
        ["transport.json"] if transport else []
    ):
        path = root / name
        files[name] = {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    remote = {item.rfilename: item for item in api.dataset_info(repo, files_metadata=True).siblings}

    def matches(entry: dict, item) -> bool:
        if item is None or item.size != entry["bytes"]:
            return False
        if item.lfs:
            return item.lfs.sha256 == entry["sha256"]
        data = (root / entry["path"]).read_bytes()
        return item.blob_id == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()

    pending = [entry for entry in files.values() if not matches(entry, remote.get(entry["path"]))]
    pending.sort(key=lambda entry: (entry["bytes"], entry["path"]))
    log(
        "resuming",
        pending=len(pending),
        verified_existing=len(files) - len(pending),
        total=len(files),
    )

    def upload(entry: dict) -> None:
        for attempt in range(1, 4):
            if stopped.is_set():
                return
            try:
                api.upload_file(
                    path_or_fileobj=root / entry["path"],
                    path_in_repo=entry["path"],
                    repo_id=repo,
                    repo_type="dataset",
                    commit_message=f"Add PAIR v1.0.0 {entry['path']}",
                )
                log("committed", path=entry["path"], bytes=entry["bytes"])
                return
            except Exception as error:
                chain = []
                cause = error
                while cause is not None and len(chain) < 5:
                    response = getattr(cause, "response", None)
                    chain.append(
                        {
                            "type": type(cause).__name__,
                            "http_status": getattr(response, "status_code", None),
                        }
                    )
                    cause = cause.__cause__ or cause.__context__
                log(
                    "retry",
                    path=entry["path"],
                    attempt=attempt,
                    error=type(error).__name__,
                    causes=chain,
                )
                if attempt == 3:
                    stopped.set()
                    raise
                time.sleep(2 * attempt)

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(upload, entry): entry["path"] for entry in pending}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                future.result()
            except Exception as error:
                failures.append(futures[future])
                log("failed", path=futures[future], error=type(error).__name__)
                stopped.set()
                for other in futures:
                    other.cancel()
    if failures:
        raise RuntimeError(
            f"Upload interrupted after failures in {len(failures)} files; rerun to resume."
        )

    info = api.dataset_info(repo, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    if set(remote) != set(files):
        raise ValueError("Remote file set differs from the reviewed release.")
    for entry in files.values():
        if not matches(entry, remote[entry["path"]]):
            raise ValueError(f"Remote digest mismatch: {entry['path']}")
    samples = [
        "metadata/stimulus_labels.csv",
        "watch_PSD_DE/sub-001_recording-01.npy",
        "watch_cleaned/sub-001_recording-01.npy",
    ]
    restored = None
    if transport:
        restored = download(
            status.parent / "download-verification", repo=repo, revision=info.sha, patterns=samples
        )
    expected = {e["path"]: e for e in manifest["files"]}
    for name in samples:
        downloaded = (
            restored / name
            if restored
            else Path(hf_hub_download(repo, name, repo_type="dataset", revision=info.sha))
        )
        if sha256(downloaded) != expected[name]["sha256"]:
            raise ValueError(f"Downloaded file mismatch: {name}")
    log(
        "verified", commit=info.sha, files=len(files), bytes=sum(e["bytes"] for e in files.values())
    )
    if public:
        tag = "v" + manifest["version"]
        existing = next(
            (ref for ref in api.list_repo_refs(repo, repo_type="dataset").tags if ref.name == tag),
            None,
        )
        if existing is not None and existing.target_commit != info.sha:
            raise ValueError(
                "Existing release tag points to a different commit; refusing to replace it."
            )
        if existing is None:
            api.create_tag(repo, tag=tag, revision=info.sha, repo_type="dataset", exist_ok=False)
        api.update_repo_settings(repo, repo_type="dataset", private=False)
        public_info = HfApi(token=False).dataset_info(repo, revision=tag)
        if public_info.private or public_info.sha != info.sha:
            raise ValueError("Public revision verification failed.")
        log("published", repo=repo, tag=tag, commit=info.sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo", default="skywalker-p/PAIR")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--status", type=Path, default=Path("outputs/dataset-upload.jsonl"))
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between one and 32")
    try:
        publish(args.folder, args.repo, args.workers, args.publish, args.status)
    except Exception as error:
        print(
            json.dumps(
                {"event": "stopped", "error": type(error).__name__, "status_log": str(args.status)}
            ),
            flush=True,
        )
        raise SystemExit(1) from None
