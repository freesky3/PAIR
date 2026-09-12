"""Upload the approved anonymous dataset privately with per-file resume and verification."""

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from pair_eeg.data.release import sha256, validate


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

    def log(event: str, **details) -> None:
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        line = json.dumps(record)
        with lock:
            with status.open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)

    log("local_validation", **validate(root, full=True))
    manifest = json.loads((root / "manifest.json").read_text())
    files = {entry["path"]: entry for entry in manifest["files"]}
    for name in ["README.md", "LICENSE", ".gitattributes", "manifest.json"]:
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
                log("retry", path=entry["path"], attempt=attempt, error=type(error).__name__)
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(upload, entry) for entry in pending]):
            future.result()

    info = api.dataset_info(repo, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    if set(remote) != set(files):
        raise ValueError("Remote file set differs from the reviewed release.")
    for entry in files.values():
        if not matches(entry, remote[entry["path"]]):
            raise ValueError(f"Remote digest mismatch: {entry['path']}")
    for name in [
        "metadata/stimulus_labels.csv",
        "watch_PSD_DE/sub-001_recording-01.npy",
        "watch_cleaned/sub-001_recording-01.npy",
    ]:
        downloaded = Path(hf_hub_download(repo, name, repo_type="dataset", revision=info.sha))
        if sha256(downloaded) != files[name]["sha256"]:
            raise ValueError(f"Downloaded file mismatch: {name}")
    log(
        "verified", commit=info.sha, files=len(files), bytes=sum(e["bytes"] for e in files.values())
    )
    if public:
        tag = "v" + manifest["version"]
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
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between one and eight")
    publish(args.folder, args.repo, args.workers, args.publish, args.status)
