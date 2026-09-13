"""Compare legacy and refactored training on a small real-data recording prefix."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pair_eeg.config import Experiment
from pair_eeg.data.epochs import recordings
from pair_eeg.training.runner import environment, run, seed_run

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "eegnet_raw_watch": Experiment(model="eegnet", task="20-c", feature="raw"),
    "mlp_psd_watch": Experiment(model="mlpnet", task="4-c", feature="psd"),
    "mlp_de_state": Experiment(model="mlpnet", task="state", feature="de", batch_size=128),
}


def load_module(path: Path, name: str):
    """Load archived source without invoking its command-line experiment loop."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_components(config: Experiment, data: Path):
    """Bind released metadata paths and explicit ablation to the archived loader.

    Args:
        config: Shared experiment settings for the comparison.
        data: Verified anonymous release directory.

    Returns:
        Legacy model definitions, loader, and original train/evaluate functions.
        Only paths and requested ablation are bound; numerical operations are unchanged.
    """
    folder = (
        ROOT / "archived_code" / ("biclassification" if config.task == "state" else "benchmark")
    )
    color_map = json.loads((ROOT / "metadata/color.json").read_text())
    metadata = types.SimpleNamespace(
        GT_label=str(data / "metadata/GT_label.npy"),
        optical_flow_score=str(data / "metadata/optical_flow_score.npy"),
        video_features=str(ROOT / "metadata/video_analysis_results.npy"),
        map_color=color_map,
        color2num={
            "Neutral Light": 0,
            "Earth & Dark": 1,
            "Cool Tones": 2,
            "Green Nature": 3,
            "Warm Vibrant": 4,
        },
        optical_flow_threshold=0.6427,
        train_ratio=0.8,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    shim = types.ModuleType("Config")
    shim.benchmark_config = lambda: metadata
    previous = sys.modules.get("Config")
    sys.modules["Config"] = shim
    try:
        loader = load_module(folder / "dataloader.py", "parity_legacy_loader")
    finally:
        if previous is None:
            del sys.modules["Config"]
        else:
            sys.modules["Config"] = previous
    if config.task == "state":
        dataset_class = loader.UnifiedDataset
        loader.UnifiedDataset = lambda feature, path: dataset_class(
            feature, path, remove_alpha=config.remove_alpha
        )
    models = load_module(folder / "model.py", "parity_legacy_models")
    source = folder / "main.py"
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"train_one_epoch", "evaluate"}
    ]
    if len(functions) != 2:
        raise ValueError("Archived train/evaluate functions were not found.")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return models, loader, namespace


def legacy_run(config: Experiment, data: Path, destination: Path, device: str, count: int) -> None:
    """Run the archived train/evaluate functions using the shared experimental contract."""
    models, loader, functions = legacy_components(config, data)
    destination.mkdir(parents=True, exist_ok=False)
    seed_run(config)
    scores = []
    for record in recordings(data)[:count]:
        output = destination / f"{record.subject}_{record.recording}"
        output.mkdir()
        condition = "watch" if config.task == "state" else config.condition
        folder = f"{condition}_cleaned" if config.feature == "raw" else f"{condition}_PSD_DE"
        path = str(data / folder / record.filename)
        if config.task == "state":
            train, valid = loader.get_dataloaders(config.feature, path)
        else:
            train, valid = loader.get_dataloaders(path, config.feature, config.task)
        split = {"train": train.dataset.indices, "validation": valid.dataset.indices}
        (output / "split.json").write_text(json.dumps(split))
        samples = 400 if condition == "watch" or config.task == "state" else 600
        if config.feature != "raw":
            samples = 4 if config.remove_alpha else 5
        kwargs = {"out_dim": config.classes}
        if config.model == "mlpnet":
            kwargs["input_dim"] = 62 * samples
        else:
            kwargs.update(C=62, T=samples)
        model = getattr(models, config.model)(**kwargs).to(device)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        best, best_epoch, patience = 0.0, 0, 0
        with (output / "epochs.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "training_loss",
                    "training_accuracy",
                    "validation_loss",
                    "validation_accuracy",
                ]
            )
            for epoch in range(config.epochs):
                train_loss, train_acc = functions["train_one_epoch"](
                    model, train, criterion, optimizer, device
                )
                val_loss, val_acc = functions["evaluate"](model, valid, criterion, device)
                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])
                handle.flush()
                if val_acc > best:
                    best, best_epoch, patience = val_acc, epoch, 0
                    torch.save(model.state_dict(), output / "best_model.pt")
                else:
                    patience += 1
                if patience >= config.patience:
                    break
        scores.append(
            {
                "source_index": record.index,
                "accuracy_percent": best,
                "best_epoch": best_epoch,
                "epochs_completed": epoch + 1,
            }
        )
        print(
            json.dumps(
                {
                    "implementation": "legacy",
                    "model": config.model,
                    "recording": record.index,
                    **scores[-1],
                }
            ),
            flush=True,
        )
    (destination / "scores.json").write_text(json.dumps(scores, indent=2))


def compare(config: Experiment, data: Path, output: Path, count: int) -> list[dict]:
    """Compare paired traces and best checkpoint tensors, separately from historical scores."""
    with (ROOT / "results/paper/session_scores.csv").open() as handle:
        archived = list(csv.DictReader(handle))
    summaries = []
    for record in recordings(data)[:count]:
        name = f"{record.subject}_{record.recording}"
        legacy, current = output / "legacy" / name, output / "refactored" / name
        split_equal = json.loads((legacy / "split.json").read_text()) == json.loads(
            (current / "split.json").read_text()
        )
        traces = []
        for folder in [legacy, current]:
            with (folder / "epochs.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            traces.append(np.array([[float(v) for v in r.values()] for r in rows]))
        same_shape = traces[0].shape == traces[1].shape
        trace_equal = same_shape and np.array_equal(*traces)
        max_trace_delta = float(np.abs(traces[0] - traces[1]).max()) if same_shape else None
        checkpoints = [
            torch.load(p / "best_model.pt", map_location="cpu", weights_only=True)
            for p in [legacy, current]
        ]
        keys_equal = checkpoints[0].keys() == checkpoints[1].keys()
        parameter_equal = keys_equal and all(
            torch.equal(checkpoints[0][key], checkpoints[1][key]) for key in checkpoints[0]
        )
        parameter_delta = (
            max(
                float((checkpoints[0][key].double() - checkpoints[1][key].double()).abs().max())
                for key in checkpoints[0]
            )
            if keys_equal
            else None
        )
        condition = (
            ("without_alpha" if config.remove_alpha else "full")
            if config.task == "state"
            else config.condition
        )
        historical = next(
            (
                float(r["accuracy_percent"])
                for r in archived
                if r["task"] == config.task
                and r["model"] == config.model
                and r["feature"] == config.feature
                and r["condition"] == condition
                and int(r["source_index"]) == record.index
            ),
            None,
        )
        best = [float(trace[:, 4].max()) for trace in traces]
        summaries.append(
            {
                "source_index": record.index,
                "subject_id": record.subject,
                "recording_id": record.recording,
                "split_equal": split_equal,
                "epochs": [len(t) for t in traces],
                "trace_exact": bool(trace_equal),
                "max_trace_absolute_difference": max_trace_delta,
                "checkpoint_exact": parameter_equal,
                "max_checkpoint_absolute_difference": parameter_delta,
                "legacy_best_validation": best[0],
                "refactored_best_validation": best[1],
                "archived_validation": historical,
                "refactored_minus_archived_pp": None
                if historical is None
                else best[1] - historical,
            }
        )
    return summaries


def main() -> None:
    """Run only explicitly selected small real-data comparisons and write a complete report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--recordings", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.recordings <= 3 or args.epochs < 1 or args.threads < 1:
        parser.error("Use 1-3 recordings and positive epoch/thread counts.")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    report = {
        "environment": environment(),
        "device": args.device,
        "description": "Real-data refactor comparison; not an independent test evaluation.",
        "archive_warning": "Historical configuration/RNG/environment were not fully archived; "
        "legacy-vs-refactored equality and archive agreement are separate checks.",
        "dataset_manifest_sha256": hashlib.sha256(
            (args.data_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "cases": {},
    }
    for case in args.cases:
        started = time.monotonic()
        config = replace(CASES[case], epochs=args.epochs, num_workers=args.num_workers)
        folder = args.output / case
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config.as_dict(), indent=2))
        print(json.dumps({"event": "case_start", "case": case}), flush=True)
        legacy_run(config, args.data_dir, folder / "legacy", args.device, args.recordings)
        run(config, args.data_dir, folder / "refactored", args.device, limit=args.recordings)
        comparison = compare(config, args.data_dir, folder, args.recordings)
        report["cases"][case] = {
            "config": config.as_dict(),
            "comparisons": comparison,
            "seconds": round(time.monotonic() - started, 2),
        }
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
        print(
            json.dumps({"event": "case_completed", "case": case, "comparisons": comparison}),
            flush=True,
        )


if __name__ == "__main__":
    main()
