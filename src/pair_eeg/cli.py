"""A small command interface for downloads, paper analysis, and explicit training."""

import argparse
import json
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    """Build the command parser without touching datasets or initializing Torch."""
    root = argparse.ArgumentParser(prog="pair-eeg", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Describe the two modes and package version.")
    fetch = commands.add_parser("download", help="Download the versioned Hugging Face dataset.")
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    fetch.add_argument("--repo", default="skywalker-p/PAIR")
    fetch.add_argument("--revision", default="v1.0.0")
    fetch.add_argument(
        "--include",
        action="append",
        help="Optional Hub glob; partial downloads are not fully validated.",
    )
    validate = commands.add_parser(
        "validate-data", help="Verify dataset manifest, shapes, and checksums."
    )
    validate.add_argument("--data-dir", type=Path, default=Path("data"))
    validate.add_argument("--headers-only", action="store_true")
    paper = commands.add_parser(
        "reproduce-paper", help="Regenerate tables/figures from archived scores; no training."
    )
    paper.add_argument("--results", type=Path, default=Path("results/paper"))
    paper.add_argument("--output", type=Path, default=Path("outputs/paper"))
    train = commands.add_parser("train", help="Explicitly launch a new training run.")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--mode", choices=["paper", "corrected"])
    train.add_argument("--data-dir", type=Path, default=Path("data"))
    train.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory; will not overwrite an existing run.",
    )
    train.add_argument("--device", default="cpu")
    train.add_argument(
        "--limit", type=int, help="Only the first N recordings; changes the run scope."
    )
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="Print effective configuration without loading Torch or data.",
    )
    return root


def main(argv: list[str] | None = None) -> None:
    """Execute one explicit command.

    Args:
        argv: CLI arguments, or None to use the process command line.
    """
    args = parser().parse_args(argv)
    if args.command == "info":
        from pair_eeg import __version__

        print(
            json.dumps(
                {
                    "version": __version__,
                    "default_mode": "paper",
                    "paper": "Archived-score reproduction; historical training behavior where supported.",
                    "corrected": "Explicit routing/dimension fixes; new results are not paper scores.",
                },
                indent=2,
            )
        )
    elif args.command == "download":
        from pair_eeg.data.release import download

        print(download(args.data_dir, args.repo, args.revision, args.include))
    elif args.command == "validate-data":
        from pair_eeg.data.release import validate

        print(json.dumps(validate(args.data_dir, not args.headers_only), indent=2))
    elif args.command == "reproduce-paper":
        from pair_eeg.analysis.sensors import plot
        from pair_eeg.analysis.statistics import plot_content, summarize

        if args.results.resolve() == args.output.resolve():
            raise ValueError("Output must differ from the immutable archived result directory.")
        summaries, report = summarize(args.results, args.output)
        plot_content(summaries, args.output / "PerceptionVsRecall.png")
        plot(args.results / "sensor_means.csv", args.output / "topomap_human.png")
        print(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "significant_content": report["significant_content"],
                },
                indent=2,
            )
        )
    elif args.command == "train":
        from pair_eeg.config import load_config

        config = load_config(args.config, args.mode)
        if args.dry_run:
            print(json.dumps(config.as_dict(), indent=2))
            return
        try:
            from pair_eeg.training.runner import run
        except ModuleNotFoundError as error:
            raise SystemExit(
                "Training dependencies are optional: run uv sync --extra train "
                "(CPU) or uv sync --extra cuda (CUDA 12.1)."
            ) from error
        run(config, args.data_dir, args.output, args.device, args.limit)


if __name__ == "__main__":
    main()
