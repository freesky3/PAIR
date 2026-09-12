"""Run an explicit list of experiment configs in isolated processes."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Parse a list of configs, then invoke one new run per configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["paper", "corrected"], default="paper")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for index, config in enumerate(args.configs):
        command = [
            sys.executable,
            "-m",
            "pair_eeg.cli",
            "train",
            "--config",
            str(config),
            "--data-dir",
            str(args.data_dir),
            "--output",
            str(args.output / f"{index:03d}-{config.stem}"),
            "--mode",
            args.mode,
            "--device",
            args.device,
        ]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
