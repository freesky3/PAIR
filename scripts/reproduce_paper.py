"""Convenience entry point for archived-score reproduction; no training."""

import sys

from pair_eeg.cli import main

if __name__ == "__main__":
    main(["reproduce-paper", *sys.argv[1:]])
