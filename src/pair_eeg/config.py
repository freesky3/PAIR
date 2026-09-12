"""Validated experiment settings with explicit historical and corrected behavior."""

import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

TASK_CLASSES = {
    "20-c": 20,
    "4-c": 4,
    "fast_slow": 2,
    "color": 5,
    "number": 3,
    "face": 2,
    "human": 2,
    "state": 2,
}
MODELS = (
    "shallownet",
    "deepnet",
    "eegnet",
    "conformer",
    "tsconv",
    "glfnet",
    "mlpnet",
    "glfnet_mlp",
    "GCN_LSTM",
    "svm",
)


@dataclass(frozen=True)
class Experiment:
    """One ordered run over recordings, with one RNG initialization.

    Attributes:
        mode: ``paper`` preserves known legacy operations; ``corrected`` fixes routing/dimensions.
        legacy_output_classes: Explicit historical output width, when a legacy entry point fixed it.
        remove_alpha: Apply the original state-ablation transform before normalization.
        seed: Initial RNG seed; complete historical RNG states were not archived.
    """

    task: str = "20-c"
    model: str = "eegnet"
    feature: str = "raw"
    condition: str = "watch"
    mode: str = "paper"
    seed: int = 42
    epochs: int = 400
    patience: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    num_workers: int = 4
    remove_alpha: bool = False
    legacy_output_classes: int | None = None
    strict_determinism: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous task, feature, and protocol combinations."""
        if self.mode not in {"paper", "corrected"} or self.task not in TASK_CLASSES:
            raise ValueError("Unknown mode or task.")
        if self.model not in MODELS or self.feature not in {"raw", "psd", "de"}:
            raise ValueError("Unknown model or feature.")
        if self.condition not in {"watch", "recall"}:
            raise ValueError("condition must be watch or recall.")
        if self.remove_alpha and self.task != "state":
            raise ValueError("Alpha ablation is defined only for the state task.")
        if min(self.epochs, self.patience, self.batch_size) < 1 or self.num_workers < 0:
            raise ValueError("Invalid epoch, patience, batch, or worker count.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters.")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must fit NumPy's unsigned 32-bit range.")
        if self.mode == "corrected" and self.legacy_output_classes is not None:
            raise ValueError("Corrected mode derives output classes from the task.")
        if self.legacy_output_classes is not None and self.legacy_output_classes < 2:
            raise ValueError("legacy_output_classes must be at least two.")
        if self.model in {"mlpnet", "glfnet_mlp"} and self.feature == "raw":
            raise ValueError("MLP baselines require PSD or DE inputs.")
        if self.model not in {"mlpnet", "glfnet_mlp", "svm"} and self.feature != "raw":
            raise ValueError("Temporal neural backbones require raw EEG.")

    @property
    def classes(self) -> int:
        """Return the effective classification-head width."""
        return self.legacy_output_classes or TASK_CLASSES[self.task]

    @property
    def actual_feature(self) -> str:
        """Return the real input feature, exposing the archived content-SVM routing bug."""
        if self.mode == "paper" and self.task != "state" and self.model == "svm":
            return "raw"
        return self.feature

    def as_dict(self) -> dict:
        """Return serializable settings and the actual feature name."""
        return {**asdict(self), "actual_feature": self.actual_feature, "classes": self.classes}


def load_config(path: Path, mode: str | None = None) -> Experiment:
    """Read a TOML experiment with optional mode override.

    Args:
        path: Configuration file containing an ``[experiment]`` table.
        mode: Explicit command-line override, or None to honor the file.

    Returns:
        Validated settings; unknown keys raise ValueError.
    """
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    if set(document) != {"experiment"}:
        raise ValueError("Expected only an [experiment] table.")
    values = document["experiment"]
    unknown = set(values) - {field.name for field in fields(Experiment)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if mode is not None:
        values["mode"] = mode
    return Experiment(**values)
