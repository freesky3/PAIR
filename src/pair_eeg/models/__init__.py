"""Model construction with preserved initialization order and explicit input sizes."""

from pathlib import Path

from torch import nn

from pair_eeg.config import Experiment

from .cnn import DeepNet, EEGNet, GLFNet, ShallowNet, TSConv
from .conformer import Conformer
from .graph import GCNLSTM
from .spectral import GLFNetMLP, MLPNet

MODEL_TYPES = {
    "shallownet": ShallowNet,
    "deepnet": DeepNet,
    "eegnet": EEGNet,
    "conformer": Conformer,
    "tsconv": TSConv,
    "glfnet": GLFNet,
    "mlpnet": MLPNet,
    "glfnet_mlp": GLFNetMLP,
    "GCN_LSTM": GCNLSTM,
}


def build_model(config: Experiment, channels: int, samples: int, adjacency: Path) -> nn.Module:
    """Construct the selected neural baseline without consuming extra RNG draws.

    Args:
        config: Validated task and protocol settings.
        channels: Number of EEG channels, normally 62.
        samples: Time samples or retained spectral features.
        adjacency: Electrode graph file required by GCN-LSTM.

    Returns:
        A model in training mode, matching the historical constructor behavior.

    Raises:
        ValueError: A historical spectral width cannot accept the requested input.
        FileNotFoundError: The required electrode graph is absent.
    """
    if config.model == "svm":
        raise ValueError("SVM is constructed by the training runner.")
    model_class = MODEL_TYPES[config.model]
    if config.model == "mlpnet":
        return model_class(out_dim=config.classes, input_dim=channels * samples)
    if config.model == "glfnet_mlp":
        bands = samples if config.mode == "corrected" else (4 if config.task == "state" else 5)
        if bands != samples:
            raise ValueError(
                f"Archived GLFNet-MLP expects {bands} local bands, received {samples}. "
                "Use --mode corrected to derive the width from input. This configuration "
                "cannot reproduce an archived table result with the available model file."
            )
        return model_class(
            out_dim=config.classes, emb_dim=128, input_dim=channels * samples, local_bands=bands
        )
    kwargs = dict(out_dim=config.classes, C=channels, T=samples)
    if config.model == "GCN_LSTM":
        if not adjacency.is_file():
            raise FileNotFoundError(adjacency)
        kwargs["adj_path"] = str(adjacency)
    return model_class(**kwargs)
