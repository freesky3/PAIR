"""Spectral MLP baselines with an explicit local frequency-band width."""

import torch
from torch import Tensor, nn


class MLPNet(nn.Module):
    """MLPNet with the archived initialization and forward operations.

    Args:
        out_dim: Number of classification outputs.
        input_dim: Flattened input width.
    """

    def __init__(self, out_dim: int, input_dim: int) -> None:
        super(MLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        out = self.net(x)
        return out


class GLFNetMLP(nn.Module):
    """GLFNetMLP with the archived initialization and forward operations.

    Args:
        out_dim: Number of classification outputs.
        emb_dim: Hidden embedding width.
        input_dim: Flattened input width.
        local_bands: Local spectral feature count.
    """

    def __init__(self, out_dim: int, emb_dim: int, input_dim: int, local_bands: int = 5) -> None:
        super(GLFNetMLP, self).__init__()
        self.globalnet = MLPNet(emb_dim, input_dim)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = MLPNet(emb_dim, 12 * local_bands)
        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        global_feature = self.globalnet(x)
        occipital_x = x[:, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
