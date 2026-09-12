"""Temporal CNN baselines; constructor dry-runs intentionally remain in training mode."""

import torch
from torch import Tensor, nn


class ShallowNet(nn.Module):
    """ShallowNet with the archived initialization and forward operations.

    Args:
        out_dim: Number of classification outputs.
        C: Number of EEG channels.
        T: Number of time samples.
    """

    def __init__(self, out_dim: int, C: int, T: int) -> None:
        super(ShallowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class DeepNet(nn.Module):
    """DeepNet with the archived initialization and forward operations.

    Args:
        out_dim: Number of classification outputs.
        C: Number of EEG channels.
        T: Number of time samples.
    """

    def __init__(self, out_dim: int, C: int, T: int) -> None:
        super(DeepNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (C, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class EEGNet(nn.Module):
    """EEGNet with the archived initialization and forward operations.

    Args:
        out_dim: Number of classification outputs.
        C: Number of EEG channels.
        T: Number of time samples.
    """

    def __init__(self, out_dim: int, C: int, T: int) -> None:
        super(EEGNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (C, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class TSConv(nn.Sequential):
    """TSConv with the archived initialization and forward operations.

    Args:
        emb_size: Embedding width.
        depth: Number of transformer encoder layers.
        out_dim: Number of classification outputs.
        C: Number of EEG channels.
        T: Number of time samples.
    """

    def __init__(
        self, emb_size: int = 40, depth: int = 3, out_dim: int = 4, C: int = 62, T: int = 400
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, C, T)
            dummy_output = self.net(dummy_input)
            self.num_flat_features = dummy_output.numel()
        self.out = nn.Linear(self.num_flat_features, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class GLFNet(nn.Module):
    """GLFNet with the archived initialization and forward operations.

    Args:
        out_dim: Number of classification outputs.
        emb_dim: Hidden embedding width.
        C: Number of EEG channels.
        T: Number of time samples.
    """

    def __init__(self, out_dim: int, emb_dim: int = 128, C: int = 62, T: int = 400) -> None:
        super(GLFNet, self).__init__()
        self.globalnet = ShallowNet(emb_dim, C, T)
        self.occipital_index = list(range(50, 62))
        self.occipital_localnet = ShallowNet(emb_dim, 12, T)
        self.out = nn.Linear(emb_dim * 2, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        global_feature = self.globalnet(x)
        global_feature = global_feature.view(x.size(0), -1)
        occipital_x = x[:, self.occipital_index, :]
        occipital_feature = self.occipital_localnet(occipital_x)
        out = self.out(torch.cat((global_feature, occipital_feature), 1))
        return out
