"""Conformer modules preserving the archived attention and pooling computations."""

import torch
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    """PatchEmbedding with the archived initialization and forward operations.

    Args:
        emb_size: Embedding width.
        C: Number of EEG channels.
    """

    def __init__(self, emb_size: int = 40, C: int = 62) -> None:
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (C, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)), Rearrange("b e (h) (w) -> b (h w) e")
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    """ResidualAdd with the archived initialization and forward operations.

    Args:
        fn: Module wrapped by the residual connection.
    """

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    """FeedForwardBlock with the archived initialization and forward operations.

    Args:
        emb_size: Embedding width.
        expansion: Feed-forward expansion factor.
        drop_p: Feed-forward dropout probability.
    """

    def __init__(self, emb_size: int, expansion: int, drop_p: float) -> None:
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention with the archived initialization and forward operations.

    Args:
        emb_size: Embedding width.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = scaling = (self.emb_size // self.num_heads) ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TransformerEncoderBlock(nn.Sequential):
    """TransformerEncoderBlock with the archived initialization and forward operations.

    Args:
        emb_size: Embedding width.
        num_heads: Number of attention heads.
        drop_p: Feed-forward dropout probability.
        forward_expansion: Model constructor setting.
        forward_drop_p: Model constructor setting.
    """

    def __init__(
        self,
        emb_size: int,
        num_heads: int = 10,
        drop_p: float = 0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ) -> None:
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    """TransformerEncoder with the archived initialization and forward operations.

    Args:
        depth: Number of transformer encoder layers.
        emb_size: Embedding width.
    """

    def __init__(self, depth: int, emb_size: int) -> None:
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    """ClassificationHead with the archived initialization and forward operations.

    Args:
        emb_size: Embedding width.
        out_dim: Number of classification outputs.
    """

    def __init__(self, emb_size: int, out_dim: int) -> None:
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce("b n e -> b e", reduction="mean"),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the archived forward computation.

        Args:
            x: Input batch in the shape expected by this module.

        Returns:
            Output tensor with the original model semantics.
        """
        out = self.clshead(x)
        return out


class Conformer(nn.Sequential):
    """Conformer with the archived initialization and forward operations.

    Args:
        emb_size: Embedding width.
        depth: Number of transformer encoder layers.
        out_dim: Number of classification outputs.
    """

    def __init__(self, emb_size: int = 40, depth: int = 3, out_dim: int = 4, **kwargs) -> None:
        C = kwargs.get("C", 62)
        super().__init__(
            PatchEmbedding(emb_size, C=C),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, out_dim),
        )
