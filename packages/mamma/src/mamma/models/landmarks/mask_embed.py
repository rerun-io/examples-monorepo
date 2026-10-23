"""SAM-style mask embedding — port of ``models_2d/mask_proc.py`` + ``layer_norm.py``."""

from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MaskEmbedding(nn.Module):
    """Downscale a 0..1 mask crop to the backbone feature grid and embed it.

    Two stride-``patch_size`` convs (net stride ``patch_size**2`` =
    backbone stride when ``patch_size = vit_patch_size // 4`` and the inner
    convs each stride 4) — layout mirrors SAM's prompt encoder. The inner
    Sequential is named ``mask_downscaling`` so checkpoint keys are
    ``mask_downscaling.mask_downscaling.N.*``.
    """

    def __init__(self, embed_dim: int, patch_size: int, mask_in_chans: int = 16) -> None:
        super().__init__()
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(x)
