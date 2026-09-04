"""RayMapEncoder.

Encodes a (B, S, 6, H, W) ray map (d_world + broadcast t) into patch-wise
features (B, S, embed_dim, H/p, W/p), added to the backbone patch embedding.

Structure:
    PixelUnshuffle(patch_size)
    Conv 3x3 -> intermediate_dims[0]
    ResidualBlock(...) x N
    Conv 1x1 -> embed_dim   (zero-initialized: ray_feat starts near 0)
    LayerNorm

The final projection is zero-initialized so the added ray_feat starts at 0 and
does not perturb the pretrained ViT patch-token distribution; the non-zero
contribution is learned during training (residual-style, as in ControlNet/LoRA).
"""

from typing import List, Sequence, Type

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """Conv-Act-Conv with shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        return self.act(out)


class RayMapEncoder(nn.Module):
    """Ray map -> patch-wise features.

    Args:
        embed_dim:         Output dimension; must equal the backbone embed_dim.
        patch_size:        Patch size (DINOv2 = 14).
        in_chans:          Ray-map channels, default 6 = (d_world(3), t(3)).
        intermediate_dims: Hidden widths.
        zero_init_proj:    Zero-initialize the final 1x1 conv (default True).

    Input:
        ray_map: (B, S, in_chans, H, W) float; H/W must be multiples of patch_size.

    Output:
        (B, S, embed_dim, H/p, W/p) float, added to the backbone patch embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 14,
        in_chans: int = 6,
        intermediate_dims: Sequence[int] = (588, 768, 1024),
        zero_init_proj: bool = True,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans

        intermediate_dims = list(intermediate_dims)
        assert len(intermediate_dims) >= 1, "intermediate_dims must have at least 1 entry"

        # Patchify: (B, C, H, W) -> (B, C*P*P, H/P, W/P)
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        # First 3x3 conv projects C*P*P to intermediate_dims[0]
        self.conv_in = nn.Conv2d(
            in_chans * patch_size * patch_size, intermediate_dims[0], 3, 1, 1
        )

        layers: List[nn.Module] = []
        for i in range(len(intermediate_dims) - 1):
            layers.append(
                _ResidualBlock(intermediate_dims[i], intermediate_dims[i + 1], act_layer=act_layer)
            )
        # Final 1x1 conv projects to embed_dim, zero-initialized by default
        proj = nn.Conv2d(intermediate_dims[-1], embed_dim, 1, 1, 0)
        if zero_init_proj:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
        layers.append(proj)
        self.encoder = nn.Sequential(*layers)

        # Final LayerNorm (over the token dimension)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        # With zero_init, LN also uses zero bias so the initial output is exactly 0
        if zero_init_proj:
            nn.init.constant_(self.norm.weight, 1.0)
            nn.init.constant_(self.norm.bias, 0.0)

    def forward(self, ray_map: torch.Tensor) -> torch.Tensor:
        """
        ray_map: (B, S, C, H, W) -> (B, S, embed_dim, H/p, W/p)
        """
        assert ray_map.dim() == 5, f"expected (B, S, C, H, W), got {ray_map.shape}"
        B, S, C, H, W = ray_map.shape
        assert C == self.in_chans, f"in_chans={self.in_chans} but ray_map has {C}"
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"H, W must be multiples of patch_size={self.patch_size}, got ({H}, {W})"
        )

        x = ray_map.reshape(B * S, C, H, W)
        x = self.unshuffle(x)              # (BS, C*P*P, H/P, W/P)
        x = self.conv_in(x)                # (BS, intermediate_dims[0], H/P, W/P)
        x = self.encoder(x)                # (BS, embed_dim, H/P, W/P)

        # LayerNorm over the token dimension (BS, N, C)
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = x.flatten(2).transpose(1, 2)   # (BS, N, embed_dim)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B * S, self.embed_dim, Hp, Wp)

        return x.reshape(B, S, self.embed_dim, Hp, Wp)
