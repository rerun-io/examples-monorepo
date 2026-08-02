# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

from collections.abc import Callable

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


class Attention(nn.Module):
    """Multi-head attention with selectable upstream inference math."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *,
        use_sdpa: bool,
    ) -> None:
        super().__init__()
        self.num_heads: int = num_heads
        head_dim: int = dim // num_heads
        self.scale: float = head_dim**-0.5
        self.use_sdpa: bool = use_sdpa

        self.qkv: nn.Linear = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop: nn.Dropout = nn.Dropout(attn_drop)
        self.proj: nn.Linear = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop: nn.Dropout = nn.Dropout(proj_drop)

    def forward(self, x_bnc: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        """Apply multi-head self-attention.

        Args:
            x_bnc: Float token tensor shaped ``b n c``.

        Returns:
            Float attended token tensor shaped ``b n c``.
        """
        batch_size: int = x_bnc.shape[0]
        token_count: int = x_bnc.shape[1]
        channels: int = x_bnc.shape[2]
        qkv_3bhnc: Float[Tensor, "3 b heads n head_dim"] = (
            self.qkv(x_bnc).reshape(batch_size, token_count, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        )

        query_bhnc: Float[Tensor, "b heads n head_dim"]
        key_bhnc: Float[Tensor, "b heads n head_dim"]
        value_bhnc: Float[Tensor, "b heads n head_dim"]
        if self.use_sdpa:
            query_bhnc, key_bhnc, value_bhnc = qkv_3bhnc.unbind(0)
            attended_bhnc: Float[Tensor, "b heads n head_dim"] = F.scaled_dot_product_attention(
                query_bhnc,
                key_bhnc,
                value_bhnc,
                None,
            )
            x_bnc = attended_bhnc.permute(0, 2, 1, 3).reshape(batch_size, token_count, channels)
        else:
            query_bhnc = qkv_3bhnc[0] * self.scale
            key_bhnc = qkv_3bhnc[1]
            value_bhnc = qkv_3bhnc[2]
            attention_bhnn: Float[Tensor, "b heads n query_n"] = query_bhnc @ key_bhnc.transpose(-2, -1)
            attention_bhnn = attention_bhnn.softmax(dim=-1)
            attention_bhnn = self.attn_drop(attention_bhnn)
            x_bnc = (attention_bhnn @ value_bhnc).transpose(1, 2).reshape(batch_size, token_count, channels)

        x_bnc = self.proj(x_bnc)
        x_bnc = self.proj_drop(x_bnc)
        return x_bnc


class Mlp(nn.Module):
    """Two-layer transformer feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        resolved_out_features: int = out_features or in_features
        resolved_hidden_features: int = hidden_features or in_features
        self.fc1: nn.Linear = nn.Linear(in_features, resolved_hidden_features, bias=bias)
        self.act: nn.Module = act_layer()
        self.fc2: nn.Linear = nn.Linear(resolved_hidden_features, resolved_out_features, bias=bias)
        self.drop: nn.Dropout = nn.Dropout(drop)

    def forward(self, x_bnc: Float[Tensor, "b n c"]) -> Float[Tensor, "b n out_c"]:
        """Apply the feed-forward network.

        Args:
            x_bnc: Float token tensor shaped ``b n c``.

        Returns:
            Float token tensor shaped ``b n out_c``.
        """
        x_bnc = self.fc1(x_bnc)
        x_bnc = self.act(x_bnc)
        x_bnc = self.drop(x_bnc)
        x_bnc = self.fc2(x_bnc)
        x_bnc = self.drop(x_bnc)
        return x_bnc


class LayerScale(nn.Module):
    """Apply a learned per-channel residual scale."""

    def __init__(self, dim: int, init_values: float | Float[Tensor, ""] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace: bool = inplace
        self.gamma: nn.Parameter = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x_bnc: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        """Scale a float token tensor shaped ``b n c`` channel-wise."""
        return x_bnc.mul_(self.gamma) if self.inplace else x_bnc * self.gamma


def _make_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        assert len(value) == 2
        return value
    return (value, value)


class PatchEmbed(nn.Module):
    """Convert a BCHW image to a sequence of patch embeddings."""

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        image_hw: tuple[int, int] = _make_2tuple(img_size)
        patch_hw: tuple[int, int] = _make_2tuple(patch_size)
        patch_grid_size: tuple[int, int] = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])

        self.img_size: tuple[int, int] = image_hw
        self.patch_size: tuple[int, int] = patch_hw
        self.num_patches: int = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans: int = in_chans
        self.embed_dim: int = embed_dim
        self.proj: nn.Conv2d = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm: nn.Module = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, image_bchw: Float[Tensor, "b c h w"]) -> Float[Tensor, "b n embed"]:
        """Embed non-overlapping image patches.

        Args:
            image_bchw: Float image tensor shaped ``b c h w``.

        Returns:
            Float patch-token tensor shaped ``b n embed``.
        """
        height: int = image_bchw.shape[-2]
        width: int = image_bchw.shape[-1]
        patch_height: int = self.patch_size[0]
        patch_width: int = self.patch_size[1]
        assert height % patch_height == 0, f"Input image height {height} is not a multiple of patch height {patch_height}"
        assert width % patch_width == 0, f"Input image width {width} is not a multiple of patch width: {patch_width}"

        tokens_bchw: Float[Tensor, "b embed patch_h patch_w"] = self.proj(image_bchw)
        tokens_bne: Float[Tensor, "b n embed"] = tokens_bchw.flatten(2).transpose(1, 2)
        tokens_bne = self.norm(tokens_bne)
        return tokens_bne


class Block(nn.Module):
    """Inference-only transformer block for plain tensor inputs."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        *,
        use_sdpa: bool,
    ) -> None:
        super().__init__()
        if drop_path != 0.0:
            raise ValueError("The inference-only DINOv2 block requires drop_path=0.0")

        self.norm1: nn.Module = norm_layer(dim)
        self.attn: Attention = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_sdpa=use_sdpa,
        )
        self.ls1: nn.Module = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2: nn.Module = norm_layer(dim)
        mlp_hidden_dim: int = int(dim * mlp_ratio)
        self.mlp: Mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2: nn.Module = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x_bnc: Float[Tensor, "b n c"]) -> Float[Tensor, "b n c"]:
        """Apply attention and feed-forward residual updates.

        Args:
            x_bnc: Float token tensor shaped ``b n c``.

        Returns:
            Float token tensor shaped ``b n c``.
        """
        x_bnc = x_bnc + self.ls1(self.attn(self.norm1(x_bnc)))
        x_bnc = x_bnc + self.ls2(self.mlp(self.norm2(x_bnc)))
        return x_bnc
