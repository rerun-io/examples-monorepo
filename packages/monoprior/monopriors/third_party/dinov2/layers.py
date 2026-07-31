# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0.

from collections.abc import Callable

import torch
import torch.nn.functional as F
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

    def forward(self, x: Tensor) -> Tensor:
        batch_size: int
        token_count: int
        channels: int
        batch_size, token_count, channels = x.shape
        qkv: Tensor = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)

        if self.use_sdpa:
            query: Tensor
            key: Tensor
            value: Tensor
            query, key, value = qkv.unbind(0)
            x = F.scaled_dot_product_attention(query, key, value, None)
            x = x.permute(0, 2, 1, 3).reshape(batch_size, token_count, channels)
        else:
            query = qkv[0] * self.scale
            key = qkv[1]
            value = qkv[2]
            attention: Tensor = query @ key.transpose(-2, -1)
            attention = attention.softmax(dim=-1)
            attention = self.attn_drop(attention)
            x = (attention @ value).transpose(1, 2).reshape(batch_size, token_count, channels)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
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
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1: nn.Linear = nn.Linear(in_features, hidden_features, bias=bias)
        self.act: nn.Module = act_layer()
        self.fc2: nn.Linear = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop: nn.Dropout = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float | Tensor = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace: bool = inplace
        self.gamma: nn.Parameter = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


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
        self.patches_resolution: tuple[int, int] = patch_grid_size
        self.num_patches: int = patch_grid_size[0] * patch_grid_size[1]
        self.in_chans: int = in_chans
        self.embed_dim: int = embed_dim
        self.proj: nn.Conv2d = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm: nn.Module = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        height: int = x.shape[-2]
        width: int = x.shape[-1]
        patch_height: int = self.patch_size[0]
        patch_width: int = self.patch_size[1]
        assert height % patch_height == 0, f"Input image height {height} is not a multiple of patch height {patch_height}"
        assert width % patch_width == 0, f"Input image width {width} is not a multiple of patch width: {patch_width}"

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


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
        self.drop_path1: nn.Identity = nn.Identity()

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
        self.drop_path2: nn.Identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
