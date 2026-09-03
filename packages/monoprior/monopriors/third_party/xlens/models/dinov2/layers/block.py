# flake8: noqa: F821
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

import logging
from collections.abc import Callable
from typing import TypeAlias

from jaxtyping import Bool, Float
from torch import Tensor, nn

from monopriors.third_party.xlens.models.dinov2.layers.attention import Attention
from monopriors.third_party.xlens.models.dinov2.layers.drop_path import DropPath
from monopriors.third_party.xlens.models.dinov2.layers.layer_scale import LayerScale
from monopriors.third_party.xlens.models.dinov2.layers.mlp import Mlp

logger = logging.getLogger("dinov2")
DwcInfo: TypeAlias = tuple[int, int, int, int, int]


class Block(nn.Module):
    """Inference transformer block with attention and feed-forward residuals."""

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
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope: nn.Module | None = None,
        ln_eps: float = 1e-6,
        use_dwc: bool = False,  # DWC bypass toggle (forwarded to Attention)
        dwc_kernel_size: int = 3,
        n_cam_types: int = 0,  # number of camera types (forwarded to Attention)
    ) -> None:
        """Build one transformer block while preserving checkpoint module names."""
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim, eps=ln_eps)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
            use_dwc=use_dwc,
            dwc_kernel_size=dwc_kernel_size,
            n_cam_types=n_cam_types,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, eps=ln_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "batch tokens features"],
        pos: Float[Tensor, "batch tokens 2"] | None = None,
        attn_mask: Float[Tensor, "batch ... tokens tokens"] | Bool[Tensor, "batch ... tokens tokens"] | None = None,
        dwc_info: DwcInfo | None = None,
    ) -> Float[Tensor, "batch tokens features"]:
        """Apply one inference transformer block.

        Args:
            x: Token features.
            pos: Optional two-dimensional rotary positions.
            attn_mask: Optional additive or Boolean attention mask.
            dwc_info: Patch layout for the optional depthwise bypass.

        Returns:
            Updated token features.
        """
        attention_residual: Float[Tensor, "batch tokens features"] = self.ls1(
            self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask, dwc_info=dwc_info)
        )
        x = x + self.drop_path1(attention_residual)
        feed_forward_residual: Float[Tensor, "batch tokens features"] = self.ls2(self.mlp(self.norm2(x)))
        return x + self.drop_path2(feed_forward_residual)
