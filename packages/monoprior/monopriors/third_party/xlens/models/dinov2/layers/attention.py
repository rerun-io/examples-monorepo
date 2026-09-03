# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
from collections.abc import Callable
from typing import TypeAlias

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

logger = logging.getLogger("dinov2")
DwcInfo: TypeAlias = tuple[int, int, int, int, int]


class Attention(nn.Module):
    """Multi-head attention with rotary positions and an optional DWC bypass."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        qk_norm: bool = False,
        rope: nn.Module | None = None,
        use_dwc: bool = False,  # DWC bypass (Agent-Attention style local inductive bias)
        dwc_kernel_size: int = 3,
        n_cam_types: int = 0,  # kept for signature compatibility; cam_type_embed /
        # calib_tokens are handled in vision_transformer
    ) -> None:
        """Build one inference attention layer."""
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        # DWC bypass: depthwise conv adds a local inductive bias to attention.
        # out = attn_out + DWC(V) (additive, not concat). Applies only to patch
        # tokens; CLS / register / scale positions contribute zero.
        self.use_dwc = use_dwc
        self.dwc: nn.Conv2d | None = None
        if use_dwc:
            self.dwc = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=dwc_kernel_size,
                padding=dwc_kernel_size // 2,
                groups=dim,
            )

        # n_cam_types retained only for the backbone's cam_type_embed.
        self.n_cam_types = n_cam_types

    def forward(
        self,
        x: Float[Tensor, "batch tokens features"],
        pos: Float[Tensor, "batch tokens 2"] | None = None,
        attn_mask: Float[Tensor, "batch ... tokens tokens"] | Bool[Tensor, "batch ... tokens tokens"] | None = None,
        dwc_info: DwcInfo | None = None,
    ) -> Float[Tensor, "batch tokens features"]:
        """Apply scaled dot-product attention and the optional DWC bypass."""
        B, N, C = x.shape
        qkv: Float[Tensor, "3 batch heads tokens head_features"] = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q: Float[Tensor, "batch heads tokens head_features"]
        k: Float[Tensor, "batch heads tokens head_features"]
        v: Float[Tensor, "batch heads tokens head_features"]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Attention bias comes only from an external attn_mask
        # (e.g. calib_mask + distortion_bias).
        effective_bias: Float[Tensor, "batch heads tokens tokens"] | Bool[Tensor, "batch heads tokens tokens"] | None = None
        if attn_mask is not None:
            # Accept (B, N, N) [broadcast over heads] or (B, H, N, N) [per-head,
            # used by DistortionBias].
            if attn_mask.dim() == 3:
                mask_expanded: Float[Tensor, "batch heads tokens tokens"] | Bool[Tensor, "batch heads tokens tokens"] = attn_mask[:, None].expand(
                    -1, self.num_heads, -1, -1
                )
            elif attn_mask.dim() == 4:
                # Already a per-head additive bias.
                mask_expanded = attn_mask
                assert mask_expanded.shape[1] in (1, self.num_heads), (
                    f"attn_mask head dim {mask_expanded.shape[1]} does not match num_heads {self.num_heads}"
                )
                if mask_expanded.shape[1] == 1:
                    mask_expanded = mask_expanded.expand(-1, self.num_heads, -1, -1)
            else:
                raise ValueError(f"attn_mask must be 3-D or 4-D, got {attn_mask.dim()}")

            if effective_bias is None:
                # Pass attn_mask through (may be bool).
                effective_bias = mask_expanded
            else:
                # Add attn_mask onto the existing bias (bool -> additive -inf).
                if mask_expanded.dtype == torch.bool:
                    # bool mask: True=keep, False=mask out -> -inf for masked
                    additive = torch.zeros_like(effective_bias)
                    additive.masked_fill_(~mask_expanded, float("-inf"))
                    effective_bias = effective_bias + additive
                else:
                    effective_bias = effective_bias + mask_expanded.to(effective_bias.dtype)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, attn_mask=effective_bias)

        x = x.transpose(1, 2).reshape(B, N, C)

        # DWC bypass: x = attn_out + DWC(V), before proj.
        if self.dwc is not None and dwc_info is not None:
            x = x + self._dwc_bypass(v, dwc_info)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _dwc_bypass(
        self,
        v: Float[Tensor, "batch heads tokens head_features"],
        dwc_info: DwcInfo,
    ) -> Float[Tensor, "batch tokens features"]:
        """
        Depthwise conv over V. Applies only to patch tokens; other positions
        (CLS/register/scale) contribute 0.

        Args:
            v: (B', num_heads, N, head_dim) — attention V.
            dwc_info: tuple (H_p, W_p, prefix, suffix, S_per_batch)
              H_p, W_p:       per-view patch grid size.
              prefix:         special tokens at the start of each view (CLS + register).
              suffix:         special tokens at the end of each view (scale token, 1 once injected).
              S_per_batch:    views packed into the N dimension (local=1, global=S).

        Returns:
            (B', N, C) — same shape as the attention output; 0 at non-patch positions.
        """
        H_p, W_p, prefix, suffix, S_per_batch = dwc_info
        B_prime, num_heads, N, head_dim = v.shape
        C = num_heads * head_dim
        per_view_N = N // S_per_batch
        per_view_patches = H_p * W_p
        assert per_view_N == prefix + per_view_patches + suffix, (
            f"DWC: per_view_N={per_view_N} != prefix({prefix})+patches({per_view_patches})+suffix({suffix})"
        )

        # Merge heads: (B', H, N, D) -> (B', N, C)
        v_merged: Float[Tensor, "batch tokens features"] = v.transpose(1, 2).reshape(B_prime, N, C)
        v_split: Float[Tensor, "batch views per_view_tokens features"] = v_merged.reshape(B_prime, S_per_batch, per_view_N, C)
        # Take patches: (B', S, H_p*W_p, C)
        v_patches: Float[Tensor, "batch views patches features"] = v_split[:, :, prefix : prefix + per_view_patches, :].contiguous()
        # To 2D: (B'*S, C, H_p, W_p)
        v_2d: Float[Tensor, "packed_views features patch_height patch_width"] = (
            v_patches.reshape(B_prime * S_per_batch, H_p, W_p, C).permute(0, 3, 1, 2).contiguous()
        )
        # DWC (cast to conv weight dtype, e.g. bf16 input vs float32 weight).
        if self.dwc is None:
            raise RuntimeError("DWC bypass requested without a depthwise convolution")
        orig_dtype = v_2d.dtype
        if v_2d.dtype != self.dwc.weight.dtype:
            v_2d = v_2d.to(self.dwc.weight.dtype)
        v_dwc: Float[Tensor, "packed_views features patch_height patch_width"] = self.dwc(v_2d).to(orig_dtype)
        # Back to sequence: (B', S, H_p*W_p, C)
        v_dwc = v_dwc.permute(0, 2, 3, 1).reshape(B_prime, S_per_batch, per_view_patches, C)
        # Scatter back into full N; non-patch positions stay 0.
        out: Float[Tensor, "batch views per_view_tokens features"] = torch.zeros(
            B_prime, S_per_batch, per_view_N, C, dtype=v_dwc.dtype, device=v_dwc.device
        )
        out[:, :, prefix : prefix + per_view_patches, :] = v_dwc
        return out.reshape(B_prime, N, C)
