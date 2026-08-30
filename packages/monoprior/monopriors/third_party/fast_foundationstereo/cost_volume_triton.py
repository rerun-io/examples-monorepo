"""Triton groupwise-correlation cost-volume construction for Fast-FoundationStereo."""

import functools
from typing import Any

import torch
import triton
import triton.language as tl
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


@functools.cache
def _create_gwc_triton_kernel() -> Any:
    """Create the cached Triton autotuner outside module scope.

    Returns:
        Triton autotuner for groupwise correlation.
    """

    def gwc_triton_kernel(
        ref_ptr,
        tar_ptr,
        ref_norm_ptr,
        tar_norm_ptr,
        out_ptr,
        BH,
        C,
        W,
        D,
        G,
        K,
        stride_rn,
        stride_rw,
        stride_rc,
        stride_tn,
        stride_tw,
        stride_tc,
        stride_nn,
        stride_ng,
        stride_nw,
        stride_on,
        stride_og,
        stride_od,
        stride_ow,
        NORMALIZE,
        BLOCK_C,
        BLOCK_W,
        BLOCK_D,
    ):
        program_group = tl.program_id(0)
        disparity_block = tl.program_id(1)
        width_block = tl.program_id(2)
        batch_height = program_group // G
        group = program_group % G
        width_offsets = width_block * BLOCK_W + tl.arange(0, BLOCK_W)
        disparity_offsets = disparity_block * BLOCK_D + tl.arange(0, BLOCK_D)
        width_mask = width_offsets < W
        source_width = width_offsets[None, :] - disparity_offsets[:, None]
        target_mask = (source_width >= 0) & width_mask[None, :]
        accumulator = tl.zeros((BLOCK_D, BLOCK_W), dtype=tl.float32)
        for channel_block in tl.static_range(0, K, BLOCK_C):
            channel_offsets = channel_block + tl.arange(0, BLOCK_C)
            channel_mask = channel_offsets < K
            channel_indices = group * K + channel_offsets
            ref_pointers = ref_ptr + batch_height * stride_rn + width_offsets[None, :] * stride_rw + channel_indices[:, None] * stride_rc
            ref_values = tl.load(ref_pointers, mask=channel_mask[:, None] & width_mask[None, :], other=0.0).to(tl.float32)
            target_pointers = (
                tar_ptr
                + batch_height * stride_tn
                + source_width[None, :, :] * stride_tw
                + channel_indices[:, None, None] * stride_tc
            )
            target_values = tl.load(
                target_pointers,
                mask=channel_mask[:, None, None] & target_mask[None, :, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(target_values * ref_values[:, None, :], axis=0)
        if NORMALIZE:
            norm_offset = batch_height * stride_nn + group * stride_ng
            ref_norm = tl.load(ref_norm_ptr + norm_offset + width_offsets * stride_nw, mask=width_mask, other=1.0).to(tl.float32)
            target_norm = tl.load(tar_norm_ptr + norm_offset + source_width * stride_nw, mask=target_mask, other=1.0).to(tl.float32)
            denominator = ref_norm[None, :] * target_norm + 1e-5
            accumulator = accumulator / denominator
        output_pointers = (
            out_ptr
            + batch_height * stride_on
            + group * stride_og
            + disparity_offsets[:, None] * stride_od
            + width_offsets[None, :] * stride_ow
        )
        tl.store(output_pointers, accumulator, mask=width_mask[None, :])

    gwc_triton_kernel.__annotations__ = {
        "D": tl.constexpr,
        "G": tl.constexpr,
        "K": tl.constexpr,
        "NORMALIZE": tl.constexpr,
        "BLOCK_C": tl.constexpr,
        "BLOCK_W": tl.constexpr,
        "BLOCK_D": tl.constexpr,
    }
    jit_kernel: Any = triton.jit(gwc_triton_kernel)
    autotuned_kernel: Any = triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 4, "BLOCK_W": 128, "BLOCK_D": 8}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_C": 8, "BLOCK_W": 128, "BLOCK_D": 8}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_C": 16, "BLOCK_W": 128, "BLOCK_D": 8}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_C": 64, "BLOCK_W": 128, "BLOCK_D": 8}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_C": 128, "BLOCK_W": 64, "BLOCK_D": 8}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_C": 128, "BLOCK_W": 128, "BLOCK_D": 8}, num_warps=8, num_stages=2),
        ],
        key=["C", "W", "D", "G", "K", "NORMALIZE"],
    )(jit_kernel)
    return autotuned_kernel


@torch.no_grad()
def build_gwc_volume_triton(
    refimg_fea_bchw: Float[Tensor, "b channels h w"],
    targetimg_fea_bchw: Float[Tensor, "b channels h w"],
    maxdisp: int,
    num_groups: int,
    normalize: bool = True,
) -> Float[Tensor, "b groups disparities h w"]:
    """Build the groupwise-correlation volume with the Triton kernel.

    Args:
        refimg_fea_bchw: Left features with shape ``(batch, channels, height, width)``.
        targetimg_fea_bchw: Right features with shape ``(batch, channels, height, width)``.
        maxdisp: Number of feature-space disparity candidates.
        num_groups: Number of channel groups.
        normalize: Whether to normalize each channel group before correlation.

    Returns:
        Correlation volume with shape ``(batch, groups, disparities, height, width)``.
    """
    batch_size: int = refimg_fea_bchw.shape[0]
    channels: int = refimg_fea_bchw.shape[1]
    height: int = refimg_fea_bchw.shape[2]
    width: int = refimg_fea_bchw.shape[3]
    if maxdisp <= 0 or channels % num_groups != 0:
        raise ValueError("maxdisp must be positive and channels must be divisible by num_groups")
    channels_per_group: int = channels // num_groups
    input_dtype: torch.dtype = refimg_fea_bchw.dtype if refimg_fea_bchw.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    if normalize:
        ref_grouped_bgkhw: Float[Tensor, "b groups channels_per_group h w"] = refimg_fea_bchw.float().view(
            batch_size,
            num_groups,
            channels_per_group,
            height,
            width,
        )
        target_grouped_bgkhw: Float[Tensor, "b groups channels_per_group h w"] = targetimg_fea_bchw.float().view(
            batch_size,
            num_groups,
            channels_per_group,
            height,
            width,
        )
        ref_norm_bghw: Float[Tensor, "b groups h w"] = ref_grouped_bgkhw.norm(dim=2)
        target_norm_bghw: Float[Tensor, "b groups h w"] = target_grouped_bgkhw.norm(dim=2)
        ref_norm_ngw: Float[Tensor, "batch_height groups w"] = rearrange(ref_norm_bghw, "b groups h w -> (b h) groups w").to(input_dtype).contiguous()
        target_norm_ngw: Float[Tensor, "batch_height groups w"] = (
            rearrange(target_norm_bghw, "b groups h w -> (b h) groups w").to(input_dtype).contiguous()
        )
    else:
        ref_norm_ngw = refimg_fea_bchw.new_empty((1, 1, 1), dtype=input_dtype)
        target_norm_ngw = refimg_fea_bchw.new_empty((1, 1, 1), dtype=input_dtype)
    ref_bhwc: Float[Tensor, "batch_height w channels"] = rearrange(
        refimg_fea_bchw.to(input_dtype),
        "b channels h w -> (b h) w channels",
    ).contiguous()
    target_bhwc: Float[Tensor, "batch_height w channels"] = rearrange(
        targetimg_fea_bchw.to(input_dtype),
        "b channels h w -> (b h) w channels",
    ).contiguous()
    output_ngdw: Float[Tensor, "batch_height groups disparities w"] = torch.empty(
        (batch_size * height, num_groups, maxdisp, width),
        device=ref_bhwc.device,
        dtype=input_dtype,
    )
    batch_height: int = batch_size * height
    effective_disparities: int = min(maxdisp, width)

    def grid(meta: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (
            batch_height * num_groups,
            triton.cdiv(effective_disparities, meta["BLOCK_D"]),
            triton.cdiv(width, meta["BLOCK_W"]),
        )

    kernel: Any = _create_gwc_triton_kernel()
    kernel[grid](
        ref_bhwc,
        target_bhwc,
        ref_norm_ngw,
        target_norm_ngw,
        output_ngdw,
        batch_height,
        channels,
        width,
        effective_disparities,
        num_groups,
        channels_per_group,
        ref_bhwc.stride(0),
        ref_bhwc.stride(1),
        ref_bhwc.stride(2),
        target_bhwc.stride(0),
        target_bhwc.stride(1),
        target_bhwc.stride(2),
        ref_norm_ngw.stride(0),
        ref_norm_ngw.stride(1),
        ref_norm_ngw.stride(2),
        output_ngdw.stride(0),
        output_ngdw.stride(1),
        output_ngdw.stride(2),
        output_ngdw.stride(3),
        NORMALIZE=normalize,
    )
    if effective_disparities < maxdisp:
        output_ngdw[:, :, effective_disparities:, :] = 0
    volume_bgdhw: Float[Tensor, "b groups disparities h w"] = rearrange(
        output_ngdw,
        "(b h) groups disparities w -> b groups disparities h w",
        b=batch_size,
        h=height,
    ).contiguous()
    return volume_bgdhw
