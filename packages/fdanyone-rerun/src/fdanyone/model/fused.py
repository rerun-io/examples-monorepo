"""Fused Triton kernels for the DiT's memory-bound glue (inference only, CUDA).

Each kernel does one row per program and keeps the rounding points of the eager code it replaces: the same BF16
casts in the same places, FP32 arithmetic in between, IEEE division where eager divides. They differ from eager
only by FP32 reduction order and contraction (a few FP32 ulps before the BF16 rounding).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from jaxtyping import BFloat16, Float32, Shaped
from torch import Tensor
from triton.language.extra import libdevice


@triton.jit
def _bf16(x):
    """Round FP32 to BF16 and back (RTNE). Written as PTX: LLVM folds a ``.to(bf16).to(fp32)`` pair that feeds
    further FP32 arithmetic, silently dropping the rounding that eager BF16 code performs."""

    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _quantize_rows_kernel(x_ptr, q_ptr, scale_ptr, n_cols, x_stride, q_stride, GELU: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
    if GELU:
        inner = 0.7978845608028654 * (x + 0.044715 * (x * x * x))
        x = _bf16(0.5 * x * (1.0 + libdevice.tanh(inner)))
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 7.888609052210118e-31)  # 2**-100
    inverse_scale = _bf16(tl.math.div_rn(448.0, amax))
    quantized = _bf16(x * inverse_scale).to(tl.float8e4nv)
    tl.store(q_ptr + row * q_stride + cols, quantized, mask=mask)
    tl.store(scale_ptr + row, tl.math.div_rn(1.0, inverse_scale))


def quantize_rows_fp8(
    rows: BFloat16[Tensor, "m k"], *, gelu: bool = False
) -> tuple[Shaped[Tensor, "m k"], Float32[Tensor, "m 1"]]:
    """Row-wise FP8 quantization as ``Fp8RowwiseLinear`` does it eagerly (optionally of tanh-GELU(rows) first).

    Returns the e4m3fn rows and the FP32 per-row dequantization scale (``scale_a`` of ``torch._scaled_mm``).
    """

    m, k = rows.shape
    if rows.stride(1) != 1:
        raise ValueError("Row quantization needs unit column stride.")
    quantized = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=rows.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=rows.device)
    block = triton.next_power_of_2(k)
    _quantize_rows_kernel[(m,)](
        rows, quantized, scale, k, rows.stride(0), quantized.stride(0), GELU=gelu, BLOCK=block, num_warps=8 if block <= 4096 else 16
    )
    return quantized, scale


@triton.jit
def _rms_norm_rope_kernel(
    x_ptr, weight_ptr, cos_ptr, sin_ptr, seq_len, eps, N_COLS: tl.constexpr, HALF_HEAD: tl.constexpr, ROPE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    pairs = tl.arange(0, BLOCK)
    mask = pairs < N_COLS // 2
    base = x_ptr + row * N_COLS
    even = tl.load(base + 2 * pairs, mask=mask, other=0.0).to(tl.float32)
    odd = tl.load(base + 2 * pairs + 1, mask=mask, other=0.0).to(tl.float32)
    rstd = libdevice.rsqrt((tl.sum(even * even, axis=0) + tl.sum(odd * odd, axis=0)) / N_COLS + eps)
    w_even = tl.load(weight_ptr + 2 * pairs, mask=mask, other=0.0).to(tl.float32)
    w_odd = tl.load(weight_ptr + 2 * pairs + 1, mask=mask, other=0.0).to(tl.float32)
    even = _bf16(_bf16(even * rstd) * w_even)
    odd = _bf16(_bf16(odd * rstd) * w_odd)
    if ROPE:
        token = row % seq_len
        freq = token * HALF_HEAD + pairs % HALF_HEAD
        cos = tl.load(cos_ptr + freq, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + freq, mask=mask, other=0.0)
        even, odd = even * cos - odd * sin, even * sin + odd * cos
    tl.store(base + 2 * pairs, even.to(tl.bfloat16), mask=mask)
    tl.store(base + 2 * pairs + 1, odd.to(tl.bfloat16), mask=mask)


def rms_norm_rope_(
    x: BFloat16[Tensor, "b s c"],
    weight: BFloat16[Tensor, " c"],
    eps: float,
    rope: tuple[Float32[Tensor, "s half"], Float32[Tensor, "s half"]] | None,
) -> BFloat16[Tensor, "b s c"]:
    """In place: ``RMSNorm`` (FP32 statistics, BF16 result times the BF16 weight), then the FP32 rotary rotation.

    ``rope`` holds the per-token cos and sin of each head's channel pair (the FP32 casts of the complex table).
    """

    if not x.is_contiguous():
        raise ValueError("Fused RMSNorm needs a contiguous projection buffer.")
    b, s, c = x.shape
    half = 1 if rope is None else rope[0].shape[1]
    cos, sin = (weight, weight) if rope is None else rope
    block = triton.next_power_of_2(c // 2)
    _rms_norm_rope_kernel[(b * s,)](x, weight, cos, sin, s, eps, N_COLS=c, HALF_HEAD=half, ROPE=rope is not None, BLOCK=block, num_warps=8)
    return x


def rope_tables(freqs: Tensor) -> tuple[Float32[Tensor, "s half"], Float32[Tensor, "s half"]]:
    """FP32 cos and sin of a complex ``[s, 1, half]`` rotary table, as ``_rotate_rope_chunk`` casts them."""

    flat = freqs.reshape(freqs.shape[0], -1)
    return flat.real.to(torch.float32).contiguous(), flat.imag.to(torch.float32).contiguous()


@triton.jit
def _frame_major_row(row, rows_per_view, views, frames, FRAME_MAJOR: tl.constexpr):
    """Row of token ``row`` of a view-major ``[v, (f hw)]`` stream in its frame-major ``[f, (v hw)]`` copy."""

    if not FRAME_MAJOR:
        return row
    hw = rows_per_view // frames
    within = row % rows_per_view
    return (within // hw) * (views * hw) + (row // rows_per_view) * hw + within % hw


@triton.jit
def _layer_norm_modulation_kernel(
    x_ptr, shift_ptr, scale_ptr, out_ptr, rows_per_view, views, frames, eps,
    N_COLS: tl.constexpr, AFFINE: tl.constexpr, FRAME_MAJOR: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    view = 0 if AFFINE else row // rows_per_view
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    x = tl.load(x_ptr + row * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N_COLS
    centered = tl.where(mask, x - mean, 0.0)
    rstd = libdevice.rsqrt(tl.sum(centered * centered, axis=0) / N_COLS + eps)
    scale = tl.load(scale_ptr + view * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + view * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    if not AFFINE:
        scale = _bf16(1.0 + scale)
    out_row = _frame_major_row(row, rows_per_view, views, frames, FRAME_MAJOR)
    tl.store(out_ptr + out_row * N_COLS + cols, (centered * rstd * scale + shift).to(tl.bfloat16), mask=mask)


def layer_norm_modulation(
    x: BFloat16[Tensor, "v n c"], shift: BFloat16[Tensor, "v 1 c"], scale: BFloat16[Tensor, "v 1 c"], eps: float, frames: int = 0
) -> BFloat16[Tensor, "..."]:
    """BF16 ``modulate(LayerNorm(x), shift, scale)``: FP32 normalization (autocast's LayerNorm), BF16 ``1 + scale``.

    With ``frames`` the result is written frame-major, ``[f, (v hw), c]`` for the per-frame multiview attention.
    """

    v, n, c = x.shape
    if not x.is_contiguous():
        raise ValueError("Fused LayerNorm modulation needs a contiguous token stream.")
    out = torch.empty_like(x) if not frames else x.new_empty((frames, v * (n // frames), c))
    shift = shift.expand(v, 1, c).contiguous()
    scale = scale.expand(v, 1, c).contiguous()
    _layer_norm_modulation_kernel[(v * n,)](
        x, shift, scale, out, n, v, max(frames, 1), eps, N_COLS=c, AFFINE=False, FRAME_MAJOR=frames > 0, BLOCK=triton.next_power_of_2(c), num_warps=8
    )
    return out


def layer_norm_affine(x: BFloat16[Tensor, "v n c"], weight: Tensor, bias: Tensor, eps: float) -> BFloat16[Tensor, "v n c"]:
    """``LayerNorm(x)`` with its affine in FP32 (autocast), rounded once to BF16 as the following linear casts it."""

    v, n, c = x.shape
    if not x.is_contiguous():
        raise ValueError("Fused LayerNorm needs a contiguous token stream.")
    out = torch.empty_like(x)
    _layer_norm_modulation_kernel[(v * n,)](
        x, bias, weight, out, n, v, 1, eps, N_COLS=c, AFFINE=True, FRAME_MAJOR=False, BLOCK=triton.next_power_of_2(c), num_warps=8
    )
    return out


@triton.jit
def _gated_residual_kernel(
    x_ptr, branch_ptr, gate_ptr, rows_per_view, views, frames, N_COLS: tl.constexpr, FRAME_MAJOR: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    view = row // rows_per_view
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS
    x = tl.load(x_ptr + row * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    branch_row = _frame_major_row(row, rows_per_view, views, frames, FRAME_MAJOR)
    branch = tl.load(branch_ptr + branch_row * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + view * N_COLS + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + row * N_COLS + cols, (x + _bf16(branch * gate)).to(tl.bfloat16), mask=mask)


def gated_residual_(
    x: BFloat16[Tensor, "v n c"], branch: BFloat16[Tensor, "..."], gate: BFloat16[Tensor, "v 1 c"], frames: int = 0
) -> BFloat16[Tensor, "v n c"]:
    """In place ``x += bf16(branch * gate)``, the two BF16 roundings of the eager ``mul_`` then ``add_``.

    With ``frames`` the branch is frame-major, ``[f, (v hw), c]`` (the multiview attention's output).
    """

    v, n, c = x.shape
    if not (x.is_contiguous() and branch.is_contiguous()):
        raise ValueError("Fused gated residual needs contiguous token streams.")
    gate = gate.expand(v, 1, c).contiguous()
    _gated_residual_kernel[(v * n,)](
        x, branch, gate, n, v, max(frames, 1), N_COLS=c, FRAME_MAJOR=frames > 0, BLOCK=triton.next_power_of_2(c), num_warps=8
    )
    return x
