"""Fused rotary position encoding (one Triton kernel per tensor).

``apply_rotary_enc`` upcasts q/k to fp32, views them as complex, multiplies by
the (repeated) complex frequencies, casts back, and the caller copies the
rotated keys back into a slice: five full-size passes over the memory keys
per attention call. Memory-attention cross-attention keys are large (one
1024-token grid per memory frame), so this dominates its elementwise time.

This kernel reads bf16/fp16/fp32 q or k once, rotates each (real, imag) pair
in fp32 with the same products as the complex multiply, and writes the
result in the input dtype (in place for keys). Results match
``apply_rotary_enc`` up to fp32 FMA contraction (at most one ulp before the
cast back).

Off by default; ``ENABLED`` is set by the caller (the mamma tracker config).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # the fused path needs Triton; the eager path does not
    triton = None

ENABLED: bool = False
"""When True, ``RoPEAttention`` uses the fused kernel on CUDA tensors."""

_cos_sin_cache: dict[tuple[int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
"""(data_ptr, shape) -> (freqs_cis kept alive so its address is not reused, cos, sin)."""


if triton is not None:

    @triton.jit
    def _rope_kernel(
        x_ptr,
        out_ptr,
        cos_ptr,
        sin_ptr,
        n_heads,
        n_tokens,
        n_freq_rows,
        stride_xb,
        stride_xh,
        stride_xn,
        stride_ob,
        stride_oh,
        stride_on,
        HALF: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_n = tl.program_id(1)
        b = pid_bh // n_heads
        h = pid_bh % n_heads
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        j = tl.arange(0, HALF)
        mask = n[:, None] < n_tokens
        # Keys longer than the query grid repeat the frequencies (rope_k_repeat).
        pos = n % n_freq_rows
        x_row = x_ptr + b * stride_xb + h * stride_xh + n[:, None] * stride_xn
        xr = tl.load(x_row + 2 * j[None, :], mask=mask).to(tl.float32)
        xi = tl.load(x_row + 2 * j[None, :] + 1, mask=mask).to(tl.float32)
        c = tl.load(cos_ptr + pos[:, None] * HALF + j[None, :], mask=mask)
        s = tl.load(sin_ptr + pos[:, None] * HALF + j[None, :], mask=mask)
        out_r = xr * c - xi * s
        out_i = xr * s + xi * c
        o_row = out_ptr + b * stride_ob + h * stride_oh + n[:, None] * stride_on
        tl.store(o_row + 2 * j[None, :], out_r.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(o_row + 2 * j[None, :] + 1, out_i.to(out_ptr.dtype.element_ty), mask=mask)


def _cos_sin(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous fp32 cos/sin tables of a complex (rows, half) frequency tensor."""
    key = (freqs_cis.data_ptr(), tuple(freqs_cis.shape))
    cached = _cos_sin_cache.get(key)
    if cached is None:
        cached = (freqs_cis, freqs_cis.real.contiguous().float(), freqs_cis.imag.contiguous().float())
        _cos_sin_cache[key] = cached
    return cached[1], cached[2]


def rope_(x: torch.Tensor, freqs_cis: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Rotate ``x`` (B, H, N, D) by ``freqs_cis`` (R, D/2); token n uses row n % R.

    Writes into ``out`` (``x`` itself for in-place) or a new tensor.
    """
    if out is None:
        out = torch.empty_like(x)
    b, h, n, d = x.shape
    if n == 0:
        return out
    assert x.stride(-1) == 1 and out.stride(-1) == 1 and d % 2 == 0
    cos, sin = _cos_sin(freqs_cis)
    assert cos.shape[1] * 2 == d
    block_n = 32
    grid = (b * h, triton.cdiv(n, block_n))
    _rope_kernel[grid](
        x,
        out,
        cos,
        sin,
        h,
        n,
        cos.shape[0],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HALF=d // 2,
        BLOCK_N=block_n,
    )
    return out
