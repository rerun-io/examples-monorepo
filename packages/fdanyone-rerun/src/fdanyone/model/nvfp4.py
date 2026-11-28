"""NVFP4 W4A4 linears for the DiT: cuBLASLt block-scaled FP4 GEMM through ``torch.nn.functional.scaled_mm``.

NVFP4 (NVIDIA convention, as ``trtkit.nvfp4``): E2M1 values, one E4M3 decode scale per 16 consecutive inputs and an
FP32 per-tensor decode scale. With ``g = 448 * 6 / amax``: ``sb = e4m3(amax(block) / 6 * g)``,
``q = e2m1(x * g / sb)``, ``x ~= q * sb / g``. Two codes share a byte, element ``2i`` in the low nibble. cuBLASLt reads
the block scales in its 128x4 tiled layout (``to_blocked``), which the activation kernel writes directly.

Weights are quantized once (per-tensor global scale, per-block scale picked by dequantized error from a few
candidates); activations are quantized per call by one Triton kernel after one max reduction (a dynamic per-tensor
global scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.nn.functional import ScalingType, SwizzleType
from triton.language.extra import libdevice

BLOCK: int = 16
"""Inputs per E4M3 block scale."""
FP4_MAX: float = 6.0
FP8_MAX: float = 448.0
_RANGE: float = FP4_MAX * FP8_MAX
_TINY_AMAX: float = 1e-30
_E2M1_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
WEIGHT_SCALE_CANDIDATES: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
"""Block scale multipliers tried per weight block (``c = 1.5`` maps the block maximum to 4); the one with the least
dequantized squared error wins (trtkit's Four-Over-Six style search)."""
_SPAN: int = 256
"""Columns per program of the activation kernel; ``in_features`` must be a multiple of it."""


@dataclass(frozen=True, slots=True)
class Nvfp4Tensor:
    """One NVFP4 GEMM operand ``[rows, cols]``."""

    packed: torch.Tensor
    """``float4_e2m1fn_x2`` ``[rows, cols / 2]``."""
    block_scale: torch.Tensor
    """``float8_e4m3fn`` block decode scales in the 128x4 tiled layout, flat (rows padded to 128, blocks to 4)."""
    decode_scale: torch.Tensor
    """FP32 scalar per-tensor decode scale ``1 / g``."""


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """``[rows, blocks]`` scales into cuBLASLt's tiled block-scale layout (zero padded), flat."""

    rows, cols = scales.shape
    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros((row_tiles * 128, col_tiles * 4), device=scales.device, dtype=scales.dtype)
    padded[:rows, :cols] = scales
    return padded.view(row_tiles, 128, col_tiles, 4).permute(0, 2, 1, 3).reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def from_blocked(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of :func:`to_blocked`."""

    row_tiles, col_tiles = -(-rows // 128), -(-cols // 4)
    tiles = flat.view(row_tiles, col_tiles, 32, 4, 4).transpose(2, 3).reshape(row_tiles, col_tiles, 128, 4)
    return tiles.permute(0, 2, 1, 3).reshape(row_tiles * 128, col_tiles * 4)[:rows, :cols]


def round_to_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """FP32 values already scaled into [-6, 6] -> uint8 codes 0..15 (bit 3 = sign), round to nearest even."""

    magnitude = values.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for threshold, value, inclusive in ((0.25, 1, False), (0.75, 2, True), (1.25, 3, False), (1.75, 4, True), (2.5, 5, False), (3.5, 6, True), (5.0, 7, False)):
        code = torch.where(magnitude >= threshold if inclusive else magnitude > threshold, value, code).to(torch.uint8)
    return torch.where(values < 0, code | 8, code).to(torch.uint8)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)


def dequantize(operand: Nvfp4Tensor, rows: int, cols: int) -> torch.Tensor:
    """FP32 ``[rows, cols]`` values of an operand (tests and error studies)."""

    raw = operand.packed.view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).view(rows, cols)
    grid = torch.tensor(_E2M1_GRID, device=raw.device)
    magnitude = grid[(codes & 7).long()]
    values = torch.where((codes & 8) > 0, -magnitude, magnitude)
    scales = from_blocked(operand.block_scale, rows, cols // BLOCK).float()
    return (values.view(rows, cols // BLOCK, BLOCK) * scales[..., None] * operand.decode_scale).view(rows, cols)


def quantize_weight(weight: torch.Tensor, candidates: tuple[float, ...] = WEIGHT_SCALE_CANDIDATES) -> Nvfp4Tensor:
    """NVFP4 weight ``[out, in]`` with a per-block scale search; eager, once per layer."""

    rows, cols = weight.shape
    blocks = weight.float().view(rows, cols // BLOCK, BLOCK)
    # Headroom of max(candidates) on the encode scale keeps every candidate block scale inside E4M3.
    encode = _RANGE / max(candidates) / blocks.abs().amax().clamp(min=_TINY_AMAX)
    block_amax = blocks.abs().amax(-1)
    grid = torch.tensor(_E2M1_GRID, device=weight.device)
    best_error = torch.full_like(block_amax, float("inf"))
    best_scale = torch.zeros_like(block_amax)
    best_codes = torch.zeros_like(blocks, dtype=torch.uint8)
    for candidate in candidates:
        scale = (block_amax / FP4_MAX * candidate * encode).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float()
        codes = round_to_e2m1_codes(blocks * torch.where(scale == 0, 0.0, encode / scale)[..., None])
        magnitude = grid[(codes & 7).long()]
        error = ((torch.where((codes & 8) > 0, -magnitude, magnitude) * (scale / encode)[..., None] - blocks) ** 2).sum(-1)
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        best_scale = torch.where(better, scale, best_scale)
        best_codes = torch.where(better[..., None], codes, best_codes)
    return Nvfp4Tensor(_pack(best_codes.view(rows, cols)), to_blocked(best_scale.to(torch.float8_e4m3fn)), (1.0 / encode).reshape(()).float())


def quantize_activation_reference(rows: torch.Tensor) -> Nvfp4Tensor:
    """Eager twin of :func:`quantize_activation` (tests)."""

    values = rows.float()
    encode = _RANGE / values.abs().amax().clamp(min=_TINY_AMAX)
    blocks = values.view(rows.shape[0], -1, BLOCK)
    scale = (blocks.abs().amax(-1) / FP4_MAX * encode).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    codes = round_to_e2m1_codes(blocks * torch.where(scale_f == 0, 0.0, encode / scale_f)[..., None]).view(rows.shape)
    return Nvfp4Tensor(_pack(codes), to_blocked(scale), (1.0 / encode).reshape(()).float())


@triton.jit
def _bf16(x):
    """Round FP32 to BF16 and back (RTNE), as eager BF16 code does between ops."""

    return tl.inline_asm_elementwise("{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _quantize_kernel(x_ptr, q_ptr, s_ptr, amax_ptr, x_stride, half_cols, tile_cols, GELU: tl.constexpr, SPAN: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    span = tl.program_id(1)
    cols = span * SPAN + tl.arange(0, SPAN)
    x = tl.load(x_ptr + row * x_stride + cols).to(tl.float32)
    if GELU:
        inner = 0.7978845608028654 * (x + 0.044715 * (x * x * x))
        x = _bf16(0.5 * x * (1.0 + libdevice.tanh(inner)))
    encode = 2688.0 / tl.maximum(tl.load(amax_ptr), 1e-30)
    blocks = tl.reshape(x, (SPAN // 16, 16))
    scale = tl.minimum(tl.max(tl.abs(blocks), axis=1) / 6.0 * encode, 448.0).to(tl.float8e4nv)
    scale_f = scale.to(tl.float32)
    v = blocks * tl.where(scale_f == 0.0, 0.0, encode / scale_f)[:, None]
    a = tl.abs(v)
    code = tl.where(a > 0.25, 1, 0)
    code = tl.where(a >= 0.75, 2, code)
    code = tl.where(a > 1.25, 3, code)
    code = tl.where(a >= 1.75, 4, code)
    code = tl.where(a > 2.5, 5, code)
    code = tl.where(a >= 3.5, 6, code)
    code = tl.where(a > 5.0, 7, code)
    code = tl.where(v < 0, code | 8, code)
    low, high = tl.split(tl.reshape(code, (SPAN // 2, 2)))
    tl.store(q_ptr + row * half_cols + span * (SPAN // 2) + tl.arange(0, SPAN // 2), (low | (high << 4)).to(tl.uint8))
    block = span * (SPAN // 16) + tl.arange(0, SPAN // 16)
    within = row % 128
    tl.store(s_ptr + ((row // 128) * tile_cols + block // 4) * 512 + (within % 32) * 16 + (within // 32) * 4 + block % 4, scale)


def quantize_activation(rows: torch.Tensor, *, gelu: bool = False) -> Nvfp4Tensor:
    """Dynamic NVFP4 quantization of BF16 rows ``[m, k]`` (``k`` a multiple of 256), optionally of tanh-GELU(rows).

    The per-tensor scale comes from ``amax(rows)``; under GELU that is an upper bound of the GELU output's maximum
    (``|gelu(x)| <= max(|x|, 0.17)``), which only costs block-scale resolution, never range.
    """

    m, k = rows.shape
    if k % _SPAN or rows.stride(1) != 1:
        raise ValueError(f"NVFP4 activation quantization needs unit-stride rows with k % {_SPAN} == 0, got {tuple(rows.shape)}.")
    amax = torch.linalg.vector_norm(rows, ord=float("inf"), dtype=torch.float32)
    if gelu:
        amax = amax.clamp(min=0.17)
    packed = torch.empty((m, k // 2), dtype=torch.uint8, device=rows.device)
    tile_cols = -(-(k // BLOCK) // 4)
    # Padded rows of the last 128-row tile must hold finite scales: zeros.
    scales = torch.zeros((-(-m // 128) * 128 * tile_cols * 4,), dtype=torch.float8_e4m3fn, device=rows.device)
    _quantize_kernel[(m, k // _SPAN)](rows, packed, scales, amax, rows.stride(0), k // 2, tile_cols, GELU=gelu, SPAN=_SPAN, num_warps=4)
    return Nvfp4Tensor(packed.view(torch.float4_e2m1fn_x2), scales, (amax / _RANGE).reshape(()))


def nvfp4_gemm(a: Nvfp4Tensor, b: Nvfp4Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """BF16 ``a @ b.T + bias`` of two NVFP4 operands (``b`` is a weight ``[out, in]``)."""

    return F.scaled_mm(
        a.packed,
        b.packed.t(),
        [a.block_scale, a.decode_scale],
        [ScalingType.BlockWise1x16, ScalingType.TensorWise],
        [b.block_scale, b.decode_scale],
        [ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        bias=bias,
        output_dtype=torch.bfloat16,
    )


class Nvfp4Linear(torch.nn.Module):
    """W4A4 NVFP4 linear with a BF16 output, built from a BF16 ``nn.Linear`` (after the Turbo merge)."""

    def __init__(self, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.in_features: int = linear.in_features
        self.out_features: int = linear.out_features
        if self.in_features % _SPAN or self.out_features % 32:
            raise ValueError(f"NVFP4 GEMM needs in_features % {_SPAN} == 0 and out_features % 32 == 0, got {tuple(linear.weight.shape)}.")
        weight = quantize_weight(linear.weight.detach())
        self.register_buffer("weight_packed", weight.packed)
        self.register_buffer("weight_block_scale", weight.block_scale)
        self.register_buffer("weight_decode_scale", weight.decode_scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(torch.bfloat16))

    @property
    def weight_operand(self) -> Nvfp4Tensor:
        return Nvfp4Tensor(self.weight_packed, self.weight_block_scale, self.weight_decode_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._project(x, gelu=False)

    def forward_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(gelu_tanh(x))`` with the GELU inside the activation quantization."""

        return self._project(x, gelu=True)

    def _project(self, x: torch.Tensor, *, gelu: bool) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = nvfp4_gemm(quantize_activation(rows.to(torch.bfloat16), gelu=gelu), self.weight_operand, self.bias)
        return out.view(*x.shape[:-1], self.out_features)
