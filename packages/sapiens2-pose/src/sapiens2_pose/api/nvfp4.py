"""Reference (pure torch) NVFP4 quantisation: e2m1 values, ue4m3 per-16 block scales, fp32 per-tensor global scale.

Encoding (NVIDIA NVFP4 / modelopt convention):
    g  = 448 * 6 / amax(x)                       per-tensor global encode scale
    sb: Tensor = e4m3(amax(block) / 6 * g)               per-16 block decode scale (stored)
    q  = round_e2m1(x * g / sb)                  values in {0, .5, 1, 1.5, 2, 3, 4, 6} * sign
    x ~= q * sb / g
Packing: two e2m1 codes per byte, element 2i in the low nibble (torch.float4_e2m1fn_x2 / cuBLASLt convention).
"""

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLOCK = 16


def round_e2m1_codes(v: Float[Tensor, "..."]) -> UInt8[Tensor, "..."]:
    """fp32 values (already scaled into [-6, 6]) -> uint8 codes 0..15 (bit 3 = sign), round-to-nearest-even."""
    a: Tensor = v.abs().clamp(max=6.0)
    # thresholds between grid points; ties go to the even code (0, 1.0, 2.0, 4.0 have even mantissa)
    # midpoints: .25 .75 1.25 1.75 2.5 3.5 5
    code: Tensor = torch.zeros_like(a, dtype=torch.uint8)
    code = torch.where(a > 0.25, 1, code)
    code = torch.where(a >= 0.75, 2, code)
    code = torch.where(a > 1.25, 3, code)
    code = torch.where(a >= 1.75, 4, code)
    code = torch.where(a > 2.5, 5, code)
    code = torch.where(a >= 3.5, 6, code)
    code = torch.where(a > 5.0, 7, code)
    code = code.to(torch.uint8)
    return torch.where(v < 0, code | 8, code).to(torch.uint8)


def quant_nvfp4_mse(
    x: Float[Tensor, "m k"], cands: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5), tensor_div: float = 6.0 * 256.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Weight-only NVFP4 with a per-block scale search (Four-Over-Six style + MSE sweep): block scale = bamax/6 * c, c from
    ``cands`` (c = 1.5 maps the block max to 4), chosen per block by dequantised MSE. Global encode scale g = tensor_div/amax
    (6*256 leaves e4m3 headroom for c > 1). Returns packed E2M1 values, E4M3 block scales, and a global encode scale; decode is q * sb / g."""
    m: int = int(x.shape[0])
    k: int = int(x.shape[1])
    xf: Tensor = x.float()
    g: Tensor = tensor_div / xf.abs().amax().clamp(min=1e-12)
    blocks: Tensor = xf.view(m, k // BLOCK, BLOCK)
    bamax: Tensor = blocks.abs().amax(-1)
    grid: Tensor = E2M1_GRID.to(x.device)
    best_err: Tensor = torch.full_like(bamax, float("inf"))
    best_sb: Tensor = torch.zeros_like(bamax)
    best_codes: Tensor = torch.zeros_like(blocks, dtype=torch.uint8)
    for c in cands:
        sb = (bamax / 6.0 * c * g).clamp(min=2.0**-9, max=448.0).to(torch.float8_e4m3fn)
        sbf: Tensor = sb.float()
        scaled: Tensor = blocks * (g / torch.where(sbf == 0, torch.ones_like(sbf), sbf))[..., None]
        codes: Tensor = round_e2m1_codes(scaled)
        mag: Tensor = grid[(codes & 7).long()]
        deq: Tensor = torch.where((codes & 8) > 0, -mag, mag) * (sbf / g)[..., None]
        err: Tensor = ((deq - blocks) ** 2).sum(-1)
        better: Tensor = err < best_err
        best_err = torch.where(better, err, best_err)
        best_sb = torch.where(better, sbf, best_sb)
        best_codes = torch.where(better[..., None], codes, best_codes)
    codes = best_codes.view(m, k)
    packed: Tensor = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    return packed, best_sb.to(torch.float8_e4m3fn), g


def bf16_round_up(value: float) -> float:
    """Return the smallest bf16 scale at least as large as a positive value."""
    scale: Tensor = torch.tensor([value], dtype=torch.float32).to(torch.bfloat16)
    if float(scale) < value:
        scale = (scale.view(torch.int16) + 1).view(torch.bfloat16)
    return float(scale.float())
