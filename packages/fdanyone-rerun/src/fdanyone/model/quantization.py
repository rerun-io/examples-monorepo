"""Selective quantization policies for the 4DAnyone DiT."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

_FP8_WEIGHT_ONLY_PATTERN: re.Pattern[str] = re.compile(
    r"blocks\.(?P<block>\d+)\."
    r"(?:(?:self_attn|self_attn_mvs|cross_attn)\.(?:q|k|v|o)|"
    r"ffn\.(?:0|2))"
)

QUANTIZED_BLOCK_RANGE: tuple[int, int] = (3, 26)
"""Inclusive interior DiT block range the guide approves for FP8 projections."""

FP8_DTYPE: torch.dtype = torch.float8_e4m3fn
FP8_MAX: float = float(torch.finfo(torch.float8_e4m3fn).max)
"""448: the largest finite e4m3fn value."""

FP8_ROW_CHUNK_BUDGET_BYTES: int = 1024**3
"""Bound on one row chunk's activation temporaries: a BF16 scaled copy (2 B) and an FP8 copy (1 B) per element."""

_TINY_AMAX: float = 2.0**-100
"""Floor of a row maximum, so an all-zero row gets a finite scale."""


def is_quantizable_dit_projection(fqn: str) -> bool:
    """Return whether an FQN is on the safe interior-projection allowlist.

    W8A8 uses the same boundary-protected projection set established by the
    earlier weight-only experiment; the name describes the surviving policy.
    """

    match: re.Match[str] | None = _FP8_WEIGHT_ONLY_PATTERN.fullmatch(fqn)
    if match is None:
        return False
    block_index: int = int(match.group("block"))
    first_block, last_block = QUANTIZED_BLOCK_RANGE
    return first_block <= block_index <= last_block


@dataclass(frozen=True, slots=True)
class FP8W8A8Report:
    """Summary of one W8A8 conversion."""

    module_count: int
    """Number of guide-approved Linear modules replaced by FP8 linears."""


class Fp8RowwiseLinear(torch.nn.Module):
    """W8A8 linear: per-output-channel FP8 weight, per-token dynamic FP8 activation, BF16 output.

    Rows are quantized and multiplied in chunks, so the only temporaries beyond the BF16
    output are one chunk's BF16 scaled copy and FP8 copy. Every row has its own scale, so
    the result does not depend on the chunk size. The scale is applied in BF16 as
    ``bf16(448 / amax)``; the GEMM gets its exact FP32 reciprocal, and a row maximum lands
    at most 0.4% above 448, far below the 464 where e4m3fn rounding would reach NaN.
    """

    def __init__(self, linear: torch.nn.Linear, *, chunk_budget_bytes: int = FP8_ROW_CHUNK_BUDGET_BYTES) -> None:
        super().__init__()
        weight = linear.weight.detach()
        self.in_features: int = linear.in_features
        self.out_features: int = linear.out_features
        if self.in_features % 16 or self.out_features % 16:
            raise ValueError(f"FP8 GEMM needs dimensions divisible by 16, got {tuple(weight.shape)}.")
        weight_scale = weight.float().abs().amax(dim=1, keepdim=True).clamp_(min=_TINY_AMAX) / FP8_MAX
        self.register_buffer("weight", (weight.float() / weight_scale).to(FP8_DTYPE))  # [N, K]
        self.register_buffer("weight_scale", weight_scale.t().contiguous())  # [1, N] fp32
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().to(torch.bfloat16))
        self.rows_per_chunk: int = max(16, chunk_budget_bytes // (3 * self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        out = torch.empty((rows.shape[0], self.out_features), dtype=torch.bfloat16, device=x.device)
        # Autocast would run the row-maximum reduction on an FP32 copy of each chunk.
        with torch.autocast(device_type=x.device.type, enabled=False):
            for start in range(0, rows.shape[0], self.rows_per_chunk):
                self._project_chunk(rows[start : start + self.rows_per_chunk], out[start : start + self.rows_per_chunk])
        return out.view(*x.shape[:-1], self.out_features)

    def _project_chunk(self, rows: torch.Tensor, out: torch.Tensor) -> None:
        """Quantize one row chunk and write its BF16 product into ``out``; temporaries end here."""

        chunk = rows.to(torch.bfloat16)  # the BF16 autocast contract of the replaced nn.Linear
        amax = torch.linalg.vector_norm(chunk, ord=float("inf"), dim=-1, keepdim=True, dtype=torch.float32)
        inverse_scale = (FP8_MAX / amax.clamp_(min=_TINY_AMAX)).to(torch.bfloat16)  # [m, 1]
        quantized = chunk.mul(inverse_scale).to(FP8_DTYPE)
        torch._scaled_mm(
            quantized,
            self.weight.t(),  # column-major [K, N]
            scale_a=inverse_scale.float().reciprocal(),  # [m, 1] fp32
            scale_b=self.weight_scale,  # [1, N] fp32
            bias=self.bias,
            out_dtype=torch.bfloat16,
            out=out,
        )


def quantize_dit_fp8_w8a8(dit: torch.nn.Module) -> FP8W8A8Report:
    """Replace the approved projections with chunked row-wise FP8 W8A8 linears."""

    selected: list[str] = [
        fqn
        for fqn, module in dit.named_modules()
        if isinstance(module, torch.nn.Linear) and is_quantizable_dit_projection(fqn)
    ]
    for fqn in selected:
        parent_name, _, child_name = fqn.rpartition(".")
        parent = dit.get_submodule(parent_name)
        setattr(parent, child_name, Fp8RowwiseLinear(parent.get_submodule(child_name)))
    return FP8W8A8Report(module_count=len(selected))
