"""Selective quantization policies for the 4DAnyone DiT."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch

_DIT_PROJECTION_PATTERN: re.Pattern[str] = re.compile(
    r"blocks\.(?P<block>\d+)\."
    r"(?:(?:self_attn|self_attn_mvs|cross_attn)\.(?:q|k|v|o)|"
    r"ffn\.(?:0|2))"
)

QUANTIZED_BLOCK_RANGE: tuple[int, int] = (3, 26)
"""Inclusive interior DiT block range quantized to FP8; the first and last blocks stay BF16."""

NVFP4_KINDS: tuple[str, ...] = ("ffn", "self_attn", "self_attn_mvs", "cross_attn")
"""Projection groups that can run NVFP4 W4A4 instead of FP8 (``InferenceConfig.dit_nvfp4_projections``)."""

FP8_DTYPE: torch.dtype = torch.float8_e4m3fn
FP8_MAX: float = float(torch.finfo(torch.float8_e4m3fn).max)
"""448: the largest finite e4m3fn value."""

FP8_ROW_CHUNK_BUDGET_BYTES: int = 1024**3
"""Bound on one row chunk's activation temporaries: a BF16 scaled copy (2 B) and an FP8 copy (1 B) per element."""

_TINY_AMAX: float = 2.0**-100
"""Floor of a row maximum, so an all-zero row gets a finite scale."""


def projection_kind(fqn: str) -> tuple[int, str] | None:
    """Block index and group (``ffn``, ``self_attn``, ``self_attn_mvs``, ``cross_attn``) of a DiT projection FQN."""

    match: re.Match[str] | None = _DIT_PROJECTION_PATTERN.fullmatch(fqn)
    if match is None:
        return None
    return int(match.group("block")), fqn.split(".")[2]


def is_quantizable_dit_projection(fqn: str) -> bool:
    """Return whether an FQN is an attention or FFN projection inside ``QUANTIZED_BLOCK_RANGE``."""

    match: re.Match[str] | None = _DIT_PROJECTION_PATTERN.fullmatch(fqn)
    if match is None:
        return False
    block_index: int = int(match.group("block"))
    first_block, last_block = QUANTIZED_BLOCK_RANGE
    return first_block <= block_index <= last_block


@dataclass(frozen=True, slots=True)
class FP8W8A8Report:
    """Summary of one W8A8 conversion."""

    module_count: int
    """Number of Linear modules replaced by FP8 linears."""
    nvfp4_count: int = 0
    """Number of Linear modules replaced by NVFP4 linears instead."""


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
        self.chunk_budget_bytes: int = chunk_budget_bytes
        self.rows_per_chunk: int = max(16, chunk_budget_bytes // (3 * self.in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return project_fp8_shared((self,), x)[0]

    def forward_gelu(self, x: torch.Tensor) -> torch.Tensor:
        """``forward(gelu_tanh(x))``, the GELU fused into the activation quantization (eagerly it overwrites ``x``)."""

        return project_fp8_shared((self,), x, gelu=True)[0]

    def _gemm(self, quantized: torch.Tensor, scale_a: torch.Tensor, out: torch.Tensor) -> None:
        torch._scaled_mm(
            quantized, self.weight.t(), scale_a=scale_a, scale_b=self.weight_scale, bias=self.bias, out_dtype=torch.bfloat16, out=out
        )

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


def project_fp8_shared(
    linears: tuple[Fp8RowwiseLinear, ...], x: torch.Tensor, *, gelu: bool = False
) -> list[torch.Tensor]:
    """Apply FP8 linears that read the same input, quantizing each row chunk of it once."""

    first = linears[0]
    rows = x.reshape(-1, first.in_features)
    outs = [torch.empty((rows.shape[0], linear.out_features), dtype=torch.bfloat16, device=x.device) for linear in linears]
    from fdanyone.model import fused as fused_kernels

    fused = fused_kernels.active(x)
    if gelu and not fused:
        from fdanyone.vendor.diffsynth.models.wan_video_dit import gelu_tanh

        rows = gelu_tanh(rows)
    # The fused kernel keeps only the FP8 copy (1 B per element) of a chunk; eager also keeps a BF16 scaled copy.
    rows_per_chunk = max(16, first.chunk_budget_bytes // first.in_features) if fused else first.rows_per_chunk
    # Autocast would run the row-maximum reduction on an FP32 copy of each chunk.
    with torch.autocast(device_type=x.device.type, enabled=False):
        for start in range(0, rows.shape[0], rows_per_chunk):
            chunk = rows[start : start + rows_per_chunk]
            if fused:
                quantized, scale_a = fused_kernels.quantize_rows_fp8(chunk.to(torch.bfloat16), gelu=gelu)
            for linear, out in zip(linears, outs, strict=True):
                if fused:
                    linear._gemm(quantized, scale_a, out[start : start + rows_per_chunk])
                else:
                    linear._project_chunk(chunk, out[start : start + rows_per_chunk])
    return [out.view(*x.shape[:-1], linear.out_features) for linear, out in zip(linears, outs, strict=True)]


def quantize_dit_fp8_w8a8(
    dit: torch.nn.Module, *, nvfp4_projections: tuple[str, ...] | None = None, nvfp4_blocks: tuple[int, int] | None = None
) -> FP8W8A8Report:
    """Replace the approved projections with chunked row-wise FP8 W8A8 linears; groups in ``nvfp4_projections``
    inside the inclusive ``nvfp4_blocks`` range get NVFP4 W4A4 linears instead (defaults: ``InferenceConfig``)."""

    from fdanyone.config import INFERENCE

    nvfp4_projections = INFERENCE.dit_nvfp4_projections if nvfp4_projections is None else nvfp4_projections
    nvfp4_blocks = INFERENCE.dit_nvfp4_blocks if nvfp4_blocks is None else nvfp4_blocks

    unknown = set(nvfp4_projections) - set(NVFP4_KINDS)
    if unknown:
        raise ValueError(f"Unknown NVFP4 projection groups {sorted(unknown)}; choose from {NVFP4_KINDS}.")

    selected: list[str] = [
        fqn
        for fqn, module in dit.named_modules()
        if isinstance(module, torch.nn.Linear) and is_quantizable_dit_projection(fqn)
    ]
    nvfp4_count = 0
    for fqn in selected:
        parent_name, _, child_name = fqn.rpartition(".")
        parent = dit.get_submodule(parent_name)
        linear = parent.get_submodule(child_name)
        block, kind = projection_kind(fqn) or (-1, "")
        if kind in nvfp4_projections and nvfp4_blocks[0] <= block <= nvfp4_blocks[1]:
            from fdanyone.model.nvfp4 import Nvfp4Linear

            setattr(parent, child_name, Nvfp4Linear(linear))
            nvfp4_count += 1
        else:
            setattr(parent, child_name, Fp8RowwiseLinear(linear))
    if nvfp4_count:
        logging.getLogger(__name__).info("NVFP4 W4A4 DiT: %d projections (%s, blocks %d-%d)", nvfp4_count, ",".join(nvfp4_projections), *nvfp4_blocks)
    return FP8W8A8Report(module_count=len(selected) - nvfp4_count, nvfp4_count=nvfp4_count)
