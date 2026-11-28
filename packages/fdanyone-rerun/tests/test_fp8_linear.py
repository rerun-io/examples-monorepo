"""CPU tests of the chunked row-wise FP8 W8A8 linear (torch._scaled_mm has a CPU kernel)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fdanyone.model.quantization import (
    FP8_DTYPE,
    FP8_MAX,
    Fp8RowwiseLinear,
    quantize_dit_fp8_w8a8,
)


def _linear(in_features: int, out_features: int, *, bias: bool = True) -> torch.nn.Linear:
    torch.manual_seed(0)
    return torch.nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)


def _budget(rows: int, in_features: int) -> int:
    return rows * 3 * in_features


def _dequantized_reference(fp8: Fp8RowwiseLinear, x: torch.Tensor) -> torch.Tensor:
    """The same quantization, unchunked, as FP32 dequantize-and-matmul."""

    rows = x.reshape(-1, fp8.in_features).to(torch.bfloat16)
    amax = rows.float().abs().amax(dim=-1, keepdim=True).clamp(min=2.0**-100)
    inverse_scale = (FP8_MAX / amax).to(torch.bfloat16)
    activation = (rows * inverse_scale).to(FP8_DTYPE).float() / inverse_scale.float()
    weight = fp8.weight.float() * fp8.weight_scale.t()
    bias = None if fp8.bias is None else fp8.bias.float()
    return F.linear(activation, weight, bias).to(torch.bfloat16).view(*x.shape[:-1], fp8.out_features)


@pytest.mark.parametrize("rows_per_chunk", [16, 48, 10_000])
def test_output_does_not_depend_on_the_chunk_size(rows_per_chunk: int) -> None:
    linear = _linear(64, 32)
    x = torch.randn(3, 37, 64, dtype=torch.bfloat16) * 5  # 111 rows: the last chunk is ragged
    whole = Fp8RowwiseLinear(linear, chunk_budget_bytes=10**9)
    chunked = Fp8RowwiseLinear(linear, chunk_budget_bytes=_budget(rows_per_chunk, 64))
    assert chunked.rows_per_chunk == rows_per_chunk
    torch.testing.assert_close(chunked(x), whole(x), rtol=0, atol=0)


def test_matches_the_dequantized_reference_and_stays_near_bf16() -> None:
    linear = _linear(128, 48)
    x = torch.randn(2, 50, 128, dtype=torch.bfloat16)
    fp8 = Fp8RowwiseLinear(linear, chunk_budget_bytes=_budget(32, 128))
    out = fp8(x)
    assert out.shape == (2, 50, 48) and out.dtype is torch.bfloat16
    torch.testing.assert_close(out, _dequantized_reference(fp8, x), rtol=2e-2, atol=2e-2)
    bf16 = linear(x).float()
    assert (out.float() - bf16).norm() / bf16.norm() < 0.05  # e4m3 W8A8: a few percent


def test_fp32_input_and_autocast_follow_the_bf16_contract() -> None:
    fp8 = Fp8RowwiseLinear(_linear(64, 32, bias=False))
    x = torch.randn(40, 64)
    expected = fp8(x.to(torch.bfloat16))
    torch.testing.assert_close(fp8(x), expected, rtol=0, atol=0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        torch.testing.assert_close(fp8(x), expected, rtol=0, atol=0)


def test_zero_tiny_huge_and_outlier_rows_stay_finite() -> None:
    linear = _linear(64, 32)
    torch.manual_seed(1)
    magnitudes = torch.pow(10.0, torch.empty(500, 1).uniform_(-20, 20))
    x = (torch.randn(500, 64) * magnitudes).to(torch.bfloat16)
    x[0] = 0.0
    x[3, 5] = 3.0e4
    fp8 = Fp8RowwiseLinear(linear, chunk_budget_bytes=_budget(64, 64))
    out = fp8(x)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out[0], linear.bias, rtol=0, atol=0)
    torch.testing.assert_close(out, _dequantized_reference(fp8, x), rtol=2e-2, atol=2e-2)


def test_temporaries_stay_inside_the_chunk_budget() -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    class NewStorages(TorchDispatchMode):
        """Record the byte size of every storage an op allocates; views share their input's storage."""

        def __init__(self, known: set[int]) -> None:
            super().__init__()
            self.known = known
            self.sizes: list[int] = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor) and result.untyped_storage().data_ptr() not in self.known:
                self.known.add(result.untyped_storage().data_ptr())
                self.sizes.append(result.untyped_storage().nbytes())
            return result

    in_features, out_features, rows = 256, 64, 1000
    budget = _budget(50, in_features)
    fp8 = Fp8RowwiseLinear(_linear(in_features, out_features), chunk_budget_bytes=budget)
    x = torch.randn(rows, in_features, dtype=torch.bfloat16)
    with NewStorages({x.untyped_storage().data_ptr()}) as mode:
        fp8(x)
    output_bytes = rows * out_features * 2
    assert mode.sizes.count(output_bytes) == 1
    assert max(size for size in mode.sizes if size != output_bytes) <= budget
    assert x.numel() * x.element_size() > 10 * budget  # the bound is meaningful


def test_dit_swap_covers_exactly_the_interior_projections() -> None:
    from fdanyone.vendor.diffsynth.models.wan_video_dit import FourDAnyoneDiT

    with torch.device("meta"):
        dit = FourDAnyoneDiT(attention_backend="sdpa").to(torch.bfloat16)
    report = quantize_dit_fp8_w8a8(dit)
    assert report.module_count == 24 * 14  # blocks 3-26: 12 attention projections + 2 FFN projections
    assert isinstance(dit.get_submodule("blocks.3.ffn.2"), Fp8RowwiseLinear)
    assert isinstance(dit.get_submodule("blocks.26.self_attn_mvs.q"), Fp8RowwiseLinear)
    assert isinstance(dit.get_submodule("blocks.2.ffn.0"), torch.nn.Linear)
    assert isinstance(dit.get_submodule("blocks.27.cross_attn.o"), torch.nn.Linear)
    assert dit.get_buffer("blocks.3.ffn.2.weight").dtype is FP8_DTYPE


def test_dit_swap_with_fp8_edges_covers_every_block() -> None:
    from fdanyone.vendor.diffsynth.models.wan_video_dit import FourDAnyoneDiT

    with torch.device("meta"):
        dit = FourDAnyoneDiT(attention_backend="sdpa").to(torch.bfloat16)
    report = quantize_dit_fp8_w8a8(dit, fp8_edges=True)
    assert report.module_count == 30 * 14
    assert isinstance(dit.get_submodule("blocks.0.self_attn.q"), Fp8RowwiseLinear)
    assert isinstance(dit.get_submodule("blocks.29.ffn.2"), Fp8RowwiseLinear)


def test_prepare_on_device_keeps_fp8_weights_after_acceleration() -> None:
    from fdanyone.model.loader import Denoiser

    model = torch.nn.Sequential(Fp8RowwiseLinear(_linear(64, 32)))
    denoiser = Denoiser(model=model, scheduler=None, dtype=torch.bfloat16)  # type: ignore[arg-type]
    denoiser.accelerated = True
    denoiser.prepare_on_device("cpu")
    assert model[0].weight.dtype is FP8_DTYPE
    assert model[0].weight_scale.dtype is torch.float32
