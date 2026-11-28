"""The fused Triton glue kernels of the DiT against the eager code they replace (CUDA, small shapes)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (Triton kernels)"),
]

DEVICE = "cuda"


def _bf16_flip_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of elements whose BF16 values differ (FP32 reduction order may flip a rounding)."""

    return float((a != b).float().mean())


@pytest.mark.parametrize("cols", [3072, 14336])
@pytest.mark.parametrize("gelu", [False, True])
def test_row_quantization_matches_the_eager_fp8_path(cols: int, gelu: bool) -> None:
    from fdanyone.model.fused import quantize_rows_fp8
    from fdanyone.model.quantization import _TINY_AMAX, FP8_DTYPE, FP8_MAX

    torch.manual_seed(0)
    rows = (torch.randn(257, cols, device=DEVICE) * 3).to(torch.bfloat16)
    rows[5] = 0  # an all-zero row gets the floor scale
    source = torch.nn.functional.gelu(rows, approximate="tanh") if gelu else rows
    amax = torch.linalg.vector_norm(source, ord=float("inf"), dim=-1, keepdim=True, dtype=torch.float32)
    inverse_scale = (FP8_MAX / amax.clamp_(min=_TINY_AMAX)).to(torch.bfloat16)
    expected = source.mul(inverse_scale).to(FP8_DTYPE)
    quantized, scale = quantize_rows_fp8(rows, gelu=gelu)
    torch.testing.assert_close(scale, inverse_scale.float().reciprocal(), rtol=0, atol=0)
    if gelu:  # libdevice tanh vs aten's: a rare BF16 flip of the activation, then at most one FP8 step
        assert _bf16_flip_fraction(quantized.view(torch.uint8), expected.view(torch.uint8)) < 1e-3
        torch.testing.assert_close(quantized.float(), expected.float(), rtol=0.07, atol=0.0)
    else:
        assert torch.equal(quantized.view(torch.uint8), expected.view(torch.uint8))


def _rope_reference(x: torch.Tensor, norm, freqs: torch.Tensor | None) -> torch.Tensor:
    from fdanyone.vendor.diffsynth.models.wan_video_dit import rope_apply

    normalized = norm.forward_inplace(x.clone())
    return normalized if freqs is None else rope_apply(normalized, freqs, 24)


@pytest.mark.parametrize("rope", [False, True])
def test_rms_norm_rope_matches_rmsnorm_then_rope_apply(rope: bool) -> None:
    from fdanyone.model.fused import rms_norm_rope_, rope_tables
    from fdanyone.vendor.diffsynth.models.wan_video_dit import RMSNorm, precompute_freqs_cis_3d

    torch.manual_seed(0)
    b, s, c = 3, 50, 3072
    norm = RMSNorm(c, eps=1e-6).to(DEVICE, torch.bfloat16)
    norm.weight.data.uniform_(0.5, 1.5)
    f0, f1, f2 = precompute_freqs_cis_3d(128)
    freqs = torch.cat([f0[:2].view(2, 1, 1, -1).expand(2, 5, 5, -1), f1[:5].view(1, 5, 1, -1).expand(2, 5, 5, -1),
                       f2[:5].view(1, 1, 5, -1).expand(2, 5, 5, -1)], dim=-1).reshape(s, 1, -1).to(DEVICE)
    x = (torch.randn(b, s, c, device=DEVICE) * 4).to(torch.bfloat16)
    with torch.inference_mode():
        expected = _rope_reference(x, norm, freqs if rope else None)
        out = rms_norm_rope_(x.clone(), norm.weight, norm.eps, rope_tables(freqs) if rope else None)
    assert _bf16_flip_fraction(out, expected) < 2e-3
    torch.testing.assert_close(out.float(), expected.float(), rtol=1.6e-2, atol=1e-2)


def test_layer_norm_modulation_matches_the_autocast_eager_path() -> None:
    from fdanyone.model.fused import layer_norm_modulation
    from fdanyone.vendor.diffsynth.models.wan_video_dit import normalized_modulation

    torch.manual_seed(0)
    v, n, c = 3, 40, 3072
    norm = torch.nn.LayerNorm(c, eps=1e-6, elementwise_affine=False).to(DEVICE)
    x = (torch.randn(v, n, c, device=DEVICE) * 2 + 0.5).to(torch.bfloat16)
    shift = (torch.randn(v, 1, c, device=DEVICE) * 0.2).to(torch.bfloat16)
    scale = (torch.randn(v, 1, c, device=DEVICE) * 0.2).to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = normalized_modulation(norm, x, shift, scale)
        out = layer_norm_modulation(x, shift, scale, 1e-6)
    assert out.dtype is torch.bfloat16
    assert _bf16_flip_fraction(out, expected) < 2e-3
    torch.testing.assert_close(out.float(), expected.float(), rtol=1.6e-2, atol=1e-2)


def test_gated_residual_is_bit_exact() -> None:
    from fdanyone.model.fused import gated_residual_

    torch.manual_seed(0)
    v, n, c = 3, 40, 3072
    x = torch.randn(v, n, c, device=DEVICE).to(torch.bfloat16)
    branch = torch.randn(v, n, c, device=DEVICE).to(torch.bfloat16)
    gate = torch.randn(v, 1, c, device=DEVICE).to(torch.bfloat16)
    expected = x.clone().add_(branch.clone().mul_(gate))
    assert torch.equal(gated_residual_(x, branch, gate), expected)


def test_layer_norm_affine_matches_autocast_layer_norm_then_the_bf16_cast() -> None:
    from fdanyone.model.fused import layer_norm_affine

    torch.manual_seed(0)
    v, n, c = 3, 40, 3072
    norm = torch.nn.LayerNorm(c, eps=1e-6).to(DEVICE, torch.bfloat16)
    norm.weight.data.uniform_(0.5, 1.5)
    norm.bias.data.uniform_(-0.2, 0.2)
    x = (torch.randn(v, n, c, device=DEVICE) * 2 + 0.5).to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = norm(x).to(torch.bfloat16)
        out = layer_norm_affine(x, norm.weight, norm.bias, norm.eps)
    assert _bf16_flip_fraction(out, expected) < 2e-3
    torch.testing.assert_close(out.float(), expected.float(), rtol=1.6e-2, atol=1e-2)
