"""NVFP4 W4A4 linear: CPU tests of the encoding, CUDA tests of the Triton quantization and the block-scaled GEMM."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from fdanyone.model.nvfp4 import (  # noqa: E402
    Nvfp4Linear,
    dequantize,
    from_blocked,
    quantize_activation,
    quantize_activation_reference,
    quantize_weight,
    round_to_e2m1_codes,
    to_blocked,
)
from fdanyone.model.quantization import quantize_dit_fp8_w8a8  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (Triton kernel, cuBLASLt FP4 GEMM)")


def test_e2m1_rounding_hits_the_grid_and_ties_to_even() -> None:
    values = torch.tensor([0.0, 0.25, 0.26, 0.75, 1.25, 1.3, 1.75, 2.5, 2.6, 3.5, 5.0, 5.1, 7.0, -0.5, -6.0])
    codes = round_to_e2m1_codes(values)
    assert codes.tolist() == [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 9, 15]


def test_block_layout_round_trips_with_padding() -> None:
    scales = torch.arange(300 * 9, dtype=torch.float32).view(300, 9)  # ragged in both tile dimensions
    flat = to_blocked(scales)
    assert flat.numel() == 384 * 12
    torch.testing.assert_close(from_blocked(flat, 300, 9), scales, rtol=0, atol=0)


def test_weight_quantization_is_close_and_keeps_the_maximum() -> None:
    torch.manual_seed(0)
    weight = torch.randn(64, 512) * 0.02
    weight[3, 17] = 0.5  # an outlier sets the per-tensor scale
    deq = dequantize(quantize_weight(weight), 64, 512)
    assert float((deq - weight).norm() / weight.norm()) < 0.1  # E2M1: about 8 % relative error on a Gaussian
    torch.testing.assert_close(deq[3, 17], weight[3, 17], rtol=0.05, atol=0)


def test_the_scale_search_does_not_lose_to_the_plain_block_maximum() -> None:
    torch.manual_seed(1)
    weight = torch.randn(32, 256)
    searched = dequantize(quantize_weight(weight), 32, 256)
    plain = dequantize(quantize_weight(weight, candidates=(1.0,)), 32, 256)
    assert (searched - weight).norm() <= (plain - weight).norm()


@cuda
@pytest.mark.integration
@pytest.mark.parametrize("rows", [128, 300])
def test_triton_quantization_matches_the_eager_reference(rows: int) -> None:
    torch.manual_seed(0)
    x = (torch.randn(rows, 512, device="cuda") * 3).to(torch.bfloat16)
    x[7] = 0
    fast, slow = quantize_activation(x), quantize_activation_reference(x)
    torch.testing.assert_close(fast.decode_scale, slow.decode_scale)
    assert torch.equal(fast.block_scale.view(torch.uint8), slow.block_scale.view(torch.uint8))
    # FP32 contraction can move a value across a rounding threshold: rare single-code differences only.
    assert float((fast.packed.view(torch.uint8) != slow.packed.view(torch.uint8)).float().mean()) < 1e-3


@cuda
@pytest.mark.integration
def test_linear_matches_its_dequantized_reference_and_stays_near_bf16() -> None:
    torch.manual_seed(0)
    linear = torch.nn.Linear(512, 384, device="cuda").to(torch.bfloat16)
    x = torch.randn(2, 150, 512, device="cuda", dtype=torch.bfloat16)
    nv = Nvfp4Linear(linear)
    out = nv(x)
    assert out.shape == (2, 150, 384) and out.dtype is torch.bfloat16
    a = dequantize(quantize_activation(x.view(-1, 512)), 300, 512)
    w = dequantize(nv.weight_operand, 384, 512)
    expected = torch.nn.functional.linear(a, w, linear.bias.float()).view(2, 150, 384)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
    bf16 = linear(x).float()
    assert float((out.float() - bf16).norm() / bf16.norm()) < 0.15


@cuda
@pytest.mark.integration
def test_gelu_path_equals_gelu_then_project() -> None:
    torch.manual_seed(0)
    nv = Nvfp4Linear(torch.nn.Linear(512, 256, device="cuda").to(torch.bfloat16))
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 2
    fused = nv.forward_gelu(x)
    # The fused path takes the per-tensor scale from amax(x), an upper bound of amax(gelu(x)).
    amax = x.float().abs().amax().clamp(min=0.17)
    a = dequantize(quantize_activation_reference(torch.nn.functional.gelu(x, approximate="tanh"), amax), 256, 512)
    expected = torch.nn.functional.linear(a, dequantize(nv.weight_operand, 256, 512), nv.bias.float())
    torch.testing.assert_close(fused.float(), expected, rtol=2e-2, atol=2e-2)


def test_policy_puts_only_the_chosen_groups_in_nvfp4(monkeypatch: pytest.MonkeyPatch) -> None:
    import fdanyone.model.nvfp4 as nvfp4

    class Marker(torch.nn.Module):
        def __init__(self, linear: torch.nn.Linear) -> None:
            super().__init__()
            self.in_features = linear.in_features

    monkeypatch.setattr(nvfp4, "Nvfp4Linear", Marker)
    blocks = torch.nn.Module()
    blocks.blocks = torch.nn.ModuleList()
    for _ in range(5):
        block = torch.nn.Module()
        block.ffn = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 32))
        block.self_attn = torch.nn.Module()
        block.self_attn.q = torch.nn.Linear(32, 32)
        blocks.blocks.append(block)
    report = quantize_dit_fp8_w8a8(blocks.to(torch.bfloat16), nvfp4_projections=("ffn",), nvfp4_blocks=(4, 4))
    assert (report.module_count, report.nvfp4_count) == (4, 2)  # blocks 3 and 4 (three linears each) are quantized; block 4 FFN is NVFP4
    assert isinstance(blocks.blocks[4].ffn[0], Marker) and isinstance(blocks.blocks[4].ffn[2], Marker)
    assert not isinstance(blocks.blocks[3].ffn[0], Marker) and not isinstance(blocks.blocks[4].self_attn.q, Marker)
    with pytest.raises(ValueError, match="Unknown NVFP4"):
        quantize_dit_fp8_w8a8(blocks, nvfp4_projections=("mlp",))


@cuda
@pytest.mark.integration
def test_shared_projection_equals_separate_projections() -> None:
    from fdanyone.model.nvfp4 import project_shared

    torch.manual_seed(0)
    layers = tuple(Nvfp4Linear(torch.nn.Linear(512, 256, device="cuda").to(torch.bfloat16)) for _ in range(3))
    x = torch.randn(2, 100, 512, device="cuda", dtype=torch.bfloat16)
    for shared, alone in zip(project_shared(layers, x), (layer(x) for layer in layers), strict=True):
        torch.testing.assert_close(shared, alone, rtol=0, atol=0)


def test_step_policy_swaps_nvfp4_and_fp8_twins_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    import fdanyone.model.nvfp4 as nvfp4
    from fdanyone.model.quantization import Fp8RowwiseLinear, select_step_precision

    class Marker(torch.nn.Module):
        def __init__(self, linear: torch.nn.Linear) -> None:
            super().__init__()
            self.in_features = linear.in_features

    monkeypatch.setattr(nvfp4, "Nvfp4Linear", Marker)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList()
    for _ in range(4):
        block = torch.nn.Module()
        block.ffn = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 32))
        dit.blocks.append(block)
    report = quantize_dit_fp8_w8a8(dit.to(torch.bfloat16), nvfp4_projections=("ffn",), nvfp4_blocks=(3, 3), nvfp4_steps=(3,))
    assert (report.module_count, report.nvfp4_count) == (0, 2)
    select_step_precision(dit, 0)
    assert isinstance(dit.blocks[3].ffn[0], Fp8RowwiseLinear) and isinstance(dit.blocks[3].ffn[2], Fp8RowwiseLinear)
    select_step_precision(dit, 3)
    assert isinstance(dit.blocks[3].ffn[0], Marker) and isinstance(dit.blocks[3].ffn[2], Marker)
    select_step_precision(torch.nn.Linear(2, 2), 3)  # no step policy: a no-op
