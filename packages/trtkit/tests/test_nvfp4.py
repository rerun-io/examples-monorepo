"""NVFP4 encoding and scale bounds checked against known representable values."""

import torch

from trtkit.nvfp4 import bf16_round_up, quant_nvfp4_mse, round_e2m1_codes


def test_e2m1_rounding_ties_and_sign_bits() -> None:
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -6.0])
    assert round_e2m1_codes(values).tolist() == [0, 2, 2, 4, 4, 6, 6, 15]


def test_quantized_zero_blocks_are_finite_and_have_zero_codes() -> None:
    packed, scales, global_scale = quant_nvfp4_mse(torch.zeros((2, 32)))
    assert packed.shape == (2, 16)
    assert packed.view(torch.uint8).eq(0).all()
    assert torch.isfinite(scales.float()).all()
    assert torch.isfinite(global_scale)


def test_bf16_scale_rounds_up_without_an_extra_ulp() -> None:
    for value in (0.0007, 0.015625, 0.34, 100.3):
        rounded = bf16_round_up(value)
        assert rounded >= value
        previous = (torch.tensor(rounded, dtype=torch.bfloat16).view(torch.int16) - 1).view(torch.bfloat16)
        assert float(previous) < value
