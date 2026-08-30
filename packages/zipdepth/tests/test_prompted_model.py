"""Behavior tests for the prompted metric ZipDepth wrapper."""

from pathlib import Path

import torch
from jaxtyping import Float32, UInt8
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from monopriors.third_party.zipdepth.architecture import ZipDepth, create_model
from torch import Tensor


def _prompt_with_fixed_range(*, flip: bool = False) -> Float32[Tensor, "1 1 192 256"]:
    prompt_b1hw: Float32[Tensor, "1 1 192 256"] = torch.linspace(0.2, 3.8, 192 * 256, dtype=torch.float32).reshape(1, 1, 192, 256)
    if flip:
        prompt_b1hw = torch.flip(prompt_b1hw, dims=(-1,))
    prompt_b1hw[:, :, 40:80, 60:120] = 0.0
    return prompt_b1hw


def test_zero_initialized_prompt_injection_has_no_spatial_contribution() -> None:
    model: ZipDepthPrompt = ZipDepthPrompt().eval()
    image_b3hw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)

    with torch.inference_mode():
        first_b1hw: Float32[Tensor, "1 1 64 96"] = model(image_b3hw, _prompt_with_fixed_range())
        second_b1hw: Float32[Tensor, "1 1 64 96"] = model(image_b3hw, _prompt_with_fixed_range(flip=True))

    assert torch.equal(first_b1hw, second_b1hw)


def test_prompted_model_accepts_uint8_and_float_rgb() -> None:
    model: ZipDepthPrompt = ZipDepthPrompt().eval()
    image_u8_b3hw: UInt8[Tensor, "1 3 64 96"] = torch.randint(0, 256, (1, 3, 64, 96), dtype=torch.uint8)
    image_f32_b3hw: Float32[Tensor, "1 3 64 96"] = image_u8_b3hw.to(dtype=torch.float32) / 255.0
    prompt_b1hw: Float32[Tensor, "1 1 192 256"] = _prompt_with_fixed_range()

    with torch.inference_mode():
        uint8_depth_b1hw: Float32[Tensor, "1 1 64 96"] = model(image_u8_b3hw, prompt_b1hw)
        float_depth_b1hw: Float32[Tensor, "1 1 64 96"] = model(image_f32_b3hw, prompt_b1hw)

    assert uint8_depth_b1hw.shape == (1, 1, 64, 96)
    torch.testing.assert_close(uint8_depth_b1hw, float_depth_b1hw, rtol=0.0, atol=0.0)
    assert float(uint8_depth_b1hw.min()) >= 0.2
    assert float(uint8_depth_b1hw.max()) <= 3.8


def test_degenerate_prompts_produce_finite_metric_depth() -> None:
    model: ZipDepthPrompt = ZipDepthPrompt().eval()
    image_b3hw: Float32[Tensor, "2 3 64 96"] = torch.rand(2, 3, 64, 96)
    prompt_b1hw: Float32[Tensor, "2 1 192 256"] = torch.zeros(2, 1, 192, 256)
    prompt_b1hw[0] = 1.25

    with torch.inference_mode():
        depth_b1hw: Float32[Tensor, "2 1 64 96"] = model(image_b3hw, prompt_b1hw)

    assert depth_b1hw.shape == (2, 1, 64, 96)
    assert torch.isfinite(depth_b1hw).all()


def test_released_zipdepth_state_loads_strictly_into_backbone(tmp_path: Path) -> None:
    released: ZipDepth = create_model(variant="base")
    checkpoint: Path = tmp_path / "zipdepth_base.pth"
    torch.save(released.state_dict(), checkpoint)

    prompted: ZipDepthPrompt = load_zipdepth_prompt(checkpoint)

    prompted_backbone_state: dict[str, Tensor] = prompted.backbone.state_dict()
    released_state: dict[str, Tensor] = released.state_dict()
    assert prompted_backbone_state.keys() == released_state.keys()
    for key, released_tensor in released_state.items():
        assert torch.equal(prompted_backbone_state[key], released_tensor), key
