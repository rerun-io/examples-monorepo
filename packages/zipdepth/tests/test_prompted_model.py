"""Behavior tests for the prompted metric ZipDepth wrapper."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest
import torch
from jaxtyping import Bool, Float32, UInt8
from monopriors.models.depth_completion.base_completion_depth import MAX_METRIC_DEPTH_M, MIN_METRIC_DEPTH_M
from monopriors.models.depth_completion.zipdepth_prompt import ZipDepthPrompt, load_zipdepth_prompt
from monopriors.models.zipdepth_checkpoint import RANGE_MARGIN_COVERAGE_MAX_KEY, RANGE_MARGIN_M_KEY
from monopriors.third_party.zipdepth.architecture import FastConvexUpsample, ZipDepth, create_model
from torch import Tensor

from zipdepth.catalog.ultrawide import DEFAULT_ULTRAWIDE_PROMPT_SCALE, PromptPlacement, prompt_placement

PromptRange: TypeAlias = tuple[Float32[Tensor, "b 1 1 1"], Float32[Tensor, "b 1 1 1"]]
ModelOutputs: TypeAlias = tuple[Float32[Tensor, "b 1 h w"], Float32[Tensor, "b 1 1 1"], Float32[Tensor, "b 1 1 1"]]


def _prompt_with_fixed_range(*, flip: bool = False) -> Float32[Tensor, "1 1 192 256"]:
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = torch.linspace(0.2, 3.8, 192 * 256, dtype=torch.float32).reshape(1, 1, 192, 256)
    if flip:
        prompt_bchw = torch.flip(prompt_bchw, dims=(-1,))
    prompt_bchw[:, :, 40:80, 60:120] = 0.0
    return prompt_bchw


def _prompt_bounded_range(prompt_bchw: Float32[Tensor, "b 1 192 256"]) -> PromptRange:
    """Return the pre-margin per-image prompt range, verbatim from the original head."""
    valid_bchw: Bool[Tensor, "b 1 192 256"] = (
        torch.isfinite(prompt_bchw) & (prompt_bchw >= MIN_METRIC_DEPTH_M) & (prompt_bchw <= MAX_METRIC_DEPTH_M)
    )
    valid_any_b: Bool[Tensor, "b 1 1 1"] = valid_bchw.any(dim=(1, 2, 3), keepdim=True)
    masked_min_b: Float32[Tensor, "b 1 1 1"] = torch.where(valid_bchw, prompt_bchw, torch.inf).amin(dim=(1, 2, 3), keepdim=True)
    masked_max_b: Float32[Tensor, "b 1 1 1"] = torch.where(valid_bchw, prompt_bchw, -torch.inf).amax(dim=(1, 2, 3), keepdim=True)
    min_depth_b: Float32[Tensor, "b 1 1 1"] = torch.where(valid_any_b, masked_min_b, MIN_METRIC_DEPTH_M)
    max_depth_b: Float32[Tensor, "b 1 1 1"] = torch.where(valid_any_b, masked_max_b, MAX_METRIC_DEPTH_M)
    return min_depth_b, max_depth_b


def _run_capturing_logits(
    model: ZipDepthPrompt,
    image_bchw: Float32[Tensor, "b 3 h w"],
    prompt_bchw: Float32[Tensor, "b 1 192 256"],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModelOutputs, Float32[Tensor, "b 1 h w"]]:
    """Run the model and also return the pre-sigmoid depth logits it rescaled.

    The convex upsampler is called as a plain method rather than through
    ``__call__``, so no forward hook sees its output; wrapping the method is the
    only way to hold the exact logits the output range is applied to.
    """
    convex_up: FastConvexUpsample = model.backbone.decoder.convex_up
    original_unfold: Callable[[Tensor, Tensor], Tensor] = convex_up._forward_unfold
    captured: list[Float32[Tensor, "b 1 h w"]] = []

    def capturing_unfold(feature_bchw: Tensor, depth_logits_bchw: Tensor) -> Float32[Tensor, "b 1 h w"]:
        logits_bchw: Float32[Tensor, "b 1 h w"] = original_unfold(feature_bchw, depth_logits_bchw)
        captured.append(logits_bchw)
        return logits_bchw

    monkeypatch.setattr(convex_up, "_forward_unfold", capturing_unfold)
    with torch.inference_mode():
        outputs: ModelOutputs = model.forward_with_range(image_bchw, prompt_bchw)
    monkeypatch.undo()
    return outputs, captured[0]


def _prompted_model(range_margin_m: float, *, range_margin_coverage_max: float = 1.0, seed: int = 0) -> ZipDepthPrompt:
    """Build a deterministically initialized prompted model with live prompt branches."""
    torch.manual_seed(seed)
    model: ZipDepthPrompt = ZipDepthPrompt(
        range_margin_m=range_margin_m, range_margin_coverage_max=range_margin_coverage_max
    ).eval()
    # The released initialization zeroes every branch output, which would hide any
    # change in the prompt conditioning; give the branches a signal to carry.
    for parameter in model.prompt_branches.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    return model


def test_zero_initialized_prompt_injection_has_no_spatial_contribution() -> None:
    model: ZipDepthPrompt = ZipDepthPrompt().eval()
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)

    with torch.inference_mode():
        first_depth_bchw: Float32[Tensor, "1 1 64 96"] = model(image_bchw, _prompt_with_fixed_range())
        second_depth_bchw: Float32[Tensor, "1 1 64 96"] = model(image_bchw, _prompt_with_fixed_range(flip=True))

    assert torch.equal(first_depth_bchw, second_depth_bchw)


def test_prompted_model_accepts_raw_and_normalized_rgb() -> None:
    model: ZipDepthPrompt = ZipDepthPrompt().eval()
    raw_image_bchw: UInt8[Tensor, "1 3 64 96"] = torch.randint(0, 256, (1, 3, 64, 96), dtype=torch.uint8)
    normalized_image_bchw: Float32[Tensor, "1 3 64 96"] = raw_image_bchw.to(dtype=torch.float32) / 255.0
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = _prompt_with_fixed_range()

    with torch.inference_mode():
        raw_input_depth_bchw: Float32[Tensor, "1 1 64 96"] = model(raw_image_bchw, prompt_bchw)
        normalized_input_depth_bchw: Float32[Tensor, "1 1 64 96"] = model(normalized_image_bchw, prompt_bchw)

    assert raw_input_depth_bchw.shape == (1, 1, 64, 96)
    torch.testing.assert_close(raw_input_depth_bchw, normalized_input_depth_bchw, rtol=0.0, atol=0.0)
    assert float(raw_input_depth_bchw.min()) >= 0.2
    assert float(raw_input_depth_bchw.max()) <= 3.8


def test_degenerate_prompts_produce_finite_metric_depth() -> None:
    model: ZipDepthPrompt = ZipDepthPrompt().eval()
    image_bchw: Float32[Tensor, "2 3 64 96"] = torch.rand(2, 3, 64, 96)
    prompt_bchw: Float32[Tensor, "2 1 192 256"] = torch.zeros(2, 1, 192, 256)
    prompt_bchw[0] = 1.25

    with torch.inference_mode():
        depth_bchw: Float32[Tensor, "2 1 64 96"] = model(image_bchw, prompt_bchw)

    assert depth_bchw.shape == (2, 1, 64, 96)
    assert torch.isfinite(depth_bchw).all()


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



def _prompt_spanning(low_m: float, high_m: float) -> Float32[Tensor, "1 1 192 256"]:
    """Return an otherwise-invalid prompt whose valid pixels span exactly ``[low_m, high_m]``."""
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = torch.zeros(1, 1, 192, 256)
    prompt_bchw[:, :, 80:112, 96:160] = torch.linspace(low_m, high_m, 32 * 64).reshape(1, 1, 32, 64)
    return prompt_bchw


def _full_canvas_prompt(low_m: float, high_m: float) -> Float32[Tensor, "1 1 192 256"]:
    """Return a wide-lane prompt: every canvas pixel valid, spanning ``[low_m, high_m]``."""
    return torch.linspace(low_m, high_m, 192 * 256, dtype=torch.float32).reshape(1, 1, 192, 256)


def _padded_ultrawide_prompt(low_m: float, high_m: float) -> Float32[Tensor, "1 1 192 256"]:
    """Return the wide LiDAR block padded into its ultrawide footprint, zeros elsewhere."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = torch.zeros(1, 1, 192, 256)
    block_hw: Float32[Tensor, "1 1 block_h block_w"] = torch.linspace(
        low_m, high_m, placement.height * placement.width, dtype=torch.float32
    ).reshape(1, 1, placement.height, placement.width)
    prompt_bchw[
        :,
        :,
        placement.top : placement.top + placement.height,
        placement.left : placement.left + placement.width,
    ] = block_hw
    return prompt_bchw


def _prompt_coverage(prompt_bchw: Float32[Tensor, "1 1 192 256"]) -> float:
    """Return the fraction of the canvas the model counts as a valid prompt."""
    valid_bchw: Bool[Tensor, "1 1 192 256"] = (
        torch.isfinite(prompt_bchw) & (prompt_bchw >= MIN_METRIC_DEPTH_M) & (prompt_bchw <= MAX_METRIC_DEPTH_M)
    )
    return float(valid_bchw.to(dtype=torch.float32).mean())


def test_zero_range_margin_reproduces_the_prompt_bounded_head_bit_for_bit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee the default margin leaves today's forward pass untouched."""
    model: ZipDepthPrompt = _prompted_model(0.0)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "2 3 64 96"] = torch.rand(2, 3, 64, 96)
    prompt_bchw: Float32[Tensor, "2 1 192 256"] = torch.cat([_prompt_with_fixed_range(), _prompt_with_fixed_range(flip=True)])

    outputs: ModelOutputs
    logits_bchw: Float32[Tensor, "2 1 64 96"]
    outputs, logits_bchw = _run_capturing_logits(model, image_bchw, prompt_bchw, monkeypatch)

    reference_range: PromptRange = _prompt_bounded_range(prompt_bchw)
    reference_min_b: Float32[Tensor, "2 1 1 1"] = reference_range[0]
    reference_max_b: Float32[Tensor, "2 1 1 1"] = reference_range[1]
    reference_span_b: Float32[Tensor, "2 1 1 1"] = (reference_max_b - reference_min_b).clamp_min(1.0e-6)
    reference_depth_bchw: Float32[Tensor, "2 1 64 96"] = torch.sigmoid(logits_bchw) * reference_span_b + reference_min_b

    assert torch.equal(outputs[1], reference_min_b)
    assert torch.equal(outputs[2], reference_max_b)
    assert torch.equal(outputs[0], reference_depth_bchw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_zero_range_margin_parity_holds_under_cuda_float16_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeat the parity guarantee in the reduced precision the export graph uses."""
    model: ZipDepthPrompt = _prompted_model(0.0).cuda()
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96, device="cuda")
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = _prompt_with_fixed_range().cuda()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs: ModelOutputs
        logits_bchw: Float32[Tensor, "1 1 64 96"]
        outputs, logits_bchw = _run_capturing_logits(model, image_bchw, prompt_bchw, monkeypatch)

    reference_range: PromptRange = _prompt_bounded_range(prompt_bchw)
    reference_min_b: Float32[Tensor, "1 1 1 1"] = reference_range[0]
    reference_max_b: Float32[Tensor, "1 1 1 1"] = reference_range[1]
    reference_span_b: Float32[Tensor, "1 1 1 1"] = (reference_max_b - reference_min_b).clamp_min(1.0e-6)
    reference_depth_bchw: Float32[Tensor, "1 1 64 96"] = torch.sigmoid(logits_bchw) * reference_span_b + reference_min_b

    assert torch.equal(outputs[1], reference_min_b)
    assert torch.equal(outputs[2], reference_max_b)
    assert torch.equal(outputs[0], reference_depth_bchw)


def test_range_margin_widens_the_reachable_output_range_by_absolute_metres(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the periphery of an ultrawide frame leave the prompt's own range."""
    model: ZipDepthPrompt = _prompted_model(0.5)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = _prompt_spanning(1.0, 1.4)

    outputs: ModelOutputs
    logits_bchw: Float32[Tensor, "1 1 64 96"]
    outputs, logits_bchw = _run_capturing_logits(model, image_bchw, prompt_bchw, monkeypatch)

    # 0.5 m on each side of [1.0, 1.4], not 0.5 spans (which would only reach [0.8, 1.6]).
    expected_min_b: Float32[Tensor, "1 1 1 1"] = torch.full((1, 1, 1, 1), 0.5)
    expected_max_b: Float32[Tensor, "1 1 1 1"] = torch.full((1, 1, 1, 1), 1.9)
    expected_depth_bchw: Float32[Tensor, "1 1 64 96"] = torch.sigmoid(logits_bchw) * (expected_max_b - expected_min_b) + expected_min_b

    torch.testing.assert_close(outputs[1], expected_min_b)
    torch.testing.assert_close(outputs[2], expected_max_b)
    torch.testing.assert_close(outputs[0], expected_depth_bchw)


def test_range_margin_widens_a_narrow_prompt_as_much_as_a_wide_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a close-up prompt the same extra reach a span-relative margin would deny it."""
    model: ZipDepthPrompt = _prompted_model(0.5)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)

    narrow: ModelOutputs = _run_capturing_logits(model, image_bchw, _prompt_spanning(1.0, 1.1), monkeypatch)[0]
    wide: ModelOutputs = _run_capturing_logits(model, image_bchw, _prompt_spanning(1.0, 2.0), monkeypatch)[0]

    assert float(narrow[1]) == pytest.approx(0.5)
    assert float(wide[1]) == pytest.approx(0.5)
    assert float(narrow[2]) == pytest.approx(1.6)
    assert float(wide[2]) == pytest.approx(2.5)


def test_range_margin_never_leaves_the_shared_metric_validity_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clip the widened range to the 0.1--4.0 m window the family agrees on."""
    model: ZipDepthPrompt = _prompted_model(0.5)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "2 3 64 96"] = torch.rand(2, 3, 64, 96)
    # Image 1 has no valid prompt pixel at all, so its range starts at the window edges.
    prompt_bchw: Float32[Tensor, "2 1 192 256"] = torch.cat([_prompt_spanning(0.3, 3.8), torch.zeros(1, 1, 192, 256)])

    outputs: ModelOutputs = _run_capturing_logits(model, image_bchw, prompt_bchw, monkeypatch)[0]

    assert outputs[1].flatten().tolist() == pytest.approx([MIN_METRIC_DEPTH_M, MIN_METRIC_DEPTH_M])
    assert outputs[2].flatten().tolist() == pytest.approx([MAX_METRIC_DEPTH_M, MAX_METRIC_DEPTH_M])
    assert torch.isfinite(outputs[0]).all()
    assert float(outputs[0].min()) >= MIN_METRIC_DEPTH_M
    assert float(outputs[0].max()) <= MAX_METRIC_DEPTH_M


def test_a_margin_of_the_full_window_width_opens_the_whole_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Document the ultrawide fine-tune setting: 3.9 m reaches every valid depth."""
    model: ZipDepthPrompt = _prompted_model(MAX_METRIC_DEPTH_M - MIN_METRIC_DEPTH_M)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)

    outputs: ModelOutputs = _run_capturing_logits(model, image_bchw, _prompt_spanning(1.0, 1.1), monkeypatch)[0]

    assert float(outputs[1]) == pytest.approx(MIN_METRIC_DEPTH_M)
    assert float(outputs[2]) == pytest.approx(MAX_METRIC_DEPTH_M)


def test_range_margin_leaves_the_prompt_conditioning_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalize the injected prompt over its own range, not the widened one."""
    narrow: ZipDepthPrompt = _prompted_model(0.0)
    wide: ZipDepthPrompt = _prompted_model(0.5)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = _prompt_with_fixed_range()

    narrow_logits_bchw: Float32[Tensor, "1 1 64 96"] = _run_capturing_logits(narrow, image_bchw, prompt_bchw, monkeypatch)[1]
    wide_logits_bchw: Float32[Tensor, "1 1 64 96"] = _run_capturing_logits(wide, image_bchw, prompt_bchw, monkeypatch)[1]

    assert torch.equal(narrow_logits_bchw, wide_logits_bchw)


def test_range_margin_rejects_negative_and_non_finite_values() -> None:
    """Fail loudly instead of inverting or poisoning the output range."""
    with pytest.raises(ValueError, match="range_margin_m"):
        ZipDepthPrompt(range_margin_m=-0.1)
    with pytest.raises(ValueError, match="range_margin_m"):
        ZipDepthPrompt(range_margin_m=float("nan"))


def test_trainer_checkpoints_carry_the_range_margin_to_inference(tmp_path: Path) -> None:
    """Rebuild the trained head from the checkpoint without repeating the flag."""
    trained: ZipDepthPrompt = ZipDepthPrompt(range_margin_m=3.9)
    checkpoint: Path = tmp_path / "final_model.pth"
    torch.save({"model_state_dict": trained.state_dict(), RANGE_MARGIN_M_KEY: 3.9}, checkpoint)

    assert load_zipdepth_prompt(checkpoint).range_margin_m == 3.9
    assert load_zipdepth_prompt(checkpoint, range_margin_m=0.0).range_margin_m == 0.0


def test_checkpoints_without_a_recorded_margin_stay_prompt_bounded(tmp_path: Path) -> None:
    """Keep zdpda-v4 and every bare released state dict on the original head."""
    released: ZipDepth = create_model(variant="base")
    bare: Path = tmp_path / "zipdepth_base.pth"
    torch.save(released.state_dict(), bare)
    legacy: Path = tmp_path / "zdpda_v4.pth"
    torch.save({"model_state_dict": ZipDepthPrompt().state_dict(), "global_step": 70_000}, legacy)

    assert load_zipdepth_prompt(bare).range_margin_m == 0.0
    assert load_zipdepth_prompt(legacy).range_margin_m == 0.0


def test_the_ultrawide_and_wide_prompts_sit_on_opposite_sides_of_the_gate() -> None:
    """Anchor the 0.6 threshold in the two coverages the ``both`` lane actually produces."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)

    assert (placement.height, placement.width) == (91, 122)
    assert _prompt_coverage(_full_canvas_prompt(1.0, 1.4)) == 1.0
    # 91 * 122 / (192 * 256): the whole wide field of view inside an ultrawide frame.
    assert _prompt_coverage(_padded_ultrawide_prompt(1.0, 1.4)) == pytest.approx(0.2259, abs=1e-4)


def test_a_gated_margin_leaves_a_fully_prompted_frame_bit_identical() -> None:
    """Keep the wide lane on exactly the head v4 trained, tensor for tensor.

    An unconditional 3.9 m margin broke this lane (AbsRel 0.19 against 0.013),
    so the gate has to be an identity for a full-canvas prompt, not merely close.
    """
    bounded: ZipDepthPrompt = _prompted_model(0.0)
    gated: ZipDepthPrompt = _prompted_model(MAX_METRIC_DEPTH_M - MIN_METRIC_DEPTH_M, range_margin_coverage_max=0.6)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)
    prompt_bchw: Float32[Tensor, "1 1 192 256"] = _full_canvas_prompt(1.0, 1.4)

    with torch.inference_mode():
        bounded_outputs: ModelOutputs = bounded.forward_with_range(image_bchw, prompt_bchw)
        gated_outputs: ModelOutputs = gated.forward_with_range(image_bchw, prompt_bchw)

    assert torch.equal(gated_outputs[1], bounded_outputs[1])
    assert torch.equal(gated_outputs[2], bounded_outputs[2])
    assert torch.equal(gated_outputs[0], bounded_outputs[0])


def test_a_gated_margin_opens_the_whole_window_for_a_padded_ultrawide_prompt() -> None:
    """Let the ultrawide periphery reach depth the centre prompt never saw."""
    model: ZipDepthPrompt = _prompted_model(MAX_METRIC_DEPTH_M - MIN_METRIC_DEPTH_M, range_margin_coverage_max=0.6)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)

    with torch.inference_mode():
        outputs: ModelOutputs = model.forward_with_range(image_bchw, _padded_ultrawide_prompt(1.0, 1.4))

    assert float(outputs[1]) == pytest.approx(MIN_METRIC_DEPTH_M)
    assert float(outputs[2]) == pytest.approx(MAX_METRIC_DEPTH_M)
    assert float(outputs[0].min()) >= MIN_METRIC_DEPTH_M
    assert float(outputs[0].max()) <= MAX_METRIC_DEPTH_M


def test_the_coverage_gate_decides_per_batch_element() -> None:
    """Train both cameras in one batch: the ``both`` lane interleaves them."""
    model: ZipDepthPrompt = _prompted_model(MAX_METRIC_DEPTH_M - MIN_METRIC_DEPTH_M, range_margin_coverage_max=0.6)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "2 3 64 96"] = torch.rand(2, 3, 64, 96)
    prompt_bchw: Float32[Tensor, "2 1 192 256"] = torch.cat(
        [_full_canvas_prompt(1.0, 1.4), _padded_ultrawide_prompt(1.0, 1.4)]
    )

    with torch.inference_mode():
        outputs: ModelOutputs = model.forward_with_range(image_bchw, prompt_bchw)

    assert outputs[1].flatten().tolist() == pytest.approx([1.0, MIN_METRIC_DEPTH_M])
    assert outputs[2].flatten().tolist() == pytest.approx([1.4, MAX_METRIC_DEPTH_M])
    assert torch.isfinite(outputs[0]).all()


def test_the_default_coverage_ceiling_widens_every_image() -> None:
    """Keep the pre-gate behaviour reachable: 1.0 is the largest coverage there is."""
    model: ZipDepthPrompt = _prompted_model(MAX_METRIC_DEPTH_M - MIN_METRIC_DEPTH_M)
    torch.manual_seed(1)
    image_bchw: Float32[Tensor, "1 3 64 96"] = torch.rand(1, 3, 64, 96)

    with torch.inference_mode():
        outputs: ModelOutputs = model.forward_with_range(image_bchw, _full_canvas_prompt(1.0, 1.4))

    assert model.range_margin_coverage_max == 1.0
    assert float(outputs[1]) == pytest.approx(MIN_METRIC_DEPTH_M)
    assert float(outputs[2]) == pytest.approx(MAX_METRIC_DEPTH_M)


def test_range_margin_coverage_max_rejects_values_outside_the_unit_interval() -> None:
    """A coverage is a fraction; anything else would silently disable or force the margin."""
    with pytest.raises(ValueError, match="range_margin_coverage_max"):
        ZipDepthPrompt(range_margin_coverage_max=-0.1)
    with pytest.raises(ValueError, match="range_margin_coverage_max"):
        ZipDepthPrompt(range_margin_coverage_max=1.5)
    with pytest.raises(ValueError, match="range_margin_coverage_max"):
        ZipDepthPrompt(range_margin_coverage_max=float("nan"))


def test_trainer_checkpoints_carry_the_coverage_gate_to_inference(tmp_path: Path) -> None:
    """Rebuild the trained gate from the checkpoint without repeating the flag."""
    trained: ZipDepthPrompt = ZipDepthPrompt(range_margin_m=3.9, range_margin_coverage_max=0.6)
    checkpoint: Path = tmp_path / "final_model.pth"
    torch.save(
        {
            "model_state_dict": trained.state_dict(),
            RANGE_MARGIN_M_KEY: 3.9,
            RANGE_MARGIN_COVERAGE_MAX_KEY: 0.6,
        },
        checkpoint,
    )

    assert load_zipdepth_prompt(checkpoint).range_margin_coverage_max == 0.6
    assert load_zipdepth_prompt(checkpoint, range_margin_coverage_max=1.0).range_margin_coverage_max == 1.0


def test_checkpoints_without_a_recorded_coverage_gate_widen_every_image(tmp_path: Path) -> None:
    """Keep every pre-gate margin checkpoint on the behaviour it was trained with."""
    legacy: Path = tmp_path / "zdpda_uw_v1.pth"
    torch.save({"model_state_dict": ZipDepthPrompt().state_dict(), RANGE_MARGIN_M_KEY: 3.9}, legacy)

    assert load_zipdepth_prompt(legacy).range_margin_coverage_max == 1.0
