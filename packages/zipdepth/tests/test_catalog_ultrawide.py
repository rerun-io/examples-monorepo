"""Ultrawide prompt placement, mask policy, sampling, and wide-lane bit-identity."""

import numpy as np
import pytest
import torch
from jaxtyping import Bool, Float32, UInt8, UInt16
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal
from torch import Tensor

from zipdepth.catalog.ultrawide import (
    DEFAULT_ULTRAWIDE_PROMPT_SCALE,
    PROMPT_CANVAS_HW,
    ULTRAWIDE_FOV_RATIO,
    PromptPlacement,
    UltrawidePolicy,
    erode_valid,
    footprint_box,
    interleave,
    place_wide_prompt,
    prompt_placement,
    subsample_indices,
    valid_fraction,
)

pytest.importorskip("arkitscenes_download", reason="ARKitScenes catalog dependencies live in the zipdepth catalog lane")

from arkitscenes_download.ingest.depth import encode_depth_png  # noqa: E402

from zipdepth.catalog.builders import CpuSampleBuilder  # noqa: E402
from zipdepth.catalog.targets import build_eval_transform  # noqa: E402

AGGRESSIVE_POLICY: UltrawidePolicy = UltrawidePolicy(min_valid_fraction=1.0, valid_erosion_px=8, prompt_scale=0.25)
"""A policy extreme enough that any leak into the wide lane changes its output."""


def test_prompt_placement_centres_the_measured_ultrawide_footprint() -> None:
    """Scale the canvas by the measured field-of-view ratio and centre the block."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)

    assert (placement.height, placement.width) == (91, 122)
    assert (placement.top, placement.left) == (50, 67)
    # Horizontal margins are equal, so mirroring the canvas and mirroring the
    # block land on the same pixels at the default scale.
    assert PROMPT_CANVAS_HW[1] - placement.left - placement.width == placement.left
    assert pytest.approx(1.0 / ULTRAWIDE_FOV_RATIO) == DEFAULT_ULTRAWIDE_PROMPT_SCALE


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5, 1.0e-4])
def test_prompt_placement_rejects_scales_outside_the_usable_range(scale: float) -> None:
    """Reject a non-positive, oversized, or vanishing prompt scale."""
    with pytest.raises(ValueError):
        prompt_placement(scale)


def test_ultrawide_policy_rejects_impossible_settings() -> None:
    """Reject a fraction outside the unit interval and a negative erosion radius."""
    with pytest.raises(ValueError):
        UltrawidePolicy(min_valid_fraction=1.5)
    with pytest.raises(ValueError):
        UltrawidePolicy(valid_erosion_px=-1)
    with pytest.raises(ValueError):
        UltrawidePolicy(prompt_scale=0.0)


def test_place_wide_prompt_scales_the_block_and_zeroes_the_rest() -> None:
    """Write the resampled prompt inside the footprint and leave zeros outside."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)
    prompt_mm_hw: UInt16[Tensor, "192 256"] = torch.full(PROMPT_CANVAS_HW, 1500, dtype=torch.int32)
    confidence_hw: UInt8[Tensor, "192 256"] = torch.full(PROMPT_CANVAS_HW, 2, dtype=torch.uint8)

    placed: tuple[Float32[Tensor, "192 256"], Bool[Tensor, "192 256"]] = place_wide_prompt(
        prompt_mm_hw, confidence_hw, placement, flipped=False
    )
    depth_hw: Float32[Tensor, "192 256"] = placed[0]
    valid_hw: Bool[Tensor, "192 256"] = placed[1]
    inside_hw: Bool[Tensor, "192 256"] = torch.zeros(PROMPT_CANVAS_HW, dtype=torch.bool)
    inside_hw[placement.top : placement.top + placement.height, placement.left : placement.left + placement.width] = True

    assert torch.equal(valid_hw, inside_hw)
    assert torch.all(depth_hw[inside_hw] == 1.5)
    assert torch.all(depth_hw[~inside_hw] == 0.0)


def test_place_wide_prompt_mirrors_the_block_not_the_canvas() -> None:
    """Mirror the scaled block so a flip stays exact for any canvas margin."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)
    prompt_mm_hw: UInt16[Tensor, "192 256"] = (
        torch.arange(PROMPT_CANVAS_HW[0] * PROMPT_CANVAS_HW[1], dtype=torch.int32).reshape(PROMPT_CANVAS_HW) % 3000 + 200
    )
    confidence_hw: UInt8[Tensor, "192 256"] = torch.full(PROMPT_CANVAS_HW, 2, dtype=torch.uint8)

    upright: tuple[Float32[Tensor, "192 256"], Bool[Tensor, "192 256"]] = place_wide_prompt(
        prompt_mm_hw, confidence_hw, placement, flipped=False
    )
    flipped: tuple[Float32[Tensor, "192 256"], Bool[Tensor, "192 256"]] = place_wide_prompt(
        prompt_mm_hw, confidence_hw, placement, flipped=True
    )

    assert torch.equal(flipped[0], torch.flip(upright[0], dims=(-1,)))
    assert torch.equal(flipped[1], torch.flip(upright[1], dims=(-1,)))


def test_place_wide_prompt_rejects_a_prompt_off_the_canvas_grid() -> None:
    """Reject prompt planes that are not already on the canvas grid."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)
    with pytest.raises(ValueError):
        place_wide_prompt(
            torch.zeros((256, 192), dtype=torch.int32),
            torch.zeros((256, 192), dtype=torch.uint8),
            placement,
            flipped=False,
        )


def test_erode_valid_removes_only_the_boundary_ring() -> None:
    """Erode one pixel from a solid block and leave the interior untouched."""
    valid_chw: Bool[Tensor, "1 h w"] = torch.zeros((1, 7, 7), dtype=torch.bool)
    valid_chw[0, 2:5, 2:5] = True

    eroded_chw: Bool[Tensor, "1 h w"] = erode_valid(valid_chw, 1)

    expected_chw: Bool[Tensor, "1 h w"] = torch.zeros((1, 7, 7), dtype=torch.bool)
    expected_chw[0, 3, 3] = True
    assert torch.equal(eroded_chw, expected_chw)


def test_erode_valid_keeps_the_image_border() -> None:
    """Leave a fully valid frame untouched: only real holes erode inward."""
    valid_chw: Bool[Tensor, "1 h w"] = torch.ones((1, 5, 6), dtype=torch.bool)

    assert torch.equal(erode_valid(valid_chw, 2), valid_chw)


def test_erode_valid_zero_radius_is_the_identity() -> None:
    """Return the input mask unchanged when no erosion is configured."""
    valid_chw: Bool[Tensor, "1 h w"] = torch.rand((1, 4, 5)) > 0.5

    assert erode_valid(valid_chw, 0) is valid_chw
    with pytest.raises(ValueError):
        erode_valid(valid_chw, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA erosion parity needs a GPU")
def test_erode_valid_matches_between_cpu_and_cuda() -> None:
    """Run the same pooling kernel on both devices and get the same mask."""
    valid_chw: Bool[Tensor, "1 h w"] = torch.rand((1, 33, 41), generator=torch.Generator().manual_seed(5)) > 0.3

    eroded_cpu_chw: Bool[Tensor, "1 h w"] = erode_valid(valid_chw, 2)
    eroded_cuda_chw: Bool[Tensor, "1 h w"] = erode_valid(valid_chw.to(device="cuda"), 2)

    assert torch.equal(eroded_cuda_chw.cpu(), eroded_cpu_chw)


def test_valid_fraction_counts_the_mask() -> None:
    """Report the fraction of the mask that is valid."""
    valid_chw: Bool[Tensor, "1 h w"] = torch.zeros((1, 2, 5), dtype=torch.bool)
    valid_chw[0, 0, :3] = True

    assert valid_fraction(valid_chw) == pytest.approx(0.3)


def test_footprint_box_scales_the_placement_to_the_output() -> None:
    """Map the canvas block onto the training output resolution."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)

    box: tuple[int, int, int, int] = footprint_box(placement, (768, 1024))

    assert box == (200, 268, 564, 756)
    assert (box[2] - box[0]) / 768 == pytest.approx(placement.height / PROMPT_CANVAS_HW[0], abs=1.0e-3)
    assert (box[3] - box[1]) / 1024 == pytest.approx(placement.width / PROMPT_CANVAS_HW[1], abs=1.0e-3)


def test_subsample_indices_is_seeded_sorted_and_bounded() -> None:
    """Draw a reproducible, ordered subset that a fresh seed changes."""
    first: list[int] = subsample_indices(100, 30, seed=11)
    again: list[int] = subsample_indices(100, 30, seed=11)
    other: list[int] = subsample_indices(100, 30, seed=12)

    assert first == again
    assert first != other
    assert first == sorted(first)
    assert len(set(first)) == 30
    assert max(first) < 100
    assert subsample_indices(5, 9, seed=1) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        subsample_indices(-1, 1, seed=1)


def test_interleave_alternates_until_every_source_is_drained() -> None:
    """Round-robin the sources and keep every item exactly once."""
    assert list(interleave([iter([1, 3, 5]), iter([2, 4, 6])])) == [1, 2, 3, 4, 5, 6]
    assert list(interleave([iter([1, 2, 3]), iter(["a"])])) == [1, "a", 2, 3]
    assert list(interleave([])) == []


FRAME_HW: tuple[int, int] = (48, 64)
"""Synthetic frame size used by the builder tests; large enough for a visible hole."""

BuilderInputs = tuple[UInt8[Tensor, "3 h w"], UInt8[ndarray, "target_n"], UInt8[ndarray, "prompt_n"], UInt8[ndarray, "confidence_n"]]
"""One synthetic frame, target blob, prompt blob, and flattened confidence."""


def _builder_inputs() -> BuilderInputs:
    """Build one synthetic frame whose target carries a hole the erosion can widen."""
    rgb_chw: UInt8[Tensor, "3 h w"] = torch.arange(3 * FRAME_HW[0] * FRAME_HW[1], dtype=torch.int32).remainder(256).to(
        dtype=torch.uint8
    ).reshape(3, *FRAME_HW)
    target_mm_hw: UInt16[ndarray, "h w"] = (
        np.arange(FRAME_HW[0] * FRAME_HW[1], dtype=np.uint16).reshape(FRAME_HW) % 3000 + 300
    ).astype(np.uint16)
    target_mm_hw[10:20, 12:30] = 0
    target_blob_bytes: UInt8[ndarray, "target_n"] = np.frombuffer(encode_depth_png(target_mm_hw), dtype=np.uint8)
    prompt_mm_hw: UInt16[ndarray, "192 256"] = (np.arange(192 * 256, dtype=np.uint16).reshape(192, 256) % 3000 + 300).astype(np.uint16)
    prompt_blob_bytes: UInt8[ndarray, "prompt_n"] = np.frombuffer(encode_depth_png(prompt_mm_hw), dtype=np.uint8)
    confidence: UInt8[ndarray, "confidence_n"] = np.full(192 * 256, 2, dtype=np.uint8)
    return rgb_chw, target_blob_bytes, prompt_blob_bytes, confidence


def test_wide_samples_are_bit_identical_with_and_without_an_ultrawide_policy() -> None:
    """Prove the wide lane never consults the ultrawide policy.

    The second builder carries a policy that would reject every frame and erode
    eight pixels if the wide path touched it, so an identical sample is proof
    that ``cameras="wide"`` still produces the released lane's tensors.
    """
    inputs: BuilderInputs = _builder_inputs()
    baseline: CpuSampleBuilder = CpuSampleBuilder(build_eval_transform(*FRAME_HW), min_depth_span=0.0, target_mode="metric")
    with_policy: CpuSampleBuilder = CpuSampleBuilder(
        build_eval_transform(*FRAME_HW), min_depth_span=0.0, target_mode="metric", ultrawide_policy=AGGRESSIVE_POLICY
    )

    expected: dict[str, Tensor] | None = baseline(*inputs, quarter_turns=0, sample_seed=7)
    actual: dict[str, Tensor] | None = with_policy(*inputs, quarter_turns=0, sample_seed=7)

    assert expected is not None
    assert actual is not None
    assert sorted(actual) == sorted(expected)
    key: str
    for key in expected:
        assert_array_equal(actual[key].numpy(), expected[key].numpy(), err_msg=f"{key} changed in the wide lane")


def test_ultrawide_camera_without_a_policy_is_rejected() -> None:
    """Refuse an ultrawide sample from a builder that has no ultrawide policy."""
    inputs: BuilderInputs = _builder_inputs()
    builder: CpuSampleBuilder = CpuSampleBuilder(build_eval_transform(*FRAME_HW), min_depth_span=0.0, target_mode="metric")

    with pytest.raises(ValueError):
        builder(*inputs, quarter_turns=0, sample_seed=1, camera="ultrawide")


def test_ultrawide_sample_pads_the_prompt_and_erodes_the_target_mask() -> None:
    """Place the prompt in its footprint and erode the surviving target mask."""
    inputs: BuilderInputs = _builder_inputs()
    policy: UltrawidePolicy = UltrawidePolicy(min_valid_fraction=0.0, valid_erosion_px=1)
    builder: CpuSampleBuilder = CpuSampleBuilder(
        build_eval_transform(*FRAME_HW), min_depth_span=0.0, target_mode="metric", ultrawide_policy=policy
    )
    placement: PromptPlacement = prompt_placement(policy.prompt_scale)

    wide_sample: dict[str, Tensor] | None = builder(*inputs, quarter_turns=0, sample_seed=3)
    ultrawide_sample: dict[str, Tensor] | None = builder(*inputs, quarter_turns=0, sample_seed=3, camera="ultrawide")

    assert wide_sample is not None
    assert ultrawide_sample is not None
    # Only the prompt and the target mask differ; the image and target depth do not.
    assert_array_equal(ultrawide_sample["image"].numpy(), wide_sample["image"].numpy())
    assert_array_equal(ultrawide_sample["target_depth"].numpy(), wide_sample["target_depth"].numpy())
    assert bool(wide_sample["prompt_valid"].all())
    placed_valid_hw: Bool[ndarray, "192 256"] = ultrawide_sample["prompt_valid"][0].numpy()
    assert not placed_valid_hw[: placement.top].any()
    assert placed_valid_hw[placement.top : placement.top + placement.height, placement.left : placement.left + placement.width].all()
    assert placed_valid_hw.mean() == pytest.approx(placement.height * placement.width / (192 * 256))
    # The eroded ultrawide mask is a strict subset of the wide one: the target's
    # hole grows by one pixel on every side.
    wide_valid_hw: Bool[ndarray, "h w"] = wide_sample["target_valid"][0].numpy()
    ultrawide_valid_hw: Bool[ndarray, "h w"] = ultrawide_sample["target_valid"][0].numpy()
    assert not bool((ultrawide_valid_hw & ~wide_valid_hw).any())
    assert int(ultrawide_valid_hw.sum()) < int(wide_valid_hw.sum())


def test_ultrawide_sample_drops_a_frame_below_the_valid_fraction() -> None:
    """Reject a sparse ultrawide frame and count it separately from flat frames."""
    rgb_chw: UInt8[Tensor, "3 h w"] = torch.zeros((3, *FRAME_HW), dtype=torch.uint8)
    target_mm_hw: UInt16[ndarray, "h w"] = np.zeros(FRAME_HW, dtype=np.uint16)
    target_mm_hw[:8] = 1500
    target_blob_bytes: UInt8[ndarray, "target_n"] = np.frombuffer(encode_depth_png(target_mm_hw), dtype=np.uint8)
    prompt_mm_hw: UInt16[ndarray, "192 256"] = np.full((192, 256), 1000, dtype=np.uint16)
    prompt_blob_bytes: UInt8[ndarray, "prompt_n"] = np.frombuffer(encode_depth_png(prompt_mm_hw), dtype=np.uint8)
    confidence: UInt8[ndarray, "confidence_n"] = np.full(192 * 256, 2, dtype=np.uint8)
    builder: CpuSampleBuilder = CpuSampleBuilder(
        build_eval_transform(*FRAME_HW),
        min_depth_span=0.0,
        target_mode="metric",
        ultrawide_policy=UltrawidePolicy(min_valid_fraction=0.7, valid_erosion_px=0),
    )

    sample: dict[str, Tensor] | None = builder(
        rgb_chw, target_blob_bytes, prompt_blob_bytes, confidence, quarter_turns=0, sample_seed=1, camera="ultrawide"
    )

    assert sample is None
    assert builder.stats.skipped_low_valid_frames == 1
    assert builder.stats.skipped_flat_frames == 0
    assert builder.stats.samples_built == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA builder parity needs a GPU")
def test_cuda_and_cpu_builders_place_the_same_ultrawide_prompt() -> None:
    """Match the CPU builder's placed prompt exactly on the CUDA builder."""
    from zipdepth.catalog.builders import CudaSampleBuilder
    from zipdepth.catalog.targets import AugmentPolicy

    inputs: BuilderInputs = _builder_inputs()
    policy: UltrawidePolicy = UltrawidePolicy(min_valid_fraction=0.0, valid_erosion_px=1)
    no_augmentation: AugmentPolicy = AugmentPolicy(flip_p=0.0, channel_shuffle_p=0.0, gray_p=0.0)
    cpu_builder: CpuSampleBuilder = CpuSampleBuilder(
        build_eval_transform(*FRAME_HW), min_depth_span=0.0, target_mode="metric", ultrawide_policy=policy
    )
    cuda_builder: CudaSampleBuilder = CudaSampleBuilder(
        FRAME_HW, no_augmentation, 0.0, torch.device("cuda"), target_mode="metric", ultrawide_policy=policy
    )

    expected: dict[str, Tensor] | None = cpu_builder(*inputs, quarter_turns=0, sample_seed=3, camera="ultrawide")
    actual: dict[str, Tensor] | None = cuda_builder(
        inputs[0].to(device="cuda"), inputs[1], inputs[2], inputs[3], quarter_turns=0, sample_seed=3, camera="ultrawide"
    )

    assert expected is not None
    assert actual is not None
    # Placement, resampling, and masking agree exactly; the metre conversion is a
    # float32 divide, whose last bit differs between the CPU and CUDA kernels.
    assert_allclose(actual["prompt_depth"].cpu().numpy(), expected["prompt_depth"].numpy(), rtol=1.0e-6, atol=0.0)
    assert_array_equal(actual["prompt_valid"].cpu().numpy(), expected["prompt_valid"].numpy())
    assert_array_equal(actual["target_valid"].cpu().numpy(), expected["target_valid"].numpy())
