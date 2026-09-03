"""Behavioral tests for catalog depth targets."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from jaxtyping import Bool, Float32, Int16, Int32, UInt8, UInt16
from numpy import ndarray
from numpy.testing import assert_allclose, assert_array_equal
from torch import Tensor

from zipdepth.catalog.targets import (
    INV_DEPTH_SCALE,
    AugmentPolicy,
    build_eval_transform,
    build_metric_training_sample,
    build_train_transform,
    build_training_sample,
    build_training_sample_cuda,
    depth_span_ratio,
    depth_span_ratio_cuda,
    inverse_depth_from_mm,
    quantize_inverse_depth,
)
from zipdepth.data.transforms import AlbumentationsWrapper, TransformResult, get_train_transforms


def test_inverse_depth_masks_holes_and_out_of_range_values() -> None:
    """Keep only 100--4000 mm targets and convert them to inverse metres."""
    depth_mm_hw: UInt16[ndarray, "1 7"] = np.array([[0, 99, 100, 1000, 4000, 4001, 65535]], dtype=np.uint16)

    inverse_depth_hw: Float32[ndarray, "1 7"] = inverse_depth_from_mm(depth_mm_hw)

    assert inverse_depth_hw.dtype == np.float32
    assert_array_equal(inverse_depth_hw, np.array([[0.0, 0.0, 10.0, 1.0, 0.25, 0.0, 0.0]], dtype=np.float32))


def test_augment_policy_is_the_single_probability_source() -> None:
    """Keep CPU and CUDA augmentation defaults in one immutable value."""
    policy: AugmentPolicy = AugmentPolicy()

    assert policy == AugmentPolicy(flip_p=0.5, channel_shuffle_p=0.05, gray_p=0.05)


def test_inverse_depth_quantization_roundtrips_valid_depth_without_overflow() -> None:
    """Keep the full valid metric range within 1% after uint16 quantization."""
    depth_mm_hw: UInt16[ndarray, "1 6"] = np.array([[100, 101, 500, 1000, 3999, 4000]], dtype=np.uint16)
    inverse_depth_hw: Float32[ndarray, "1 6"] = inverse_depth_from_mm(depth_mm_hw)

    quantized_hw: UInt16[ndarray, "1 6"] = quantize_inverse_depth(inverse_depth_hw)
    reconstructed_mm_hw: Float32[ndarray, "1 6"] = 1000.0 / (quantized_hw.astype(np.float32) / INV_DEPTH_SCALE)

    assert quantized_hw.dtype == np.uint16
    assert int(quantized_hw[0, 0]) == 60000
    assert_allclose(reconstructed_mm_hw, depth_mm_hw.astype(np.float32), rtol=0.01)


def test_depth_span_ratio_uses_the_stride_16_spatial_sample() -> None:
    """Estimate p95/p5 from the fixed spatial sample instead of all image pixels."""
    depth_mm_hw: UInt16[ndarray, "32 32"] = np.full((32, 32), 1000, dtype=np.uint16)
    depth_mm_hw[0, 0] = 100
    depth_mm_hw[0, 16] = 200
    depth_mm_hw[16, 0] = 300
    depth_mm_hw[16, 16] = 400

    ratio: float = depth_span_ratio(depth_mm_hw)

    assert ratio == pytest.approx(385.0 / 115.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA depth-span filtering needs a GPU")
def test_depth_span_ratio_cuda_matches_cpu_stride_sample() -> None:
    """Keep the CPU percentile semantics when the flat-frame filter moves to CUDA."""
    depth_mm_hw: UInt16[ndarray, "32 32"] = np.full((32, 32), 1000, dtype=np.uint16)
    depth_mm_hw[0, 0] = 100
    depth_mm_hw[0, 16] = 200
    depth_mm_hw[16, 0] = 300
    depth_mm_hw[16, 16] = 400

    ratio_cuda: Float32[Tensor, ""] = depth_span_ratio_cuda(torch.from_numpy(depth_mm_hw.astype(np.int32)).to(device="cuda"))

    assert ratio_cuda.device.type == "cuda"
    assert float(ratio_cuda.item()) == pytest.approx(depth_span_ratio(depth_mm_hw))


def test_build_training_sample_matches_upstream_layout_through_augmentation() -> None:
    """Produce collatable uint8 RGB, uint16 inverse depth, and a binary validity mask."""
    height: int = 768
    width: int = 1024
    rgb_hwc: UInt8[ndarray, "height width 3"] = np.full((height, width, 3), 127, dtype=np.uint8)
    depth_mm_hw: UInt16[ndarray, "height width"] = np.full((height, width), 1000, dtype=np.uint16)
    depth_mm_hw[:, : width // 2] = 0
    transform: AlbumentationsWrapper = get_train_transforms(height, width)

    sample: dict[str, torch.Tensor] = build_training_sample(rgb_hwc, depth_mm_hw, transform)

    assert sample["image"].shape == (3, height, width)
    assert sample["image"].dtype == torch.uint8
    assert sample["image"].is_contiguous()
    assert sample["depth"].shape == (1, height, width)
    assert sample["depth"].dtype == torch.uint16
    assert sample["depth"].is_contiguous()
    assert sample["mask"].shape == (1, height, width)
    assert sample["mask"].dtype == torch.bool
    assert sample["mask"].is_contiguous()
    assert set(sample["mask"].unique().tolist()) == {False, True}
    depth_hw: UInt16[ndarray, "height width"] = sample["depth"].numpy()[0]
    assert_array_equal(sample["mask"].numpy()[0], depth_hw > 0)
    assert np.any(depth_hw == 0)


def test_build_training_sample_does_not_create_depth_in_holes() -> None:
    """Keep an all-hole target empty through the real random augmentation pipeline."""
    height: int = 768
    width: int = 1024
    rgb_hwc: UInt8[ndarray, "height width 3"] = np.full((height, width, 3), 127, dtype=np.uint8)
    holes_mm_hw: UInt16[ndarray, "height width"] = np.zeros((height, width), dtype=np.uint16)

    sample: dict[str, torch.Tensor] = build_training_sample(rgb_hwc, holes_mm_hw, get_train_transforms(height, width))

    assert np.count_nonzero(sample["depth"].numpy()) == 0
    assert not bool(sample["mask"].any())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sample construction needs a GPU")
def test_build_training_sample_cuda_matches_deterministic_cpu_path() -> None:
    """Match production-size target geometry and bound CPU/GPU RGB filter drift."""
    image_path: Path = Path(__file__).parents[1] / "assets/examples/im0.jpg"
    image_bgr_hwc: UInt8[ndarray, "asset_h asset_w 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image_bgr_hwc is not None
    asset_rgb_hwc: UInt8[ndarray, "asset_h asset_w 3"] = cv2.cvtColor(image_bgr_hwc, cv2.COLOR_BGR2RGB)
    rgb_hwc: UInt8[ndarray, "source_h=1440 source_w=1920 3"] = cv2.resize(
        asset_rgb_hwc, (1920, 1440), interpolation=cv2.INTER_CUBIC
    )
    gray_hw: UInt8[ndarray, "source_h source_w"] = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2GRAY)
    depth_mm_hw: UInt16[ndarray, "source_h source_w"] = (500 + gray_hw.astype(np.uint16) * 10).astype(np.uint16)
    transform: AlbumentationsWrapper = build_eval_transform(768, 1024)
    expected: dict[str, torch.Tensor] = build_training_sample(rgb_hwc, depth_mm_hw, transform)
    rgb_chw: UInt8[ndarray, "3 source_h source_w"] = np.ascontiguousarray(np.moveaxis(rgb_hwc, -1, 0))
    generator: torch.Generator = torch.Generator(device="cuda").manual_seed(101)

    actual: dict[str, torch.Tensor] = build_training_sample_cuda(
        torch.from_numpy(rgb_chw).to(device="cuda"),
        torch.from_numpy(depth_mm_hw.astype(np.int32)).to(device="cuda"),
        quarter_turns=0,
        out_hw=(768, 1024),
        generator=generator,
        policy=AugmentPolicy(flip_p=0.0, channel_shuffle_p=0.0, gray_p=0.0),
    )

    assert actual["image"].dtype == torch.uint8
    assert actual["depth"].dtype == torch.uint16
    assert actual["mask"].dtype == torch.bool
    assert all(value.device.type == "cuda" for value in actual.values())
    assert_array_equal(actual["depth"].cpu().numpy(), expected["depth"].numpy())
    assert_array_equal(actual["mask"].cpu().numpy(), expected["mask"].numpy())
    image_error: Int16[ndarray, "3 h w"] = np.abs(
        actual["image"].cpu().numpy().astype(np.int16) - expected["image"].numpy().astype(np.int16)
    )
    assert int(image_error.max()) <= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sample construction needs a GPU")
@pytest.mark.parametrize("quarter_turns", [0, 1, 2, 3])
def test_build_training_sample_cuda_rotates_counter_clockwise_like_numpy(quarter_turns: int) -> None:
    """Use the same counter-clockwise quarter-turn convention as NumPy."""
    rgb_hwc: UInt8[ndarray, "source_h source_w 3"] = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
    depth_mm_hw: UInt16[ndarray, "source_h source_w"] = np.full((4, 7), 1000, dtype=np.uint16)
    expected_hwc: UInt8[ndarray, "rotated_h rotated_w 3"] = np.ascontiguousarray(np.rot90(rgb_hwc, quarter_turns))
    rgb_chw: UInt8[ndarray, "3 source_h source_w"] = np.ascontiguousarray(np.moveaxis(rgb_hwc, -1, 0))
    generator: torch.Generator = torch.Generator(device="cuda").manual_seed(11)

    actual: dict[str, torch.Tensor] = build_training_sample_cuda(
        torch.from_numpy(rgb_chw).to(device="cuda"),
        torch.from_numpy(depth_mm_hw.astype(np.int32)).to(device="cuda"),
        quarter_turns=quarter_turns,
        out_hw=expected_hwc.shape[:2],
        generator=generator,
        policy=AugmentPolicy(flip_p=0.0, channel_shuffle_p=0.0, gray_p=0.0),
    )

    actual_hwc: UInt8[ndarray, "h w 3"] = np.moveaxis(actual["image"].cpu().numpy(), 0, -1)
    assert_array_equal(actual_hwc, expected_hwc)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sample construction needs a GPU")
def test_build_training_sample_cuda_is_reproducible_from_the_sample_generator() -> None:
    """Repeat stochastic augmentation exactly from an equal per-sample seed."""
    rgb_chw: UInt8[Tensor, "3 8 12"] = (
        torch.arange(3 * 8 * 12, dtype=torch.int32, device="cuda").remainder(256).to(dtype=torch.uint8).reshape(3, 8, 12)
    )
    depth_mm_hw: Int32[Tensor, "8 12"] = torch.full((8, 12), 1000, dtype=torch.int32, device="cuda")

    first: dict[str, torch.Tensor] = build_training_sample_cuda(
        rgb_chw,
        depth_mm_hw,
        quarter_turns=0,
        out_hw=(8, 12),
        generator=torch.Generator().manual_seed(317),
        policy=AugmentPolicy(flip_p=0.5, channel_shuffle_p=1.0, gray_p=0.5),
    )
    second: dict[str, torch.Tensor] = build_training_sample_cuda(
        rgb_chw,
        depth_mm_hw,
        quarter_turns=0,
        out_hw=(8, 12),
        generator=torch.Generator().manual_seed(317),
        policy=AugmentPolicy(flip_p=0.5, channel_shuffle_p=1.0, gray_p=0.5),
    )

    assert all(torch.equal(first[key], second[key]) for key in first)


def test_build_train_transform_is_the_eval_resize_with_optional_mirroring() -> None:
    """Match the eval transform when every augmentation is off, and mirror both targets on flip."""
    rgb_hwc: UInt8[ndarray, "12 16 3"] = np.random.default_rng(0).integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    depth_mm_hw: UInt16[ndarray, "12 16"] = np.random.default_rng(1).integers(100, 4000, size=(12, 16), dtype=np.uint16)
    off: AugmentPolicy = AugmentPolicy(flip_p=0.0, channel_shuffle_p=0.0, gray_p=0.0)
    mirror: AugmentPolicy = AugmentPolicy(flip_p=1.0, channel_shuffle_p=0.0, gray_p=0.0)

    expected: dict[str, torch.Tensor] = build_training_sample(rgb_hwc, depth_mm_hw, build_eval_transform(6, 8))
    actual: dict[str, torch.Tensor] = build_training_sample(rgb_hwc, depth_mm_hw, build_train_transform(6, 8, off))
    flipped: dict[str, torch.Tensor] = build_training_sample(rgb_hwc, depth_mm_hw, build_train_transform(6, 8, mirror))

    key: str
    for key in ("image", "depth", "mask"):
        assert torch.equal(actual[key], expected[key])
        assert_array_equal(flipped[key].numpy(), np.flip(expected[key].numpy(), axis=-1))

    transformed: tuple[np.ndarray, np.ndarray, bool] = build_train_transform(6, 8, mirror).apply_with_flip(
        rgb_hwc, depth_mm_hw.astype(np.float32)
    )
    assert transformed[2] is True


def test_replay_transform_result_names_the_aligned_depth() -> None:
    """Expose typed depth and replay fields without leaking Albumentations' mask key."""
    rgb_hwc: UInt8[ndarray, "4 6 3"] = np.zeros((4, 6, 3), dtype=np.uint8)
    depth_hw: Float32[ndarray, "4 6"] = np.ones((4, 6), dtype=np.float32)

    result: TransformResult = build_train_transform(
        4,
        6,
        AugmentPolicy(flip_p=1.0, channel_shuffle_p=0.0, gray_p=0.0),
    ).apply(rgb_hwc, depth_hw)

    assert set(result) == {"image", "depth", "replay"}


def test_target_resize_samples_source_pixel_centres_without_mean_phase_offset() -> None:
    """Centre the target samples instead of keeping legacy nearest's -0.875 px phase."""
    source_width: int = 1920
    output_width: int = 1024
    source_height: int = 4
    rgb_hwc: UInt8[ndarray, "source_h source_w 3"] = np.zeros((source_height, source_width, 3), dtype=np.uint8)
    ramp_hw: Float32[ndarray, "source_h source_w"] = np.broadcast_to(
        np.arange(source_width, dtype=np.float32), (source_height, source_width)
    ).copy()

    _, resized_ramp_hw = build_eval_transform(source_height, output_width)(rgb_hwc, ramp_hw)

    assert resized_ramp_hw is not None
    scale: float = source_width / output_width
    ideal_source_x: Float32[ndarray, "output_w"] = ((np.arange(output_width, dtype=np.float32) + 0.5) * scale - 0.5).astype(np.float32)
    source_offset_x: Float32[ndarray, "output_w"] = resized_ramp_hw[0] - ideal_source_x
    legacy_ramp_hw: Float32[ndarray, "source_h output_w"] = cv2.resize(
        ramp_hw, (output_width, source_height), interpolation=cv2.INTER_NEAREST
    )
    legacy_offset_x: Float32[ndarray, "output_w"] = legacy_ramp_hw[0] - ideal_source_x

    assert abs(float(source_offset_x.mean())) <= 0.5
    assert float(source_offset_x.mean()) == pytest.approx(0.0, abs=1.0e-6)
    assert float(legacy_offset_x.mean()) == pytest.approx(-0.875, abs=1.0e-6)


def test_target_resize_commutes_exactly_with_horizontal_flip() -> None:
    """Make mirrored training labels identical to mirrored centre-aligned labels."""
    rng: np.random.Generator = np.random.default_rng(883)
    depth_hw: Float32[ndarray, "source_h source_w"] = rng.random((23, 37), dtype=np.float32)
    rgb_hwc: UInt8[ndarray, "source_h source_w 3"] = np.zeros((*depth_hw.shape, 3), dtype=np.uint8)
    transform: AlbumentationsWrapper = build_eval_transform(17, 29)

    _, resized_hw = transform(rgb_hwc, depth_hw)
    _, resized_flipped_hw = transform(np.flip(rgb_hwc, axis=1), np.flip(depth_hw, axis=1))

    assert resized_hw is not None
    assert resized_flipped_hw is not None
    assert_array_equal(np.flip(resized_flipped_hw, axis=1), resized_hw)


def test_rgb_and_target_step_edges_land_in_the_same_output_column() -> None:
    """Keep an aligned RGB/depth boundary co-located after their distinct filters."""
    source_height: int = 24
    source_width: int = 1920
    output_width: int = 1024
    edge_column: int = 913
    rgb_hwc: UInt8[ndarray, "source_h source_w 3"] = np.zeros((source_height, source_width, 3), dtype=np.uint8)
    rgb_hwc[:, edge_column:] = 255
    depth_hw: Float32[ndarray, "source_h source_w"] = np.ones((source_height, source_width), dtype=np.float32)
    depth_hw[:, edge_column:] = 2.0

    resized_rgb_hwc, resized_depth_hw = build_eval_transform(12, output_width)(rgb_hwc, depth_hw)

    assert resized_depth_hw is not None
    rgb_edge_column: int = int(np.flatnonzero(resized_rgb_hwc[0, :, 0] >= 128)[0])
    depth_edge_column: int = int(np.flatnonzero(resized_depth_hw[0] == 2.0)[0])
    assert rgb_edge_column == depth_edge_column


def test_metric_training_sample_emits_metres_and_flips_every_spatial_input() -> None:
    """Keep metric target and prompt aligned while preserving the native prompt grid."""
    rgb_hwc: UInt8[ndarray, "4 6 3"] = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    target_mm_hw: UInt16[ndarray, "4 6"] = np.array(
        [
            [0, 100, 500, 1000, 4000, 4001],
            [1000, 1000, 1000, 1000, 1000, 1000],
            [2000, 2000, 2000, 2000, 2000, 2000],
            [3000, 3000, 3000, 3000, 3000, 3000],
        ],
        dtype=np.uint16,
    )
    prompt_mm_hw: UInt16[ndarray, "192 256"] = np.full((192, 256), 1500, dtype=np.uint16)
    prompt_mm_hw[:, :128] = 500
    confidence_hw: UInt8[ndarray, "192 256"] = np.full((192, 256), 2, dtype=np.uint8)
    confidence_hw[:, :64] = 0
    mirror: AugmentPolicy = AugmentPolicy(flip_p=1.0, channel_shuffle_p=0.0, gray_p=0.0)

    sample: dict[str, torch.Tensor] = build_metric_training_sample(
        rgb_hwc,
        target_mm_hw,
        prompt_mm_hw,
        confidence_hw,
        build_train_transform(4, 6, mirror),
    )

    assert set(sample) == {"image", "target_depth", "target_valid", "prompt_depth", "prompt_valid"}
    assert sample["target_depth"].shape == (1, 4, 6)
    assert sample["target_depth"].dtype == torch.float32
    assert sample["target_valid"].dtype == torch.bool
    assert sample["prompt_depth"].shape == (1, 192, 256)
    assert sample["prompt_depth"].dtype == torch.float32
    assert sample["prompt_valid"].shape == (1, 192, 256)
    assert sample["prompt_valid"].dtype == torch.bool
    assert_array_equal(sample["target_valid"].numpy()[0], np.flip((target_mm_hw >= 100) & (target_mm_hw <= 4000), axis=-1))
    # Depth keeps every real teacher value while target_valid supplies the fixed metric window.
    assert_allclose(sample["target_depth"].numpy()[0], np.flip(np.where(target_mm_hw > 0, target_mm_hw / 1000.0, 0.0), axis=-1))
    assert_allclose(sample["prompt_depth"].numpy()[0], np.flip(prompt_mm_hw / 1000.0, axis=-1))
    expected_prompt_valid_hw: Bool[ndarray, "192 256"] = (confidence_hw >= 1) & (prompt_mm_hw >= 100) & (prompt_mm_hw <= 4000)
    assert_array_equal(sample["prompt_valid"].numpy()[0], np.flip(expected_prompt_valid_hw, axis=-1))
