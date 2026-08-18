"""Behavior contracts for fused-publication error metrics."""

import numpy as np
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray

from gauss_surf.train_gsplat.layer_metrics import TriageErrors, angular_error_degrees, colorize_error, compute_triage_errors, viridis_like


def test_viridis_like_has_stable_endpoints() -> None:
    values_n: Float32[ndarray, "n=2"] = np.asarray([0.0, 1.0], dtype=np.float32)

    colors_n3: UInt8[ndarray, "n=2 3"] = viridis_like(values_n)

    np.testing.assert_array_equal(colors_n3[0], [68, 1, 84])
    np.testing.assert_array_equal(colors_n3[1], [253, 231, 37])


def test_angular_error_is_zero_for_identical_and_ninety_for_orthogonal() -> None:
    first_hw3: Float32[ndarray, "h=1 w=2 3"] = np.asarray([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    second_hw3: Float32[ndarray, "h=1 w=2 3"] = np.asarray([[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]], dtype=np.float32)

    result: tuple[Float32[ndarray, "h=1 w=2"], Bool[ndarray, "h=1 w=2"]] = angular_error_degrees(first_hw3, second_hw3)
    error_degrees_hw: Float32[ndarray, "h=1 w=2"] = result[0]
    valid_hw: Bool[ndarray, "h=1 w=2"] = result[1]

    np.testing.assert_allclose(error_degrees_hw, [[0.0, 90.0]], atol=1e-5)
    np.testing.assert_array_equal(valid_hw, True)


def test_error_maps_are_black_outside_each_validity_mask() -> None:
    source_rgb_hw3: UInt8[ndarray, "h=1 w=2 3"] = np.asarray([[[255, 255, 255], [255, 255, 255]]], dtype=np.uint8)
    splat_rgb_hw3: UInt8[ndarray, "h=1 w=2 3"] = np.zeros((1, 2, 3), dtype=np.uint8)
    splat_depth_m_hw: Float32[ndarray, "h=1 w=2"] = np.asarray([[1.0, 0.0]], dtype=np.float32)
    mesh_depth_m_hw: Float32[ndarray, "h=1 w=2"] = np.asarray([[1.5, 1.0]], dtype=np.float32)
    splat_normal_hw3: Float32[ndarray, "h=1 w=2 3"] = np.asarray([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
    moge_normal_hw3: Float32[ndarray, "h=1 w=2 3"] = np.asarray([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)

    errors: TriageErrors = compute_triage_errors(
        source_rgb_hw3, splat_rgb_hw3, mesh_depth_m_hw, splat_depth_m_hw, moge_normal_hw3, splat_normal_hw3
    )

    for error_map_hw3 in (errors.rgb_map_hw3, errors.depth_map_hw3, errors.normal_map_hw3):
        np.testing.assert_array_equal(error_map_hw3[0, 1], [0, 0, 0])
        assert np.any(error_map_hw3[0, 0] != 0)
    assert errors.rgb_mean == 1.0
    assert errors.depth_mean_m == 0.5
    assert errors.normal_mean_degrees == 90.0


def test_colorize_error_clips_scale_and_masks_invalid_pixels() -> None:
    error_hw: Float32[ndarray, "h=1 w=3"] = np.asarray([[0.0, 0.5, 2.0]], dtype=np.float32)
    valid_hw: Bool[ndarray, "h=1 w=3"] = np.asarray([[True, False, True]], dtype=np.bool_)

    color_hw3: UInt8[ndarray, "h=1 w=3 3"] = colorize_error(error_hw, valid_hw, maximum=1.0)

    np.testing.assert_array_equal(color_hw3[0, 0], [68, 1, 84])
    np.testing.assert_array_equal(color_hw3[0, 1], [0, 0, 0])
    np.testing.assert_array_equal(color_hw3[0, 2], [253, 231, 37])
