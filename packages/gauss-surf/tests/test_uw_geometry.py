"""Behavior checks for rectified ultrawide geometry and depth storage."""

from io import BytesIO

import numpy as np
import open3d as o3d
from arkitscenes_download.ingest.depth import encode_depth_png
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from PIL import Image

from gauss_surf.uw_geometry import (
    build_apple_undistortion,
    compose_world_from_ultrawide,
    depth_meters_to_uint16_mm,
    pose_indices_at_or_before,
    raycast_z_depth,
    undistort_rgb,
)

REAL_ULTRAWIDE_COEFFICIENTS_8: Float32[ndarray, "8"] = np.array(
    [-0.08353952, -6.350463, 5.248554, -1.9897834, 0.5831521, -0.10259675, 0.009266047, -0.0003309832],
    dtype=np.float32,
)


def test_zero_apple_polynomial_builds_an_identity_remap() -> None:
    """Zero Apple coefficients leave the pinhole and every source pixel unchanged."""
    K_33: Float32[ndarray, "3 3"] = np.array(
        [[8.0, 0.0, 3.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    undistortion = build_apple_undistortion(
        K_33,
        np.zeros(8, dtype=np.float32),
        distortion_center_reference_xy=np.array([3.0, 2.0], dtype=np.float32),
        reference_dimensions_wh=(7, 5),
        image_wh=(7, 5),
    )

    expected_x_hw: Float32[ndarray, "h=5 w=7"] = np.tile(np.arange(7, dtype=np.float32), (5, 1))
    expected_y_hw: Float32[ndarray, "h=5 w=7"] = np.tile(np.arange(5, dtype=np.float32)[:, None], (1, 7))
    np.testing.assert_array_equal(undistortion.K_rect_33, K_33)
    np.testing.assert_array_equal(undistortion.source_x_hw, expected_x_hw)
    np.testing.assert_array_equal(undistortion.source_y_hw, expected_y_hw)


def test_real_apple_remap_crops_only_enough_to_keep_every_sample_in_bounds() -> None:
    """The real 640×480 remap has no invalid sampled source pixels."""
    K_uw_33: Float32[ndarray, "3 3"] = np.array(
        [[245.3910065, 0.0, 321.8739929], [0.0, 245.3910065, 238.0269928], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    undistortion = build_apple_undistortion(
        K_uw_33,
        REAL_ULTRAWIDE_COEFFICIENTS_8,
        distortion_center_reference_xy=np.array([1853.1506, 1371.0317], dtype=np.float32),
        reference_dimensions_wh=(3680, 2760),
        image_wh=(640, 480),
    )

    assert np.all((undistortion.source_x_hw >= 0.0) & (undistortion.source_x_hw <= 639.0))
    assert np.all((undistortion.source_y_hw >= 0.0) & (undistortion.source_y_hw <= 479.0))
    assert K_uw_33[0, 0] < undistortion.K_rect_33[0, 0] < 1.1 * K_uw_33[0, 0]
    assert K_uw_33[1, 1] < undistortion.K_rect_33[1, 1] < 1.1 * K_uw_33[1, 1]


def test_raycast_returns_camera_z_depth_at_oblique_pixels_and_zero_on_miss() -> None:
    """Open3D hit parameters become pinhole z-depth, not Euclidean ray length."""
    vertices_n3: Float32[ndarray, "n=4 3"] = np.array(
        [
            [-1.1, -0.75, 2.0],
            [1.1, -0.75, 2.0],
            [1.1, 0.75, 2.0],
            [-1.1, 0.75, 2.0],
        ],
        dtype=np.float32,
    )
    triangles_m3: ndarray = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh: o3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices_n3),
        o3d.core.Tensor(triangles_m3),
    )
    scene: o3d.t.geometry.RaycastingScene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    K_33: Float32[ndarray, "3 3"] = np.array(
        [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    world_from_camera_44: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)

    depth_hw: Float32[ndarray, "h=4 w=4"] = raycast_z_depth(
        scene,
        K_33,
        world_from_camera_44,
        image_wh=(4, 4),
    )

    np.testing.assert_allclose(depth_hw[1, 1:3], 2.0, atol=1e-4, rtol=0.0)
    assert depth_hw[0, 0] == 0.0


def test_pose_composition_places_camera_ray_in_world_coordinates() -> None:
    """The rig pose composes with wide-from-ultrawide before rays enter the world."""
    world_from_rig_44: Float32[ndarray, "4 4"] = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    wide_from_ultrawide_44: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    wide_from_ultrawide_44[0, 3] = 0.012

    world_from_ultrawide_44: Float32[ndarray, "4 4"] = compose_world_from_ultrawide(
        world_from_rig_44[None],
        wide_from_ultrawide_44,
    )[0]
    ray_origin_world_3: Float32[ndarray, "3"] = world_from_ultrawide_44[:3, 3]
    ray_direction_world_3: Float32[ndarray, "3"] = world_from_ultrawide_44[:3, :3] @ np.array([1.0, 0.0, 1.0], dtype=np.float32)

    np.testing.assert_allclose(ray_origin_world_3, [1.0, 2.012, 3.0], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(ray_direction_world_3, [0.0, 1.0, 1.0], atol=1e-6, rtol=0.0)


def test_pose_lookup_uses_nearest_sample_at_or_before_each_ultrawide_time() -> None:
    """Jittered ultrawide times use causal 60 Hz poses and never a future sample."""
    pose_times_n: ndarray = np.array([0, 16_666_667, 33_333_334, 50_000_001], dtype="timedelta64[ns]")
    ultrawide_times_n: ndarray = np.array([1_000_000, 16_666_667, 32_900_000, 50_100_000], dtype="timedelta64[ns]")

    indices_n: ndarray = pose_indices_at_or_before(pose_times_n, ultrawide_times_n)

    np.testing.assert_array_equal(indices_n, [0, 1, 1, 3])
    assert np.all(pose_times_n[indices_n] <= ultrawide_times_n)


def test_zero_apple_polynomial_rectification_is_pixel_exact() -> None:
    """A zero Apple radial model leaves the RGB pixels unchanged."""
    rgb_hw3: UInt8[ndarray, "h=5 w=7 3"] = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    K_33: Float32[ndarray, "3 3"] = np.array(
        [[8.0, 0.0, 3.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    coefficients_8: Float32[ndarray, "8"] = np.zeros(8, dtype=np.float32)

    undistortion = build_apple_undistortion(
        K_33,
        coefficients_8,
        distortion_center_reference_xy=np.array([3.0, 2.0], dtype=np.float32),
        reference_dimensions_wh=(7, 5),
        image_wh=(7, 5),
    )
    rectified_hw3: UInt8[ndarray, "h=5 w=7 3"] = undistort_rgb(rgb_hw3, undistortion)

    np.testing.assert_array_equal(rectified_hw3, rgb_hw3)


def test_depth_png_preserves_zero_sentinel_and_clamps_uint16_overflow() -> None:
    """Metre depth rounds to millimetres while invalid and overflowing values stay safe."""
    depth_m_hw: Float32[ndarray, "h=2 w=3"] = np.array(
        [[0.0, 1.2344, 1.2346], [np.inf, -1.0, 70.0]],
        dtype=np.float32,
    )

    depth_mm_hw: UInt16[ndarray, "h=2 w=3"] = depth_meters_to_uint16_mm(depth_m_hw)
    png_bytes: bytes = encode_depth_png(depth_mm_hw)
    with Image.open(BytesIO(png_bytes)) as image:
        decoded_mm_hw: UInt16[ndarray, "h=2 w=3"] = np.asarray(image, dtype=np.uint16)

    np.testing.assert_array_equal(decoded_mm_hw, [[0, 1234, 1235], [0, 0, 65535]])
