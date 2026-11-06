"""metrictrail.fusion: per-frame normals ride along ViPE's ``fuse_point_cloud`` into the world frame (CPU, synthetic)."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation

pointcloud = pytest.importorskip("vipe.metrictrail.pointcloud", reason="vipe comes from the metrictrail env (recipes/vipe-src)")

from metrictrail.artifacts import ColoredCloud  # noqa: E402
from metrictrail.fusion import NormalVoxels, fuse_world_cloud  # noqa: E402

HEIGHT, WIDTH = 24, 32


def test_fused_points_carry_their_normals_rotated_by_the_camera_pose() -> None:
    depth: Float32[ndarray, "2 h w"] = np.full((2, HEIGHT, WIDTH), 2.0, dtype=np.float32)
    intrinsics: Float64[ndarray, "2 4"] = np.tile([30.0, 30.0, WIDTH / 2, HEIGHT / 2], (2, 1))
    c2w: Float64[ndarray, "2 4 4"] = np.tile(np.eye(4), (2, 1, 1))
    c2w[1, :3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    c2w[1, :3, 3] = [5.0, 0.0, 0.0]
    camera_normal: Float32[ndarray, "3"] = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # facing the camera
    rgb: UInt8[ndarray, "h w 3"] = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    settings = pointcloud.fusion_settings("default", min_consistent=0)
    cloud: ColoredCloud = fuse_world_cloud(
        depth, intrinsics, c2w, lambda index: rgb, settings, normals=lambda index: np.broadcast_to(camera_normal, (HEIGHT, WIDTH, 3)).copy()
    )
    assert cloud.normals_xyz is not None and cloud.normals_xyz.shape == cloud.points_xyz.shape
    first: np.ndarray = cloud.points_xyz[:, 0] < 2.5  # frame 0 sees a plane at z = 2 (x within +-1.1), frame 1 one at x = 7 facing -x
    np.testing.assert_allclose(cloud.normals_xyz[first], np.tile([0.0, 0.0, -1.0], (int(first.sum()), 1)), atol=1e-6)
    np.testing.assert_allclose(cloud.normals_xyz[~first], np.tile([-1.0, 0.0, 0.0], (int((~first).sum()), 1)), atol=1e-6)
    plain: ColoredCloud = fuse_world_cloud(depth, intrinsics, c2w, lambda index: rgb, settings)
    assert plain.normals_xyz is None
    np.testing.assert_array_equal(plain.points_xyz, cloud.points_xyz)
    np.testing.assert_array_equal(plain.colors_rgb, cloud.colors_rgb)


def test_normal_voxels_match_vipes_voxels_across_compactions() -> None:
    """More frames than one compaction holds, overlapping views: the normal voxels are ViPE's voxels, in ViPE's order."""
    frames: int = 40
    generator = np.random.default_rng(3)
    depth: Float32[ndarray, "f h w"] = generator.uniform(1.5, 2.5, (frames, HEIGHT, WIDTH)).astype(np.float32)
    intrinsics: Float64[ndarray, "f 4"] = np.tile([30.0, 30.0, WIDTH / 2, HEIGHT / 2], (frames, 1))
    c2w: Float64[ndarray, "f 4 4"] = np.tile(np.eye(4), (frames, 1, 1))
    c2w[:, :3, :3] = Rotation.from_euler("y", np.linspace(0.0, 30.0, frames)[:, None], degrees=True).as_matrix()
    c2w[:, :3, 3] = np.stack([np.linspace(0.0, 0.5, frames), np.zeros(frames), np.zeros(frames)], axis=1)
    normals: Float32[ndarray, "f h w 3"] = generator.normal(size=(frames, HEIGHT, WIDTH, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    rgb: UInt8[ndarray, "h w 3"] = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    settings = pointcloud.fusion_settings("default", min_consistent=0, voxel_size_m=0.05)
    cloud: ColoredCloud = fuse_world_cloud(depth, intrinsics, c2w, lambda index: rgb, settings, normals=lambda index: normals[index])
    assert cloud.normals_xyz is not None
    np.testing.assert_allclose(np.linalg.norm(cloud.normals_xyz, axis=1), 1.0, atol=1e-5)
    reference = pointcloud.fuse_point_cloud(depth, intrinsics, c2w, lambda index: rgb, settings, world_frame=True)
    np.testing.assert_array_equal(cloud.points_xyz, reference.points)


def test_normal_voxels_refuse_positions_that_are_not_vipes() -> None:
    points: Float32[ndarray, "2 3"] = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    voxels = NormalVoxels(0.1)
    voxels.add(points, np.array([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(voxels.finish(points), [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    swapped = NormalVoxels(0.1)
    swapped.add(points, points)
    with pytest.raises(RuntimeError, match="do not match"):
        swapped.finish(points[::-1].copy())
