"""recipes/vipe-src patch 0004: ``fuse_point_cloud`` carries per-frame normals into the world frame (CPU, synthetic)."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation

pointcloud = pytest.importorskip("vipe.metrictrail.pointcloud", reason="vipe comes from the metrictrail env (recipes/vipe-src)")


def test_fused_points_carry_their_normals_rotated_by_the_camera_pose() -> None:
    height, width = 24, 32
    depth: Float32[ndarray, "2 h w"] = np.full((2, height, width), 2.0, dtype=np.float32)
    intrinsics: Float64[ndarray, "2 4"] = np.tile([30.0, 30.0, width / 2, height / 2], (2, 1))
    c2w: Float64[ndarray, "2 4 4"] = np.tile(np.eye(4), (2, 1, 1))
    c2w[1, :3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    c2w[1, :3, 3] = [5.0, 0.0, 0.0]
    camera_normal: Float32[ndarray, "3"] = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # facing the camera
    rgb: UInt8[ndarray, "h w 3"] = np.full((height, width, 3), 128, dtype=np.uint8)
    settings = pointcloud.fusion_settings("default", min_consistent=0)
    cloud = pointcloud.fuse_point_cloud(
        depth,
        intrinsics,
        c2w,
        lambda index: rgb,
        settings,
        world_frame=True,
        normals=lambda index: np.broadcast_to(camera_normal, (height, width, 3)).copy(),
    )
    assert cloud.normals is not None and cloud.normals.shape == cloud.points.shape
    first: np.ndarray = cloud.points[:, 0] < 2.5  # frame 0 sees a plane at z = 2 (x within +-1.1), frame 1 one at x = 7 facing -x
    np.testing.assert_allclose(cloud.normals[first], np.tile([0.0, 0.0, -1.0], (int(first.sum()), 1)), atol=1e-6)
    np.testing.assert_allclose(cloud.normals[~first], np.tile([-1.0, 0.0, 0.0], (int((~first).sum()), 1)), atol=1e-6)
    plain = pointcloud.fuse_point_cloud(depth, intrinsics, c2w, lambda index: rgb, settings, world_frame=True)
    assert plain.normals is None
    np.testing.assert_array_equal(plain.points, cloud.points)
