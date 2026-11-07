"""metrictrail.surface: the TSDF mesh of a synthetic wall lands on the wall, in its colour, facing the cameras."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray

from metrictrail.surface import ColoredMesh, TsdfSettings, fuse_tsdf_mesh, integrate_frame, new_volume

HEIGHT, WIDTH = 48, 64
WALL_M = 2.0
WALL_RGB = (200, 40, 10)


def test_a_wall_seen_by_two_cameras_becomes_one_mesh_at_its_depth() -> None:
    depth: Float32[ndarray, "3 h w"] = np.full((3, HEIGHT, WIDTH), WALL_M, dtype=np.float32)
    depth[2] = np.nan  # a frame with no depth contributes nothing
    intrinsics: Float64[ndarray, "3 4"] = np.tile([60.0, 60.0, WIDTH / 2, HEIGHT / 2], (3, 1))
    world_T_cam: Float64[ndarray, "3 4 4"] = np.tile(np.eye(4), (3, 1, 1))
    world_T_cam[1, 0, 3] = 0.1  # the second camera 10 cm to the right, same wall depth
    color: UInt8[ndarray, "h w 3"] = np.broadcast_to(np.array(WALL_RGB, dtype=np.uint8), (HEIGHT, WIDTH, 3)).copy()
    mesh: ColoredMesh | None = fuse_tsdf_mesh(depth, intrinsics, world_T_cam, lambda _index: color, TsdfSettings(voxel_size_m=0.02))
    assert mesh is not None and mesh.triangles.shape[0] > 0
    np.testing.assert_allclose(mesh.vertices_xyz[:, 2], WALL_M, atol=0.02)
    assert np.abs(mesh.colors_rgb.astype(int) - np.array(WALL_RGB)).max() <= 2
    np.testing.assert_allclose(np.linalg.norm(mesh.normals_xyz, axis=1), 1.0, atol=1e-5)
    # Interior vertices face the cameras (-Z: the cameras look down +Z at the wall)...
    assert (mesh.normals_xyz[:, 2] < -0.9).mean() > 0.9
    # ...and so do the triangles: counter-clockwise seen from the cameras.
    corners: Float32[ndarray, "t 3 3"] = mesh.vertices_xyz[mesh.triangles]
    face_z: Float32[ndarray, "t"] = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2]
    assert (face_z <= 0.0).all()


def test_out_of_range_depth_and_non_finite_poses_give_no_mesh() -> None:
    depth: Float32[ndarray, "2 h w"] = np.full((2, HEIGHT, WIDTH), 40.0, dtype=np.float32)  # beyond max_depth_m
    intrinsics: Float64[ndarray, "2 4"] = np.tile([60.0, 60.0, WIDTH / 2, HEIGHT / 2], (2, 1))
    world_T_cam: Float64[ndarray, "2 4 4"] = np.tile(np.eye(4), (2, 1, 1))
    world_T_cam[1] = np.nan
    color: UInt8[ndarray, "h w 3"] = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert fuse_tsdf_mesh(depth, intrinsics, world_T_cam, lambda _index: color) is None


def test_integrate_frame_skips_a_non_finite_camera() -> None:
    settings = TsdfSettings(voxel_size_m=0.02)
    volume = new_volume(settings)
    depth: Float32[ndarray, "h w"] = np.full((HEIGHT, WIDTH), WALL_M, dtype=np.float32)
    rgb: UInt8[ndarray, "h w 3"] = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    intrinsics: Float64[ndarray, "4"] = np.array([60.0, 60.0, WIDTH / 2, HEIGHT / 2])
    bad_pose: Float64[ndarray, "4 4"] = np.full((4, 4), np.nan)
    assert not integrate_frame(volume, settings, depth, intrinsics, bad_pose, rgb)
    assert not integrate_frame(volume, settings, depth, np.array([np.nan, 60.0, 32.0, 24.0]), np.eye(4), rgb)
    assert volume.extract_mesh()[1].shape[0] == 0
    assert integrate_frame(volume, settings, depth, intrinsics, np.eye(4), rgb)
    assert volume.extract_mesh()[1].shape[0] > 0


def test_tsdf_rs_refuses_fortran_ordered_arrays() -> None:
    """A transposed view must raise, not be read as its transpose (K.T of a pinhole K is not a pinhole K)."""
    volume = new_volume(TsdfSettings())
    depth: Float32[ndarray, "h w"] = np.full((HEIGHT, WIDTH), WALL_M, dtype=np.float32)
    K: Float32[ndarray, "3 3"] = np.array([[60.0, 0.0, WIDTH / 2], [0.0, 60.0, HEIGHT / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    cam_T_world: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    with pytest.raises(ValueError, match="K_33 must be C-contiguous"):
        volume.integrate(depth, np.asfortranarray(K), cam_T_world)
    with pytest.raises(ValueError, match="depth_hw must be C-contiguous"):
        volume.integrate(np.asfortranarray(depth), K, cam_T_world)
    with pytest.raises(ValueError, match="K_33 must have shape"):
        volume.integrate(depth, np.eye(4, dtype=np.float32), cam_T_world)
    with pytest.raises(ValueError, match="max_depth"):
        volume.integrate(depth, K, cam_T_world, min_depth=2.0, max_depth=1.0)
    volume.integrate(depth, K, cam_T_world)
