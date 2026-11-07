"""metrictrail.surface: the TSDF mesh of a synthetic wall lands on the wall, in its colour, facing the cameras."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray

pytest.importorskip("tsdf_rs", reason="tsdf-rs is the metrictrail env's source package (packages/tsdf-rs)")

from metrictrail.surface import ColoredMesh, TsdfSettings, fuse_tsdf_mesh, vertex_normals  # noqa: E402

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
    # Interior vertices face the cameras (-Z, RDF), up to the triangle winding's sign.
    interior = np.abs(mesh.normals_xyz[:, 2]) > 0.5
    assert interior.mean() > 0.9


def test_out_of_range_depth_and_non_finite_poses_give_no_mesh() -> None:
    depth: Float32[ndarray, "2 h w"] = np.full((2, HEIGHT, WIDTH), 40.0, dtype=np.float32)  # beyond max_depth_m
    intrinsics: Float64[ndarray, "2 4"] = np.tile([60.0, 60.0, WIDTH / 2, HEIGHT / 2], (2, 1))
    world_T_cam: Float64[ndarray, "2 4 4"] = np.tile(np.eye(4), (2, 1, 1))
    world_T_cam[1] = np.nan
    color: UInt8[ndarray, "h w 3"] = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert fuse_tsdf_mesh(depth, intrinsics, world_T_cam, lambda _index: color) is None


def test_vertex_normals_are_unit_and_area_weighted() -> None:
    vertices: Float32[ndarray, "4 3"] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.uint32)
    normals: Float32[ndarray, "4 3"] = vertex_normals(vertices, triangles)
    np.testing.assert_allclose(normals[:3], np.tile([0.0, 0.0, 1.0], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(normals[3], 0.0)
