"""World placement and display-mesh logic are independent of model weights."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("egohand_inference", reason="requires the egohand Pixi environment")

from posekit.apis.egohand_comparison import EgoCamera
from posekit.apis.egohand_world import camera_world_poses, make_mesh_lod, mesh_normals


def test_camera_world_pose_interpolates_and_composes_stereo_offset() -> None:
    track = np.repeat(np.eye(4)[None], 2, axis=0)
    track[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    track[1, 0, 3] = 2.0
    stereo = np.eye(4)
    stereo[0, 3] = 0.1
    camera = EgoCamera(4, 1, np.eye(3), np.zeros(14), (1280, 720), stereo, np.array([0.0, 0.08]), track)
    poses, valid = camera_world_poses(camera, np.array([-0.1, 0.04, 0.2]))
    np.testing.assert_array_equal(valid, [False, True, False])
    np.testing.assert_allclose(poses[1, :3, 3], [1.0 + np.sqrt(0.005), np.sqrt(0.005), 0.0], atol=1e-8)
    np.testing.assert_allclose(poses[1, :3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix(), atol=1e-8)


def test_lod_preserves_rigid_motion_and_has_consistent_normals() -> None:
    xy = np.stack(np.meshgrid(np.arange(5), np.arange(5)), axis=-1).reshape(-1, 2)
    vertices = np.c_[xy, np.zeros(25)].astype(np.float32)
    faces = np.array([[i, i + 1, i + 5] for i in range(20) if i % 5 < 4] + [[i + 1, i + 6, i + 5] for i in range(20) if i % 5 < 4], np.uint32)
    lod = make_mesh_lod(vertices, faces, 12)
    reduced = lod.apply(vertices)
    assert len(reduced) <= 12
    assert lod.faces.max() < len(reduced)
    np.testing.assert_allclose(lod.apply(vertices + np.array([1.0, 2.0, 3.0], np.float32)), reduced + [1.0, 2.0, 3.0])
    normals = mesh_normals(reduced, lod.faces)
    np.testing.assert_allclose(normals[:, :2], 0.0)
    np.testing.assert_allclose(normals[:, 2], 1.0)


def test_world_distance_separates_depth_from_articulation() -> None:
    from posekit.apis.egohand_world_metrics import summarize_distances

    reference = np.zeros((2, 2, 21, 3), dtype=np.float32)
    reference[..., 2] = 1.0
    predicted = reference.copy()
    predicted[..., 2] += 0.1
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    result = summarize_distances(predicted, reference, poses)
    assert result.pairs == 84
    assert result.median_mm is not None
    assert result.depth_bias_mm is not None
    assert abs(result.median_mm - 100.0) < 1e-3
    assert abs(result.depth_bias_mm - 100.0) < 1e-3
    assert result.articulation_median_mm == 0.0
