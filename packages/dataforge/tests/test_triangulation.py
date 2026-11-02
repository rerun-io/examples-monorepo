import numpy as np
import torch

from dataforge.triangulation import Camera, project, triangulate_frames


def test_two_view_triangulation_recovers_point_without_third_view(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    right_pose = np.eye(4)
    right_pose[0, 3] = 1.0
    cameras = [Camera(intrinsics, np.eye(4), (0.0, 0.0), "a"), Camera(intrinsics, right_pose, (0.0, 0.0), "b")]
    point = np.array([[0.0, 0.0, 5.0]])
    pixels = np.array([[[[50.0, 50.0]], [[30.0, 50.0]]]])
    result = triangulate_frames(cameras, pixels, np.ones((1, 2, 1)))
    np.testing.assert_allclose(result.xyz[0], point, atol=1e-5)
    np.testing.assert_allclose(project(cameras[1], point)[0], [30.0, 50.0, 5.0])


def test_two_views_facing_each_other_give_no_point(monkeypatch):
    """Phones on opposite sides of the wearer see a point near the line between them along almost one line (0.29 degrees
    here, rays 179.7 degrees apart): depth is free along it, so that pair alone must not place the point."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    intrinsics = np.array([[900.0, 0.0, 960.0], [0.0, 900.0, 540.0], [0.0, 0.0, 1.0]])
    facing = np.eye(4)
    facing[:3, :3], facing[:3, 3] = np.diag([-1.0, 1.0, -1.0]), [0.0, 0.0, 8.0]
    cameras = [Camera(intrinsics, np.eye(4), (0.0, 0.0), "a"), Camera(intrinsics, facing, (0.0, 0.0), "b")]
    point = np.array([[0.01, 0.0, 4.0]])
    pixels = np.stack([project(c, point)[:, :2] for c in cameras])[None]
    assert triangulate_frames(cameras, pixels, np.ones((1, 2, 1))).n_views[0, 0] == 0
