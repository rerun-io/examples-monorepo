"""Joint camera refinement recovers physical synthetic observations."""

import numpy as np
from scipy.spatial.transform import Rotation

from selfcap_curate.exo_calibration import Camera
from selfcap_curate.exo_landmarks import Landmark, Landmarks
from selfcap_curate.exo_refine import RefinementProblem, solve
from selfcap_curate.quest_fit import Correspondences


def test_joint_solver_recovers_pose_focal_and_head_offset() -> None:
    rng = np.random.default_rng(42)
    cameras = tuple(Camera(np.array([0.0, angle, 0.0, x, 0.0, 3.0, focal, 960.0, 540.0, 0.0, 0.0]))
                    for angle, x, focal in ((0.2, -0.8, 800.0), (-0.3, 0.9, 950.0)))
    offset = np.array([-0.03, -0.05, 0.02])
    world = rng.uniform(-0.6, 0.6, (80, 3))
    hands = tuple(Correspondences(world, camera.project(world)[:, :2], np.ones(80)) for camera in cameras)
    labels, centres, rotations = [], [], []
    for view, camera in enumerate(cameras):
        for i in range(20):
            rotation = Rotation.from_rotvec(rng.normal(0.0, 0.3, 3)).as_matrix()
            centre = rng.uniform(-0.3, 0.3, 3)
            point = centre + rotation @ offset
            uv = camera.project(point[None])[0]
            labels.append(Landmark(view, float(i), "quest0", float(uv[0]), float(uv[1]), 1.0))
            centres.append(centre)
            rotations.append(rotation)
        other = cameras[1 - view].world_T_cam[:3, 3]
        uv = camera.project(other[None])[0]
        labels.append(Landmark(view, 0.0, f"phone{1-view}", float(uv[0]), float(uv[1]), 1.0))
        centres.append(np.zeros(3))
        rotations.append(np.eye(3))
    seeds = []
    for camera in cameras:
        p = camera.parameters.copy()
        p[:6] += rng.normal(0.0, 0.02, 6)
        p[6] *= 0.9
        seeds.append(Camera(p))
    problem = RefinementProblem(tuple(seeds), hands, Landmarks(labels, "synthetic", "exact", "2026-10-29"),
                                np.asarray(centres), np.asarray(rotations))
    result = solve(problem, focal_source="fit")
    np.testing.assert_allclose(result.head_translation_m, offset, atol=1e-5)
    for actual, expected in zip(result.cameras, cameras, strict=True):
        np.testing.assert_allclose(actual.parameters, expected.parameters, atol=1e-4)
    assert result.label_layer_hash == problem.landmarks.digest
