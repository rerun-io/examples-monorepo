"""Geometric invariants of exo-calib's Ops (hypothesis, CPU only)."""

import numpy as np
from conftest import observe_points, ring_rig
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from exo_calib.ops.bundle_adjust import refine_extrinsics
from exo_calib.ops.eval import camera_centers
from exo_calib.ops.triangulation import triangulate_robust

K = np.array([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]])
POINTS = arrays(np.float64, st.tuples(st.integers(8, 30), st.just(3)), elements=st.floats(-0.5, 0.5))


@settings(max_examples=60, deadline=None)
@given(angles=arrays(np.float64, (3, 3), elements=st.floats(-np.pi, np.pi)), translation=arrays(np.float64, (3, 3), elements=st.floats(-5.0, 5.0)))
def test_camera_centers_are_the_translation_of_the_se3_inverse(angles: np.ndarray, translation: np.ndarray) -> None:
    from scipy.spatial.transform import Rotation

    cam_T_world = np.tile(np.eye(4), (3, 1, 1))
    cam_T_world[:, :3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    cam_T_world[:, :3, 3] = translation
    np.testing.assert_allclose(camera_centers(cam_T_world), np.linalg.inv(cam_T_world)[:, :3, 3], atol=1e-9)


@settings(max_examples=25, deadline=None)
@given(points=POINTS, n_views=st.integers(4, 6), fixed=st.integers(0, 3))
def test_ba_on_exact_observations_keeps_poses_and_never_moves_a_fixed_pose(points: np.ndarray, n_views: int, fixed: int) -> None:
    cam_T_world = ring_rig(n_views, 4.0)
    intrinsics = np.repeat(K[None], n_views, axis=0)
    result = refine_extrinsics(cam_T_world, intrinsics, observe_points(points, intrinsics, cam_T_world), points, rounds=1,
                               robust="cauchy", robust_scale_px=1.0, max_iterations=20, fixed_pose_indices=(fixed,))  # fmt: skip
    assert result.mean_reproj_px_per_round[-1] < 1e-3
    np.testing.assert_array_equal(result.cam_T_world[fixed], cam_T_world[fixed])
    np.testing.assert_allclose(result.cam_T_world, cam_T_world, atol=1e-4)


@settings(max_examples=40, deadline=None)
@given(points=POINTS, view=st.integers(0, 4), shift=st.floats(30.0, 200.0), seed=st.integers(0, 2**16))
def test_robust_triangulation_drops_an_injected_outlier(points: np.ndarray, view: int, shift: float, seed: int) -> None:
    cam_T_world = ring_rig(5, 4.0)  # no two cameras face each other: an opposite pair cannot tell which of them is off
    intrinsics = np.repeat(K[None], 5, axis=0)
    obs = observe_points(points, intrinsics, cam_T_world)
    target: int = int(np.random.default_rng(seed).integers(len(points)))
    obs.obs_xy[target * 5 + view, 0] += shift
    robust = triangulate_robust(obs, intrinsics, cam_T_world, reproj_threshold_px=10.0)
    kept = robust.obs.obs_view_idx[robust.obs.obs_point_idx == target]
    assert view not in kept.tolist() and kept.size == 4 and robust.obs.obs_point_idx.size == obs.obs_point_idx.size - 1
    np.testing.assert_allclose(robust.points_xyz, points, atol=1e-6)
