"""Geometric invariants of exo-calib's Ops (hypothesis, CPU only)."""

import numpy as np
from conftest import observe_points, ring_rig
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from exo_calib.ops.bundle_adjust import refine_extrinsics, revert_large_pose_updates
from exo_calib.ops.eval import camera_centers
from exo_calib.ops.triangulation import triangulate_robust

K = np.array([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]])
POINTS = arrays(np.float64, st.tuples(st.integers(8, 30), st.just(3)), elements=st.floats(-0.5, 0.5))
ROTVEC = arrays(np.float64, 3, elements=st.floats(-1.0, 1.0)).filter(lambda v: np.linalg.norm(v) > 0.1)


def rotated_rig(cam_T_world: np.ndarray, common: np.ndarray, per_camera: np.ndarray) -> np.ndarray:
    """Rotate the whole world by ``common`` (a gauge move), then each camera by its own ``per_camera`` rotation."""
    out = cam_T_world.copy()
    out[:, :3, :3] = per_camera @ cam_T_world[:, :3, :3] @ common.T
    out[:, :3, 3] = np.einsum("vij,vj->vi", per_camera, cam_T_world[:, :3, 3])
    return out


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


@settings(max_examples=60, deadline=None)
@given(n_views=st.integers(3, 8), axis=ROTVEC, angle_deg=st.floats(0.0, 40.0), small=arrays(np.float64, (8, 3), elements=st.floats(-1.0, 1.0)))
def test_rotation_guard_ignores_a_rotation_the_whole_rig_shares(n_views: int, axis: np.ndarray, angle_deg: float, small: np.ndarray) -> None:
    from scipy.spatial.transform import Rotation

    initial = ring_rig(n_views, 4.0)
    common = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.deg2rad(angle_deg)).as_matrix()
    own = Rotation.from_rotvec(small[:n_views] * np.deg2rad(3.0) / np.sqrt(3.0)).as_matrix()  # <= 3 deg per camera
    refined = rotated_rig(initial, common, own)
    guarded, reverted = revert_large_pose_updates(refined, initial, maximum_rotation_deg=10.0)
    assert reverted == ()
    np.testing.assert_array_equal(guarded, refined)


@settings(max_examples=60, deadline=None)
@given(n_views=st.integers(3, 8), axis=ROTVEC, angle_deg=st.floats(0.0, 5.0), small=arrays(np.float64, (8, 3), elements=st.floats(-1.0, 1.0)),
       bad=st.integers(0, 7), bad_axis=ROTVEC, bad_deg=st.floats(30.0, 120.0))  # fmt: skip
def test_rotation_guard_still_reverts_a_camera_that_diverges_from_the_rig(
    n_views: int, axis: np.ndarray, angle_deg: float, small: np.ndarray, bad: int, bad_axis: np.ndarray, bad_deg: float
) -> None:
    from scipy.spatial.transform import Rotation

    initial = ring_rig(n_views, 4.0)
    common = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.deg2rad(angle_deg)).as_matrix()
    own = Rotation.from_rotvec(small[:n_views] * np.deg2rad(3.0) / np.sqrt(3.0)).as_matrix()
    own[bad % n_views] = Rotation.from_rotvec(bad_axis / np.linalg.norm(bad_axis) * np.deg2rad(bad_deg)).as_matrix() @ own[bad % n_views]
    refined = rotated_rig(initial, common, own)
    guarded, reverted = revert_large_pose_updates(refined, initial, maximum_rotation_deg=10.0)
    assert reverted == (bad % n_views,)
    np.testing.assert_array_equal(guarded[bad % n_views], initial[bad % n_views])
