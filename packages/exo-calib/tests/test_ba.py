import cv2
import numpy as np
from conftest import FakeScene, observe_points, ring_rig
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from exo_calib.ops.bundle_adjust import (
    PosePrior,
    RigFit,
    mean_reprojection_error_px,
    one_focal_per_camera_model,
    refine_rig,
    refine_through_lenses,
)
from exo_calib.ops.correspondences import ObservationSet
from exo_calib.ops.eval import camera_centers, evaluate_rig
from exo_calib.types import PINHOLE, Lens


def test_mean_reprojection_error_px_uses_confidence_weights_and_ignores_invalid() -> None:
    observations: ObservationSet = ObservationSet(
        point_frame_idx=np.array([0], dtype=np.int64),
        point_joint_idx=np.array([0], dtype=np.int64),
        obs_point_idx=np.array([0, 0, 0], dtype=np.int64),
        obs_view_idx=np.array([0, 1, 1], dtype=np.int64),
        obs_xy=np.array([[0.0, 0.0], [6.0, 0.0], [np.nan, 0.0]], dtype=np.float64),
        obs_conf=np.array([1.0, 0.5, 1.0], dtype=np.float64),
    )
    points_xyz: Float64[ndarray, "1 3"] = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    intrinsics: Float64[ndarray, "2 3 3"] = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    cam_T_world: Float64[ndarray, "2 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)

    actual_error_px: float = mean_reprojection_error_px(observations, points_xyz, intrinsics, cam_T_world)

    assert actual_error_px == 2.0


def test_refine_rig_recovers_the_poses_and_a_shared_focal_15_percent_low() -> None:
    n_views: int = 8
    n_points: int = 200
    true_cam_T_world: Float64[ndarray, "8 4 4"] = ring_rig(n_views, radius_m=4.0)
    intrinsics: Float64[ndarray, "8 3 3"] = np.zeros((n_views, 3, 3), dtype=np.float64)
    intrinsics[:, 0, 0] = 750.0 + 10.0 * np.arange(n_views, dtype=np.float64)
    intrinsics[:, 1, 1] = 770.0 + 8.0 * np.arange(n_views, dtype=np.float64)
    intrinsics[:, 0, 2] = 320.0
    intrinsics[:, 1, 2] = 240.0
    intrinsics[:, 2, 2] = 1.0
    rng: np.random.Generator = np.random.default_rng(12)
    # A person walking a 2.4 m square: perspective across the volume is what tells a focal change from a move.
    true_points_xyz: Float64[ndarray, "200 3"] = rng.uniform([-1.2, -1.2, -0.9], [1.2, 1.2, 0.9], size=(n_points, 3))
    observations: ObservationSet = observe_points(true_points_xyz, intrinsics, true_cam_T_world)

    initial_cam_T_world: Float64[ndarray, "8 4 4"] = true_cam_T_world.copy()
    for view_idx in range(n_views):
        turn: Float64[ndarray, "3 3"] = Rotation.from_rotvec(rng.normal(size=3) * np.deg2rad(2.0) / np.sqrt(3.0)).as_matrix()
        initial_cam_T_world[view_idx, :3, :3] = turn @ true_cam_T_world[view_idx, :3, :3]
        initial_cam_T_world[view_idx, :3, 3] += rng.normal(scale=0.05, size=3)
    initial_intrinsics: Float64[ndarray, "8 3 3"] = intrinsics.copy()
    initial_intrinsics[:, :2, :2] *= 0.85  # one shared error, as a whole rig's Stage A focal is off on a fisheye
    prior = PosePrior(camera_centers(initial_cam_T_world), sigma_m=0.3)

    fit: RigFit = refine_rig(initial_cam_T_world, initial_intrinsics, observations, pose_prior=prior, robust_scale_px=1.0, device="cpu")

    errors = evaluate_rig(fit.cam_T_world, true_cam_T_world, mode="sim3")  # the prior pins the gauge to the perturbed centres
    assert np.max(errors.rotation_error_deg) < 0.01 and np.max(errors.translation_error_cm) < 0.1
    np.testing.assert_allclose(fit.intrinsics, intrinsics, rtol=1e-5)
    assert abs(fit.focal_scale - 1.0 / 0.85) < 1e-5


def test_cameras_of_one_model_start_from_one_focal_and_another_model_keeps_its_own() -> None:
    """Stage A's per-camera focals of one camera model differ by a few percent for no physical reason; the rig starts every
    camera of a model (resolution) at their median, and a camera of another resolution at its own model's."""
    intrinsics: Float64[ndarray, "4 3 3"] = np.repeat(np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])[None], 4, axis=0)
    intrinsics[:, [0, 1], [0, 1]] = np.array([[980.0, 980.0], [1000.0, 1000.0], [1040.0, 1040.0], [600.0, 600.0]])
    resolution_wh: Int64[ndarray, "4 2"] = np.array([[1920, 1080], [1920, 1080], [1920, 1080], [1280, 720]])

    shared: Float64[ndarray, "4 3 3"] = one_focal_per_camera_model(intrinsics, resolution_wh)

    np.testing.assert_array_equal(shared[:, 0, 0], [1000.0, 1000.0, 1000.0, 600.0])
    np.testing.assert_array_equal(shared[:, 1, 1], [1000.0, 1000.0, 1000.0, 600.0])
    np.testing.assert_array_equal(shared[:, :2, 2], intrinsics[:, :2, 2])  # the principal points stay


def test_refine_undistorts_each_camera_through_its_lens() -> None:
    """Keypoints seen through a fisheye lens give back the exact rig when the lens is known; read as pinhole pixels, the lens
    near the axis passes for a focal change."""
    scene = FakeScene()
    lens = Lens("kannala_brandt", (-0.3, 0.1, 0.0, 0.0))
    K: Float64[ndarray, "3 3"] = scene.rig.intrinsics[0]
    exact: Float64[ndarray, "v t 133 2"] = np.stack([scene.project(frame) for frame in range(0, 48, 4)], axis=1)
    normalized = ((exact - K[:2, 2]) / K[0, 0]).reshape(-1, 1, 2)
    through_lens: Float64[ndarray, "v t 133 2"] = cv2.fisheye.distortPoints(normalized, K, np.asarray(lens.coefficients)).reshape(exact.shape)
    conf: Float64[ndarray, "v t 133"] = np.full(exact.shape[:3], 0.95)

    with_lens = refine_through_lenses([lens] * 4, through_lens, conf, scene.rig, device="cpu")
    as_pinhole = refine_through_lenses([PINHOLE] * 4, through_lens, conf, scene.rig, device="cpu")

    np.testing.assert_allclose(with_lens.cam_T_world, scene.rig.cam_T_world, atol=2e-3)
    np.testing.assert_allclose(with_lens.intrinsics, scene.rig.intrinsics, rtol=1e-4)
    assert abs(as_pinhole.intrinsics[0, 0, 0] / K[0, 0] - 1.0) > 1e-3
