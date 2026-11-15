import numpy as np
from conftest import observe_points
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.correspondences import ObservationSet
from exo_calib.focal import FocalRefinementResult, refine_focals


def test_refine_focals_recovers_per_camera_scale_with_fixed_poses_and_points() -> None:
    n_views: int = 3
    n_points: int = 80
    cam_T_world: Float64[ndarray, "3 4 4"] = np.repeat(np.eye(4, dtype=np.float64)[None], n_views, axis=0)
    cam_T_world[1, 0, 3] = -0.8
    cam_T_world[2, 1, 3] = -0.6
    true_intrinsics: Float64[ndarray, "3 3 3"] = np.zeros((n_views, 3, 3), dtype=np.float64)
    true_intrinsics[:, 0, 0] = np.array([700.0, 760.0, 820.0], dtype=np.float64)
    true_intrinsics[:, 1, 1] = np.array([710.0, 750.0, 840.0], dtype=np.float64)
    true_intrinsics[:, 0, 2] = 640.0
    true_intrinsics[:, 1, 2] = 360.0
    true_intrinsics[:, 2, 2] = 1.0
    rng: np.random.Generator = np.random.default_rng(18)
    points_xyz: Float64[ndarray, "80 3"] = np.column_stack(
        (rng.uniform(-1.0, 1.0, n_points), rng.uniform(-0.8, 0.8, n_points), rng.uniform(3.0, 6.0, n_points))
    )
    observations: ObservationSet = observe_points(points_xyz, true_intrinsics, cam_T_world)
    initial_intrinsics: Float64[ndarray, "3 3 3"] = true_intrinsics.copy()
    initial_scale: Float64[ndarray, "3"] = np.array([0.95, 1.04, 0.96], dtype=np.float64)
    initial_intrinsics[:, 0, 0] *= initial_scale
    initial_intrinsics[:, 1, 1] *= initial_scale

    result: FocalRefinementResult = refine_focals(
        initial_intrinsics,
        cam_T_world,
        points_xyz,
        observations,
        iterations=400,
        learning_rate=0.05,
        robust_scale_px=2.0,
        device="cpu",
    )

    recovered_error: Float64[ndarray, "3"] = result.intrinsics[:, 0, 0] / true_intrinsics[:, 0, 0] - 1.0
    assert np.max(np.abs(recovered_error)) < 0.002
    np.testing.assert_allclose(
        result.intrinsics[:, 1, 1] / result.intrinsics[:, 0, 0],
        initial_intrinsics[:, 1, 1] / initial_intrinsics[:, 0, 0],
        atol=1e-12,
    )
    assert result.final_loss < result.initial_loss
