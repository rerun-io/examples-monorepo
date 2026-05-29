"""Property tests for Brown–Conrady batched undistortion."""

from __future__ import annotations

import cv2
import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from simplecv.camera_parameters import BrownConradyDistortion
from simplecv.sensors.camera.brown_conrady import _distort_normalized_points, undistort_brown_conrady_batch


@st.composite
def brown_conrady_case(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, BrownConradyDistortion]:
    fx = draw(st.floats(min_value=300.0, max_value=1500.0))
    fy = draw(st.floats(min_value=300.0, max_value=1500.0))
    cx = draw(st.floats(min_value=200.0, max_value=1000.0))
    cy = draw(st.floats(min_value=150.0, max_value=900.0))
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    dist_vals = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(14,),
            elements=st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
        )
    )
    distortion = BrownConradyDistortion(
        k1=float(dist_vals[0]),
        k2=float(dist_vals[1]),
        p1=float(dist_vals[2]),
        p2=float(dist_vals[3]),
        k3=float(dist_vals[4]),
        k4=float(dist_vals[5]),
        k5=float(dist_vals[6]),
        k6=float(dist_vals[7]),
        s1=float(dist_vals[8]),
        s2=float(dist_vals[9]),
        s3=float(dist_vals[10]),
        s4=float(dist_vals[11]),
        tau_x=float(dist_vals[12]),
        tau_y=float(dist_vals[13]),
    )

    n_frames = draw(st.integers(min_value=1, max_value=2))
    n_kpts = draw(st.integers(min_value=1, max_value=32))

    uv_norm = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_frames, 1, n_kpts, 2),
            elements=st.floats(min_value=-0.8, max_value=0.8, allow_nan=False, allow_infinity=False),
        )
    )
    uv_dist_norm = _distort_normalized_points(uv_norm.reshape(-1, 2), distortion).reshape(uv_norm.shape)
    uv_dist_pix = np.empty_like(uv_dist_norm)
    uv_dist_pix[..., 0] = uv_dist_norm[..., 0] * fx + cx
    uv_dist_pix[..., 1] = uv_dist_norm[..., 1] * fy + cy

    return K, uv_dist_pix, distortion


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(brown_conrady_case())
def test_brown_conrady_undistort_matches_opencv(case: tuple[np.ndarray, np.ndarray, BrownConradyDistortion]) -> None:
    K, uv_dist_pix, distortion = case
    n_frames, _, n_kpts, _ = uv_dist_pix.shape

    uv_ud = undistort_brown_conrady_batch(
        uv_distorted=uv_dist_pix,
        intrinsics_stack=np.repeat(K[None, ...], repeats=1, axis=0),
        distortions=[distortion],
    )

    pts_flat = uv_dist_pix.reshape(-1, 2)
    cv_out = cv2.undistortPoints(
        pts_flat[:, None, :],
        cameraMatrix=K,
        distCoeffs=np.array(
            [
                distortion.k1,
                distortion.k2,
                distortion.p1,
                distortion.p2,
                distortion.k3,
                distortion.k4,
                distortion.k5,
                distortion.k6,
                distortion.s1,
                distortion.s2,
                distortion.s3,
                distortion.s4,
                distortion.tau_x,
                distortion.tau_y,
            ],
            dtype=np.float64,
        ),
        P=K,
    ).reshape(n_frames, 1, n_kpts, 2)

    np.testing.assert_allclose(uv_ud, cv_out, rtol=1e-7, atol=1e-7)


def test_brown_conrady_undistort_no_distortion_identity() -> None:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    uv = np.array([[[[100.0, 150.0], [200.0, 250.0]]]], dtype=np.float64)
    uv_ud = undistort_brown_conrady_batch(uv, intrinsics_stack=np.repeat(K[None, ...], 1, axis=0), distortions=[None])
    np.testing.assert_allclose(uv_ud, uv, rtol=0, atol=0)
