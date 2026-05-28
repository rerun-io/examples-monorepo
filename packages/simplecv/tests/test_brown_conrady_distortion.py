"""Parity checks for Brown–Conrady batched projection against OpenCV."""

from __future__ import annotations

import cv2
import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.sensors.camera.brown_conrady import project_brown_conrady_diagonal, project_brown_conrady_grid


@composite
def brown_conrady_case(draw: DrawFn) -> tuple[np.ndarray, list[PinholeParameters]]:
    """Random batched scenario with per-view Brown–Conrady coefficients."""

    n_frames = draw(st.integers(min_value=1, max_value=2))
    n_points = draw(st.integers(min_value=1, max_value=48))
    n_views = draw(st.integers(min_value=1, max_value=3))

    xyz_world = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_frames, n_points, 3),
            elements=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False),
        )
    )
    # keep points in front of camera to avoid divide-by-zero in projection
    xyz_world[..., 2] = np.abs(xyz_world[..., 2]) + 0.5

    pinholes: list[PinholeParameters] = []
    for _ in range(n_views):
        fx = draw(st.floats(min_value=200.0, max_value=1500.0))
        fy = draw(st.floats(min_value=200.0, max_value=1500.0))
        cx = draw(st.floats(min_value=100.0, max_value=1200.0))
        cy = draw(st.floats(min_value=100.0, max_value=800.0))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        intrinsics = Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=K, height=1080, width=1920)

        dist_vals = draw(
            hnp.arrays(
                dtype=np.float64,
                shape=(14,),
                elements=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
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

        extrinsics = Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3))
        pinholes.append(PinholeParameters(name="view", intrinsics=intrinsics, extrinsics=extrinsics, distortion=distortion))

    return xyz_world, pinholes


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(brown_conrady_case())
def test_brown_conrady_matches_opencv(case: tuple[np.ndarray, list[PinholeParameters]]) -> None:
    """NumPy Brown–Conrady projection should match OpenCV's projectPoints for each view."""

    xyz_world, pinholes = case
    n_frames, n_points, _ = xyz_world.shape
    n_views = len(pinholes)

    uv_bc = project_brown_conrady_grid(
        xyz_stack_world=xyz_world.astype(np.float64),
        pinholes_per_view=pinholes,
        filter_invalid=False,
    )
    assert uv_bc.shape == (n_frames, n_views, n_points, 2)

    uv_cv = np.empty_like(uv_bc)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    for view_idx, pinhole in enumerate(pinholes):
        assert pinhole.distortion is not None
        pts3d = xyz_world.reshape(-1, 3)
        dist_vec = np.array(
            [
                pinhole.distortion.k1,
                pinhole.distortion.k2,
                pinhole.distortion.p1,
                pinhole.distortion.p2,
                pinhole.distortion.k3,
                pinhole.distortion.k4,
                pinhole.distortion.k5,
                pinhole.distortion.k6,
                pinhole.distortion.s1,
                pinhole.distortion.s2,
                pinhole.distortion.s3,
                pinhole.distortion.s4,
                pinhole.distortion.tau_x,
                pinhole.distortion.tau_y,
            ],
            dtype=np.float64,
        )
        uv_flat, _ = cv2.projectPoints(
            pts3d,
            rvec,
            tvec,
            np.asarray(pinhole.intrinsics.k_matrix, dtype=np.float64),
            dist_vec,
        )
        uv_cv[:, view_idx, :, :] = uv_flat.reshape(n_frames, n_points, 2)

    np.testing.assert_allclose(uv_bc, uv_cv, rtol=1e-9, atol=1e-9)


def test_brown_conrady_diagonal_matches_opencv_with_shared_distortion() -> None:
    """Frame-aligned projection should match OpenCV when distortion is shared."""

    xyz_world = np.array(
        [
            [[0.1, -0.2, 1.5], [0.3, 0.1, 2.0]],
            [[-0.2, 0.15, 1.2], [0.05, -0.1, 1.8]],
            [[0.25, 0.05, 2.4], [-0.15, 0.2, 1.6]],
        ],
        dtype=np.float64,
    )
    K = np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intrinsics = Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=K, height=720, width=1280)
    distortion = BrownConradyDistortion(
        k1=0.01,
        k2=-0.005,
        p1=0.0003,
        p2=-0.0002,
        k3=0.0001,
        k4=0.0002,
        k5=-0.0001,
        k6=0.00005,
    )
    pinholes: list[PinholeParameters] = []
    for frame_idx in range(xyz_world.shape[0]):
        extrinsics = Extrinsics(
            cam_R_world=np.eye(3),
            cam_t_world=np.array([0.01 * frame_idx, -0.02 * frame_idx, 0.0], dtype=np.float64),
        )
        pinholes.append(
            PinholeParameters(
                name=f"frame_{frame_idx}",
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                distortion=distortion,
            )
        )

    uv_bc = project_brown_conrady_diagonal(xyz_world, pinholes, filter_invalid=False)

    uv_cv = np.empty_like(uv_bc)
    rvec = np.zeros((3, 1), dtype=np.float64)
    dist_vec = np.array(
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
    )
    for frame_idx, pinhole in enumerate(pinholes):
        xyz_h = np.concatenate([xyz_world[frame_idx], np.ones((xyz_world.shape[1], 1), dtype=np.float64)], axis=-1)
        xyz_cam_h = xyz_h @ pinhole.extrinsics.cam_T_world.T
        xyz_cam = xyz_cam_h[:, :3] / xyz_cam_h[:, 3:]
        uv_flat, _ = cv2.projectPoints(
            xyz_cam,
            rvec,
            np.zeros((3, 1), dtype=np.float64),
            K,
            dist_vec,
        )
        uv_cv[frame_idx] = uv_flat.reshape(xyz_world.shape[1], 2)

    np.testing.assert_allclose(uv_bc, uv_cv, rtol=1e-9, atol=1e-9)
