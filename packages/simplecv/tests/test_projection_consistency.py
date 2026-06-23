"""Property-based checks for camera projection helpers."""

from __future__ import annotations

import hypothesis.extra.numpy as hnp
import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.sensors.camera.base_camera import world_to_cam_batched
from simplecv.sensors.camera.brown_conrady import cam_to_image_batched


@st.composite
def projection_case(draw: st.DataObject) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random multi-view projection scenario."""

    n_frames = draw(st.integers(min_value=1, max_value=3))
    n_points = draw(st.integers(min_value=1, max_value=6))
    n_views = draw(st.integers(min_value=1, max_value=4))

    xyz_world = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_frames, n_points, 3),
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        )
    )

    cam_T_world = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_views, 4, 4),
            elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        )
    )
    # enforce last row [0,0,0,1]
    cam_T_world[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0])

    K = draw(
        hnp.arrays(
            dtype=np.float64,
            shape=(n_views, 3, 3),
            elements=st.floats(min_value=50.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
        )
    )
    for view in range(n_views):
        K[view] = np.array(
            [
                [max(K[view, 0, 0], 50.0), 0.0, 0.0],
                [0.0, max(K[view, 1, 1], 50.0), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    xyz_hom = np.concatenate([xyz_world, np.ones((n_frames, n_points, 1), dtype=np.float64)], axis=-1)

    return xyz_world, xyz_hom, cam_T_world, K


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(projection_case())
def test_cam_projection_matches_matrix(case: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    xyz_world, xyz_hom, cam_T_world, K = case
    n_views = cam_T_world.shape[0]

    projection_matrices = np.einsum("vij,vjk->vik", K, cam_T_world[:, :3, :])

    uv_matrix = proj_3d_vectorized(xyz_hom=xyz_hom, P=projection_matrices)
    xyz_cam = world_to_cam_batched(xyz_world=xyz_world, cam_T_world=cam_T_world)
    uv_factored = cam_to_image_batched(xyz_cam=xyz_cam, K=K)

    assert uv_matrix.shape == uv_factored.shape == (xyz_world.shape[0], n_views, xyz_world.shape[1], 2)

    stable_depth = np.abs(xyz_cam[..., 2:3]) > 1e-6
    mask = np.isfinite(uv_matrix) & stable_depth
    assume(mask.any())
    np.testing.assert_allclose(uv_factored[mask], uv_matrix[mask], rtol=1e-6, atol=1e-6)
