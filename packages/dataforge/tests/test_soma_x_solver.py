"""The fast paths of the SOMA-X fit match what they replace: scipy's axis-angle, upstream's normal-equation solver."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

pytest.importorskip("soma", reason="SOMA-X is installed only in the dataforge-soma-x environments")
torch = pytest.importorskip("torch")

from soma.fitting import pose_inversion  # noqa: E402

from dataforge.soma_x_fit import rotvecs_from_matrices, rotvecs_to_matrices, solve_normal_equations  # noqa: E402


def test_axis_angle_matches_scipy_near_zero_and_near_pi() -> None:
    rotvecs: Float64[ndarray, "n 3"] = Rotation.random(20_000, random_state=1).as_rotvec()
    rotvecs[:500] *= 1e-5
    rotvecs[500:1000] *= np.pi / np.linalg.norm(rotvecs[500:1000], axis=1, keepdims=True) * (1 - 1e-7)
    matrices: Float32[ndarray, "n 3 3"] = Rotation.from_rotvec(rotvecs).as_matrix().astype(np.float32)
    expected: Float64[ndarray, "n 3"] = Rotation.from_matrix(matrices.astype(np.float64)).as_rotvec()
    got: Float64[ndarray, "n 3"] = rotvecs_from_matrices(torch.as_tensor(matrices)).numpy()
    near_pi = np.linalg.norm(expected, axis=1) > np.pi - 1e-3
    np.testing.assert_allclose(got[~near_pi], expected[~near_pi], atol=1e-6)
    # At pi the axis sign is a free choice: compare the rotations.
    np.testing.assert_allclose(Rotation.from_rotvec(got[near_pi]).as_matrix(), matrices[near_pi], atol=1e-6)


def test_axis_angle_round_trips_through_rotvecs_to_matrices() -> None:
    rotations: Float32[ndarray, "4 5 3"] = np.random.default_rng(3).normal(0.0, 1.0, (4, 5, 3)).astype(np.float32)
    back: Float64[ndarray, "4 5 3"] = rotvecs_from_matrices(torch.as_tensor(rotvecs_to_matrices(rotations))).numpy()
    np.testing.assert_allclose(Rotation.from_rotvec(back.reshape(-1, 3)).as_matrix(), rotvecs_to_matrices(rotations).reshape(-1, 3, 3), atol=1e-6)


def test_cholesky_solve_matches_upstream_and_falls_back_on_a_singular_frame() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    jacobians: Float32[ndarray, "4 40 12"] = rng.normal(size=(4, 40, 12)).astype(np.float32)
    jtj = torch.as_tensor(jacobians.transpose(0, 2, 1) @ jacobians)
    jtj[3] = 0.0
    jtj[3, 0, 0] = 1.0  # rank one: Cholesky fails, upstream's damping ladder and diagonal fallback take it
    rhs = torch.as_tensor(rng.normal(size=(4, 12)).astype(np.float32))
    got = solve_normal_equations(jtj, rhs)
    expected = pose_inversion._solve_lie_gn_normal_equations(jtj, rhs)
    torch.testing.assert_close(got[:3], expected[:3], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[3], expected[3])
