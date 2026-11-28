"""The numpy reference solver: its linearisation against dense algebra and finite differences, and its optimum."""

from __future__ import annotations

import numpy as np
import pytest

from caspar_mojo.problem import BaOptions, BaProblem, CameraModel
from caspar_mojo.reference.solver import State, jtj_cross, linearize, make_groups, reference_solve, retract
from caspar_mojo.reference.synthetic import make_problem


def _state(problem: BaProblem) -> State:
    return State(problem.rig_from_world.copy(), problem.sensor_from_rig.copy(), problem.intrinsics.copy(), problem.points.copy())


def _dense_jacobian(problem: BaProblem, groups, jacobians) -> np.ndarray:
    n = len(problem.pixels)
    width = sum(len(g.nodes) * g.size for g in groups)
    dense = np.zeros((2 * n, width))
    offset = 0
    rows = np.arange(n)
    for g, jacobian in zip(groups, jacobians, strict=True):
        seen = g.observation_slot >= 0
        for r in range(2):
            for i in range(g.size):
                dense[2 * rows[seen] + r, offset + g.observation_slot[seen] * g.size + i] = jacobian[seen, r, i]
        offset += len(g.nodes) * g.size
    return dense


SMALL = {"n_frames": 5, "n_points": 23, "seed": 7, "rig_cameras": 2, "intrinsics_free": True, "sensor_free": True, "focal_noise": 0.01}


@pytest.mark.parametrize("model", list(CameraModel))
def test_gradient_is_minus_cost_gradient_through_retraction(model: CameraModel) -> None:
    problem, _ = make_problem(model, **SMALL)
    groups = make_groups(problem)
    state = _state(problem)
    lin = linearize(problem, state, groups)
    for index, g in enumerate(groups):
        for slot in range(0, len(g.nodes), max(1, len(g.nodes) // 3)):
            for i in range(g.size):
                step = 1e-6
                plus = [np.zeros((len(h.nodes), h.size)) for h in groups]
                minus = [np.zeros((len(h.nodes), h.size)) for h in groups]
                plus[index][slot, i] = step
                minus[index][slot, i] = -step
                numeric = (
                    linearize(problem, retract(problem, state, groups, plus), groups).cost
                    - linearize(problem, retract(problem, state, groups, minus), groups).cost
                ) / (2 * step)
                assert lin.gradient[index][slot, i] == pytest.approx(-numeric, rel=1e-5, abs=1e-6 * np.max(np.abs(lin.gradient[index]))), (
                    g.name,
                    slot,
                    i,
                )


@pytest.mark.parametrize("model", list(CameraModel))
def test_block_diagonal_and_cross_product_split_jtj(model: CameraModel) -> None:
    problem, _ = make_problem(model, **SMALL)
    groups = make_groups(problem)
    lin = linearize(problem, _state(problem), groups)
    dense = _dense_jacobian(problem, groups, lin.jacobians)
    jtj = dense.T @ dense
    block_part = np.zeros_like(jtj)
    offset = 0
    for g, blocks in zip(groups, lin.block_diagonal, strict=True):
        for slot in range(len(g.nodes)):
            start = offset + slot * g.size
            block_part[start : start + g.size, start : start + g.size] = blocks[slot]
        offset += len(g.nodes) * g.size
    np.testing.assert_allclose(block_part, np.where(block_part != 0, jtj, 0), rtol=1e-12, atol=1e-9 * np.abs(jtj).max())

    rng = np.random.default_rng(0)
    p = [rng.normal(size=(len(g.nodes), g.size)) for g in groups]
    w = [np.zeros_like(x) for x in p]
    jtj_cross(groups, lin, p, w)
    flat_p = np.concatenate([x.ravel() for x in p])
    np.testing.assert_allclose(np.concatenate([x.ravel() for x in w]), (jtj - block_part) @ flat_p, rtol=1e-9, atol=1e-9 * np.abs(jtj).max())


def _rms_floor(problem: BaProblem, noise_px: float) -> float:
    """Expected RMS per residual component at the optimum: the noise less what the free parameters absorb."""
    groups = make_groups(problem)
    residuals = 2 * len(problem.pixels)
    parameters = sum(len(g.nodes) * g.size for g in groups)
    return noise_px * np.sqrt((residuals - parameters) / residuals)


CONFIGS = [
    pytest.param(CameraModel.PINHOLE, {}, id="pinhole"),
    pytest.param(CameraModel.SIMPLE_RADIAL, {}, id="simple_radial"),
    pytest.param(CameraModel.OPENCV_FISHEYE, {}, id="opencv_fisheye"),
    pytest.param(CameraModel.FULL_OPENCV, {}, id="full_opencv"),
    pytest.param(CameraModel.SIMPLE_RADIAL, {"intrinsics_free": True, "focal_noise": 0.01}, id="simple_radial_intrinsics"),
    pytest.param(CameraModel.OPENCV_FISHEYE, {"intrinsics_free": True, "focal_noise": 0.01}, id="fisheye_intrinsics"),
    pytest.param(CameraModel.PINHOLE, {"rig_cameras": 3, "sensor_free": True}, id="pinhole_rig_sensor_free"),
    pytest.param(CameraModel.FULL_OPENCV, {"rig_cameras": 2}, id="full_opencv_rig_fixed"),
]


@pytest.mark.parametrize(("model", "flags"), CONFIGS)
def test_reaches_noise_floor(model: CameraModel, flags: dict) -> None:
    problem, _ = make_problem(model, n_frames=13, n_points=1031, seed=int(model), **flags)
    result = reference_solve(problem, BaOptions(solver_iter_max=25))
    assert result.final_cost < 0.05 * result.initial_cost
    costs = [record.cost_best for record in result.trace]
    assert all(b <= a for a, b in zip(costs, costs[1:], strict=False))
    rms = np.sqrt(2 * result.final_cost / (2 * len(problem.pixels)))
    assert rms == pytest.approx(_rms_floor(problem, 0.5), rel=0.05)


def test_reduced_and_full_system_reach_the_same_optimum() -> None:
    problem, _ = make_problem(CameraModel.PINHOLE, n_frames=17, n_points=1021, seed=3)
    reduced = reference_solve(problem, BaOptions(solver_iter_max=25))
    full = reference_solve(problem, BaOptions(solver_iter_max=25, reduced_camera_system=False))
    assert full.final_cost == pytest.approx(reduced.final_cost, rel=1e-3)


@pytest.mark.parametrize("model", list(CameraModel))
def test_noise_free_problem_converges_to_zero(model: CameraModel) -> None:
    problem, _ = make_problem(model, n_frames=11, n_points=1025, seed=5, noise_px=0.0)
    result = reference_solve(problem, BaOptions(solver_iter_max=40))
    assert result.final_cost < 1e-12 * result.initial_cost


def test_float32_run_follows_float64() -> None:
    problem, _ = make_problem(CameraModel.PINHOLE, n_frames=13, n_points=1023, seed=2)
    result64 = reference_solve(problem, BaOptions(solver_iter_max=25))
    result32 = reference_solve(problem, BaOptions(solver_iter_max=25), dtype=np.float32)
    trace32 = result32.trace
    assert result32.initial_cost == pytest.approx(result64.initial_cost, rel=1e-5)
    assert result32.final_cost == pytest.approx(result64.final_cost, rel=1e-4)
    assert result32.exit_reason == "CONVERGED_DIAG_EXIT"
    assert not trace32[-1].accepted


def test_fixed_nodes_come_back_unchanged() -> None:
    problem, _ = make_problem(CameraModel.PINHOLE, n_frames=7, n_points=101, seed=4, rig_cameras=2)
    result = reference_solve(problem, BaOptions(solver_iter_max=5))
    np.testing.assert_array_equal(result.rig_from_world[0], problem.rig_from_world[0])
    np.testing.assert_array_equal(result.sensor_from_rig, problem.sensor_from_rig)
    np.testing.assert_array_equal(result.intrinsics, problem.intrinsics)
    np.testing.assert_allclose(np.linalg.norm(result.rig_from_world[:, :4], axis=1), 1.0, rtol=1e-12)
