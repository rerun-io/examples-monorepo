"""The Mojo solver against the numpy reference on synthetic problems.

Tolerances (the lead's parity ruling). The Mojo solver runs in float32 like CASPAR. Once the cost has converged,
every trial changes it by ~1e-6 relative, so float32 summation order decides its accept or reject (in CASPAR too)
and the length of that tail varies from run to run. So:
- the trace is compared trial by trial only up to convergence, the float32 reference's first trial whose best cost
  is within `CONVERGED` of its final cost (same accept flags, best cost within 1e-3, PCG iterations within 1). That
  trial itself is left out: its damping comes from the quality ratio of a decrease near the float32 noise;
- costs against the float64 reference: initial within 1e-4, final within 1 %;
- the exit reason equals the float32 reference's, and the trial count is within a factor 2 of it (a gross check);
- the state is compared through what it predicts, the pixels, because each tail accept moves it along a nearly flat
  direction (the scale gauge, point depths) by more than float32 moves it elsewhere. The float64 reference ends at
  the minimum, where cost(x) - cost_min = |J (x - x_min)|^2 / 2 to second order: a state on the same minimum
  differs from it in pixels (L2 over every residual) by sqrt(2 (cost - cost_min)) and no more. Both costs are
  evaluated here in float64. A state on another minimum, or one moved in a direction the cost does not see, fails.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from caspar_mojo.problem import BaIteration, BaOptions, BaProblem, BaResult, CameraModel
from caspar_mojo.reference.solver import State, make_groups, reference_linearize, reference_solve, residuals_and_jacobians
from caspar_mojo.reference.synthetic import make_problem

pytestmark = pytest.mark.integration

OPTIONS = BaOptions(solver_iter_max=40)
CONVERGED = 1e-5
"""Relative distance to the final cost at which a trace counts as converged."""
QUADRATIC_SLACK = 1.1
"""Room for the third-order terms of the cost around its minimum (measured: at most 1.01 on 12 cases x 2 GPUs)."""

CASES = {
    "pinhole": (CameraModel.PINHOLE, ()),
    "pinhole_1031": (CameraModel.PINHOLE, (("n_frames", 17), ("n_points", 1031))),
    "simple_radial": (CameraModel.SIMPLE_RADIAL, ()),
    "opencv_fisheye": (CameraModel.OPENCV_FISHEYE, ()),
    "full_opencv": (CameraModel.FULL_OPENCV, ()),
    "pinhole_intrinsics_free": (CameraModel.PINHOLE, (("intrinsics_free", True), ("focal_noise", 0.01))),
    "simple_radial_intrinsics_free": (CameraModel.SIMPLE_RADIAL, (("intrinsics_free", True), ("focal_noise", 0.01))),
    "opencv_fisheye_intrinsics_free": (CameraModel.OPENCV_FISHEYE, (("intrinsics_free", True), ("focal_noise", 0.01))),
    "full_opencv_intrinsics_free": (CameraModel.FULL_OPENCV, (("intrinsics_free", True), ("focal_noise", 0.01))),
    "pinhole_rig_fixed": (CameraModel.PINHOLE, (("rig_cameras", 3),)),
    "pinhole_rig_sensor_free": (CameraModel.PINHOLE, (("rig_cameras", 3), ("sensor_free", True))),
    "fisheye_rig_sensor_free": (CameraModel.OPENCV_FISHEYE, (("rig_cameras", 2), ("sensor_free", True))),
}


@cache
def _problem(name: str) -> BaProblem:
    model, flags = CASES[name]
    sizes = {"n_frames": 13, "n_points": 1023, **dict(flags)}
    return make_problem(model, seed=11, **sizes)[0]


@cache
def _reference(name: str, dtype: type) -> tuple[BaResult, list]:
    return reference_solve(_problem(name), OPTIONS, dtype=dtype)


def _solver():
    solver = pytest.importorskip("caspar_mojo.solver", reason="caspar_mojo.solver is not written yet (core worker)")
    if not hasattr(solver, "solve"):
        pytest.skip("caspar_mojo.solver has no solve() yet")
    pytest.importorskip("caspar_mojo._caspar_mojo", reason="the Mojo extension is not built: pixi run -e caspar-mojo-dev caspar-mojo-build")
    return solver


def _solve(solver, problem: BaProblem) -> BaResult:
    try:
        return solver.solve(problem, OPTIONS)
    except RuntimeError as error:
        if "GPU" in str(error) or "device" in str(error).lower():
            pytest.skip(f"no GPU answered: {error}")
        raise


def _converged(trace: list) -> int:
    """Trials before the first whose best cost is within `CONVERGED` of the final one."""
    final = trace[-1].cost_best
    return next(i for i, trial in enumerate(trace) if trial.cost_best <= final * (1 + CONVERGED))


def _residuals(problem: BaProblem, result: BaResult) -> np.ndarray:
    state = State(result.rig_from_world, result.sensor_from_rig, result.intrinsics, result.points)
    residual, _ = residuals_and_jacobians(problem, state, make_groups(problem), with_jacobians=False)
    return residual


@pytest.mark.parametrize("name", CASES)
def test_solve_matches_reference(name: str) -> None:
    solver = _solver()
    problem = _problem(name)
    mojo = _solve(solver, problem)
    reference64, _ = _reference(name, np.float64)
    reference32, trace32 = _reference(name, np.float32)

    converged = _converged(trace32)
    assert len(mojo.trace) >= converged, (len(mojo.trace), converged)
    for index, (trial, expected) in enumerate(zip(mojo.trace[:converged], trace32[:converged], strict=True)):
        assert isinstance(trial, BaIteration)
        assert trial.accepted == expected.accepted, (index, trial, expected)
        assert trial.cost_best == pytest.approx(expected.cost_best, rel=1e-3), (index, trial, expected)
        assert abs(trial.pcg_iterations - expected.pcg_iterations) <= 1, (index, trial, expected)
    assert mojo.initial_cost == pytest.approx(reference64.initial_cost, rel=1e-4)
    assert mojo.final_cost == pytest.approx(reference64.final_cost, rel=1e-2)
    assert mojo.exit_reason == reference32.exit_reason
    assert len(trace32) / 2 <= len(mojo.trace) <= 2 * len(trace32), (len(mojo.trace), len(trace32))

    residual64 = _residuals(problem, reference64)
    residual = _residuals(problem, mojo)
    cost_gap = 0.5 * float(np.sum(residual**2) - np.sum(residual64**2))
    pixel_distance = float(np.linalg.norm(residual - residual64))
    assert pixel_distance <= QUADRATIC_SLACK * np.sqrt(2 * max(cost_gap, 0.0)) + 1e-9, (pixel_distance, cost_gap)


@pytest.mark.parametrize("name", CASES)
def test_linearize_matches_reference(name: str) -> None:
    solver = _solver()
    if not hasattr(solver, "linearize"):
        pytest.skip("caspar_mojo.solver exposes no linearize() hook")
    problem = _problem(name)
    cost, gradient, blocks = solver.linearize(problem)
    reference_cost, reference_gradient, reference_blocks = reference_linearize(problem)
    assert cost == pytest.approx(reference_cost, rel=1e-4)
    gradient = np.asarray(gradient, dtype=np.float64).ravel()
    assert gradient.shape == reference_gradient.shape
    # float32 accumulation over ~2k observations per pose: 1e-4 of the largest entry is far above its rounding.
    np.testing.assert_allclose(gradient, reference_gradient, rtol=0, atol=1e-4 * np.abs(reference_gradient).max())
    blocks = [np.asarray(b, dtype=np.float64) for b in blocks if len(b)]
    assert [b.shape for b in blocks] == [b.shape for b in reference_blocks]
    for block, reference_block in zip(blocks, reference_blocks, strict=True):
        np.testing.assert_allclose(block, reference_block, rtol=0, atol=1e-4 * np.abs(reference_block).max())
