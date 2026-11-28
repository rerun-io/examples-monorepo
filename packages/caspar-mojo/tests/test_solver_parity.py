"""The Mojo solver against the numpy reference on synthetic problems.

Tolerances (the lead's parity ruling). The Mojo solver runs in float32 like CASPAR. Once the cost has converged,
every trial changes it by ~1e-6 relative, so float32 summation order decides its accept or reject (in CASPAR too)
and the length of that tail varies from run to run. So:
- the trace is compared trial by trial only up to convergence, the float32 reference's first trial whose best cost
  is within `CONVERGED` of its final cost (same accept flags, best cost within 1e-3). That trial itself is left out:
  its damping comes from the quality ratio of a decrease near the float32 noise. PCG iterations are compared per trial
  (within 1) only while the best cost is more than `PCG_CONVERGED` above the final one; closer to it the Q stop meets
  its threshold on float32 noise (the joint rig + intrinsics case: the same trial runs 11 to 19 iterations over five
  Mojo runs on the 3060), so there only the prefix's total is compared, within 30 %;
- costs against the float64 reference: initial within 1e-4, final within 1 %;
- the exit reason equals the float32 reference's, and the trial count is within a factor 2 of it (a gross check);
- the state is compared through what it predicts, the pixels, because each tail accept moves it along a nearly flat
  direction (the scale gauge, point depths) by more than float32 moves it elsewhere. The float64 reference ends at
  the minimum, where cost(x) - cost_min = |J (x - x_min)|^2 / 2 to second order: a state on the same minimum
  differs from it in pixels (L2 over every residual) by sqrt(2 (cost - cost_min)) and no more. Both costs are
  evaluated here in float64. A state on another minimum fails. The check sees only projections, so a move the pixels
  do not see passes it; fixed rows are therefore compared separately, exactly.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np
import pytest

from caspar_mojo.problem import BaIteration, BaOptions, BaProblem, BaResult, CameraModel
from caspar_mojo.reference.solver import COST_DECREASE_FLOOR, State, make_groups, reference_linearize, reference_solve, residuals_and_jacobians
from caspar_mojo.reference.synthetic import make_problem
from caspar_mojo.solver import linearize, solve

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("native")]

OPTIONS = BaOptions(solver_iter_max=40)
CONVERGED = 1e-5
"""Relative distance to the final cost at which a trace counts as converged."""
PCG_CONVERGED = 1e-4
"""Relative distance to the final cost below which the per-trial PCG count is float32 noise at the Q stop."""
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
    "joint_rig_intrinsics_free": (CameraModel.PINHOLE, (("rig_cameras", 2), ("sensor_free", True), ("intrinsics_free", True), ("focal_noise", 0.01))),
    "mixed_masks": (
        CameraModel.SIMPLE_RADIAL,
        (("rig_cameras", 3), ("sensor_free", True), ("intrinsics_free", True), ("focal_noise", 0.01), ("mixed", True)),
    ),
    "points_fixed": (CameraModel.PINHOLE, (("points_fixed", True),)),
}
"""Name -> (model, `make_problem` flags). Two flags edit the masks afterwards: `mixed` fixes every fourth point and
the last camera's intrinsics (both specialisations stay on, with fixed nodes inside them); `points_fixed` fixes every
point, so the reduced system has nothing to eliminate."""


@cache
def _problem(name: str) -> BaProblem:
    model, flags = CASES[name]
    options = {"n_frames": 13, "n_points": 1023, **dict(flags)}
    mixed, points_fixed = options.pop("mixed", False), options.pop("points_fixed", False)
    problem = make_problem(model, seed=11, **options)[0]
    if mixed:
        intrinsics_free = problem.intrinsics_free.copy()
        intrinsics_free[-1] = False
        problem = dataclasses.replace(problem, point_free=np.arange(len(problem.points)) % 4 != 0, intrinsics_free=intrinsics_free)
    if points_fixed:
        problem = dataclasses.replace(problem, point_free=np.zeros(len(problem.points), dtype=bool))
    return problem


@cache
def _reference(name: str, dtype: type, cost_decrease_floor: float = COST_DECREASE_FLOOR) -> BaResult:
    return reference_solve(_problem(name), OPTIONS, dtype=dtype, cost_decrease_floor=cost_decrease_floor)


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
    problem = _problem(name)
    mojo = solve(problem, OPTIONS)
    # The state bound measures the distance to the exact minimum, so the float64 reference polishes without the floor.
    reference64 = _reference(name, np.float64, 0.0)
    reference32 = _reference(name, np.float32)
    trace32 = reference32.trace

    converged = _converged(trace32)
    assert len(mojo.trace) >= converged, (len(mojo.trace), converged)
    for index, (trial, expected) in enumerate(zip(mojo.trace[:converged], trace32[:converged], strict=True)):
        assert isinstance(trial, BaIteration)
        assert trial.accepted == expected.accepted, (index, trial, expected)
        assert trial.cost_best == pytest.approx(expected.cost_best, rel=1e-3), (index, trial, expected)
        if expected.cost_best > trace32[-1].cost_best * (1 + PCG_CONVERGED):
            assert abs(trial.pcg_iterations - expected.pcg_iterations) <= 1, (index, trial, expected)
    prefix_pcg = sum(t.pcg_iterations for t in mojo.trace[:converged])
    reference_pcg = sum(t.pcg_iterations for t in trace32[:converged])
    assert prefix_pcg == pytest.approx(reference_pcg, rel=0.3), (prefix_pcg, reference_pcg)
    assert mojo.initial_cost == pytest.approx(reference64.initial_cost, rel=1e-4)
    assert mojo.final_cost == pytest.approx(reference64.final_cost, rel=1e-2)
    assert mojo.exit_reason == reference32.exit_reason
    assert len(trace32) / 2 <= len(mojo.trace) <= 2 * len(trace32), (len(mojo.trace), len(trace32))

    residual64 = _residuals(problem, reference64)
    residual = _residuals(problem, mojo)
    cost_gap = 0.5 * float(np.sum(residual**2) - np.sum(residual64**2))
    pixel_distance = float(np.linalg.norm(residual - residual64))
    assert pixel_distance <= QUADRATIC_SLACK * np.sqrt(2 * max(cost_gap, 0.0)) + 1e-9, (pixel_distance, cost_gap)

    for refined, original, free in (
        (mojo.points, problem.points, problem.point_free),
        (mojo.rig_from_world, problem.rig_from_world, problem.pose_free),
        (mojo.sensor_from_rig, problem.sensor_from_rig, problem.sensor_free),
        (mojo.intrinsics, problem.intrinsics, problem.intrinsics_free),
    ):
        np.testing.assert_array_equal(refined[~free], original[~free])


def _with_observations(problem: BaProblem, rows: np.ndarray) -> BaProblem:
    return dataclasses.replace(
        problem,
        pixels=problem.pixels[rows],
        observation_frame=problem.observation_frame[rows],
        observation_camera=problem.observation_camera[rows],
        observation_point=problem.observation_point[rows],
    )


def test_observation_order_does_not_change_the_solve() -> None:
    """The solver regroups observations by (camera, frame) itself; only float32 summation order may differ."""
    problem = _problem("pinhole_rig_fixed")
    shuffled = _with_observations(problem, np.random.default_rng(0).permutation(len(problem.pixels)))
    ordered, permuted = solve(problem, OPTIONS), solve(shuffled, OPTIONS)
    assert permuted.final_cost == pytest.approx(ordered.final_cost, rel=1e-4)
    converged = _converged(list(ordered.trace))
    assert [t.accepted for t in permuted.trace[:converged]] == [t.accepted for t in ordered.trace[:converged]]


def test_empty_problem_returns_its_start() -> None:
    problem = _problem("pinhole")
    result = solve(_with_observations(problem, np.arange(0)), OPTIONS)
    assert (result.initial_cost, result.final_cost, result.exit_reason) == (0.0, 0.0, "CONVERGED_SCORE_THRESHOLD")
    np.testing.assert_allclose(result.points, problem.points, rtol=1e-6)
    np.testing.assert_allclose(result.rig_from_world, problem.rig_from_world, rtol=1e-6, atol=1e-7)


def test_two_trial_cap_matches_reference() -> None:
    """The last allowed trial is not relinearised (CASPAR's rule): pins the solver's two-slot state schedule."""
    problem = _problem("pinhole")
    options = dataclasses.replace(OPTIONS, solver_iter_max=2)
    mojo = solve(problem, options)
    trace32 = reference_solve(problem, options, dtype=np.float32).trace
    assert [t.accepted for t in mojo.trace] == [t.accepted for t in trace32]
    for trial, expected in zip(mojo.trace, trace32, strict=True):
        assert trial.cost_best == pytest.approx(expected.cost_best, rel=1e-3), (trial, expected)
    assert mojo.final_cost == pytest.approx(trace32[-1].cost_best, rel=1e-3)


@pytest.mark.parametrize("name", CASES)
def test_linearize_matches_reference(name: str) -> None:
    problem = _problem(name)
    cost, gradient, blocks = linearize(problem)
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
