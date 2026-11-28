"""The Mojo solver against the numpy reference on synthetic problems.

Tolerances. The Mojo solver runs in float32 like CASPAR, so:
- costs are compared with the float64 reference (initial within 1e-4, final within 1 %);
- the iteration count and the exit reason are compared with the float32 reference run: in float64 the strict
  `trial < best` test keeps accepting tiny decreases until `solver_iter_max`, while float32 ends on the damping
  exit, so only the float32 run predicts the trajectory's length;
- poses and points may differ from the float64 reference by 10x what float32 alone moves them (the float32
  reference against the float64 one), plus a floor of 1e-6 of the scene extent: no float32 solver can do better
  than that, and a real bug moves the state by far more.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest
from reference.solver import reference_linearize, reference_solve
from reference.synthetic import make_problem

from caspar_mojo.problem import BaOptions, BaProblem, BaResult, CameraModel

pytestmark = pytest.mark.integration

OPTIONS = BaOptions(solver_iter_max=40)
NOT_YET = pytest.mark.xfail(strict=False, reason="core: not implemented yet")

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
SUPPORTED = {"pinhole", "pinhole_1031"}
"""Cases the core worker has implemented; the rest are expected to fail until it flips them on here."""


def _params() -> list:
    return [pytest.param(name, id=name, marks=() if name in SUPPORTED else NOT_YET) for name in CASES]


@cache
def _problem(name: str) -> BaProblem:
    model, flags = CASES[name]
    sizes = {"n_frames": 13, "n_points": 1023, **dict(flags)}
    return make_problem(model, seed=11, **sizes)[0]


@cache
def _reference(name: str, dtype: type) -> tuple[BaResult, int]:
    result, trace = reference_solve(_problem(name), OPTIONS, dtype=dtype)
    return result, len(trace)


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


def _rotation_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (rad) between the rotations of pose rows."""
    cosine = np.abs(np.sum(a[:, :4] * b[:, :4], axis=1)) / (np.linalg.norm(a[:, :4], axis=1) * np.linalg.norm(b[:, :4], axis=1))
    return 2 * np.arccos(np.clip(cosine, 0.0, 1.0))


def _state_differences(a: BaResult, b: BaResult) -> dict[str, float]:
    return {
        "rotation": float(np.max(_rotation_difference(a.rig_from_world, b.rig_from_world))),
        "translation": float(np.max(np.abs(a.rig_from_world[:, 4:] - b.rig_from_world[:, 4:]))),
        "sensor_rotation": float(np.max(_rotation_difference(a.sensor_from_rig, b.sensor_from_rig))),
        "sensor_translation": float(np.max(np.abs(a.sensor_from_rig[:, 4:] - b.sensor_from_rig[:, 4:]))),
        "points": float(np.max(np.abs(a.points - b.points))),
        "intrinsics": float(np.max(np.abs(a.intrinsics - b.intrinsics) / np.maximum(np.abs(b.intrinsics), 1.0))),
    }


@pytest.mark.parametrize("name", _params())
def test_solve_matches_reference(name: str) -> None:
    solver = _solver()
    problem = _problem(name)
    mojo = _solve(solver, problem)
    reference64, _ = _reference(name, np.float64)
    reference32, trace32_length = _reference(name, np.float32)

    assert mojo.initial_cost == pytest.approx(reference64.initial_cost, rel=1e-4)
    assert mojo.final_cost == pytest.approx(reference64.final_cost, rel=1e-2)
    assert mojo.exit_reason == reference32.exit_reason
    assert abs(mojo.iterations - reference32.iterations) <= 2, (mojo.iterations, reference32.iterations, trace32_length)

    extent = float(np.ptp(problem.points, axis=0).max())
    float32_effect = _state_differences(reference32, reference64)
    error = _state_differences(mojo, reference64)
    for key, value in error.items():
        floor = 1e-6 if key in ("rotation", "sensor_rotation", "intrinsics") else 1e-6 * extent
        assert value <= 10 * float32_effect[key] + floor, (key, value, float32_effect[key])


@pytest.mark.parametrize("name", _params())
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
