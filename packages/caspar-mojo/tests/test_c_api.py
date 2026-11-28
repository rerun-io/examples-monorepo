"""The C ABI (`libcaspar_mojo.so`, `include/caspar_mojo.h`) against the Python door on the same problem.

Both call the same Mojo `solve`; only the boundary differs (ctypes structs vs `(address, *shape)` tuples), so the
initial cost must agree to float32 summation noise and the final state to the float32 tail of the LM run.
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest
from caspar_mojo.reference.synthetic import make_problem

from caspar_mojo.problem import BaOptions, CameraModel

pytestmark = pytest.mark.integration

LIBRARY = Path(__file__).parents[1] / "caspar_mojo" / "libcaspar_mojo.so"

F32P = ctypes.POINTER(ctypes.c_float)
U8P = ctypes.POINTER(ctypes.c_uint8)
I32P = ctypes.POINTER(ctypes.c_int32)


class Problem(ctypes.Structure):
    _fields_ = [
        ("camera_model", ctypes.c_int64),
        ("num_frames", ctypes.c_int64),
        ("num_cameras", ctypes.c_int64),
        ("num_points", ctypes.c_int64),
        ("num_observations", ctypes.c_int64),
        ("intrinsics", F32P),
        ("intrinsics_free", U8P),
        ("sensor_from_rig", F32P),
        ("sensor_free", U8P),
        ("rig_from_world", F32P),
        ("pose_free", U8P),
        ("points", F32P),
        ("point_free", U8P),
        ("pixels", F32P),
        ("observation_frame", I32P),
        ("observation_camera", I32P),
        ("observation_point", I32P),
    ]


class Options(ctypes.Structure):
    _fields_ = [
        ("device_id", ctypes.c_int64),
        ("solver_iter_max", ctypes.c_int64),
        ("diag_init", ctypes.c_double),
        ("diag_min", ctypes.c_double),
        ("diag_scaling_up", ctypes.c_double),
        ("diag_scaling_down", ctypes.c_double),
        ("diag_exit_value", ctypes.c_double),
        ("score_exit_value", ctypes.c_double),
        ("solver_rel_decrease_min", ctypes.c_double),
        ("reduced_pcg_iter_max", ctypes.c_int64),
        ("reduced_pcg_eta", ctypes.c_double),
    ]


class State(ctypes.Structure):
    _fields_ = [("intrinsics", F32P), ("sensor_from_rig", F32P), ("rig_from_world", F32P), ("points", F32P)]


class Stats(ctypes.Structure):
    _fields_ = [
        ("initial_cost", ctypes.c_double),
        ("final_cost", ctypes.c_double),
        ("iterations", ctypes.c_int64),
        ("pcg_iterations", ctypes.c_int64),
        ("exit_reason", ctypes.c_int64),
        ("solve_seconds", ctypes.c_double),
        ("total_seconds", ctypes.c_double),
        ("launches", ctypes.c_int64),
    ]


def _ptr(array: np.ndarray, pointer_type: type) -> object:
    return array.ctypes.data_as(pointer_type)


def test_c_abi_matches_python_door() -> None:
    if not LIBRARY.exists():
        pytest.skip(f"{LIBRARY.name} is not built: pixi run -e caspar-mojo-dev caspar-mojo-build-c")
    solver = pytest.importorskip("caspar_mojo.solver", reason="the Mojo extension is not built")
    library = ctypes.CDLL(str(LIBRARY))
    library.caspar_mojo_solve.restype = ctypes.c_int32
    library.caspar_mojo_abi_version.restype = ctypes.c_int32
    assert library.caspar_mojo_abi_version() == 1

    problem = make_problem(CameraModel.PINHOLE, n_frames=13, n_points=1023, seed=11)[0]
    options = BaOptions(solver_iter_max=40)
    expected = solver.solve(problem, options)

    f32 = [
        np.ascontiguousarray(a, dtype=np.float32)
        for a in (problem.intrinsics, problem.sensor_from_rig, problem.rig_from_world, problem.points, problem.pixels)
    ]
    u8 = [np.ascontiguousarray(a, dtype=np.uint8) for a in (problem.intrinsics_free, problem.sensor_free, problem.pose_free, problem.point_free)]
    i32 = [np.ascontiguousarray(a, dtype=np.int32) for a in (problem.observation_frame, problem.observation_camera, problem.observation_point)]
    c_problem = Problem(
        int(CameraModel.PINHOLE),
        len(problem.rig_from_world),
        len(problem.intrinsics),
        len(problem.points),
        len(problem.pixels),
        _ptr(f32[0], F32P),
        _ptr(u8[0], U8P),
        _ptr(f32[1], F32P),
        _ptr(u8[1], U8P),
        _ptr(f32[2], F32P),
        _ptr(u8[2], U8P),
        _ptr(f32[3], F32P),
        _ptr(u8[3], U8P),
        _ptr(f32[4], F32P),
        *(_ptr(a, I32P) for a in i32),
    )
    c_options = Options(
        0,
        options.solver_iter_max,
        options.diag_init,
        options.diag_min,
        options.diag_scaling_up,
        options.diag_scaling_down,
        options.diag_exit_value,
        options.score_exit_value,
        options.solver_rel_decrease_min,
        options.reduced_pcg_iter_max,
        options.reduced_pcg_eta,
    )
    outputs = [np.zeros_like(a) for a in f32[:4]]
    state = State(*(_ptr(a, F32P) for a in outputs))
    stats = Stats()
    error = ctypes.create_string_buffer(1024)
    code = library.caspar_mojo_solve(ctypes.byref(c_problem), ctypes.byref(c_options), ctypes.byref(state), ctypes.byref(stats), error, len(error))
    assert code == 0, error.value

    assert stats.initial_cost == pytest.approx(expected.initial_cost, rel=1e-5)
    assert stats.final_cost == pytest.approx(expected.final_cost, rel=1e-3)
    assert ("MAX_ITERATIONS", "CONVERGED_DIAG_EXIT", "CONVERGED_SCORE_THRESHOLD")[stats.exit_reason] == expected.exit_reason
    extent = float(np.ptp(problem.points, axis=0).max())
    np.testing.assert_allclose(outputs[3], expected.points, rtol=0, atol=1e-4 * extent)
    np.testing.assert_allclose(outputs[2][:, 4:], expected.rig_from_world[:, 4:], rtol=0, atol=1e-4 * extent)
    np.testing.assert_array_equal(outputs[2][~problem.pose_free], f32[2][~problem.pose_free])
