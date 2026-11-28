"""The C ABI (`libcaspar_mojo.so`) against its one header, `include/caspar_mojo.h`, and against the Python door.

The header is the single source of the ABI. A C probe compiled against it reports every field's offset and type and
every constant; the ctypes declarations here and the Python constants must agree with it (a reordered same-width
field or a renumbered constant fails here, without a GPU). The integration tests then run the C program built against
the header, and solve one problem through ctypes and through the Python door: both call the same Mojo `solve`, so the
initial cost agrees to float32 summation noise and the final state to the float32 tail of the LM run.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from caspar_mojo.problem import MAX_INTRINSICS, BaOptions, CameraModel
from caspar_mojo.reference.synthetic import make_problem
from caspar_mojo.solver import EXIT_REASONS, solve

ROOT = Path(__file__).parents[1]
HEADER = ROOT / "include" / "caspar_mojo.h"
LIBRARY = ROOT / "caspar_mojo" / "libcaspar_mojo.so"

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


STRUCTS: dict[str, type[ctypes.Structure]] = {
    "CasparMojoProblem": Problem,
    "CasparMojoOptions": Options,
    "CasparMojoState": State,
    "CasparMojoStats": Stats,
}
"""Header struct name -> its ctypes declaration."""

KINDS: dict[object, str] = {ctypes.c_int64: "i64", ctypes.c_double: "f64", F32P: "f32*", U8P: "u8*", I32P: "i32*"}
"""ctypes type -> the probe's name for the C type."""

CONSTANTS: dict[str, int] = {
    "CASPAR_MOJO_MAX_INTRINSICS": MAX_INTRINSICS,
    **{f"CASPAR_MOJO_{model.name}": int(model) for model in CameraModel},
    **{f"CASPAR_MOJO_EXIT_{name}": code for code, name in enumerate(EXIT_REASONS)},
}
"""Header constants and the Python values they must equal."""

PROBE_PRELUDE = """#include <stddef.h>
#include <stdio.h>
#include "caspar_mojo.h"
#define KIND(x) _Generic((x), int64_t: "i64", double: "f64", const float*: "f32*", float*: "f32*", \\
                         const uint8_t*: "u8*", const int32_t*: "i32*", default: "?")
int main(void) {
"""


def _compiler() -> str:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler (cc) on PATH: the header checks compile against include/caspar_mojo.h")
    return compiler


def _probe(tmp_path: Path) -> dict[str, str]:
    """`name -> value` lines printed by a C program compiled against the header."""
    lines: list[str] = [PROBE_PRELUDE]
    for struct, declaration in STRUCTS.items():
        lines.append(f'  printf("{struct} %zu\\n", sizeof({struct}));')
        for field, _ in declaration._fields_:
            lines.append(f'  printf("{struct}.{field} %zu %s\\n", offsetof({struct}, {field}), KIND((({struct}*)0)->{field}));')
    lines += [f'  printf("{name} %d\\n", {name});' for name in ("CASPAR_MOJO_ABI_VERSION", *CONSTANTS)]
    lines.append("  return 0;\n}\n")
    source = tmp_path / "probe.c"
    source.write_text("\n".join(lines))
    binary = tmp_path / "probe"
    subprocess.run([_compiler(), "-std=c11", f"-I{HEADER.parent}", str(source), "-o", str(binary)], check=True)
    output = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    return dict(line.split(" ", 1) for line in output.splitlines())


def test_ctypes_and_python_constants_match_the_header(tmp_path: Path) -> None:
    probe = _probe(tmp_path)
    for struct, declaration in STRUCTS.items():
        assert int(probe[struct]) == ctypes.sizeof(declaration), struct
        for field, kind in declaration._fields_:
            assert probe[f"{struct}.{field}"] == f"{getattr(declaration, field).offset} {KINDS[kind]}", (struct, field)
    for name, value in CONSTANTS.items():
        assert int(probe[name]) == value, name


def _abi_version(tmp_path: Path) -> int:
    return int(_probe(tmp_path)["CASPAR_MOJO_ABI_VERSION"])


@pytest.mark.integration
def test_c_program_runs_against_the_header(c_library: ctypes.CDLL, tmp_path: Path) -> None:
    binary = tmp_path / "test_c_api"
    command = [_compiler(), "-O1", f"-I{HEADER.parent}", str(ROOT / "tests" / "c" / "test_c_api.c"), f"-L{LIBRARY.parent}", "-lcaspar_mojo"]
    subprocess.run([*command, f"-Wl,-rpath,{LIBRARY.parent}", "-o", str(binary)], check=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr


def _ptr(array: np.ndarray, pointer_type: type) -> object:
    return array.ctypes.data_as(pointer_type)


@pytest.mark.integration
def test_c_abi_matches_python_door(c_library: ctypes.CDLL, tmp_path: Path) -> None:
    c_library.caspar_mojo_solve.restype = ctypes.c_int32
    c_library.caspar_mojo_abi_version.restype = ctypes.c_int32
    assert c_library.caspar_mojo_abi_version() == _abi_version(tmp_path)

    problem = make_problem(CameraModel.PINHOLE, n_frames=13, n_points=1023, seed=11)[0]
    options = BaOptions(solver_iter_max=40)
    expected = solve(problem, options)

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
    code = c_library.caspar_mojo_solve(ctypes.byref(c_problem), ctypes.byref(c_options), ctypes.byref(state), ctypes.byref(stats), error, len(error))
    assert code == 0, error.value

    assert stats.initial_cost == pytest.approx(expected.initial_cost, rel=1e-5)
    assert stats.final_cost == pytest.approx(expected.final_cost, rel=1e-3)
    assert EXIT_REASONS[stats.exit_reason] == expected.exit_reason
    extent = float(np.ptp(problem.points, axis=0).max())
    np.testing.assert_allclose(outputs[3], expected.points, rtol=0, atol=1e-4 * extent)
    np.testing.assert_allclose(outputs[2][:, 4:], expected.rig_from_world[:, 4:], rtol=0, atol=1e-4 * extent)
    np.testing.assert_array_equal(outputs[2][~problem.pose_free], f32[2][~problem.pose_free])
