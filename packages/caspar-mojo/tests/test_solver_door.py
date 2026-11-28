"""The Python door's checks that need no GPU and no built extension.

`_FakeExtension` stands in for `_caspar_mojo`: like the Mojo solver it reads each array through its address as a
flat row-major buffer and writes each output the same way, so the door's layout and write-back contract is tested
on the CPU.
"""

import ctypes
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from caspar_mojo import solver as door
from caspar_mojo.problem import FOCAL_AND_EXTRA_COLUMNS, BaOptions, BaProblem, CameraModel
from caspar_mojo.reference.synthetic import make_problem

STEP = 0.25
"""What the fake solver adds to every state value it writes back."""


def _flat(array: np.ndarray) -> np.ndarray:
    """The float32 buffer behind a door array, as the native side sees it: from its address, row-major."""
    pointer = ctypes.cast(array.ctypes.data, ctypes.POINTER(ctypes.c_float))
    return np.ctypeslib.as_array(pointer, shape=(array.size,))


class _FakeExtension:
    def __init__(self) -> None:
        self.packets: list[object] = []

    def solve(self, packet: object, options: BaOptions, result: object) -> dict[str, object]:
        self.packets.append(packet)
        for name in ("intrinsics", "sensor_from_rig", "rig_from_world", "points"):
            _flat(getattr(result, name))[:] = _flat(getattr(packet, name)) + STEP
        return {
            "initial_cost": 2.0,
            "final_cost": 1.0,
            "iterations": 1,
            "pcg_iterations": 3,
            "exit_reason": 1,
            "solve_seconds": 0.0,
            "launches": 0,
            "trace": [(1.0, 1.0, 1.0, 3, True)],
        }


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakeExtension:
    extension = _FakeExtension()
    monkeypatch.setattr(door, "_extension", lambda: extension)
    return extension


def _small(model: CameraModel = CameraModel.PINHOLE, rig_cameras: int = 1, free: bool = False) -> BaProblem:
    return make_problem(model, n_frames=4, n_points=29, seed=3, rig_cameras=rig_cameras, intrinsics_free=free, sensor_free=free)[0]


def test_full_system_pcg_is_refused() -> None:
    with pytest.raises(NotImplementedError):
        door.solve(_small(), BaOptions(reduced_camera_system=False))


def test_mixed_camera_models_are_refused() -> None:
    problem = _small(rig_cameras=2)
    mixed = dataclasses.replace(problem, camera_model=np.array([CameraModel.PINHOLE, CameraModel.SIMPLE_RADIAL], dtype=np.int32))
    with pytest.raises(ValueError, match="one camera model per solve"):
        door.solve(mixed)


def test_fortran_order_state_comes_back_row_for_row(fake: _FakeExtension) -> None:
    problem = _small()
    fortran = dataclasses.replace(problem, points=np.asfortranarray(problem.points), rig_from_world=np.asfortranarray(problem.rig_from_world))
    result = door.solve(fortran)
    free = problem.point_free
    np.testing.assert_allclose(result.points[free], problem.points[free] + STEP, rtol=0, atol=1e-5)


def test_fixed_state_comes_back_bit_exact(fake: _FakeExtension) -> None:
    problem = _small(CameraModel.OPENCV_FISHEYE, rig_cameras=2, free=True)
    # float64 values that a float32 round trip would change
    point_free = np.arange(len(problem.points)) % 3 != 0
    intrinsics = problem.intrinsics + 1e-7
    problem = dataclasses.replace(problem, points=problem.points + 0.1, point_free=point_free, intrinsics=intrinsics)
    result = door.solve(problem)

    np.testing.assert_array_equal(result.points[~point_free], problem.points[~point_free])
    np.testing.assert_allclose(result.points[point_free], problem.points[point_free] + STEP, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(result.rig_from_world[~problem.pose_free], problem.rig_from_world[~problem.pose_free])
    np.testing.assert_array_equal(result.sensor_from_rig[~problem.sensor_free], problem.sensor_from_rig[~problem.sensor_free])
    refined = list(FOCAL_AND_EXTRA_COLUMNS[CameraModel.OPENCV_FISHEYE])
    kept = [c for c in range(intrinsics.shape[1]) if c not in refined]
    np.testing.assert_array_equal(result.intrinsics[:, kept], intrinsics[:, kept])
    np.testing.assert_allclose(result.intrinsics[:, refined], intrinsics[:, refined] + STEP, rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(result.rig_from_world[problem.pose_free, :4], axis=1), 1.0, rtol=1e-12)
    assert result.trace[0].pcg_iterations == 3
    assert result.exit_reason == "CONVERGED_DIAG_EXIT"


def test_packet_names_every_array_in_native_layout(fake: _FakeExtension) -> None:
    door.solve(_small())
    packet = fake.packets[0]
    assert packet.camera_model == CameraModel.PINHOLE
    for name, dtype in (("intrinsics", np.float32), ("pose_free", np.uint8), ("pixels", np.float32), ("observation_point", np.int32)):
        array = getattr(packet, name)
        assert array.dtype == dtype and array.flags.c_contiguous, name


# Each array passes its own annotation; only the cross-array contract is broken.
MALFORMED = {
    "short_index": (lambda p: {"observation_frame": p.observation_frame[:-1]}, "observation_frame"),
    "point_out_of_range": (
        lambda p: {"observation_point": np.where(np.arange(len(p.pixels)) == 5, len(p.points), p.observation_point).astype(np.int32)},
        "observation_point",
    ),
    "negative_camera": (lambda p: {"observation_camera": np.full(len(p.pixels), -1, dtype=np.int32)}, "observation_camera"),
    "short_mask": (lambda p: {"point_free": p.point_free[:-1]}, "point_free"),
    "long_mask": (lambda p: {"sensor_free": np.ones(len(p.sensor_free) + 1, dtype=bool)}, "sensor_free"),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_malformed_problems_are_refused_before_native_code(fake: _FakeExtension, name: str) -> None:
    change, message = MALFORMED[name]
    problem = _small()
    with pytest.raises(ValueError, match=message):
        door.solve(dataclasses.replace(problem, **change(problem)))
    assert not fake.packets


def test_linearize_is_refused_on_a_malformed_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(door, "_extension", lambda: SimpleNamespace())
    problem = _small()
    with pytest.raises(ValueError, match="observation_frame"):
        door.linearize(dataclasses.replace(problem, observation_frame=problem.observation_frame[:-1]))
