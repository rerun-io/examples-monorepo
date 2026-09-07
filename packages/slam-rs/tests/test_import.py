"""The compiled core imports and its boundary behaves as the stub promises."""

from typing import cast

import numpy as np
import pytest
from jaxtyping import UInt8
from numpy.typing import NDArray

from slam_rs import _core


def test_core_reports_a_version() -> None:
    assert _core.__version__


def test_a_frame_without_imu_needs_more_imu() -> None:
    vio: _core.Vio = _core.Vio(camera_count=2, min_imu_samples=1)
    assert vio.camera_count == 2
    image: UInt8[np.ndarray, "8 8"] = np.zeros((8, 8), dtype=np.uint8)
    result: _core.VioResult = vio.track(1_000, [image, image])
    assert result.status == _core.VioStatus.NeedMoreImu
    assert result.t_ns == 1_000
    assert result.world_from_rig.shape == (7,)
    assert result.velocity.shape == (3,)
    np.testing.assert_allclose(result.world_from_rig, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def test_push_imu_rejects_a_non_monotonic_timestamp() -> None:
    vio: _core.Vio = _core.Vio(camera_count=1)
    vio.push_imu(1_000, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    with pytest.raises(ValueError, match="does not follow"):
        vio.push_imu(1_000, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])


def test_push_imu_batch_accepts_arrays_and_lifts_the_status() -> None:
    vio: _core.Vio = _core.Vio(camera_count=1, min_imu_samples=2)
    t_ns: np.ndarray = np.arange(4, dtype=np.int64) * 1_000
    gyro: np.ndarray = np.zeros((4, 3), dtype=np.float64)
    accel: np.ndarray = np.tile(np.array([0.0, 0.0, 9.81]), (4, 1))
    vio.push_imu_batch(t_ns, gyro, accel)
    image: UInt8[np.ndarray, "4 6"] = np.zeros((4, 6), dtype=np.uint8)
    assert vio.track(10_000, [image]).status == _core.VioStatus.NotInitialised


def test_wrong_dtype_and_rank_raise_value_error() -> None:
    vio: _core.Vio = _core.Vio(camera_count=1)
    # The casts feed the boundary exactly what the stub forbids: that is the point
    # of the test, and the checker would otherwise reject the call statically.
    float_image: NDArray[np.uint8] = cast("NDArray[np.uint8]", np.zeros((4, 4), dtype=np.float32))
    int32_times: NDArray[np.int64] = cast("NDArray[np.int64]", np.zeros(2, dtype=np.int32))
    with pytest.raises(ValueError, match="2-D uint8"):
        vio.track(0, [float_image])
    with pytest.raises(ValueError, match="2-D uint8"):
        vio.track(0, [np.zeros(4, dtype=np.uint8)])
    with pytest.raises(ValueError, match="1-D int64"):
        vio.push_imu_batch(int32_times, np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        vio.push_imu_batch(np.zeros(2, dtype=np.int64), np.zeros((2, 4)), np.zeros((2, 3)))


def test_non_contiguous_rows_raise_value_error() -> None:
    vio: _core.Vio = _core.Vio(camera_count=1)
    strided: UInt8[np.ndarray, "8 4"] = np.zeros((8, 8), dtype=np.uint8)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        vio.track(0, [strided])


def test_a_frameset_of_the_wrong_width_raises_value_error() -> None:
    vio: _core.Vio = _core.Vio(camera_count=2)
    image: UInt8[np.ndarray, "4 4"] = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected 2 images"):
        vio.track(0, [image])
