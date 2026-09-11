"""The compiled core imports, and the estimator's boundary behaves as the stub promises.

The rig, its frames and the pipeline over it are :mod:`conftest` fixtures, which
pytest injects: the factory aliases below are declared here rather than imported
from it, because ``tests`` is not on the typechecker's search path and every
module in this directory therefore stands alone. Everything runs on the 200x200
synthetic rig, so the whole file stays inside the default, seconds-long suite;
the reference segments are the V2 gate's business.
"""

from typing import cast

import numpy as np
import pytest
from fixture_types import FRAME, FRAME_PERIOD_NS, IMU_PERIOD_NS, PipelineFactory, RigFactory, TextureFactory, gravity_batch
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from numpy.typing import NDArray

from slam_rs import _core


def test_core_reports_a_version() -> None:
    assert _core.__version__


def test_a_frame_without_imu_needs_more_imu(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """Insufficient IMU coverage returns before any frontend mutation (D17).
    Retrying after samples arrive must match a run that had them from the start.
    """
    vio: _core.Vio = pipeline(2)
    assert vio.camera_count == 2
    result: _core.VioResult = vio.track(1_000, [texture(0, 0), texture(1, 0)])
    assert result.status == _core.VioStatus.NeedMoreImu
    assert result.t_ns == 1_000
    assert result.world_from_rig.shape == (7,)
    assert result.velocity.shape == (3,)
    np.testing.assert_allclose(result.world_from_rig, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # Nothing measured, so there is no window and no statistics to snapshot; and
    # the frontend never ran, so there are no keypoints either.
    assert vio.snapshot() is None
    assert vio.flow_frame() is None


def test_a_rig_of_one_camera_is_refused(rig: RigFactory) -> None:
    """``optical_flow.h:210``: the epipolar filter needs a second camera."""
    with pytest.raises(ValueError, match="expected 2 cameras, got 1"):
        _core.Vio(rig(1), _core.VioConfig())


def test_a_config_the_port_does_not_implement_is_refused(rig: RigFactory) -> None:
    """Offline mode cannot drop framesets without letting arrival order reach a decision."""
    raw: str = _core.VioConfig().to_json().replace('"config.vio_enforce_realtime": false', '"config.vio_enforce_realtime": true')
    with pytest.raises(ValueError, match="vio_enforce_realtime"):
        _core.Vio(rig(2), _core.VioConfig.from_json(raw))


def test_push_imu_rejects_a_non_monotonic_timestamp(pipeline: PipelineFactory) -> None:
    vio: _core.Vio = pipeline(2)
    vio.push_imu(1_000, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
    with pytest.raises(ValueError, match="does not follow"):
        vio.push_imu(1_000, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])


def test_a_frameset_retried_after_its_imu_tracks_as_if_it_had_it(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """D17: no arrival order may reach the trajectory.

    One pipeline gets every frameset's samples before the frameset. The other
    gets only the samples up to the frame time — which do not cover it — is
    refused, then gets the rest and tracks the same frameset again. The two
    trajectories agree bit for bit, which is what makes holding a refused
    frameset and retrying it the right thing for a feed to do.
    """
    frames: int = 8
    covered: _core.Vio = pipeline(2)
    retried: _core.Vio = pipeline(2)
    for index in range(frames):
        t_ns: int = index * FRAME_PERIOD_NS
        images: list[UInt8[ndarray, "h w"]] = [texture(index, 0), texture(index + 1, 0)]
        samples: Int64[ndarray, " n_samples"] = np.arange(t_ns, t_ns + FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
        gyro, accel = gravity_batch(samples)
        covered.push_imu_batch(samples, gyro, accel)
        first: _core.VioResult = covered.track(t_ns, images)

        short: Int64[ndarray, " n_short"] = samples[samples <= t_ns]
        retried.push_imu_batch(short, *gravity_batch(short))
        assert retried.track(t_ns, images).status == _core.VioStatus.NeedMoreImu
        rest: Int64[ndarray, " n_rest"] = samples[samples > t_ns]
        retried.push_imu_batch(rest, *gravity_batch(rest))
        second: _core.VioResult = retried.track(t_ns, images)

        assert first.status == second.status == _core.VioStatus.Tracking
        np.testing.assert_array_equal(first.world_from_rig, second.world_from_rig)
        np.testing.assert_array_equal(first.velocity, second.velocity)
        np.testing.assert_array_equal(first.gyro_bias, second.gyro_bias)


def test_a_refused_frameset_leaves_the_last_accepted_keypoints_alone(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """``flow_frame`` still describes the last frameset the frontend actually ran on.

    ``num_new`` is derived from the keypoint counter as it stood before that
    frameset, so a refused one must not overwrite it: the keypoints would all
    look old.
    """
    vio: _core.Vio = pipeline(2)
    samples: Int64[ndarray, " n_samples"] = np.arange(0, FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
    gyro, accel = gravity_batch(samples)
    vio.push_imu_batch(samples, gyro, accel)
    assert vio.track(0, [texture(0, 0), texture(1, 0)]).status == _core.VioStatus.Tracking
    accepted: _core.FlowFrame | None = vio.flow_frame()
    assert accepted is not None

    assert vio.track(FRAME_PERIOD_NS, [texture(1, 0), texture(2, 0)]).status == _core.VioStatus.NeedMoreImu
    refused: _core.FlowFrame | None = vio.flow_frame()
    assert refused is not None
    assert refused.t_ns == accepted.t_ns == 0
    assert refused.num_new(0) == accepted.num_new(0) > 0


def test_the_pipeline_tracks_a_shifted_scene_and_reports_its_window(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """End to end on synthetic pixels: the estimator initialises, tracks and fills a window.

    The scene translates by one pixel a frame, which is not a metric trajectory —
    the numbers the port is measured against are the reference segments (the V2
    gate). What this pins is that the whole path runs from Python and that every
    array the Rerun rung reads comes back with the shapes the stub declares.
    """
    vio: _core.Vio = pipeline(2)
    frames: int = 12
    tracked: int = 0
    for index in range(frames):
        t_ns: int = index * FRAME_PERIOD_NS
        # The samples of one frame period, pushed before the frameset they cover:
        # the estimator needs one strictly past ``t_ns`` or it reports NeedMoreImu.
        samples: Int64[ndarray, " n_samples"] = np.arange(t_ns, t_ns + FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
        gyro, accel = gravity_batch(samples)
        vio.push_imu_batch(samples, gyro, accel)
        result: _core.VioResult = vio.track(t_ns, [texture(index, 0), texture(index + 1, 0)])
        if result.status == _core.VioStatus.Tracking:
            tracked += 1
            current: _core.VioSnapshot | None = vio.snapshot()
            assert current is not None
            is_keyframe: bool = t_ns in current.kf_ids
            assert (current.timings_ms["keyframe"] > 0.0) == is_keyframe
            assert current.timings_ms["optimize"] >= sum(current.timings_ms[key] for key in ("linearize", "solver", "back_substitution", "error"))
            assert current.timings_ms["measure"] >= sum(current.timings_ms[key] for key in ("keyframe", "optimize", "marginalize"))
    assert tracked >= frames - 2, "only the framesets before the first covered one may fail to track"

    snapshot: _core.VioSnapshot | None = vio.snapshot()
    assert snapshot is not None
    window: int = len(snapshot.window_t_ns)
    assert window > 0
    assert snapshot.t_ns == snapshot.window_t_ns.max()
    assert snapshot.window_poses.shape == (window, 7)
    # Every keyframe is a window frame; the reverse is not true.
    assert set(snapshot.kf_ids.tolist()) <= set(snapshot.window_t_ns.tolist())
    # The per-frame flags and the id list are the same fact, so a frame is a
    # keyframe exactly where its timestamp is in the list.
    assert snapshot.window_keyframe.shape == (window,)
    assert snapshot.window_long_term.shape == (window,)
    np.testing.assert_array_equal(snapshot.window_keyframe, np.isin(snapshot.window_t_ns, snapshot.kf_ids))
    # A long-term keyframe is a keyframe.
    assert not np.any(snapshot.window_long_term & ~snapshot.window_keyframe)

    landmarks: int = len(snapshot.landmark_ids)
    assert landmarks > 0, "a textured scene should triangulate something"
    assert snapshot.landmark_positions.shape == (landmarks, 3)
    assert snapshot.landmark_hosts.shape == (landmarks,)
    assert set(snapshot.landmark_hosts.tolist()) <= set(snapshot.kf_ids.tolist()), "a landmark is hosted by a keyframe"
    assert snapshot.num_observations >= landmarks

    # The four LM scalars the Rerun rung logs, in the types the stub declares.
    assert isinstance(snapshot.lm_iterations, int)
    assert all(isinstance(value, float) for value in (snapshot.lm_lambda, snapshot.lm_error_before, snapshot.lm_error_after))
    # Estimator and frontend stage durations share one public map.
    assert set(snapshot.timings_ms) == {
        "back_substitution",
        "error",
        "linearize",
        "marginalize",
        "measure",
        "solver",
        "predict",
        "keyframe",
        "optimize",
        "frontend_stereo",
        "frontend_pyramid",
        "frontend_detect",
        "frontend_track",
        "frontend_imu",
    }
    assert all(milliseconds >= 0.0 for milliseconds in snapshot.timings_ms.values())
    # The three the frontend measures itself ran on every frameset this drive
    # tracked; the preintegration only runs once the estimator has published a
    # state, which it has by the last of them.
    for stage in ("frontend_pyramid", "frontend_detect", "frontend_track", "frontend_imu"):
        assert snapshot.timings_ms[stage] > 0.0, stage

    frame: _core.FlowFrame | None = vio.flow_frame()
    assert frame is not None
    assert frame.t_ns == (frames - 1) * FRAME_PERIOD_NS
    assert frame.num_tracks(0) > 0


def test_wrong_dtype_and_rank_raise_value_error(pipeline: PipelineFactory) -> None:
    vio: _core.Vio = pipeline(2)
    # The casts feed the boundary exactly what it forbids: that is the point of
    # the test, and the checker would otherwise reject the call statically.
    float_image: NDArray[np.uint8] = cast("NDArray[np.uint8]", np.zeros((FRAME, FRAME), dtype=np.float32))
    int32_times: NDArray[np.int64] = cast("NDArray[np.int64]", np.zeros(2, dtype=np.int32))
    with pytest.raises(ValueError, match="2-D uint8"):
        vio.track(0, [float_image, float_image])
    with pytest.raises(ValueError, match="1-D int64"):
        vio.push_imu_batch(int32_times, np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        vio.push_imu_batch(np.zeros(2, dtype=np.int64), np.zeros((2, 4)), np.zeros((2, 3)))


def test_fortran_order_arrays_are_rejected(pipeline: PipelineFactory) -> None:
    """Fortran bytes run down the columns; copying them as rows would transpose the data."""
    vio: _core.Vio = pipeline(2)
    fortran_image: UInt8[ndarray, "h w"] = np.asfortranarray(np.zeros((FRAME, FRAME), dtype=np.uint8))
    with pytest.raises(ValueError, match="C-contiguous"):
        vio.track(0, [fortran_image, fortran_image])

    t_ns: NDArray[np.int64] = np.arange(2, dtype=np.int64)
    fortran_gyro: NDArray[np.float64] = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3))
    contiguous: NDArray[np.float64] = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="C-contiguous"):
        vio.push_imu_batch(t_ns, fortran_gyro, contiguous)
    with pytest.raises(ValueError, match="C-contiguous"):
        vio.push_imu_batch(t_ns, contiguous, fortran_gyro)
    with pytest.raises(ValueError, match="C-contiguous"):
        vio.push_imu_batch(np.arange(4, dtype=np.int64)[::2], contiguous, contiguous)


def test_a_frameset_of_the_wrong_width_raises_value_error(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    vio: _core.Vio = pipeline(2)
    with pytest.raises(ValueError, match="expected 2 images"):
        vio.track(0, [texture(0, 0)])


def test_a_frame_that_is_not_the_calibrated_size_is_refused(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """The calibration is the geometry: a cropped frame means other bearings.

    Every rule ``track`` has is decided before anything moves, so this one does
    not need the inertial samples either; they are pushed because the refusal a
    covered frameset gets is the one the retry below depends on.
    """
    vio: _core.Vio = pipeline(2)
    samples: Int64[ndarray, " n_samples"] = np.arange(0, FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
    gyro, accel = gravity_batch(samples)
    vio.push_imu_batch(samples, gyro, accel)
    with pytest.raises(ValueError, match=f"the calibration is for {FRAME}x{FRAME} frames"):
        vio.track(0, [texture(0, 0), np.zeros((64, 64), dtype=np.uint8)])


def test_a_refused_frameset_and_its_retry_are_the_clean_run_bit_for_bit(pipeline: PipelineFactory, texture: TextureFactory) -> None:
    """The binding's promise, over a whole run: a refusal costs the trajectory nothing.

    A cropped frameset is offered before every good one and then corrected. The
    frontend undoes its own passes, but the refusal has to be decided before
    ``track`` spends the frontend's own preintegration on the interval, which is
    not undone: with that check behind it, the corrected retry parted from a
    clean run by 6e-8 m in the pose by the seventh frameset. Twelve framesets, so
    the probe lands both before the estimator has a state and after.
    """
    frames: int = 12
    cropped: UInt8[ndarray, "h w"] = np.zeros((64, 64), dtype=np.uint8)
    runs: list[list[Float64[ndarray, " 16"]]] = []
    for probe in (False, True):
        vio: _core.Vio = pipeline(2)
        states: list[Float64[ndarray, " 16"]] = []
        for index in range(frames):
            t_ns: int = index * FRAME_PERIOD_NS
            samples: Int64[ndarray, " n_samples"] = np.arange(t_ns, t_ns + FRAME_PERIOD_NS, IMU_PERIOD_NS, dtype=np.int64)
            gyro, accel = gravity_batch(samples)
            vio.push_imu_batch(samples, gyro, accel)
            images: list[UInt8[ndarray, "h w"]] = [texture(index, 0), texture(index + 1, 0)]
            if probe:
                with pytest.raises(ValueError, match=f"the calibration is for {FRAME}x{FRAME} frames"):
                    vio.track(t_ns, [images[0], cropped])
            result: _core.VioResult = vio.track(t_ns, images)
            states.append(np.concatenate([result.world_from_rig, result.velocity, result.gyro_bias, result.accel_bias]))
        runs.append(states)
    for index, (clean, probed) in enumerate(zip(runs[0], runs[1], strict=True)):
        np.testing.assert_array_equal(probed, clean, err_msg=f"frameset {index}: the pose, velocity or a bias moved with a refusal before it")
