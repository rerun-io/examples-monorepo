"""Property tests for the optical-flow boundary, driven through ``slam_rs._core`` (D23).

The calibration is built from the feed's own ``CameraCalib``/``ImuCalib``
dataclasses, so these tests also pin the field-name contract
``Calibration.from_catalog`` reads across the boundary: rename a field in
:mod:`slam_rs.catalog_feed` and this file fails rather than the estimator.

Everything here runs on small synthetic frames, so the whole file stays inside
the default, seconds-long suite.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jaxtyping import Float64, UInt8
from numpy import ndarray
from numpy.typing import NDArray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, ImuCalib

MAX_EXAMPLES: int = 25
FRAME: int = 200
"""Synthetic frame size: four whole 50-pixel detection cells per side."""

sizes = st.integers(min_value=1, max_value=32)
camera_counts = st.integers(min_value=1, max_value=3)


def camera(index: int, baseline_m: float = 0.0) -> CameraCalib:
    """A 200x200 pinhole-like camera: kb4 with every coefficient zero."""
    imu_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    imu_T_cam[0, 3] = baseline_m
    return CameraCalib(
        index=index,
        width=FRAME,
        height=FRAME,
        frequency_hz=30.0,
        fx=100.0,
        fy=100.0,
        cx=FRAME / 2,
        cy=FRAME / 2,
        model="kb4",
        distortion=np.zeros(4, dtype=np.float64),
        distortion_valid_radius=None,
        imu_T_cam=imu_T_cam,
        image_rotation_cw_deg=0,
    )


def imu() -> ImuCalib:
    """The Index device's frozen noise model, which no test here is sensitive to."""
    return ImuCalib(
        frequency_hz=1000.0,
        gyro_noise_std=0.000282,
        accel_noise_std=0.016,
        gyro_bias_std=0.0001,
        accel_bias_std=0.001,
        cam_time_offset_ns=0,
        imu_T_body=np.eye(4, dtype=np.float64),
    )


def frontend(camera_count: int = 1) -> _core.OpticalFlow:
    """A frontend on a rig of ``camera_count`` identical cameras, 10 cm apart."""
    cameras: list[CameraCalib] = [camera(index, 0.1 * index) for index in range(camera_count)]
    return _core.OpticalFlow(_core.Calibration.from_catalog(cameras, imu()), _core.VioConfig())


def texture(shift_x: int = 0, shift_y: int = 0) -> UInt8[ndarray, "h w"]:
    """One fixed blocky-noise scene, shifted by whole pixels.

    Blocky **noise**, not a lattice: a repeating pattern gives every corner the
    same score, and basalt's suppression is strictly-greater-than, so a tie kills
    both sides and a perfectly regular scene detects almost nothing.
    """
    blocks: UInt8[ndarray, "b b"] = np.random.default_rng(20250907).integers(0, 256, (FRAME // 2, FRAME // 2), dtype=np.uint8)
    image: UInt8[ndarray, "h w"] = np.repeat(np.repeat(blocks, 2, axis=0), 2, axis=1)
    return np.ascontiguousarray(np.roll(image, (shift_y, shift_x), axis=(0, 1)))


def test_the_calibration_comes_across_field_by_field() -> None:
    calibration: _core.Calibration = _core.Calibration.from_catalog([camera(0), camera(1, 0.1)], imu())
    assert calibration.camera_count == 2
    assert calibration.resolution == [(FRAME, FRAME), (FRAME, FRAME)]
    # basalt's own JSON is the round trip, so a reader can check what was pushed.
    assert _core.Calibration.from_json(calibration.to_json()).resolution == calibration.resolution


def test_an_extrinsic_that_is_not_a_rotation_is_rejected() -> None:
    mirrored: CameraCalib = camera(0)
    mirrored.imu_T_cam[0, 0] = -1.0
    with pytest.raises(ValueError, match="rotation"):
        _core.Calibration.from_catalog([mirrored], imu())


def test_a_config_asking_for_another_pattern_is_rejected() -> None:
    config: _core.VioConfig = _core.VioConfig()
    raw: str = config.to_json().replace('"config.optical_flow_pattern": 51', '"config.optical_flow_pattern": 24')
    with pytest.raises(ValueError, match="pattern"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0)], imu()), raw)


def test_the_image_safe_radius_survives_the_json_round_trip() -> None:
    config: _core.VioConfig = _core.VioConfig()
    config.optical_flow_image_safe_radius = 472.0
    assert _core.VioConfig.from_json(config.to_json()).optical_flow_image_safe_radius == 472.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(dtype=st.sampled_from([np.float32, np.float64, np.int16, np.uint16]), height=sizes, width=sizes)
def test_images_of_the_wrong_dtype_are_rejected(dtype: type, height: int, width: int) -> None:
    wrong: NDArray[np.uint8] = cast("NDArray[np.uint8]", np.zeros((height, width), dtype=dtype))
    with pytest.raises(ValueError, match="2-D uint8"):
        frontend().process(0, [wrong])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(ndim=st.sampled_from([1, 3, 4]), size=st.integers(min_value=1, max_value=4))
def test_images_of_the_wrong_rank_are_rejected(ndim: int, size: int) -> None:
    with pytest.raises(ValueError, match="2-D uint8"):
        frontend().process(0, [np.zeros((size,) * ndim, dtype=np.uint8)])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(height=st.integers(min_value=2, max_value=16), width=st.integers(min_value=2, max_value=16))
def test_a_transposed_image_is_rejected(height: int, width: int) -> None:
    """Fortran order passes ``as_slice`` but its bytes run down the columns."""
    transposed: UInt8[ndarray, "w h"] = np.zeros((height, width), dtype=np.uint8).T
    with pytest.raises(ValueError, match="C-contiguous"):
        frontend().process(0, [transposed])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(count=camera_counts, given_count=st.integers(min_value=0, max_value=4))
def test_the_wrong_number_of_images_is_rejected(count: int, given_count: int) -> None:
    assume(count != given_count)
    image: UInt8[ndarray, "h w"] = np.zeros((FRAME, FRAME), dtype=np.uint8)
    with pytest.raises(ValueError, match=f"expected {count} images"):
        frontend(count).process(0, [image] * given_count)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(first=st.integers(min_value=-(10**9), max_value=10**9), back=st.integers(min_value=0, max_value=10**6))
def test_a_frameset_that_does_not_move_the_clock_forward_is_rejected(first: int, back: int) -> None:
    flow: _core.OpticalFlow = frontend()
    image: UInt8[ndarray, "h w"] = np.zeros((FRAME, FRAME), dtype=np.uint8)
    flow.process(first, [image])
    with pytest.raises(ValueError, match="must increase"):
        flow.process(first - back, [image])


def test_a_refused_frameset_leaves_the_clock_alone() -> None:
    flow: _core.OpticalFlow = frontend()
    flow.process(1_000, [texture()])
    with pytest.raises(ValueError, match="expected 1 images"):
        flow.process(2_000, [texture(), texture()])
    assert flow.t_ns == 1_000
    assert flow.frame_counter == 1


@settings(max_examples=10, deadline=None)
@given(shift_x=st.integers(min_value=-3, max_value=3), shift_y=st.integers(min_value=-3, max_value=3))
def test_a_shifted_texture_keeps_its_track_ids(shift_x: int, shift_y: int) -> None:
    """Two frames of the same scene: the ids survive and the positions follow the shift."""
    flow: _core.OpticalFlow = frontend()
    first: _core.FlowFrame = flow.process(0, [texture()])
    second: _core.FlowFrame = flow.process(33_000_000, [texture(shift_x, shift_y)])

    # 4x4 whole cells at one point per cell, so 16 is the ceiling.
    assert first.num_tracks(0) >= 12, "the texture should offer a corner in most cells"
    assert first.num_new(0) == first.num_tracks(0), "every keypoint of the first frameset is new"
    assert second.num_tracks(0) > 0

    before: NDArray[np.int64] = first.ids(0)
    after: NDArray[np.int64] = second.ids(0)
    assert np.array_equal(np.sort(before), before), "ids come back ascending"
    kept: NDArray[np.int64] = np.intersect1d(before, after)
    assert len(kept) >= 0.8 * len(before), f"only {len(kept)} of {len(before)} ids survived a ({shift_x}, {shift_y}) px shift"

    moved: NDArray[np.float32] = second.positions(0)[np.isin(after, kept)] - first.positions(0)[np.isin(before, kept)]
    assert np.allclose(np.median(moved, axis=0), [shift_x, shift_y], atol=0.5)


def test_the_frame_reports_shapes_the_stub_promises() -> None:
    flow: _core.OpticalFlow = frontend(2)
    frame: _core.FlowFrame = flow.process(0, [texture(), texture(2, 0)])
    assert frame.t_ns == 0
    assert frame.camera_count == 2
    for index in range(2):
        count: int = frame.num_tracks(index)
        assert frame.ids(index).shape == (count,)
        assert frame.ids(index).dtype == np.int64
        assert frame.positions(index).shape == (count, 2)
        assert frame.positions(index).dtype == np.float32
        assert frame.transforms(index).shape == (count, 2, 3)
        assert frame.responses(index).shape == (count,)
        # basalt only fills pyramid_levels in the multiscale variant.
        assert frame.levels(index).shape == (0,)
        # The occupancy grid is camera 0's for every camera, as basalt's is.
        assert frame.occupancy(index).shape == (FRAME // 50 + 1, FRAME // 50 + 1)
        assert frame.occupancy(index).dtype == np.int32
    # The 2x3 warp starts at the identity with the keypoint's pixel in the last column.
    assert np.allclose(frame.transforms(0)[:, :, :2], np.eye(2), atol=1e-6)
    assert np.array_equal(frame.transforms(0)[:, :, 2], frame.positions(0))


def test_a_camera_past_the_end_of_the_rig_is_an_index_error() -> None:
    frame: _core.FlowFrame = frontend().process(0, [texture()])
    with pytest.raises(IndexError, match="past the end"):
        frame.ids(1)


HANG_GUARD_S: float = 60.0
"""How long the subprocess probe below is given before it counts as a hang."""

DETECTOR_PROBE: str = """
import sys

import numpy as np

from slam_rs import _core

flow = _core.OpticalFlow(sys.argv[1], sys.argv[2])
flow.process(0, [np.zeros((200, 200), dtype=np.uint8)])
print("processed")
"""
"""Build a frontend from JSON text and detect on a blank frame, with no error handling.

A blank frame is the worst case for the detector's threshold ladder: no cell ever
fills its budget, so every rung is walked. This runs in a subprocess because the
failure it guards against is a **hang** inside the released-GIL region, where a
Python-level ``signal`` handler never gets to run — only an outside timeout can
end it.
"""


def probe_detector(config_json: str) -> subprocess.CompletedProcess[str]:
    """Run :data:`DETECTOR_PROBE` against one config, or fail the test on a hang.

    Args:
        config_json: One of basalt's configs as text.

    Returns:
        The finished process, so the caller can assert on its output.
    """
    calibration: str = _core.Calibration.from_catalog([camera(0)], imu()).to_json()
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", DETECTOR_PROBE, calibration, config_json],
        capture_output=True,
        text=True,
        timeout=HANG_GUARD_S,
        check=False,
    )


def config_with(key: str, value: int) -> str:
    """basalt's default config as text, with one integer field replaced."""
    document: dict = json.loads(_core.VioConfig().to_json())
    assert key in document["value0"], f"{key} is not a config field: {sorted(document['value0'])}"
    document["value0"][key] = value
    return json.dumps(document)


def test_the_shipped_config_detects_on_a_blank_frame_and_returns() -> None:
    """The control for the probe below: this configuration reaches ``process`` and finishes."""
    finished: subprocess.CompletedProcess[str] = probe_detector(_core.VioConfig().to_json())
    assert finished.returncode == 0, finished.stderr
    assert "processed" in finished.stdout


@pytest.mark.parametrize("min_threshold", [0, -1, -(2**31)])
def test_a_detector_threshold_ladder_that_never_ends_is_refused(min_threshold: int) -> None:
    """``min_threshold <= 0`` hangs the detector — and basalt's own — so it is refused.

    ``keypoints.cpp:162,187`` halves the FAST threshold by integer division while
    it is at or above ``min_threshold``: zero halves to zero for ever. The C++ has
    the same non-terminating loop; the port refuses the config instead of running
    it, and floors its own last rung as a second line.
    """
    finished: subprocess.CompletedProcess[str] = probe_detector(config_with("config.optical_flow_detection_min_threshold", min_threshold))
    assert finished.returncode != 0, f"the frontend accepted min_threshold={min_threshold}: {finished.stdout}"
    assert "ValueError" in finished.stderr
    assert "optical_flow_detection_min_threshold" in finished.stderr


def test_a_detector_threshold_ladder_that_never_runs_is_refused() -> None:
    """A ladder starting below where it stops could never add a keypoint."""
    document: dict = json.loads(_core.VioConfig().to_json())
    document["value0"]["config.optical_flow_detection_min_threshold"] = 40
    document["value0"]["config.optical_flow_detection_max_threshold"] = 5
    with pytest.raises(ValueError, match="ladder starts below"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0)], imu()), json.dumps(document))


@pytest.mark.parametrize("max_keypoints", [2**20 + 1, 2**31, 2**63, 2**64 - 1])
def test_a_keypoint_budget_nothing_could_allocate_is_a_value_error(max_keypoints: int) -> None:
    """A budget is a memory request, and an impossible one used to be a Rust panic.

    ``Vec::with_capacity(2**63)`` panics with ``capacity overflow``, which crosses
    the boundary as ``pyo3_runtime.PanicException`` — a ``BaseException`` that no
    ordinary ``except Exception`` catches. The ceiling makes it an answer.
    """
    with pytest.raises(ValueError, match="ceiling"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0)], imu()), _core.VioConfig(), max_keypoints=max_keypoints)


@pytest.mark.parametrize("threads", [2**20 + 1, 100_000, 2**63])
def test_more_workers_than_the_ceiling_is_refused(threads: int) -> None:
    """rayon spawns exactly what it is asked for, so an unbounded count wedges the machine."""
    with pytest.raises(ValueError, match="ceiling"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0)], imu()), _core.VioConfig(), threads=threads)


@pytest.mark.parametrize("levels", [24, 1_000, 10**9])
def test_a_pyramid_deeper_than_the_ceiling_is_refused(levels: int) -> None:
    """``optical_flow_levels`` sizes every per-patch buffer; an absurd one aborted the process."""
    with pytest.raises(ValueError, match="optical_flow_levels"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0)], imu()), config_with("config.optical_flow_levels", levels))


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    height=st.integers(min_value=1, max_value=2 * FRAME),
    width=st.integers(min_value=1, max_value=2 * FRAME),
)
def test_an_image_that_is_not_the_calibrated_size_is_refused(height: int, width: int) -> None:
    """The calibration is the geometry: a cropped or resized frame means other bearings.

    The camera model projects with the calibrated intrinsics and the detection
    grid is derived from the calibrated size, so a 64x64 frame tracked against a
    200x200 calibration produces keypoints in pixels that do not exist.
    """
    assume((height, width) != (FRAME, FRAME))
    odd: UInt8[ndarray, "h w"] = np.zeros((height, width), dtype=np.uint8)
    with pytest.raises(ValueError, match=f"the calibration is for {FRAME}x{FRAME} frames, got {width}x{height}"):
        frontend().process(0, [odd])


def test_the_wrong_camera_is_named_when_only_one_image_is_the_wrong_size() -> None:
    flow: _core.OpticalFlow = frontend(2)
    with pytest.raises(ValueError, match="camera 1"):
        flow.process(0, [texture(), np.zeros((64, 64), dtype=np.uint8)])


def test_a_frame_of_the_wrong_size_leaves_the_frontend_as_it_was() -> None:
    """The refusal is transactional: the frontend is as the last accepted frame left it."""
    flow: _core.OpticalFlow = frontend()
    first: _core.FlowFrame = flow.process(1_000, [texture()])
    ids_before: int = flow.last_keypoint_id
    with pytest.raises(ValueError, match="the calibration is for"):
        flow.process(2_000, [np.zeros((FRAME + 8, FRAME), dtype=np.uint8)])
    assert flow.t_ns == 1_000
    assert flow.frame_counter == 1
    assert flow.last_keypoint_id == ids_before

    # And it still tracks against the frame it kept.
    second: _core.FlowFrame = flow.process(3_000, [texture(1, 0)])
    assert flow.frame_counter == 2
    kept: NDArray[np.int64] = np.intersect1d(first.ids(0), second.ids(0))
    assert len(kept) > 0, "the kept frame was not tracked against"


@settings(max_examples=10, deadline=None)
@given(start=st.integers(min_value=-(10**12), max_value=-1))
def test_identical_frames_at_negative_timestamps_keep_their_ids(start: int) -> None:
    """basalt reads ``t_ns < 0`` as "no previous frame"; the port's clock is an Option.

    Two identical framesets used to share **zero** ids at ``(-2, -1)`` and 14 of
    16 at ``(0, 1)``: the negative timestamp made every frameset the first one.
    """
    negative: _core.OpticalFlow = frontend()
    first: _core.FlowFrame = negative.process(start, [texture()])
    second: _core.FlowFrame = negative.process(start + 1, [texture()])
    shared: int = len(np.intersect1d(first.ids(0), second.ids(0)))

    zeroed: _core.OpticalFlow = frontend()
    third: _core.FlowFrame = zeroed.process(0, [texture()])
    fourth: _core.FlowFrame = zeroed.process(1, [texture()])
    at_zero: int = len(np.intersect1d(third.ids(0), fourth.ids(0)))

    assert shared > 0, f"no id survived an identical frameset at t = {start}"
    assert shared == at_zero, f"{shared} ids kept at t = {start} against {at_zero} at t = 0"
    assert negative.t_ns == start + 1


HOSTILE_INTS: tuple[int, ...] = (0, -1, 1, 2**31, 2**63 - 1, -(2**63), 2**63, 2**64, 10**40, -(10**40))
"""Integers a caller can type: past both ends of ``i64``, past ``u64``, and beyond."""
HOSTILE_OBJECTS: tuple[object, ...] = (None, "", "x", b"", [], {}, 0.5, float("nan"), object(), (1, 2))
"""Objects that are not what any parameter here asks for."""
HOSTILE_ARRAYS: tuple[NDArray[np.uint8], ...] = (
    np.zeros((0, 0), dtype=np.uint8),
    np.zeros((1, 0), dtype=np.uint8),
    cast("NDArray[np.uint8]", np.zeros((0,), dtype=np.uint8)),
    cast("NDArray[np.uint8]", np.zeros((2, 2, 2), dtype=np.uint8)),
    cast("NDArray[np.uint8]", np.zeros((3, 3), dtype=np.float64)),
    cast("NDArray[np.uint8]", np.zeros((FRAME, FRAME), dtype=np.uint8).T),
    np.asfortranarray(np.zeros((5, 5), dtype=np.uint8)),
)
"""Arrays of the wrong rank, dtype, layout or extent."""
EXPECTED_ERRORS: tuple[type[Exception], ...] = (ValueError, TypeError, IndexError, OverflowError)
"""What a boundary may raise. A panic is a ``BaseException`` and is none of these."""


def refuse(what: str, call: Callable[[], object], failures: list[str]) -> None:
    """Call ``call`` and record anything that is not an ordinary refusal.

    ``pyo3_runtime.PanicException`` only exists once a panic has been raised, so
    it is recognised by class name rather than imported.

    Args:
        what: How the call is named in a failure.
        call: The boundary call to make.
        failures: Where an escaping panic is recorded.
    """
    try:
        call()
    except EXPECTED_ERRORS:
        return
    except BaseException as error:  # noqa: BLE001
        failures.append(f"{what}: {type(error).__name__}({error})")


def batch_imu(t_ns: object, gyro: object, accel: object) -> None:
    """Push an IMU batch of whatever was handed in, past the stub's declared dtypes.

    ``push_imu_batch`` declares ``int64[n]`` and ``float64[n, 3]``, and the audit's
    business is what happens when it is given something else, so the three
    arguments arrive as ``object`` and are cast here rather than at a dozen call
    sites.

    Args:
        t_ns: Whatever is standing in for the timestamps.
        gyro: Whatever is standing in for the gyroscope samples.
        accel: Whatever is standing in for the accelerometer samples.
    """
    _core.Vio(1, 1).push_imu_batch(
        cast("NDArray[np.int64]", t_ns),
        cast("NDArray[np.float64]", gyro),
        cast("NDArray[np.float64]", accel),
    )


def test_no_hostile_argument_reaches_python_as_a_panic() -> None:
    """Every public entry point, against every class of hostile argument (D32).

    A panic inside the core reaches Python as ``PanicException``, which derives
    from ``BaseException`` and so passes straight through an ``except Exception``
    handler; ``max_keypoints`` did exactly that. This walks the surface rather
    than the one argument that was reported.
    """
    failures: list[str] = []
    calibration: _core.Calibration = _core.Calibration.from_catalog([camera(0), camera(1, 0.1)], imu())
    config: _core.VioConfig = _core.VioConfig()
    good: list[UInt8[ndarray, "h w"]] = [texture(), texture(1, 0)]
    flow: _core.OpticalFlow = _core.OpticalFlow(calibration, config)
    frame: _core.FlowFrame = flow.process(0, good)

    for value in HOSTILE_INTS:
        refuse(f"Vio({value})", lambda v=value: _core.Vio(v, 1), failures)
        refuse(f"Vio(min_imu_samples={value})", lambda v=value: _core.Vio(1, v), failures)
        refuse(f"push_imu({value})", lambda v=value: _core.Vio(1, 1).push_imu(v, [0.0] * 3, [0.0] * 3), failures)
        refuse(f"OpticalFlow(threads={value})", lambda v=value: _core.OpticalFlow(calibration, config, threads=v), failures)
        refuse(f"OpticalFlow(max_keypoints={value})", lambda v=value: _core.OpticalFlow(calibration, config, max_keypoints=v), failures)
        refuse(f"process(t_ns={value})", lambda v=value: _core.OpticalFlow(calibration, config).process(v, good), failures)
        for accessor in ("ids", "positions", "transforms", "responses", "levels", "occupancy", "num_new", "num_tracks"):
            refuse(f"frame.{accessor}({value})", lambda a=accessor, v=value: getattr(frame, a)(v), failures)

    # Every value below violates the type its parameter declares — that is what
    # is under test — so each is passed through the declared type. The casts are
    # the violation, not an assumption about the value.
    for thing in HOSTILE_OBJECTS:
        text: str = cast("str", thing)
        camera_like: CameraCalib = cast("CameraCalib", thing)
        imu_like: ImuCalib = cast("ImuCalib", thing)
        config_like: _core.VioConfig = cast("_core.VioConfig", thing)
        calibration_like: _core.Calibration = cast("_core.Calibration", thing)
        images_like: list[UInt8[ndarray, "h w"]] = cast("list[UInt8[ndarray, 'h w']]", thing)
        frameset_like: list[UInt8[ndarray, "h w"]] = cast("list[UInt8[ndarray, 'h w']]", [thing, thing])
        index_like: int = cast("int", thing)
        refuse(f"Calibration.from_json({thing!r})", lambda o=text: _core.Calibration.from_json(o), failures)
        refuse(f"VioConfig.from_json({thing!r})", lambda o=text: _core.VioConfig.from_json(o), failures)
        refuse(f"Calibration.from_catalog([{thing!r}])", lambda c=camera_like, i=imu_like: _core.Calibration.from_catalog([c], i), failures)
        refuse(f"Calibration.from_catalog(imu={thing!r})", lambda i=imu_like: _core.Calibration.from_catalog([camera(0)], i), failures)
        refuse(f"OpticalFlow(calibration={thing!r})", lambda c=calibration_like: _core.OpticalFlow(c, config), failures)
        refuse(f"OpticalFlow(config={thing!r})", lambda o=config_like: _core.OpticalFlow(calibration, o), failures)
        refuse(f"process(images={thing!r})", lambda o=images_like: flow.process(1, o), failures)
        refuse(f"process([{thing!r}])", lambda o=frameset_like: flow.process(1, o), failures)
        refuse(f"safe_radius = {thing!r}", lambda o=thing: setattr(config, "optical_flow_image_safe_radius", o), failures)
        refuse(f"Vio.track({thing!r})", lambda o=images_like: _core.Vio(1, 1).track(0, o), failures)
        refuse(f"push_imu_batch({thing!r})", lambda o=thing: batch_imu(o, o, o), failures)
        refuse(f"frame.ids({thing!r})", lambda o=index_like: frame.ids(o), failures)

    for array in HOSTILE_ARRAYS:
        refuse(f"process({array.shape} {array.dtype})", lambda a=array: flow.process(1, [a, a]), failures)
        refuse(f"process(mixed {array.shape})", lambda a=array: flow.process(1, [good[0], a]), failures)
        refuse(f"Vio.track({array.shape} {array.dtype})", lambda a=array: _core.Vio(1, 1).track(0, [a]), failures)
        refuse(f"push_imu_batch({array.shape} {array.dtype})", lambda a=array: batch_imu(a, a, a), failures)

    # Every numeric config field, one at a time, over the values that broke one.
    document: dict = json.loads(_core.VioConfig().to_json())
    numeric: list[str] = [key for key, value in document["value0"].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    assert len(numeric) > 20, f"only {len(numeric)} numeric config fields were found"
    for key in numeric:
        for value in (0, -1, 1, 2**31 - 1, -(2**31), 10**12):
            refuse(f"config {key}={value}", lambda k=key, v=value: _core.OpticalFlow(calibration, config_with(k, v)), failures)

    assert not failures, f"{len(failures)} calls did not refuse cleanly: {failures[:10]}"

    # A refused frameset leaves a frontend that still works, after all of that.
    assert flow.frame_counter == 1
    assert flow.process(2, good).num_tracks(0) > 0
