"""Property tests for the optical-flow boundary, driven through ``slam_rs._core`` (D23).

The rig, its frames and the frontend over it come from :mod:`conftest` as
fixtures. The calibration is built from the feed's own ``CameraCalib``/
``ImuCalib`` dataclasses, so these tests also pin the field-name contract
``Calibration.from_catalog`` reads across the boundary: rename a field in
:mod:`slam_rs.catalog_feed` and this file fails rather than the estimator.

Everything here runs on small synthetic frames, so the whole file stays inside
the default, seconds-long suite.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from fixture_types import CameraFactory, FrontendFactory, PipelineFactory, TextureFactory
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from numpy.typing import NDArray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, ImuCalib

MAX_EXAMPLES: int = 25

sizes = st.integers(min_value=1, max_value=32)
camera_counts = st.integers(min_value=2, max_value=3)
"""Camera counts both entry points accept: the estimator refuses a one-camera rig (``optical_flow.h:210``)."""
wrong_sizes = st.integers(min_value=1, max_value=400)
"""Frame sides on both sides of the synthetic rig's, so cropped and enlarged frames are both generated."""

EntryPoint: TypeAlias = Callable[[int, list[UInt8[ndarray, "h w"]]], object]
"""One frameset into a boundary call: ``(t_ns, images)``."""
EntryFactory: TypeAlias = Callable[[int], EntryPoint]
"""An entry point on a rig of the given camera count."""


@pytest.fixture(scope="session", params=["Vio.track", "OpticalFlow.process"])
def entry_point(request: pytest.FixtureRequest, frontend: FrontendFactory, pipeline: PipelineFactory) -> EntryFactory:
    """Both array-taking entry points, so one property covers the two of them.

    They share ``gray_array``'s rank, dtype and layout checks and the frontend's
    own frame-size rule, so a rule proved on one of them says nothing about the
    other unless both are driven. Every rig here has at least two cameras,
    because that is what the estimator's epipolar filter requires.
    """
    if request.param == "Vio.track":
        return lambda camera_count: pipeline(camera_count).track
    return lambda camera_count: frontend(camera_count).process


def test_the_calibration_comes_across_field_by_field(camera: CameraFactory, imu: ImuCalib) -> None:
    calibrated: int = camera(0, 0.0).width
    calibration: _core.Calibration = _core.Calibration.from_catalog([camera(0, 0.0), camera(1, 0.1)], imu)
    assert calibration.camera_count == 2
    assert calibration.resolution == [(calibrated, calibrated), (calibrated, calibrated)]
    # basalt's own JSON is the round trip, so a reader can check what was pushed.
    assert _core.Calibration.from_json(calibration.to_json()).resolution == calibration.resolution


def test_an_extrinsic_that_is_not_a_rotation_is_rejected(camera: CameraFactory, imu: ImuCalib) -> None:
    mirrored: CameraCalib = camera(0, 0.0)
    mirrored.imu_T_cam[0, 0] = -1.0
    with pytest.raises(ValueError, match="rotation"):
        _core.Calibration.from_catalog([mirrored], imu)


def test_a_config_asking_for_another_pattern_is_rejected(camera: CameraFactory, imu: ImuCalib) -> None:
    config: _core.VioConfig = _core.VioConfig()
    raw: str = config.to_json().replace('"config.optical_flow_pattern": 51', '"config.optical_flow_pattern": 24')
    with pytest.raises(ValueError, match="pattern"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0, 0.0)], imu), _core.VioConfig.from_json(raw))


def test_the_image_safe_radius_survives_the_json_round_trip() -> None:
    config: _core.VioConfig = _core.VioConfig()
    config.optical_flow_image_safe_radius = 472.0
    assert _core.VioConfig.from_json(config.to_json()).optical_flow_image_safe_radius == 472.0


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(dtype=st.sampled_from([np.float32, np.float64, np.int16, np.uint16]), height=sizes, width=sizes)
def test_images_of_the_wrong_dtype_are_rejected(entry_point: EntryFactory, dtype: type, height: int, width: int) -> None:
    wrong: NDArray[np.uint8] = cast("NDArray[np.uint8]", np.zeros((height, width), dtype=dtype))
    with pytest.raises(ValueError, match="2-D uint8"):
        entry_point(2)(0, [wrong, wrong])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(ndim=st.sampled_from([1, 3, 4]), size=st.integers(min_value=1, max_value=4))
def test_images_of_the_wrong_rank_are_rejected(entry_point: EntryFactory, ndim: int, size: int) -> None:
    wrong: UInt8[ndarray, "..."] = np.zeros((size,) * ndim, dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D uint8"):
        entry_point(2)(0, [wrong, wrong])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(height=st.integers(min_value=2, max_value=16), width=st.integers(min_value=2, max_value=16))
def test_a_transposed_image_is_rejected(frontend: FrontendFactory, height: int, width: int) -> None:
    """Fortran order passes ``as_slice`` but its bytes run down the columns."""
    transposed: UInt8[ndarray, "w h"] = np.zeros((height, width), dtype=np.uint8).T
    with pytest.raises(ValueError, match="C-contiguous"):
        frontend(1).process(0, [transposed])


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(count=camera_counts, given_count=st.integers(min_value=0, max_value=4))
def test_the_wrong_number_of_images_is_rejected(entry_point: EntryFactory, texture: TextureFactory, count: int, given_count: int) -> None:
    assume(count != given_count)
    image: UInt8[ndarray, "h w"] = texture(0, 0)
    with pytest.raises(ValueError, match=f"expected {count} images"):
        entry_point(count)(0, [image] * given_count)


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(first=st.integers(min_value=-(10**9), max_value=10**9), back=st.integers(min_value=0, max_value=10**6))
def test_a_frameset_that_does_not_move_the_clock_forward_is_rejected(
    frontend: FrontendFactory, texture: TextureFactory, first: int, back: int
) -> None:
    flow: _core.OpticalFlow = frontend(1)
    image: UInt8[ndarray, "h w"] = texture(0, 0)
    flow.process(first, [image])
    with pytest.raises(ValueError, match="must increase"):
        flow.process(first - back, [image])


def test_a_refused_frameset_leaves_the_clock_alone(frontend: FrontendFactory, texture: TextureFactory) -> None:
    flow: _core.OpticalFlow = frontend(1)
    flow.process(1_000, [texture(0, 0)])
    with pytest.raises(ValueError, match="expected 1 images"):
        flow.process(2_000, [texture(0, 0), texture(0, 0)])
    assert flow.t_ns == 1_000


@settings(max_examples=10, deadline=None)
@given(shift_x=st.integers(min_value=-3, max_value=3), shift_y=st.integers(min_value=-3, max_value=3))
def test_a_shifted_texture_keeps_its_track_ids(frontend: FrontendFactory, texture: TextureFactory, shift_x: int, shift_y: int) -> None:
    """Two frames of the same scene: the ids survive and the positions follow the shift."""
    flow: _core.OpticalFlow = frontend(1)
    first: _core.FlowFrame = flow.process(0, [texture(0, 0)])
    second: _core.FlowFrame = flow.process(33_000_000, [texture(shift_x, shift_y)])

    # 4x4 whole cells at one point per cell, so 16 is the ceiling.
    assert first.num_tracks(0) >= 12, "the texture should offer a corner in most cells"
    assert first.num_new(0) == first.num_tracks(0), "every keypoint of the first frameset is new"
    assert second.num_tracks(0) > 0

    before: Int64[ndarray, " n_first"] = first.ids(0)
    after: Int64[ndarray, " n_second"] = second.ids(0)
    assert np.array_equal(np.sort(before), before), "ids come back ascending"
    kept: Int64[ndarray, " n_kept"] = np.intersect1d(before, after)
    assert len(kept) >= 0.8 * len(before), f"only {len(kept)} of {len(before)} ids survived a ({shift_x}, {shift_y}) px shift"

    moved: Float32[ndarray, "n_kept 2"] = second.positions(0)[np.isin(after, kept)] - first.positions(0)[np.isin(before, kept)]
    assert np.allclose(np.median(moved, axis=0), [shift_x, shift_y], atol=0.5)


def test_the_frame_reports_shapes_the_stub_promises(camera: CameraFactory, frontend: FrontendFactory, texture: TextureFactory) -> None:
    cells: int = camera(0, 0.0).width // 50 + 1
    flow: _core.OpticalFlow = frontend(2)
    frame: _core.FlowFrame = flow.process(0, [texture(0, 0), texture(2, 0)])
    assert frame.t_ns == 0
    assert frame.camera_count == 2
    for index in range(2):
        count: int = frame.num_tracks(index)
        assert frame.ids(index).shape == (count,)
        assert frame.ids(index).dtype == np.int64
        assert frame.positions(index).shape == (count, 2)
        assert frame.positions(index).dtype == np.float32
        assert frame.transforms(index).shape == (count, 2, 3)
        # The occupancy grid is camera 0's for every camera, as basalt's is.
        assert frame.occupancy(index).shape == (cells, cells)
        assert frame.occupancy(index).dtype == np.int32
    # The 2x3 warp starts at the identity with the keypoint's pixel in the last column.
    assert np.allclose(frame.transforms(0)[:, :, :2], np.eye(2), atol=1e-6)
    assert np.array_equal(frame.transforms(0)[:, :, 2], frame.positions(0))


def test_a_camera_past_the_end_of_the_rig_is_an_index_error(frontend: FrontendFactory, texture: TextureFactory) -> None:
    frame: _core.FlowFrame = frontend(1).process(0, [texture(0, 0)])
    with pytest.raises(IndexError, match="past the end"):
        frame.ids(1)


HANG_GUARD_S: float = 60.0
"""How long the subprocess probe below is given before it counts as a hang."""

DETECTOR_PROBE: str = """
import sys

import numpy as np

from slam_rs import _core

flow = _core.OpticalFlow(_core.Calibration.from_json(sys.argv[1]), _core.VioConfig.from_json(sys.argv[2]))
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


def probe_detector(calibration_json: str, config_json: str) -> subprocess.CompletedProcess[str]:
    """Run :data:`DETECTOR_PROBE` against one config, or fail the test on a hang.

    Args:
        calibration_json: The rig's calibration as basalt's JSON.
        config_json: One of basalt's configs as text.

    Returns:
        The finished process, so the caller can assert on its output.
    """
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", DETECTOR_PROBE, calibration_json, config_json],
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


def test_the_shipped_config_detects_on_a_blank_frame_and_returns(camera: CameraFactory, imu: ImuCalib) -> None:
    """The control for the probe below: this configuration reaches ``process`` and finishes."""
    calibration: str = _core.Calibration.from_catalog([camera(0, 0.0)], imu).to_json()
    finished: subprocess.CompletedProcess[str] = probe_detector(calibration, _core.VioConfig().to_json())
    assert finished.returncode == 0, finished.stderr
    assert "processed" in finished.stdout


@pytest.mark.parametrize("min_threshold", [0, -1, -(2**31)])
def test_a_detector_threshold_ladder_that_never_ends_is_refused(camera: CameraFactory, imu: ImuCalib, min_threshold: int) -> None:
    """``min_threshold <= 0`` hangs the detector — and basalt's own — so it is refused.

    ``keypoints.cpp:162,187`` halves the FAST threshold by integer division while
    it is at or above ``min_threshold``: zero halves to zero for ever. The C++ has
    the same non-terminating loop; the port refuses the config instead of running
    it, and floors its own last rung as a second line.
    """
    calibration: str = _core.Calibration.from_catalog([camera(0, 0.0)], imu).to_json()
    finished: subprocess.CompletedProcess[str] = probe_detector(
        calibration, config_with("config.optical_flow_detection_min_threshold", min_threshold)
    )
    assert finished.returncode != 0, f"the frontend accepted min_threshold={min_threshold}: {finished.stdout}"
    assert "ValueError" in finished.stderr
    assert "optical_flow_detection_min_threshold" in finished.stderr


def test_a_detector_threshold_ladder_that_never_runs_is_refused(camera: CameraFactory, imu: ImuCalib) -> None:
    """A ladder starting below where it stops could never add a keypoint."""
    document: dict = json.loads(_core.VioConfig().to_json())
    document["value0"]["config.optical_flow_detection_min_threshold"] = 40
    document["value0"]["config.optical_flow_detection_max_threshold"] = 5
    with pytest.raises(ValueError, match="ladder starts below"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0, 0.0)], imu), _core.VioConfig.from_json(json.dumps(document)))


@pytest.mark.parametrize("max_keypoints", [2**20 + 1, 2**31, 2**63, 2**64 - 1])
def test_a_keypoint_budget_nothing_could_allocate_is_a_value_error(camera: CameraFactory, imu: ImuCalib, max_keypoints: int) -> None:
    """A budget is a memory request, and an impossible one used to be a Rust panic.

    ``Vec::with_capacity(2**63)`` panics with ``capacity overflow``, which crosses
    the boundary as ``pyo3_runtime.PanicException`` — a ``BaseException`` that no
    ordinary ``except Exception`` catches. The ceiling makes it an answer.
    """
    with pytest.raises(ValueError, match="ceiling"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0, 0.0)], imu), _core.VioConfig(), max_keypoints=max_keypoints)


@pytest.mark.parametrize("threads", [2**20 + 1, 100_000, 2**63])
def test_more_workers_than_the_ceiling_is_refused(camera: CameraFactory, imu: ImuCalib, threads: int) -> None:
    """rayon spawns exactly what it is asked for, so an unbounded count wedges the machine."""
    with pytest.raises(ValueError, match="ceiling"):
        _core.OpticalFlow(_core.Calibration.from_catalog([camera(0, 0.0)], imu), _core.VioConfig(), threads=threads)


@pytest.mark.parametrize("levels", [24, 1_000, 10**9])
def test_a_pyramid_deeper_than_the_ceiling_is_refused(camera: CameraFactory, imu: ImuCalib, levels: int) -> None:
    """``optical_flow_levels`` sizes every per-patch buffer; an absurd one aborted the process."""
    with pytest.raises(ValueError, match="optical_flow_levels"):
        _core.OpticalFlow(
            _core.Calibration.from_catalog([camera(0, 0.0)], imu),
            _core.VioConfig.from_json(config_with("config.optical_flow_levels", levels)),
        )


def test_a_calibration_whose_detection_grid_nothing_could_allocate_is_refused(camera: CameraFactory, imu: ImuCalib) -> None:
    """The occupancy grid is derived from the calibration, so a resolution is a memory request too.

    The re-review's finding: the grid nothing could allocate used to panic out of
    the constructor with no image in sight (``detect::MAX_CELLS`` carries the
    arithmetic). In process like the ceilings above, because nothing is allocated
    on a refusal and a ``PanicException`` fails this ``pytest.raises`` anyway.
    """
    document: dict = json.loads(_core.Calibration.from_catalog([camera(0, 0.0)], imu).to_json())
    document["value0"]["resolution"] = [[2**32 - 2, 2**32 - 2]]
    with pytest.raises(ValueError, match="occupancy grid"):
        _core.OpticalFlow(
            _core.Calibration.from_json(json.dumps(document)),
            _core.VioConfig.from_json(config_with("config.optical_flow_detection_grid_size", 1)),
        )


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(height=wrong_sizes, width=wrong_sizes)
def test_an_image_that_is_not_the_calibrated_size_is_refused(camera: CameraFactory, frontend: FrontendFactory, height: int, width: int) -> None:
    """The calibration is the geometry: a cropped or resized frame means other bearings.

    The camera model projects with the calibrated intrinsics and the detection
    grid is derived from the calibrated size, so a 64x64 frame tracked against a
    200x200 calibration produces keypoints in pixels that do not exist.
    """
    calibrated: int = camera(0, 0.0).width
    assume((height, width) != (calibrated, calibrated))
    odd: UInt8[ndarray, "h w"] = np.zeros((height, width), dtype=np.uint8)
    with pytest.raises(ValueError, match=f"the calibration is for {calibrated}x{calibrated} frames, got {width}x{height}"):
        frontend(1).process(0, [odd])


def test_the_wrong_camera_is_named_when_only_one_image_is_the_wrong_size(frontend: FrontendFactory, texture: TextureFactory) -> None:
    flow: _core.OpticalFlow = frontend(2)
    with pytest.raises(ValueError, match="camera 1"):
        flow.process(0, [texture(0, 0), np.zeros((64, 64), dtype=np.uint8)])


def test_a_frame_of_the_wrong_size_leaves_the_frontend_as_it_was(
    camera: CameraFactory, frontend: FrontendFactory, texture: TextureFactory
) -> None:
    """The refusal is transactional: the frontend is as the last accepted frame left it."""
    calibrated: int = camera(0, 0.0).width
    flow: _core.OpticalFlow = frontend(1)
    first: _core.FlowFrame = flow.process(1_000, [texture(0, 0)])
    ids_before: int = flow.last_keypoint_id
    with pytest.raises(ValueError, match="the calibration is for"):
        flow.process(2_000, [np.zeros((calibrated + 8, calibrated), dtype=np.uint8)])
    assert flow.t_ns == 1_000
    assert flow.last_keypoint_id == ids_before

    # And it still tracks against the frame it kept.
    second: _core.FlowFrame = flow.process(3_000, [texture(1, 0)])
    assert flow.t_ns == 3_000
    kept: Int64[ndarray, " n_kept"] = np.intersect1d(first.ids(0), second.ids(0))
    assert len(kept) > 0, "the kept frame was not tracked against"


@settings(max_examples=10, deadline=None)
@given(start=st.integers(min_value=-(10**12), max_value=-1))
def test_identical_frames_at_negative_timestamps_keep_their_ids(frontend: FrontendFactory, texture: TextureFactory, start: int) -> None:
    """basalt reads ``t_ns < 0`` as "no previous frame"; the port's clock is an Option.

    Two identical framesets used to share **zero** ids at ``(-2, -1)`` and 14 of
    16 at ``(0, 1)``: the negative timestamp made every frameset the first one.
    """
    negative: _core.OpticalFlow = frontend(1)
    first: _core.FlowFrame = negative.process(start, [texture(0, 0)])
    second: _core.FlowFrame = negative.process(start + 1, [texture(0, 0)])
    shared: int = len(np.intersect1d(first.ids(0), second.ids(0)))

    zeroed: _core.OpticalFlow = frontend(1)
    third: _core.FlowFrame = zeroed.process(0, [texture(0, 0)])
    fourth: _core.FlowFrame = zeroed.process(1, [texture(0, 0)])
    at_zero: int = len(np.intersect1d(third.ids(0), fourth.ids(0)))

    assert shared > 0, f"no id survived an identical frameset at t = {start}"
    assert shared == at_zero, f"{shared} ids kept at t = {start} against {at_zero} at t = 0"
    assert negative.t_ns == start + 1


HOSTILE_INTS: tuple[int, ...] = (0, -1, 1, 2**31, 2**63 - 1, -(2**63), 2**63, 2**64, 10**40, -(10**40))
"""Integers a caller can type: past both ends of ``i64``, past ``u64``, and beyond."""
HOSTILE_OBJECTS: tuple[object, ...] = (None, "", "x", b"", [], {}, 0.5, float("nan"), object(), (1, 2))
"""Objects that are not what any parameter here asks for."""
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


def batch_imu(vio: _core.Vio, t_ns: object, gyro: object, accel: object) -> None:
    """Push an IMU batch of whatever was handed in, past the stub's declared dtypes.

    ``push_imu_batch`` declares ``int64[n]`` and ``float64[n, 3]``, and the audit's
    business is what happens when it is given something else, so the three
    arguments arrive as ``object`` and are cast here rather than at a dozen call
    sites.

    Args:
        vio: The estimator to push into.
        t_ns: Whatever is standing in for the timestamps.
        gyro: Whatever is standing in for the gyroscope samples.
        accel: Whatever is standing in for the accelerometer samples.
    """
    vio.push_imu_batch(
        cast("NDArray[np.int64]", t_ns),
        cast("NDArray[np.float64]", gyro),
        cast("NDArray[np.float64]", accel),
    )


def test_no_hostile_argument_reaches_python_as_a_panic(
    camera: CameraFactory, imu: ImuCalib, texture: TextureFactory, pipeline: PipelineFactory
) -> None:
    """Every public entry point, against every class of hostile argument (D32).

    A panic inside the core reaches Python as ``PanicException``, which derives
    from ``BaseException`` and so passes straight through an ``except Exception``
    handler; ``max_keypoints`` did exactly that. This walks the surface rather
    than the one argument that was reported.
    """
    failures: list[str] = []
    calibration: _core.Calibration = _core.Calibration.from_catalog([camera(0, 0.0), camera(1, 0.1)], imu)
    config: _core.VioConfig = _core.VioConfig()
    good: list[UInt8[ndarray, "h w"]] = [texture(0, 0), texture(1, 0)]
    flow: _core.OpticalFlow = _core.OpticalFlow(calibration, config)
    frame: _core.FlowFrame = flow.process(0, good)
    vio: _core.Vio = pipeline(2)

    for value in HOSTILE_INTS:
        refuse(f"push_imu({value})", lambda v=value: pipeline(2).push_imu(v, [0.0] * 3, [0.0] * 3), failures)
        refuse(f"Vio(threads={value})", lambda v=value: _core.Vio(calibration, config, threads=v), failures)
        refuse(f"Vio(max_keypoints={value})", lambda v=value: _core.Vio(calibration, config, max_keypoints=v), failures)
        refuse(f"Vio.track(t_ns={value})", lambda v=value: pipeline(2).track(v, good), failures)
        refuse(f"OpticalFlow(threads={value})", lambda v=value: _core.OpticalFlow(calibration, config, threads=v), failures)
        refuse(f"OpticalFlow(max_keypoints={value})", lambda v=value: _core.OpticalFlow(calibration, config, max_keypoints=v), failures)
        refuse(f"process(t_ns={value})", lambda v=value: _core.OpticalFlow(calibration, config).process(v, good), failures)
        for accessor in ("ids", "positions", "transforms", "occupancy", "num_new", "num_tracks"):
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
        refuse(f"Calibration.from_catalog(imu={thing!r})", lambda i=imu_like: _core.Calibration.from_catalog([camera(0, 0.0)], i), failures)
        refuse(f"OpticalFlow(calibration={thing!r})", lambda c=calibration_like: _core.OpticalFlow(c, config), failures)
        refuse(f"OpticalFlow(config={thing!r})", lambda o=config_like: _core.OpticalFlow(calibration, o), failures)
        refuse(f"Vio(calibration={thing!r})", lambda c=calibration_like: _core.Vio(c, config), failures)
        refuse(f"Vio(config={thing!r})", lambda o=config_like: _core.Vio(calibration, o), failures)
        refuse(f"process(images={thing!r})", lambda o=images_like: flow.process(1, o), failures)
        refuse(f"process([{thing!r}])", lambda o=frameset_like: flow.process(1, o), failures)
        refuse(f"safe_radius = {thing!r}", lambda o=thing: setattr(config, "optical_flow_image_safe_radius", o), failures)
        refuse(f"Vio.track({thing!r})", lambda o=images_like: vio.track(0, o), failures)
        refuse(f"push_imu_batch({thing!r})", lambda o=thing: batch_imu(vio, o, o, o), failures)
        refuse(f"frame.ids({thing!r})", lambda o=index_like: frame.ids(o), failures)

    # Arrays of the wrong rank, dtype, layout or extent, the transposed one being
    # a frame of exactly the calibrated size whose bytes run down the columns.
    hostile_arrays: tuple[NDArray[np.uint8], ...] = (
        np.zeros((0, 0), dtype=np.uint8),
        np.zeros((1, 0), dtype=np.uint8),
        cast("NDArray[np.uint8]", np.zeros((0,), dtype=np.uint8)),
        cast("NDArray[np.uint8]", np.zeros((2, 2, 2), dtype=np.uint8)),
        cast("NDArray[np.uint8]", np.zeros((3, 3), dtype=np.float64)),
        cast("NDArray[np.uint8]", good[0].T),
        np.asfortranarray(np.zeros((5, 5), dtype=np.uint8)),
    )
    for array in hostile_arrays:
        refuse(f"process({array.shape} {array.dtype})", lambda a=array: flow.process(1, [a, a]), failures)
        refuse(f"process(mixed {array.shape})", lambda a=array: flow.process(1, [good[0], a]), failures)
        refuse(f"Vio.track({array.shape} {array.dtype})", lambda a=array: vio.track(0, [a, a]), failures)
        refuse(f"Vio.track(mixed {array.shape})", lambda a=array: vio.track(0, [good[0], a]), failures)
        refuse(f"push_imu_batch({array.shape} {array.dtype})", lambda a=array: batch_imu(vio, a, a, a), failures)

    # Every numeric config field, one at a time, over the values that broke one.
    # Both constructors: the estimator reads fields the frontend never looks at.
    document: dict = json.loads(_core.VioConfig().to_json())
    numeric: list[str] = [key for key, value in document["value0"].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    assert len(numeric) > 20, f"only {len(numeric)} numeric config fields were found"
    for key in numeric:
        for value in (0, -1, 1, 2**31 - 1, -(2**31), 10**12):
            refuse(
                f"OpticalFlow config {key}={value}",
                lambda k=key, v=value: _core.OpticalFlow(calibration, _core.VioConfig.from_json(config_with(k, v))),
                failures,
            )
            refuse(
                f"Vio config {key}={value}",
                lambda k=key, v=value: _core.Vio(calibration, _core.VioConfig.from_json(config_with(k, v))),
                failures,
            )

    assert not failures, f"{len(failures)} calls did not refuse cleanly: {failures[:10]}"

    # A refused frameset leaves a frontend that still works, after all of that.
    assert flow.t_ns == 0
    assert flow.process(2, good).num_tracks(0) > 0
    # And an estimator that still tracks: nothing above moved its clock.
    assert vio.track(1, good).status == _core.VioStatus.NeedMoreImu
