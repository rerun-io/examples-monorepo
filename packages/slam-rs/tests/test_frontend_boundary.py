"""Property tests for the optical-flow boundary, driven through ``slam_rs._core`` (D23).

The calibration is built from the feed's own ``CameraCalib``/``ImuCalib``
dataclasses, so these tests also pin the field-name contract
``Calibration.from_catalog`` reads across the boundary: rename a field in
:mod:`slam_rs.catalog_feed` and this file fails rather than the estimator.

Everything here runs on small synthetic frames, so the whole file stays inside
the default, seconds-long suite.
"""

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
