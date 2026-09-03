import numpy as np
import pytest
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

from exo_calib.keypoints import (
    CameraKeypoints,
    postprocess_confidences,
    sampled_frame_indices,
    stack_postprocessed,
)


def _camera(num_frames: int = 1) -> CameraKeypoints:
    kp_xy: Float32[ndarray, "t 133 2"] = np.full((num_frames, 133, 2), np.nan, dtype=np.float32)
    kp_xy[:, :3] = np.array([[50.0, 40.0], [5.0, 40.0], [-1.0, 40.0]], dtype=np.float32)
    conf: Float32[ndarray, "t 133"] = np.zeros((num_frames, 133), dtype=np.float32)
    conf[:, :3] = np.array([0.575, 0.8, 0.8], dtype=np.float32)
    return CameraKeypoints(
        sample_indices=np.arange(num_frames, dtype=np.int64),
        times_ns=np.arange(num_frames, dtype=np.int64),
        kp_xy=kp_xy,
        conf=conf,
        bbox_xyxy=np.tile(np.array([[0.0, 0.0, 100.0, 80.0]], dtype=np.float32), (num_frames, 1)),
        box_source=("yolox",) * num_frames,
        crop_origin_xy=np.zeros((num_frames, 2), dtype=np.float32),
        crop_size_wh=np.tile(np.array([[100.0, 80.0]], dtype=np.float32), (num_frames, 1)),
        crop_input_wh=np.array([100, 80], dtype=np.int64),
        video_wh=np.array([100, 80], dtype=np.int64),
    )


def test_postprocess_confidences_applies_crop_margin_then_threshold_rescale() -> None:
    filtered_xy, post_conf = postprocess_confidences(_camera(), median_window=1, margin_px=10.0, conf_tau=0.15)

    # margin: 0.575 (interior), 0.8 * 5/10 = 0.4 (5 px from the edge), 0 (outside); then (c - 0.15) / 0.85.
    np.testing.assert_allclose(post_conf[0, :3], np.array([0.5, 0.25 / 0.85, 0.0]), atol=1e-7)
    np.testing.assert_allclose(filtered_xy[0, :3], np.array([[50.0, 40.0], [5.0, 40.0], [-1.0, 40.0]]))
    assert np.all(post_conf[0, 3:] == 0.0)


def test_sampled_frame_indices_spreads_the_budget_evenly_and_keeps_short_captures_whole() -> None:
    short: Int64[ndarray, "t"] = sampled_frame_indices(150, 1000)
    long: Int64[ndarray, "t"] = sampled_frame_indices(4000, 1000)

    np.testing.assert_array_equal(short, np.arange(150, dtype=np.int64))
    assert long.size == 1000 and long[0] == 0 and long[-1] == 3999
    assert np.all(np.diff(long) >= 1) and np.unique(np.diff(long)).size <= 2
    with pytest.raises(ValueError, match="no frames"):
        sampled_frame_indices(0, 1000)


def test_stack_postprocessed_rejects_mismatched_frame_grids() -> None:
    with pytest.raises(ValueError, match="frame grid"):
        stack_postprocessed({"a": _camera(2), "b": _camera(3)}, ("a", "b"), 1, 10.0, 0.15)
    shifted: CameraKeypoints = _camera(2)
    shifted.sample_indices = shifted.sample_indices + 4  # same length, different frames
    with pytest.raises(ValueError, match="frame grid"):
        stack_postprocessed({"a": _camera(2), "b": shifted}, ("a", "b"), 1, 10.0, 0.15)

    kp_xy, conf = stack_postprocessed({"a": _camera(2), "b": _camera(2)}, ("a", "b"), 1, 10.0, 0.15)
    expected_conf: Float64[ndarray, "3"] = np.array([0.5, 0.25 / 0.85, 0.0])
    assert kp_xy.shape == (2, 2, 133, 2)
    np.testing.assert_allclose(conf[:, :, :3], np.broadcast_to(expected_conf, (2, 2, 3)), atol=1e-7)
