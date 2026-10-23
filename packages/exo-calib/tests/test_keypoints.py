import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

from exo_calib.ops.confidence import median_filter_keypoints
from exo_calib.ops.keypoints import AHX_MEDIAN_WINDOW, postprocess_confidences, stack_postprocessed
from exo_calib.types import MultiViewKeypoints


def _frame(views: int = 1, frame_index: int = 0, x0: float = 50.0) -> MultiViewKeypoints:
    kp_xy: Float32[ndarray, "v 133 2"] = np.full((views, 133, 2), np.nan, dtype=np.float32)
    kp_xy[:, :3] = np.array([[x0, 40.0], [5.0, 40.0], [-1.0, 40.0]], dtype=np.float32)
    conf: Float32[ndarray, "v 133"] = np.zeros((views, 133), dtype=np.float32)
    conf[:, :3] = np.array([0.575, 0.8, 0.8], dtype=np.float32)
    return MultiViewKeypoints(
        frame_index=frame_index,
        views=tuple(f"rig_{i:02d}/cam_00" for i in range(views)),
        cam_t_ns=np.zeros(views, dtype=np.int64),
        kp_xy=kp_xy,
        conf=conf,
        bbox_xyxy=np.tile(np.array([[0.0, 0.0, 100.0, 80.0]], dtype=np.float32), (views, 1)),
        box_source=("yolox",) * views,
        crop_origin_xy=np.zeros((views, 2), dtype=np.float32),
        crop_size_wh=np.tile(np.array([[100.0, 80.0]], dtype=np.float32), (views, 1)),
        crop_input_wh=(100, 80),
        video_wh=(100, 80),
    )


def test_postprocess_confidences_applies_crop_margin_then_threshold_rescale() -> None:
    f = _frame()
    filtered_xy, post_conf = postprocess_confidences(
        f.kp_xy, f.conf, f.crop_origin_xy, f.crop_size_wh, f.crop_input_wh, np.zeros(1, dtype=np.int64), margin_px=10.0
    )

    # margin: 0.575 (interior), 0.8 * 5/10 = 0.4 (5 px from the edge), 0 (outside); then (c - 0.15) / 0.85.
    np.testing.assert_allclose(post_conf[0, :3], np.array([0.5, 0.25 / 0.85, 0.0]), atol=1e-7)
    np.testing.assert_allclose(filtered_xy[0, :3], np.array([[50.0, 40.0], [5.0, 40.0], [-1.0, 40.0]]))
    assert np.all(post_conf[0, 3:] == 0.0)


def test_stack_postprocessed_puts_views_first_and_frames_second() -> None:
    kp_xy, conf = stack_postprocessed([_frame(2), _frame(2), _frame(2)])
    expected_conf: Float64[ndarray, "3"] = np.array([0.575, 0.8 * 5.0 / 32.0, 0.0])
    expected_conf = np.where(expected_conf >= 0.15, (expected_conf - 0.15) / 0.85, 0.0)
    assert kp_xy.shape == (2, 3, 133, 2) and conf.shape == (2, 3, 133)
    np.testing.assert_allclose(conf[:, :, :3], np.broadcast_to(expected_conf, (2, 3, 3)), atol=1e-7)


def test_median_never_mixes_records_farther_apart_than_its_window() -> None:
    # A frame budget samples every 84th video frame (1.4 s at 60 fps); a hand going up and down between samples must
    # be drawn and bundle-adjusted where the model saw it in each frame, not at the median of five samples.
    records = [_frame(frame_index=i * 84, x0=x) for i, x in enumerate([0.0, 100.0, 0.0, 100.0, 0.0])]

    gated_xy: Float64[ndarray, "5"] = stack_postprocessed(records).xy[0, :, 0, 0]

    np.testing.assert_array_equal(gated_xy, [0.0, 100.0, 0.0, 100.0, 0.0])


def test_median_still_filters_consecutive_video_frames() -> None:
    # At the native rate the AssemblyHands-X 5-frame median removes a one-frame spike.
    records = [_frame(frame_index=i, x0=x) for i, x in enumerate([0.0, 0.0, 100.0, 0.0, 0.0])]

    gated_xy: Float64[ndarray, "5"] = stack_postprocessed(records).xy[0, :, 0, 0]

    np.testing.assert_array_equal(gated_xy, [0.0, 0.0, 0.0, 0.0, 0.0])


@settings(max_examples=60, deadline=None)
@given(
    data=st.data(),
    window=st.sampled_from([1, 3, AHX_MEDIAN_WINDOW, 7, 9]),
    n_records=st.integers(1, 12),
)
def test_median_is_the_identity_when_records_are_farther_apart_than_half_its_window(data: st.DataObject, window: int, n_records: int) -> None:
    gaps: list[int] = data.draw(st.lists(st.integers(window // 2 + 1, 200), min_size=n_records, max_size=n_records))
    frame_index: Int64[ndarray, " t"] = np.cumsum(gaps).astype(np.int64)
    kp_xy: Float64[ndarray, "t 4 2"] = data.draw(
        arrays(np.float64, (n_records, 4, 2), elements=st.one_of(st.floats(-1e4, 1e4), st.just(np.nan)))
    )

    np.testing.assert_array_equal(median_filter_keypoints(kp_xy, frame_index, window=window), kp_xy)
