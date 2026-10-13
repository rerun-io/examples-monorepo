import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray

from exo_calib.ops.keypoints import postprocess_confidences, stack_postprocessed
from exo_calib.types import MultiViewKeypoints


def _frame(views: int = 1) -> MultiViewKeypoints:
    kp_xy: Float32[ndarray, "v 133 2"] = np.full((views, 133, 2), np.nan, dtype=np.float32)
    kp_xy[:, :3] = np.array([[50.0, 40.0], [5.0, 40.0], [-1.0, 40.0]], dtype=np.float32)
    conf: Float32[ndarray, "v 133"] = np.zeros((views, 133), dtype=np.float32)
    conf[:, :3] = np.array([0.575, 0.8, 0.8], dtype=np.float32)
    return MultiViewKeypoints(
        frame_index=0,
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
    filtered_xy, post_conf = postprocess_confidences(f.kp_xy, f.conf, f.crop_origin_xy, f.crop_size_wh, f.crop_input_wh, median_window=1, margin_px=10.0)

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
