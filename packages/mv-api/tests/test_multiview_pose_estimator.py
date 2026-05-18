import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from mv_api.multiview_pose_estimator import MultiviewBodyTracker


def test_compute_hand_bbox_expands_square_and_clips_to_image() -> None:
    hand_uv: Float32[ndarray, "4 2"] = np.array(
        [
            [10.0, 10.0],
            [20.0, 30.0],
            [15.0, 25.0],
            [np.nan, np.nan],
        ],
        dtype=np.float32,
    )

    bbox: Float32[ndarray, "4"] | None = MultiviewBodyTracker._compute_hand_bbox(
        hand_uv=hand_uv,
        image_shape=(40, 25),
        expansion_ratio=0.0,
    )

    expected_bbox: Float32[ndarray, "4"] = np.array([5.0, 10.0, 24.0, 30.0], dtype=np.float32)
    assert bbox is not None
    np.testing.assert_allclose(bbox, expected_bbox)
