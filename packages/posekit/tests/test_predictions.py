"""Public boundary tests for posekit prediction containers."""

import numpy as np
import torch

from posekit.predictions import BoxDetections, DenseLandmarks2d, Keypoints2d
from posekit.skeletons import COCO_17


def test_numpy_helpers_detach_as_float32_arrays() -> None:
    """Box and dense-landmark arrays cross the serialization boundary as float32 NumPy arrays."""
    xyxy = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32, requires_grad=True)
    detections = BoxDetections(
        xyxy=xyxy,
        scores=torch.ones((1,), dtype=torch.float32),
        frame_indices=torch.zeros((1,), dtype=torch.long),
    )
    dense = DenseLandmarks2d(
        xy=torch.tensor([[[5.0, 6.0]]], dtype=torch.float32, requires_grad=True),
        log_variance=torch.zeros((1, 1), dtype=torch.float32),
        visibility=torch.tensor([[0.7]], dtype=torch.float32, requires_grad=True),
        contact=torch.tensor([[0.4]], dtype=torch.float32, requires_grad=True),
        floor_contact=torch.tensor([[0.2]], dtype=torch.float32, requires_grad=True),
        frame_indices=torch.zeros((1,), dtype=torch.long),
    )
    keypoints = Keypoints2d(
        xy=torch.tensor([[[7.0, 8.0]]], dtype=torch.float32, requires_grad=True),
        scores=torch.tensor([[0.9]], dtype=torch.float32, requires_grad=True),
        frame_indices=torch.zeros((1,), dtype=torch.long),
        skeleton=COCO_17,
    )

    arrays: tuple[np.ndarray, ...] = (
        detections.xyxy_numpy(),
        dense.xy_numpy(),
        dense.visibility_numpy(),
        dense.contact_numpy(),
        dense.floor_contact_numpy(),
        keypoints.xy_numpy(),
        keypoints.scores_numpy(),
    )
    boxes_array, dense_xy_array, visibility_array, contact_array, floor_contact_array, keypoints_xy_array, keypoint_scores_array = arrays

    assert all(array.dtype == np.float32 for array in arrays)
    np.testing.assert_array_equal(boxes_array, [[1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_array_equal(dense_xy_array, [[[5.0, 6.0]]])
    np.testing.assert_allclose(visibility_array, [[0.7]])
    np.testing.assert_allclose(contact_array, [[0.4]])
    np.testing.assert_allclose(floor_contact_array, [[0.2]])
    np.testing.assert_array_equal(keypoints_xy_array, [[[7.0, 8.0]]])
    np.testing.assert_allclose(keypoint_scores_array, [[0.9]])
