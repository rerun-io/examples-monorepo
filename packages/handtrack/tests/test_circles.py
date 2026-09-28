import numpy as np
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray
from torch import Tensor

from handtrack.labels.circles import enclosing_circles, square_boxes


def test_enclosing_circle_uses_only_valid_points_and_preserves_batches() -> None:
    points: Float32[ndarray, '1 3 3 2'] = np.array([[[[0, 0], [4, 0], [200, 100]], [[2, 3], [99, 99], [0, 0]], [[0, 0], [0, 0], [0, 0]]]], dtype=np.float32)
    valid: Bool[ndarray, '1 3 3'] = np.array([[[True, True, False], [True, False, False], [False, False, False]]])
    circles: Float32[ndarray, '1 3 3'] = enclosing_circles(points, valid)
    np.testing.assert_allclose(circles[0, 0], [2.0, 0.0, 2.0], atol=2e-4)
    np.testing.assert_allclose(circles[0, 1], [2.0, 3.0, 0.0], atol=2e-4)
    assert np.isnan(circles[0, 2]).all()
    assert enclosing_circles(points[0, 0], valid[0, 0]).shape == (3,)


def test_square_box_enlargement() -> None:
    circle: Float32[Tensor, '3'] = torch.tensor([10.0, 20.0, 5.0])
    torch.testing.assert_close(square_boxes(circle), torch.tensor([5.0, 15.0, 15.0, 25.0]))
    torch.testing.assert_close(square_boxes(circle, 1.2), torch.tensor([4.0, 14.0, 16.0, 26.0]))
