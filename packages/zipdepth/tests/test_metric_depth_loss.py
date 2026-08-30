"""Metric distillation-loss behavior tests."""

import pytest
import torch
from jaxtyping import Bool, Float32
from torch import Tensor

from zipdepth.loss.metric_depth_loss import MetricDepthLoss


def test_metric_depth_loss_is_masked_l1_plus_half_gradient() -> None:
    """Use metre-space errors directly, without SSI normalization."""
    prediction_b1hw: Float32[Tensor, "1 1 2 2"] = torch.tensor([[[[1.0, 2.0], [1.0, 2.0]]]])
    target_b1hw: Float32[Tensor, "1 1 2 2"] = torch.ones(1, 1, 2, 2)
    valid_b1hw: Bool[Tensor, "1 1 2 2"] = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    criterion: MetricDepthLoss = MetricDepthLoss(grad_scales=1)

    loss: Tensor
    parts: dict[str, float]
    loss, parts = criterion(prediction_b1hw, target_b1hw, valid_b1hw)

    assert float(loss) == pytest.approx(0.75)
    assert parts == {"l1": pytest.approx(0.5), "grad": pytest.approx(0.5)}


def test_metric_depth_loss_applies_teacher_range_and_explicit_mask() -> None:
    """Supervise only explicit finite teacher pixels in the 0.1--4 m lane."""
    prediction_b1hw: Float32[Tensor, "1 1 1 5"] = torch.tensor([[[[2.0, 2.0, 2.0, 2.0, 2.0]]]])
    target_b1hw: Float32[Tensor, "1 1 1 5"] = torch.tensor([[[[0.0, 1.0, 3.0, 5.0, float("nan")]]]])
    valid_b1hw: Bool[Tensor, "1 1 1 5"] = torch.tensor([[[[True, True, False, True, True]]]])
    criterion: MetricDepthLoss = MetricDepthLoss(grad_scales=1)

    loss: Tensor = criterion(prediction_b1hw, target_b1hw, valid_b1hw)[0]

    assert float(loss) == pytest.approx(1.0)
