"""Metric distillation-loss behavior tests, including legacy bit-parity at defaults."""

import pytest
import torch
from jaxtyping import Bool, Float32
from torch import Tensor

from zipdepth.loss.depth_loss import ZipDepthLoss
from zipdepth.loss.metric_depth_loss import MetricDepthLoss


def test_metric_depth_loss_is_masked_l1_plus_half_gradient() -> None:
    """Use metre-space errors directly, without SSI normalization."""
    prediction_bchw: Float32[Tensor, "1 1 2 2"] = torch.tensor([[[[1.0, 2.0], [1.0, 2.0]]]])
    target_bchw: Float32[Tensor, "1 1 2 2"] = torch.ones(1, 1, 2, 2)
    valid_bchw: Bool[Tensor, "1 1 2 2"] = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    criterion: MetricDepthLoss = MetricDepthLoss(grad_scales=1)

    loss: Float32[Tensor, ""]
    parts: dict[str, Float32[Tensor, ""]]
    loss, parts = criterion(prediction_bchw, target_bchw, valid_bchw)

    assert float(loss) == pytest.approx(0.75)
    assert {key: float(value) for key, value in parts.items()} == {"l1": pytest.approx(0.5), "grad": pytest.approx(0.5)}


def test_metric_depth_loss_applies_teacher_range_and_explicit_mask() -> None:
    """Supervise only explicit finite teacher pixels in the 0.1--4 m lane."""
    prediction_bchw: Float32[Tensor, "1 1 1 5"] = torch.tensor([[[[2.0, 2.0, 2.0, 2.0, 2.0]]]])
    target_bchw: Float32[Tensor, "1 1 1 5"] = torch.tensor([[[[0.0, 1.0, 3.0, 5.0, float("nan")]]]])
    valid_bchw: Bool[Tensor, "1 1 1 5"] = torch.tensor([[[[True, True, False, True, True]]]])
    criterion: MetricDepthLoss = MetricDepthLoss(grad_scales=1)

    loss: Float32[Tensor, ""] = criterion(prediction_bchw, target_bchw, valid_bchw)[0]

    assert float(loss) == pytest.approx(1.0)


def _fixture() -> tuple[Float32[Tensor, "2 1 64 80"], Float32[Tensor, "2 1 64 80"], Bool[Tensor, "2 1 64 80"]]:
    generator: torch.Generator = torch.Generator().manual_seed(0)
    target: Float32[Tensor, "2 1 64 80"] = torch.rand((2, 1, 64, 80), generator=generator) * 6.0  # includes > 4 m values
    target[:, :, :4, :4] = 0.0  # invalid teacher pixels
    pred: Float32[Tensor, "2 1 64 80"] = target + 0.05 * torch.randn((2, 1, 64, 80), generator=generator)
    mask: Bool[Tensor, "2 1 64 80"] = torch.ones_like(target, dtype=torch.bool)
    return pred, target, mask


def test_defaults_match_legacy_objective() -> None:
    fixture: tuple[Float32[Tensor, "2 1 64 80"], Float32[Tensor, "2 1 64 80"], Bool[Tensor, "2 1 64 80"]] = _fixture()
    pred: Float32[Tensor, "2 1 64 80"] = fixture[0]
    target: Float32[Tensor, "2 1 64 80"] = fixture[1]
    mask: Bool[Tensor, "2 1 64 80"] = fixture[2]
    total: Float32[Tensor, ""]
    parts: dict[str, Float32[Tensor, ""]]
    total, parts = MetricDepthLoss(gradient_weight=0.5)(pred=pred, target=target, mask=mask)

    legacy: ZipDepthLoss = ZipDepthLoss(alpha_ssi=0.0, alpha_grad=1.0, grad_scales=4)
    valid: Bool[Tensor, "2 1 64 80"] = torch.isfinite(target) & (target >= 0.1) & (target <= 4.0) & mask
    safe_pred: Float32[Tensor, "2 1 64 80"] = torch.where(valid, pred, torch.zeros_like(pred))
    safe_target: Float32[Tensor, "2 1 64 80"] = torch.where(valid, target, torch.zeros_like(target))
    weights: Float32[Tensor, "2 1 64 80"] = valid.to(pred.dtype)
    expected_l1: Float32[Tensor, ""] = legacy._ssi_loss(safe_pred, safe_target, weights)
    expected_grad: Float32[Tensor, ""] | float = legacy._gradient_loss(safe_pred, safe_target, weights)
    assert isinstance(expected_grad, Tensor)
    assert torch.equal(parts["l1"], expected_l1)
    assert torch.equal(parts["grad"], expected_grad)
    assert torch.equal(total, expected_l1 + 0.5 * expected_grad)
