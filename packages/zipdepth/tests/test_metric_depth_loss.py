"""Metric distillation-loss behavior tests, including legacy bit-parity at defaults."""

import pytest
import torch
from jaxtyping import Bool, Float32
from torch import Tensor

from zipdepth.loss.depth_loss import ZipDepthLoss
from zipdepth.loss.metric_depth_loss import MetricDepthLoss


def test_metric_depth_loss_is_masked_l1_plus_half_gradient() -> None:
    """Use metre-space errors directly, without SSI normalization."""
    prediction_b1hw: Float32[Tensor, "1 1 2 2"] = torch.tensor([[[[1.0, 2.0], [1.0, 2.0]]]])
    target_b1hw: Float32[Tensor, "1 1 2 2"] = torch.ones(1, 1, 2, 2)
    valid_b1hw: Bool[Tensor, "1 1 2 2"] = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    criterion: MetricDepthLoss = MetricDepthLoss(grad_scales=1)

    loss: Tensor
    parts: dict[str, Tensor]
    loss, parts = criterion(prediction_b1hw, target_b1hw, valid_b1hw)

    assert float(loss) == pytest.approx(0.75)
    assert {key: float(value) for key, value in parts.items()} == {"l1": pytest.approx(0.5), "grad": pytest.approx(0.5)}


def test_metric_depth_loss_applies_teacher_range_and_explicit_mask() -> None:
    """Supervise only explicit finite teacher pixels in the 0.1--4 m lane."""
    prediction_b1hw: Float32[Tensor, "1 1 1 5"] = torch.tensor([[[[2.0, 2.0, 2.0, 2.0, 2.0]]]])
    target_b1hw: Float32[Tensor, "1 1 1 5"] = torch.tensor([[[[0.0, 1.0, 3.0, 5.0, float("nan")]]]])
    valid_b1hw: Bool[Tensor, "1 1 1 5"] = torch.tensor([[[[True, True, False, True, True]]]])
    criterion: MetricDepthLoss = MetricDepthLoss(grad_scales=1)

    loss: Tensor = criterion(prediction_b1hw, target_b1hw, valid_b1hw)[0]

    assert float(loss) == pytest.approx(1.0)


def _fixture() -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    target = torch.rand((2, 1, 64, 80), generator=generator) * 6.0  # includes > 4 m values
    target[:, :, :4, :4] = 0.0  # invalid teacher pixels
    pred = target + 0.05 * torch.randn((2, 1, 64, 80), generator=generator)
    mask = torch.ones_like(target, dtype=torch.bool)
    return pred, target, mask


def test_defaults_match_legacy_objective() -> None:
    pred, target, mask = _fixture()
    total, parts = MetricDepthLoss(gradient_weight=0.5)(pred=pred, target=target, mask=mask)

    legacy = ZipDepthLoss(alpha_ssi=0.0, alpha_grad=1.0, grad_scales=4)
    valid = torch.isfinite(target) & (target >= 0.1) & (target <= 4.0) & mask
    safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    weights = valid.to(pred.dtype)
    expected_l1 = legacy._ssi_loss(safe_pred, safe_target, weights)
    expected_grad = legacy._gradient_loss(safe_pred, safe_target, weights)
    torch.testing.assert_close(parts["l1"], expected_l1, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(parts["grad"], expected_grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(total, expected_l1 + 0.5 * expected_grad, atol=1e-6, rtol=1e-5)


def test_fullres_only_weights_match_single_scale_legacy() -> None:
    pred, target, mask = _fixture()
    _, parts = MetricDepthLoss(gradient_weight=2.0, grad_scale_weights=(1.0, 0.0, 0.0, 0.0))(pred=pred, target=target, mask=mask)

    legacy = ZipDepthLoss(alpha_ssi=0.0, alpha_grad=1.0, grad_scales=1)
    valid = torch.isfinite(target) & (target >= 0.1) & (target <= 4.0) & mask
    safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    expected_grad = legacy._gradient_loss(safe_pred, safe_target, valid.to(pred.dtype))
    torch.testing.assert_close(parts["grad"], expected_grad, atol=1e-6, rtol=1e-5)


def test_wide_gradient_window_supervises_pairs_beyond_4m() -> None:
    target = torch.full((1, 1, 32, 32), 3.5)
    target[:, :, :, 16:] = 5.0  # the strongest room edge: near object against a far wall
    pred = torch.full_like(target, 3.5)  # misses the step entirely
    mask = torch.ones_like(target, dtype=torch.bool)
    _, narrow = MetricDepthLoss(gradient_weight=2.0)(pred=pred, target=target, mask=mask)
    _, wide = MetricDepthLoss(gradient_weight=2.0, grad_max_depth_m=64.0, grad_pool_mask_threshold=0.5)(
        pred=pred, target=target, mask=mask
    )
    assert float(narrow["grad"]) == 0.0  # every gradient pair straddles the 4 m boundary and is deleted
    assert float(wide["grad"]) > 0.01  # the wide window keeps and penalizes the missed edge


def test_wide_window_requires_unwindowed_mask() -> None:
    """A pre-windowed mask (the dataset's target_valid) silently caps grad_max_depth_m at 4 m.

    This pins the trainer contract: the loss must be fed target_finite, not
    target_valid, or widening the gradient window selects nothing new.
    """
    target = torch.full((1, 1, 32, 32), 3.5)
    target[:, :, :, 16:] = 5.0
    pred = torch.full_like(target, 3.5)
    windowed_mask = (target >= 0.1) & (target <= 4.0)  # the eval-facing target_valid
    finite_mask = target > 0.0  # the loss-facing target_finite
    wide = MetricDepthLoss(gradient_weight=2.0, grad_max_depth_m=64.0, grad_pool_mask_threshold=0.5)
    _, capped = wide(pred=pred, target=target, mask=windowed_mask)
    _, widened = wide(pred=pred, target=target, mask=finite_mask)
    assert float(capped["grad"]) == 0.0
    assert float(widened["grad"]) > 0.01


def test_edge_weighted_l1_upweights_edge_errors() -> None:
    target = torch.full((1, 1, 32, 32), 1.0)
    target[:, :, :, 16:] = 2.0
    pred = target.clone()
    pred[:, :, :, 15:17] += 0.2  # error concentrated on the edge
    mask = torch.ones_like(target, dtype=torch.bool)
    _, plain = MetricDepthLoss(gradient_weight=0.0)(pred=pred, target=target, mask=mask)
    _, weighted = MetricDepthLoss(gradient_weight=0.0, edge_weight_alpha=3.0)(pred=pred, target=target, mask=mask)
    assert float(weighted["l1"]) > float(plain["l1"]) * 1.5


def test_edge_weight_ignores_window_and_hole_boundaries() -> None:
    """Invalid neighbours must not register as teacher edges in the L1 weight map."""
    target = torch.full((1, 1, 32, 32), 2.0)
    target[:, :, :, 16:] = 5.0  # beyond the 4 m window: a window boundary, not an edge
    target[:, :, 8, 8] = 0.0  # a hole
    pred = target.clone()
    pred[:, :, :, :1] += 0.2  # one column of error far from any boundary...
    pred_boundary = target.clone()
    pred_boundary[:, :, :, 15:16] += 0.2  # ...and the same column of error hugging the window boundary
    mask = target > 0.0
    criterion = MetricDepthLoss(gradient_weight=0.0, edge_weight_alpha=3.0)
    _, flat_error = criterion(pred=pred, target=target, mask=mask)
    _, boundary_error = criterion(pred=pred_boundary, target=target, mask=mask)
    # The masked gradient sees no valid-to-valid discontinuity anywhere, so neither
    # error location earns extra weight: both reduce to the same plain masked L1.
    torch.testing.assert_close(flat_error["l1"], boundary_error["l1"], atol=1e-6, rtol=1e-5)
