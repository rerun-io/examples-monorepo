"""The reallocatable metric loss must match the legacy objective at defaults."""

import torch

from zipdepth.loss.depth_loss import ZipDepthLoss
from zipdepth.loss.metric_depth_loss import MetricDepthLoss


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    torch.testing.assert_close(torch.as_tensor(parts["l1"]), expected_l1, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.as_tensor(parts["grad"]), expected_grad, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(total, expected_l1 + 0.5 * expected_grad, atol=1e-6, rtol=1e-5)


def test_fullres_only_weights_match_single_scale_legacy() -> None:
    pred, target, mask = _fixture()
    _, parts = MetricDepthLoss(gradient_weight=2.0, grad_scale_weights=(1.0, 0.0, 0.0, 0.0))(pred=pred, target=target, mask=mask)

    legacy = ZipDepthLoss(alpha_ssi=0.0, alpha_grad=1.0, grad_scales=1)
    valid = torch.isfinite(target) & (target >= 0.1) & (target <= 4.0) & mask
    safe_pred = torch.where(valid, pred, torch.zeros_like(pred))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    expected_grad = legacy._gradient_loss(safe_pred, safe_target, valid.to(pred.dtype))
    torch.testing.assert_close(torch.as_tensor(parts["grad"]), expected_grad, atol=1e-6, rtol=1e-5)


def test_wide_gradient_window_supervises_pairs_beyond_4m() -> None:
    target = torch.full((1, 1, 32, 32), 3.5)
    target[:, :, :, 16:] = 5.0  # the strongest room edge: near object against a far wall
    pred = torch.full_like(target, 3.5)  # misses the step entirely
    mask = torch.ones_like(target, dtype=torch.bool)
    _, narrow = MetricDepthLoss(gradient_weight=2.0)(pred=pred, target=target, mask=mask)
    _, wide = MetricDepthLoss(gradient_weight=2.0, grad_max_depth_m=64.0, grad_pool_mask_threshold=0.5)(
        pred=pred, target=target, mask=mask
    )
    assert narrow["grad"] == 0.0  # every gradient pair straddles the 4 m boundary and is deleted
    assert wide["grad"] > 0.01  # the wide window keeps and penalizes the missed edge


def test_edge_weighted_l1_upweights_edge_errors() -> None:
    target = torch.full((1, 1, 32, 32), 1.0)
    target[:, :, :, 16:] = 2.0
    pred = target.clone()
    pred[:, :, :, 15:17] += 0.2  # error concentrated on the edge
    mask = torch.ones_like(target, dtype=torch.bool)
    _, plain = MetricDepthLoss(gradient_weight=0.0)(pred=pred, target=target, mask=mask)
    _, weighted = MetricDepthLoss(gradient_weight=0.0, edge_weight_alpha=3.0)(pred=pred, target=target, mask=mask)
    assert weighted["l1"] > plain["l1"] * 1.5
