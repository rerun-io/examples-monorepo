"""Metric PromptDA distillation loss.

The gradient term reimplements the vendored ``ZipDepthLoss._gradient_loss``
(same pool-then-difference semantics, verified by test) so the metric lane can
reallocate weight across scales, widen the gradient validity window past the
4 m L1 window, and relax the pooled-mask erosion — the three levers behind the
edge-washout fixes. At the defaults every term matches the previous behavior.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

from zipdepth.loss.depth_loss import ZipDepthLoss

L1_MIN_DEPTH_M: float = 0.1
"""Lower edge of the metric supervision window, in metres."""
L1_MAX_DEPTH_M: float = 4.0
"""Upper edge of the metric L1 supervision window, in metres."""


class MetricDepthLoss(nn.Module):
    """Masked metre L1 plus a reallocatable multi-scale gradient loss.

    Args:
        gradient_weight: Total weight on the gradient term against L1 = 1.
        grad_scales: Number of dyadic gradient scales (k = 1, 2, 4, ...).
        grad_scale_weights: Per-scale weights, normalized to sum to one. None
            weights all scales equally (the upstream mean over scales).
        grad_max_depth_m: Upper edge of the gradient validity window. The
            default matches the L1 window; widening it keeps gradient pairs
            that straddle the 4 m boundary — the strongest room-scale edges.
        grad_pool_mask_threshold: Pooled-validity threshold at coarse scales.
            The upstream 0.99 kills every kxk block near an invalid pixel;
            0.5 keeps majority-valid blocks.
        edge_weight_alpha: Extra L1 weight on teacher edges,
            ``w = 1 + alpha * clip(g / q90(g), 0, 1)`` with detached teacher
            gradient magnitude ``g``. Zero disables the reweighting.
    """

    def __init__(
        self,
        gradient_weight: float = 0.5,
        grad_scales: int = 4,
        grad_scale_weights: tuple[float, ...] | None = None,
        grad_max_depth_m: float = 4.0,
        grad_pool_mask_threshold: float = 0.99,
        edge_weight_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if gradient_weight < 0.0:
            raise ValueError("gradient_weight must be non-negative")
        if grad_scales <= 0:
            raise ValueError("grad_scales must be positive")
        if grad_scale_weights is not None:
            if len(grad_scale_weights) != grad_scales:
                raise ValueError(f"grad_scale_weights needs {grad_scales} entries, got {len(grad_scale_weights)}")
            if any(weight < 0.0 for weight in grad_scale_weights) or sum(grad_scale_weights) <= 0.0:
                raise ValueError("grad_scale_weights must be non-negative and sum to a positive value")
        if grad_max_depth_m < L1_MAX_DEPTH_M:
            raise ValueError("grad_max_depth_m must not be narrower than the L1 window")
        if not 0.0 < grad_pool_mask_threshold < 1.0:
            raise ValueError("grad_pool_mask_threshold must be in (0, 1)")
        if edge_weight_alpha < 0.0:
            raise ValueError("edge_weight_alpha must be non-negative")
        self.gradient_weight: float = gradient_weight
        self.grad_scales: int = grad_scales
        raw_weights: tuple[float, ...] = grad_scale_weights if grad_scale_weights is not None else (1.0,) * grad_scales
        total_weight: float = sum(raw_weights)
        self.grad_scale_weights: tuple[float, ...] = tuple(weight / total_weight for weight in raw_weights)
        self.grad_max_depth_m: float = grad_max_depth_m
        self.grad_pool_mask_threshold: float = grad_pool_mask_threshold
        self.edge_weight_alpha: float = edge_weight_alpha
        self._l1: ZipDepthLoss = ZipDepthLoss(alpha_ssi=0.0, alpha_grad=1.0, grad_scales=grad_scales)

    def forward(
        self,
        pred: Float[Tensor, "b 1 h w"],
        target: Float[Tensor, "b 1 h w"],
        mask: Bool[Tensor, "b 1 h w"] | Float[Tensor, "b 1 h w"] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the metric objective.

        Args:
            pred: Predicted float depth in metres with shape
                ``(batch, 1, height, width)``.
            target: Dense teacher float depth in metres with the same shape.
                Values beyond the L1 window are used by the gradient term when
                ``grad_max_depth_m`` allows.
            mask: Optional explicit teacher-valid mask with the same shape.

        Returns:
            Scalar loss and detached ``l1``/``grad`` components.
        """
        finite_b1hw: Tensor = torch.isfinite(target) & (target >= L1_MIN_DEPTH_M)
        if mask is not None:
            finite_b1hw = finite_b1hw & (mask > 0)
        l1_valid_b1hw: Tensor = finite_b1hw & (target <= L1_MAX_DEPTH_M)
        l1_mask_b1hw: Tensor = l1_valid_b1hw.to(dtype=pred.dtype)
        safe_prediction_b1hw: Tensor = torch.where(l1_valid_b1hw, pred, torch.zeros_like(pred))
        safe_target_b1hw: Tensor = torch.where(l1_valid_b1hw, target, torch.zeros_like(target))
        if self.edge_weight_alpha > 0.0:
            l1_loss: Tensor = self._edge_weighted_l1(safe_prediction_b1hw, safe_target_b1hw, l1_mask_b1hw)
        else:
            l1_loss = self._l1._ssi_loss(safe_prediction_b1hw, safe_target_b1hw, l1_mask_b1hw)

        grad_valid_b1hw: Tensor = finite_b1hw & (target <= self.grad_max_depth_m)
        grad_mask_hw: Tensor = grad_valid_b1hw.to(dtype=pred.dtype).squeeze(1)
        grad_prediction_hw: Tensor = torch.where(grad_valid_b1hw, pred, torch.zeros_like(pred)).squeeze(1)
        grad_target_hw: Tensor = torch.where(grad_valid_b1hw, target, torch.zeros_like(target)).squeeze(1)
        gradient_loss: Tensor = self._gradient_loss(grad_prediction_hw, grad_target_hw, grad_mask_hw)

        total_loss: Tensor = l1_loss + self.gradient_weight * gradient_loss
        return total_loss, {"l1": float(l1_loss.detach().item()), "grad": float(gradient_loss.detach().item())}

    def _edge_weighted_l1(
        self,
        pred_b1hw: Float[Tensor, "b 1 h w"],
        target_b1hw: Float[Tensor, "b 1 h w"],
        mask_b1hw: Float[Tensor, "b 1 h w"],
    ) -> Tensor:
        """Masked L1 with extra weight on teacher depth discontinuities."""
        with torch.no_grad():
            gradient_hw: Tensor = torch.zeros_like(target_b1hw.squeeze(1))
            dx_hw: Tensor = (target_b1hw[..., 1:] - target_b1hw[..., :-1]).abs().squeeze(1)
            dy_hw: Tensor = (target_b1hw[..., 1:, :] - target_b1hw[..., :-1, :]).abs().squeeze(1)
            gradient_hw[:, :, :-1] += dx_hw
            gradient_hw[:, :, 1:] += dx_hw
            gradient_hw[:, :-1, :] += dy_hw
            gradient_hw[:, 1:, :] += dy_hw
            valid_gradients: Tensor = gradient_hw[mask_b1hw.squeeze(1) > 0]
            scale: Tensor = torch.quantile(valid_gradients.float(), 0.9) if valid_gradients.numel() > 0 else gradient_hw.new_ones(())
            weight_b1hw: Tensor = 1.0 + self.edge_weight_alpha * torch.clamp(gradient_hw / scale.clamp(min=1e-6), 0.0, 1.0).unsqueeze(1)
        weighted_mask_b1hw: Tensor = weight_b1hw * mask_b1hw
        return ((pred_b1hw - target_b1hw).abs() * weighted_mask_b1hw).sum() / (weighted_mask_b1hw.sum() + 1e-8)

    def _gradient_loss(
        self,
        pred_hw: Float[Tensor, "b h w"],
        target_hw: Float[Tensor, "b h w"],
        mask_hw: Float[Tensor, "b h w"],
    ) -> Tensor:
        """Upstream pool-then-difference gradient matching with per-scale weights."""
        total: Tensor = pred_hw.new_zeros(())
        for scale, scale_weight in enumerate(self.grad_scale_weights):
            if scale_weight == 0.0:
                continue
            k: int = 2**scale
            if k > 1:
                pooled_pred: Tensor = F.avg_pool2d(pred_hw.unsqueeze(1), k).squeeze(1)
                pooled_target: Tensor = F.avg_pool2d(target_hw.unsqueeze(1), k).squeeze(1)
                pooled_mask: Tensor = (F.avg_pool2d(mask_hw.unsqueeze(1), k).squeeze(1) > self.grad_pool_mask_threshold).to(pred_hw.dtype)
            else:
                pooled_pred, pooled_target, pooled_mask = pred_hw, target_hw, mask_hw
            pred_dx: Tensor = pooled_pred[:, :, 1:] - pooled_pred[:, :, :-1]
            pred_dy: Tensor = pooled_pred[:, 1:, :] - pooled_pred[:, :-1, :]
            target_dx: Tensor = pooled_target[:, :, 1:] - pooled_target[:, :, :-1]
            target_dy: Tensor = pooled_target[:, 1:, :] - pooled_target[:, :-1, :]
            mask_x: Tensor = pooled_mask[:, :, 1:] * pooled_mask[:, :, :-1]
            mask_y: Tensor = pooled_mask[:, 1:, :] * pooled_mask[:, :-1, :]
            loss_x: Tensor = ((pred_dx - target_dx).abs() * mask_x).sum()
            loss_y: Tensor = ((pred_dy - target_dy).abs() * mask_y).sum()
            total = total + scale_weight * (loss_x + loss_y) / (mask_x.sum() + mask_y.sum() + 1e-8)
        return total
