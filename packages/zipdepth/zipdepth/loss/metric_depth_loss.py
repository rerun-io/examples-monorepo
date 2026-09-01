"""Metric PromptDA distillation loss.

The gradient term reimplements the vendored ``ZipDepthLoss._gradient_loss``
(same pool-then-difference semantics, verified by test) so the metric lane can
reallocate weight across scales, widen the gradient validity window past the
4 m L1 window, and relax the pooled-mask erosion — the three levers behind the
edge-washout fixes. The masked L1 is likewise inlined rather than reaching into
the vendored class's private methods. At the defaults every term matches the
previous behavior bit for bit.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

L1_MIN_DEPTH_M: float = 0.1
"""Lower edge of the metric supervision window, in metres."""
L1_MAX_DEPTH_M: float = 4.0
"""Upper edge of the metric L1 supervision window, in metres."""
EDGE_WEIGHT_QUANTILE: float = 0.9
"""Per-frame teacher-gradient quantile normalizing the edge-weighted L1, matching eval's ``EDGE_QUANTILE``."""


class MetricDepthLoss(nn.Module):
    """Masked metre L1 plus a reallocatable multi-scale gradient loss.

    The mask argument should carry the teacher's unwindowed validity (finite
    and non-zero): the L1 window is applied here, and ``grad_max_depth_m`` can
    then genuinely widen the gradient window past it. A pre-windowed mask
    silently caps both terms at its own ceiling instead.

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
            ``w = 1 + alpha * clip(g / q(g), 0, 1)`` with detached teacher
            gradient magnitude ``g`` (validity-masked, computed from the raw
            target so window and hole boundaries do not read as edges) and a
            per-frame quantile scale ``q``. Zero disables the reweighting.
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
        raw_weights: tuple[float, ...] = grad_scale_weights if grad_scale_weights is not None else (1.0,) * grad_scales
        total_weight: float = sum(raw_weights)
        self.grad_scale_weights: tuple[float, ...] = tuple(weight / total_weight for weight in raw_weights)
        self.grad_max_depth_m: float = grad_max_depth_m
        self.grad_pool_mask_threshold: float = grad_pool_mask_threshold
        self.edge_weight_alpha: float = edge_weight_alpha

    def forward(
        self,
        pred: Float[Tensor, "b 1 h w"],
        target: Float[Tensor, "b 1 h w"],
        mask: Bool[Tensor, "b 1 h w"] | Float[Tensor, "b 1 h w"] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the metric objective.

        Args:
            pred: Predicted float depth in metres with shape
                ``(batch, 1, height, width)``.
            target: Dense teacher float depth in metres with the same shape.
                Values beyond the L1 window are used by the gradient term when
                ``grad_max_depth_m`` allows.
            mask: Optional explicit teacher-valid mask with the same shape.

        Returns:
            Scalar loss and detached 0-dim ``l1``/``grad`` components. The
            components stay on device: the caller decides when to pay the
            host sync, after the whole step is enqueued.
        """
        finite_b1hw: Tensor = torch.isfinite(target) & (target >= L1_MIN_DEPTH_M)
        if mask is not None:
            finite_b1hw = finite_b1hw & (mask > 0)
        l1_valid_b1hw: Tensor = finite_b1hw & (target <= L1_MAX_DEPTH_M)
        l1_mask_b1hw: Tensor = l1_valid_b1hw.to(dtype=pred.dtype)
        safe_prediction_b1hw: Tensor = torch.where(l1_valid_b1hw, pred, 0.0)
        safe_target_b1hw: Tensor = torch.where(l1_valid_b1hw, target, 0.0)
        if self.edge_weight_alpha > 0.0:
            weight_b1hw: Tensor = self._edge_weight(target, l1_valid_b1hw) * l1_mask_b1hw
        else:
            weight_b1hw = l1_mask_b1hw
        l1_loss: Tensor = ((safe_prediction_b1hw - safe_target_b1hw).abs() * weight_b1hw).sum() / (weight_b1hw.sum() + 1e-8)

        if self.grad_max_depth_m == L1_MAX_DEPTH_M:
            grad_prediction_b1hw, grad_target_b1hw, grad_mask_b1hw = safe_prediction_b1hw, safe_target_b1hw, l1_mask_b1hw
        else:
            grad_valid_b1hw: Tensor = finite_b1hw & (target <= self.grad_max_depth_m)
            grad_mask_b1hw = grad_valid_b1hw.to(dtype=pred.dtype)
            grad_prediction_b1hw = torch.where(grad_valid_b1hw, pred, 0.0)
            grad_target_b1hw = torch.where(grad_valid_b1hw, target, 0.0)
        gradient_loss: Tensor = self._gradient_loss(
            grad_prediction_b1hw.squeeze(1), grad_target_b1hw.squeeze(1), grad_mask_b1hw.squeeze(1)
        )

        total_loss: Tensor = l1_loss + self.gradient_weight * gradient_loss
        return total_loss, {"l1": l1_loss.detach(), "grad": gradient_loss.detach()}

    def _edge_weight(
        self,
        target_b1hw: Float[Tensor, "b 1 h w"],
        valid_b1hw: Bool[Tensor, "b 1 h w"],
    ) -> Float[Tensor, "b 1 h w"]:
        """Per-pixel L1 weight from the teacher's validity-masked gradient magnitude.

        Differences straddling an invalid pixel are dropped before accumulation,
        so window boundaries and holes do not register as (maximal) edges. The
        normalizing quantile is per frame, matching eval's edge stratification
        and staying clear of ``torch.quantile``'s 2**24-element limit.
        """
        with torch.no_grad():
            target_hw: Tensor = target_b1hw.squeeze(1)
            valid_hw: Tensor = valid_b1hw.squeeze(1)
            valid_f_hw: Tensor = valid_hw.to(dtype=target_hw.dtype)
            dx_hw: Tensor = (target_hw[..., 1:] - target_hw[..., :-1]).abs() * (valid_f_hw[..., 1:] * valid_f_hw[..., :-1])
            dy_hw: Tensor = (target_hw[..., 1:, :] - target_hw[..., :-1, :]).abs() * (valid_f_hw[..., 1:, :] * valid_f_hw[..., :-1, :])
            gradient_hw: Tensor = torch.zeros_like(target_hw)
            gradient_hw[:, :, :-1] += dx_hw
            gradient_hw[:, :, 1:] += dx_hw
            gradient_hw[:, :-1, :] += dy_hw
            gradient_hw[:, 1:, :] += dy_hw
            batch: int = gradient_hw.shape[0]
            masked_gradient_bn: Tensor = torch.where(valid_hw, gradient_hw, torch.nan).view(batch, -1).float()
            scale_b: Tensor = torch.nanquantile(masked_gradient_bn, EDGE_WEIGHT_QUANTILE, dim=1)
            scale_b = torch.where(torch.isnan(scale_b) | (scale_b <= 0.0), torch.ones_like(scale_b), scale_b)
            normalized_hw: Tensor = torch.clamp(gradient_hw / scale_b.view(batch, 1, 1).to(dtype=gradient_hw.dtype), 0.0, 1.0)
            return (1.0 + self.edge_weight_alpha * normalized_hw).unsqueeze(1)

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
