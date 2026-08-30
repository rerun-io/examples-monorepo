"""Metric PromptDA distillation loss."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from zipdepth.loss.depth_loss import ZipDepthLoss


class MetricDepthLoss(nn.Module):
    """Masked metre L1 plus the existing multi-scale gradient loss."""

    def __init__(self, gradient_weight: float = 0.5, grad_scales: int = 4) -> None:
        super().__init__()
        if gradient_weight < 0.0:
            raise ValueError("gradient_weight must be non-negative")
        if grad_scales <= 0:
            raise ValueError("grad_scales must be positive")
        self.gradient_weight: float = gradient_weight
        self._gradient: ZipDepthLoss = ZipDepthLoss(alpha_ssi=0.0, alpha_grad=1.0, grad_scales=grad_scales)

    def forward(
        self,
        pred: Float[Tensor, "b 1 h w"],
        target: Float[Tensor, "b 1 h w"],
        mask: Bool[Tensor, "b 1 h w"] | Float[Tensor, "b 1 h w"] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the fixed Stage-1 metric objective.

        Args:
            pred: Predicted float depth in metres with shape
                ``(batch, 1, height, width)``.
            target: Dense teacher float depth in metres with the same shape.
            mask: Optional explicit teacher-valid mask with the same shape.

        Returns:
            Scalar loss and detached ``l1``/``grad`` components.
        """
        valid_b1hw: Tensor = torch.isfinite(target) & (target >= 0.1) & (target <= 4.0)
        if mask is not None:
            valid_b1hw = valid_b1hw & (mask > 0)
        mask_b1hw: Tensor = valid_b1hw.to(dtype=pred.dtype)
        safe_prediction_b1hw: Tensor = torch.where(valid_b1hw, pred, torch.zeros_like(pred))
        safe_target_b1hw: Tensor = torch.where(valid_b1hw, target, torch.zeros_like(target))
        l1_loss: Tensor = self._gradient._ssi_loss(safe_prediction_b1hw, safe_target_b1hw, mask_b1hw)
        gradient_loss: Tensor = self._gradient._gradient_loss(safe_prediction_b1hw, safe_target_b1hw, mask_b1hw)
        total_loss: Tensor = l1_loss + self.gradient_weight * gradient_loss
        return total_loss, {"l1": float(l1_loss.detach().item()), "grad": float(gradient_loss.detach().item())}
