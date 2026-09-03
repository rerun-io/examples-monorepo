"""Metric PromptDA distillation loss."""

import torch
from jaxtyping import Bool, Float
from monopriors.models.depth_completion.base_completion_depth import MAX_METRIC_DEPTH_M, MIN_METRIC_DEPTH_M
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
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        """Compute the fixed Stage-1 metric objective.

        Args:
            pred: Predicted float depth in metres with shape
                ``(batch, 1, height, width)``.
            target: Dense teacher float depth in metres with the same shape.
            mask: Optional explicit teacher-valid mask with the same shape.

        Returns:
            Scalar loss and detached 0-dim ``l1``/``grad`` components.
        """
        valid_bchw: Bool[Tensor, "b 1 h w"] = torch.isfinite(target) & (target >= MIN_METRIC_DEPTH_M) & (target <= MAX_METRIC_DEPTH_M)
        if mask is not None:
            valid_bchw = valid_bchw & (mask > 0)
        mask_bchw: Float[Tensor, "b 1 h w"] = valid_bchw.to(dtype=pred.dtype)
        safe_prediction_bchw: Float[Tensor, "b 1 h w"] = torch.where(valid_bchw, pred, torch.zeros_like(pred))
        safe_target_bchw: Float[Tensor, "b 1 h w"] = torch.where(valid_bchw, target, torch.zeros_like(target))
        l1_loss: Float[Tensor, ""] = self._gradient._ssi_loss(safe_prediction_bchw, safe_target_bchw, mask_bchw)
        gradient_loss: Float[Tensor, ""] = self._gradient._gradient_loss(safe_prediction_bchw, safe_target_bchw, mask_bchw)
        total_loss: Float[Tensor, ""] = l1_loss + self.gradient_weight * gradient_loss
        return total_loss, {"l1": l1_loss.detach(), "grad": gradient_loss.detach()}
