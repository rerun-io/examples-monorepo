"""The masked-composite photometric loss of TRiGS as one function for torch.compile (PennyGS 3.5's single-pass loss).

The test suite's beartype import hook skips this module (pyproject ``beartype_skip_packages``): dynamo traces these
functions with fake tensors, which a runtime type check cannot inspect.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float32, UInt8
from torch import Tensor


def masked_composite_terms(
    pred: Float32[Tensor, "b h w 3"],
    alpha: Float32[Tensor, "b h w 1"],
    gt: UInt8[Tensor, "b h w 3"] | Float32[Tensor, "b h w 3"],
    mask: Float32[Tensor, "b h w"],
    bg: Float32[Tensor, "3"],
    alpha_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``TrigsTrainer._photometric_losses`` for ``masked_rgb_loss`` + ``masked_rgb_composite`` with a constant background and
    clamped renders, without the SSIM term, for ``b`` same-size cameras: per camera (L1, composited render, composited
    target, alpha-outside, alpha-to-mask), the composited images channel-first ``(b, 3, h, w)`` (the layout the fused SSIM
    reads, so no copy sits between the two). The target is uint8 as stored (divided by 255 here) or float in [0, 1]."""
    m: Tensor = mask[..., None]
    m = (m / torch.where(m.amax(dim=(1, 2, 3), keepdim=True) > 1.0, 255.0, 1.0)).clamp(0.0, 1.0)
    a: Tensor = alpha.clamp(0.0, 1.0)
    outside: Tensor = (1.0 - m).clamp(0.0, 1.0)
    pred_l: Tensor = (pred + (1.0 - a) * bg).clamp(0.0, 1.0)
    target: Tensor = gt.float().div(255.0) if gt.dtype == torch.uint8 else gt
    gt_l: Tensor = target * m + bg * outside
    n_px: float = float(m.shape[1] * m.shape[2])
    bce: Tensor = F.binary_cross_entropy(a.clamp(alpha_eps, 1.0 - alpha_eps), m, reduction="none")
    alpha_mask: Tensor = 0.5 * ((bce * m).sum(dim=(1, 2, 3)) / n_px + (bce * outside).sum(dim=(1, 2, 3)) / n_px)
    l1: Tensor = (pred_l - gt_l).abs().mean(dim=(1, 2, 3))
    alpha_out: Tensor = (a * outside).sum(dim=(1, 2, 3)) / n_px
    return l1, pred_l.permute(0, 3, 1, 2).contiguous(), gt_l.permute(0, 3, 1, 2).contiguous(), alpha_out, alpha_mask
