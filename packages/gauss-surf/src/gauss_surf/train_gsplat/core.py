"""Small, testable contracts used by direct gsplat training."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch


def decode_normal_uint8(encoded: torch.Tensor) -> torch.Tensor:
    """Decode normal components while mapping the central code to exact zero."""
    if encoded.dtype != torch.uint8:
        raise TypeError(f"encoded normals must be uint8, got {encoded.dtype}")
    decoded: torch.Tensor = encoded.to(torch.float32) / 255.0 * 2.0 - 1.0
    return torch.where(encoded == 128, torch.zeros_like(decoded), decoded)


def per_frame_average_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean of independent frame PSNR values."""
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError(f"prediction and target shapes must match by frame, got {prediction.shape} and {target.shape}")
    frame_dimensions: tuple[int, ...] = tuple(range(1, prediction.ndim))
    frame_mse: torch.Tensor = (prediction - target).square().mean(dim=frame_dimensions)
    return (-10.0 * torch.log10(frame_mse)).mean()


def holdout_hash(stems: tuple[str, ...]) -> str:
    """Hash ordered holdout identities with an unambiguous trailing newline."""
    return hashlib.sha256(("\n".join(stems) + "\n").encode()).hexdigest()


def opacity_export_mask(opacity_logits: torch.Tensor) -> torch.Tensor:
    """Select Gaussians whose opacity meets the required 1/255 floor."""
    threshold: torch.Tensor = torch.logit(opacity_logits.new_tensor(1.0 / 255.0))
    return opacity_logits >= threshold


def median_scale_regularization(log_scales_n3: torch.Tensor, *, maximum_ratio: float) -> torch.Tensor:
    """Return max-to-median loss with zero-volume duplicate seeds neutralized."""
    if log_scales_n3.ndim != 2 or log_scales_n3.shape[1] != 3:
        raise ValueError(f"log scales must have shape (N, 3), got {log_scales_n3.shape}")
    if maximum_ratio <= 0.0:
        raise ValueError("maximum ratio must be positive")
    scales_n3: torch.Tensor = torch.exp(log_scales_n3)
    threshold: torch.Tensor = scales_n3.new_tensor(maximum_ratio)
    ratio_n: torch.Tensor = scales_n3.amax(dim=-1) / scales_n3.median(dim=-1).values
    # A duplicated seed can have all three log-scales at -inf. It contributes
    # no rendered support, so give its undefined 0/0 ratio zero excess loss.
    ratio_n = torch.where(torch.isnan(ratio_n), threshold, ratio_n)
    return (torch.maximum(ratio_n, threshold) - threshold).mean()


def positive_neighbor_distances(average_distances_n1: np.ndarray) -> np.ndarray:
    """Replace exact duplicate-seed zero distances by the smallest positive one."""
    distances_n1: np.ndarray = np.asarray(average_distances_n1, dtype=np.float32)
    positive: np.ndarray = distances_n1[distances_n1 > 0.0]
    if positive.size == 0:
        raise ValueError("nearest-neighbor initialization has no positive distances")
    return np.where(distances_n1 > 0.0, distances_n1, positive.min()).astype(np.float32, copy=False)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    splats: torch.nn.ParameterDict,
    metadata: dict[str, Any],
) -> None:
    """Save a tensor-and-primitives checkpoint compatible with weights-only loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "splats": {name: parameter.detach().cpu() for name, parameter in splats.items()},
        "metadata": metadata,
    }
    temporary_path: Path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
