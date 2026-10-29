"""The 2D image scorer (specs/twod.md 1): upstream `basic_metrics` plus LPIPS-Alex, on the GPU.

Same math as upstream `scripts/image_compression_training.basic_metrics` and `scripts/evaluate_image.py`:
clamp once to [0, 1]; PSNR = -10 log10 of the float64 MSE over all pixels and channels; SSIM =
`fused_ssim` (11x11 Gaussian, sigma 1.5); LPIPS = `lpips.LPIPS(net="alex")` on inputs mapped to [-1, 1].
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from PIL import Image


@dataclass(frozen=True, slots=True)
class ImageScores:
    """Scores of one decoded image."""

    psnr: float
    """PSNR in dB."""
    ssim: float
    """fused_ssim."""
    lpips_alex: float
    """LPIPS AlexNet."""


class ImageScorer:
    """Holds the LPIPS network on `device` across images."""

    def __init__(self, device: torch.device) -> None:
        import lpips

        self.device: torch.device = device
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device=device).eval()

    @torch.no_grad()
    def score(self, prediction: Float[torch.Tensor, "3 h w"], target: Float[torch.Tensor, "3 h w"]) -> ImageScores:
        from fused_ssim import fused_ssim

        image: torch.Tensor = prediction.to(device=self.device, dtype=torch.float32).clamp(0, 1)
        truth: torch.Tensor = target.to(device=self.device, dtype=torch.float32)
        mse: float = float((image.double() - truth.double()).square().mean())
        return ImageScores(
            psnr=-10.0 * math.log10(mse),
            ssim=float(fused_ssim(image[None], truth[None])),
            lpips_alex=float(self.lpips(image[None] * 2 - 1, truth[None] * 2 - 1)),
        )


def load_target(path: Path, device: torch.device) -> Float[torch.Tensor, "3 h w"]:
    """Ground truth exactly as upstream reads it: `PIL.Image.open(png).convert("RGB") / 255`, float32."""
    with Image.open(path) as image:
        rgb: UInt8[ndarray, "h w 3"] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pixels: Float32[ndarray, "h w 3"] = rgb.astype(np.float32) / np.float32(255)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).to(device=device)


@torch.no_grad()
def time_decoder(decode: Callable[[], torch.Tensor], warmup: int = 10, trials: int = 5, per_trial: int = 20) -> list[float]:
    """Mean milliseconds per decode for each trial, upstream's protocol (`quality_and_timing`)."""
    for _ in range(warmup):
        decode()
    trial_ms: list[float] = []
    for _ in range(trials):
        torch.cuda.synchronize()
        started: float = perf_counter()
        for _ in range(per_trial):
            decode()
        torch.cuda.synchronize()
        trial_ms.append((perf_counter() - started) * 1000.0 / per_trial)
    return trial_ms
