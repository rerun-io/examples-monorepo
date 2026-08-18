"""Holdout metrics and schema-v3 product rendering for direct gsplat."""

import time

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from gauss_surf.contracts import RENDER_BACKGROUND
from gauss_surf.train_gsplat.cache import GpuTrainingCache, TrainingCamera
from gauss_surf.train_gsplat.core import per_frame_average_psnr
from gauss_surf.train_gsplat.renderer import RenderOutput, render_splats


def evaluate_holdout(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    cache: GpuTrainingCache,
) -> dict[str, float]:
    """Render all wide holdouts at full resolution and average per-frame metrics."""
    holdouts: tuple[TrainingCamera, ...] = tuple(cache.cameras[index] for index in cache.holdout_indices)
    if not holdouts:
        raise RuntimeError("training cache has no wide holdouts")
    device: torch.device = splats["means"].device
    background_3: torch.Tensor = torch.tensor(RENDER_BACKGROUND, dtype=torch.float32, device=device)
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    started_at: float = time.perf_counter()
    with torch.inference_mode():
        for index, camera in enumerate(holdouts):
            output: RenderOutput = render_splats(
                splats, camera, downscale=1, sh_degree=3, background_3=background_3, absgrad=False
            )
            target: torch.Tensor = cache.wide_rgb_nhw3[camera.cache_index].to(device=device, dtype=torch.float32) / 255.0
            psnr: torch.Tensor = per_frame_average_psnr(output.rgb_hw3[None], target[None])
            psnr_values.append(float(psnr.item()))
            ssim_value = structural_similarity_index_measure(
                output.rgb_hw3.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), data_range=1.0
            )
            if not isinstance(ssim_value, torch.Tensor):
                raise RuntimeError("SSIM metric unexpectedly returned a full image")
            ssim: torch.Tensor = ssim_value
            ssim_values.append(float(ssim.item()))
            if (index + 1) % 10 == 0 or index + 1 == len(holdouts):
                print(f"evaluated {index + 1}/{len(holdouts)} holdout frames", flush=True)
    torch.cuda.synchronize(device)
    return {
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "wall_seconds": time.perf_counter() - started_at,
    }
