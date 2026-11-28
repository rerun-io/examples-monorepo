"""Shared denoising math for RCP, single-GPU, and multi-GPU execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from fdanyone.model.loader import Denoiser


STEP_HOOKS: list[Callable[[int, tuple[int, ...] | None, Tensor], None]] = []
"""Observers of every group step: hook(step_index, view_indices, clean_estimate).

view_indices names the target views of the group (None for RCP proposals);
clean_estimate is the step's x_0 estimate on the denoising device. Hooks must not
modify it.
"""


def denoise_group(
    denoiser: Denoiser,
    latents: Tensor,
    source: Tensor,
    context: Tensor,
    pose_features: Tensor,
    null_pose_feature: Tensor,
    step_index: int,
    view_indices: tuple[int, ...] | None = None,
) -> Tensor:
    """Advance one camera group by one scheduler step."""

    import torch

    from fdanyone.model.quantization import select_step_precision

    select_step_precision(denoiser.model, step_index)
    timestep = denoiser.scheduler.timesteps[step_index]
    batched_timestep = timestep.unsqueeze(0).to(dtype=denoiser.dtype, device=latents.device)
    batched_timestep = torch.cat([batched_timestep] * latents.shape[0], dim=0)
    prediction = denoiser.model(
        x=latents,
        x_src=source,
        timestep=batched_timestep,
        pose_features=pose_features,
        null_pose_feature=null_pose_feature,
        context=context,
    )
    if STEP_HOOKS:
        # Flow matching: x_t = (1 - sigma) x_0 + sigma noise and the model predicts noise - x_0.
        clean_estimate = latents - denoiser.scheduler.sigmas[step_index] * prediction
        for hook in STEP_HOOKS:
            hook(step_index, view_indices, clean_estimate)
    return denoiser.scheduler.step(prediction, timestep, latents)
