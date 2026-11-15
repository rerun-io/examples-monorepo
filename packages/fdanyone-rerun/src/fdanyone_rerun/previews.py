"""Denoising previews: each step's x_0 estimate of every view, decoded cheaply by TAEW2.2.

The STEP_HOOKS seam of fdanyone.model.denoise hands over each group's clean-latent estimate; the full Wan 2.2
VAE would take seconds per view, the TAEW2.2 tiny decoder takes milliseconds for one latent frame.
"""

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy import ndarray
from taehv import TAEHV


def preview_latent_index(num_latent_frames: int) -> int:
    """The latent frame a preview shows: the middle of the clip (latent frame ``i`` decodes to pixel frame ``4 i``)."""
    return num_latent_frames // 2


@torch.inference_mode()
def decode_previews(decoder: TAEHV, clean_estimate: Float[torch.Tensor, "views 48 t h w"]) -> list[UInt8[ndarray, "height width 3"]]:
    """Decode the middle latent frame of each view to an RGB image."""
    index: int = preview_latent_index(clean_estimate.shape[2])
    frame: Float[torch.Tensor, "views 1 48 h w"] = clean_estimate[:, :, index : index + 1].transpose(1, 2)
    decoded: Float[torch.Tensor, "views frames 3 height width"] = decoder.decode_video(
        frame.to(dtype=torch.float16), parallel=False, show_progress_bar=False
    )
    rgb: UInt8[torch.Tensor, "views height width 3"] = (decoded[:, 0].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1)
    return list(np.ascontiguousarray(rgb.cpu().numpy()))
