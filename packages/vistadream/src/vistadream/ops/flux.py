from dataclasses import dataclass

import numpy as np
import torch
from diffusers import FluxFillPipeline
from jaxtyping import UInt8
from PIL import Image


@dataclass
class FluxInpaintingConfig:
    """Configuration for Flux Fill inpainting via HuggingFace diffusers."""

    offload: bool = True
    """Whether to use CPU offloading to reduce VRAM usage."""
    num_steps: int = 25
    """Number of denoising steps."""
    guidance: int | float = 30.0
    """Guidance scale for classifier-free guidance."""
    seed: int = 42
    """Random seed for reproducibility."""


class FluxInpainting:
    """Flux Fill inpainting wrapper around HuggingFace diffusers FluxFillPipeline."""

    def __init__(self, config: FluxInpaintingConfig) -> None:
        self.config: FluxInpaintingConfig = config
        self.pipe: FluxFillPipeline = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            torch_dtype=torch.bfloat16,
        )
        if config.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to("cuda")

    @torch.inference_mode()
    def __call__(
        self,
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
        mask: UInt8[np.ndarray, "h w"],
    ) -> Image.Image:
        """Run Flux Fill inpainting.

        Args:
            rgb_hw3: Input RGB image. Masked regions should contain the border/background.
            mask: Inpainting mask where 255 = fill (inpaint), 0 = keep.

        Returns:
            Inpainted PIL Image.
        """
        image: Image.Image = Image.fromarray(rgb_hw3)
        mask_image: Image.Image = Image.fromarray(mask)
        result = self.pipe(
            prompt="",
            image=image,
            mask_image=mask_image,
            height=rgb_hw3.shape[0],
            width=rgb_hw3.shape[1],
            num_inference_steps=self.config.num_steps,
            guidance_scale=self.config.guidance,
            generator=torch.Generator("cpu").manual_seed(self.config.seed),
        )
        return result.images[0]
