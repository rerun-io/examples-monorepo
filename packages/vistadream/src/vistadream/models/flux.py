"""FLUX.1 Fill (diffusers) as an inpainting role: fill the masked pixels of an RGB image."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import torch
from jaxtyping import UInt8
from numpy import ndarray
from PIL import Image
from serde import serde

Inpaint: TypeAlias = Callable[[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w"]], UInt8[ndarray, "h w 3"]]
"""``(rgb, mask) -> rgb``: fill where ``mask`` is 255, keep where it is 0. Nodes take any such callable (fakes in tests)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FluxConfig:
    """FLUX.1-Fill-dev settings."""

    offload: bool = True
    """Offload idle sub-models to the CPU (less VRAM, slower)."""
    num_steps: int = 25
    """Denoising steps."""
    guidance: float = 30.0
    """Guidance scale."""
    seed: int = 42
    """Seed of the diffusion noise (every call starts from it)."""
    local_files_only: bool = False
    """Load only from the Hugging Face cache: a missing file is an error, never a download."""


class FluxInpainting:
    """Loads ``black-forest-labs/FLUX.1-Fill-dev`` (bfloat16; about 34 GB the first time) and fills with an empty prompt."""

    def __init__(self, config: FluxConfig) -> None:
        from diffusers import FluxFillPipeline

        self.config: FluxConfig = config
        # Any: diffusers' lazy-import dummies make every FluxFillPipeline method unknown to the type checker
        self.pipe: Any = cast(Any, FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, local_files_only=config.local_files_only))
        if config.offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to("cuda")

    @torch.inference_mode()
    def __call__(self, rgb: UInt8[ndarray, "h w 3"], mask: UInt8[ndarray, "h w"]) -> UInt8[ndarray, "h w 3"]:
        result = self.pipe(
            prompt="",
            image=Image.fromarray(rgb),
            mask_image=Image.fromarray(mask),
            height=rgb.shape[0],
            width=rgb.shape[1],
            num_inference_steps=self.config.num_steps,
            guidance_scale=self.config.guidance,
            generator=torch.Generator("cpu").manual_seed(self.config.seed),
        )
        return np.asarray(result.images[0].convert("RGB"))
