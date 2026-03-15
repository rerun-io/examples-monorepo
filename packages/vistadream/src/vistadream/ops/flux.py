from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from jaxtyping import BFloat16, UInt8
from PIL import Image

from vistadream.flux.model import Flux
from vistadream.flux.modules.autoencoder import AutoEncoder
from vistadream.flux.sampling import denoise, get_noise, get_schedule, prepare_fill_empty_prompt, unpack
from vistadream.flux.util import load_ae, load_flow_model


@dataclass
class FluxInpaintingConfig:
    offload: bool = True
    num_steps: int = 25
    guidance: int | float = 30.0
    seed: int = 42


class FluxInpainting:
    def __init__(self, config: FluxInpaintingConfig) -> None:
        self.config: FluxInpaintingConfig = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device)
        self._load_model()

    def _load_model(self):
        self.model: Flux = load_flow_model("flux-dev-fill", device="cpu" if self.config.offload else self.torch_device)
        self.ae: AutoEncoder = load_ae("flux-dev-fill", device="cpu" if self.config.offload else self.torch_device)

    @torch.inference_mode
    def __call__(
        self,
        rgb_hw3: UInt8[np.ndarray, "h w 3"],
        mask: UInt8[np.ndarray, "h w"],
    ) -> Image.Image:
        height: int = rgb_hw3.shape[0]
        width: int = rgb_hw3.shape[1]
        x: BFloat16[torch.Tensor, "batch channels latent_height latent_width"] = get_noise(
            num_samples=1,
            height=height,
            width=width,
            device=self.torch_device,
            dtype=torch.bfloat16,
            seed=self.config.seed,
        )

        if self.config.offload:
            self.ae = self.ae.to(self.torch_device)

        inp: dict[str, torch.Tensor] = prepare_fill_empty_prompt(
            x,
            prompt="",
            ae=self.ae,
            img_cond=rgb_hw3,
            mask=mask,
        )

        timesteps: list[float] = get_schedule(self.config.num_steps, inp["img"].shape[1], shift=True)

        if self.config.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.torch_device)

        x = denoise(self.model, **inp, timesteps=timesteps, guidance=self.config.guidance)

        if self.config.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        torch.cuda.empty_cache()
        # Process and display result
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")
        inpainted_image: Image.Image = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return inpainted_image
