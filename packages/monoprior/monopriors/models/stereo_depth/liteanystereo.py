"""LiteAnyStereo V2 (LAS2 S/M/L/H) as a ``BaseStereoPredictor``.

Preprocessing mirrors upstream ``demo.py``: float RGB in ``[0, 255]`` (no normalisation), padded to a
multiple of 32 with the upstream ``InputPadder``, ``model(left, right, max_disp, test_mode=True)``.
"""

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float32, UInt8
from torch import nn

from monopriors.third_party.liteanystereo.liteanystereov2 import build_liteanystereo
from monopriors.third_party.liteanystereo.padding import InputPadder

from .base_stereo_depth import BaseStereoPredictor, StereoDepthPrediction, disparity_to_metric_depth

LAS2ModelSize: TypeAlias = Literal["s", "m", "l", "h"]

LITEANYSTEREO_HF_REPO = "pablovela5620/liteanystereo"
"""Unmodified mirror of ``tomtomtommi/LiteAnyStereoV2`` (LAS2_{S,M,L,H}.pth)."""
LITEANYSTEREO_HF_REVISION = "19870ad07fe89fe6e5a7c11c69348e13936adbe7"


def download_liteanystereo_checkpoint(model_size: LAS2ModelSize) -> Path:
    """Released LAS2 weights for one model size, from the pinned HF mirror."""
    return Path(hf_hub_download(repo_id=LITEANYSTEREO_HF_REPO, filename=f"LAS2_{model_size.upper()}.pth", revision=LITEANYSTEREO_HF_REVISION))


def load_liteanystereo(model_size: LAS2ModelSize, checkpoint: Path, max_disp: int = 192) -> nn.Module:
    """Build a LAS2 network and strictly load a released checkpoint (bare state dict or ``{"model"|"state_dict": ...}``)."""
    model: nn.Module = build_liteanystereo(model_size=model_size, fnet_pretrained=False, max_disp=max_disp)
    state: dict = torch.load(checkpoint, map_location="cpu")
    for key in ("model", "state_dict"):
        if key in state:
            state = state[key]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


class LiteAnyStereoPredictor(BaseStereoPredictor[nn.Module]):
    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_size: LAS2ModelSize = "m",
        checkpoint: Path | None = None,
        max_disp: int = 192,
    ) -> None:
        """
        Args:
            device: Where the network runs.
            model_size: LAS2 variant; ``s``/``m``/``l`` are feed-forward, ``h`` is the iterative ConvGRU model.
            checkpoint: Local weights; downloaded from the pinned HF mirror when None.
            max_disp: Disparity search range in pixels. Upstream S/M/L are built for 192 regardless; ``h`` is built for this value.
        """
        self.max_disp: int = max_disp
        self.model: nn.Module = load_liteanystereo(model_size, checkpoint or download_liteanystereo_checkpoint(model_size), max_disp).to(device).eval()

    def infer_disparity(self, left_rgb: UInt8[np.ndarray, "h w 3"], right_rgb: UInt8[np.ndarray, "h w 3"]) -> Float32[np.ndarray, "h w"]:
        """Left-view disparity in pixels at input resolution."""
        device: torch.device = next(self.model.parameters()).device
        left_13hw: Float32[torch.Tensor, "1 3 h w"] = rearrange(torch.from_numpy(left_rgb).to(device).float(), "h w c -> 1 c h w")
        right_13hw: Float32[torch.Tensor, "1 3 h w"] = rearrange(torch.from_numpy(right_rgb).to(device).float(), "h w c -> 1 c h w")
        padder: InputPadder = InputPadder(left_13hw.shape, divis_by=32)
        padded: list[Float32[torch.Tensor, "1 3 hp wp"]] = padder.pad(left_13hw, right_13hw)
        with torch.no_grad():
            disparity_11hw: Float32[torch.Tensor, "1 1 hp wp"] = self.model(padded[0], padded[1], max_disp=self.max_disp, test_mode=True)
        return rearrange(padder.unpad(disparity_11hw.float()), "1 1 h w -> h w").cpu().numpy()

    def __call__(
        self,
        left_rgb: UInt8[np.ndarray, "h w 3"],
        right_rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float32[np.ndarray, "3 3"] | None = None,
        baseline_m: float | None = None,
    ) -> StereoDepthPrediction:
        disparity_hw: Float32[np.ndarray, "h w"] = self.infer_disparity(left_rgb, right_rgb)
        depth_hw: Float32[np.ndarray, "h w"] | None = None
        if K_33 is not None and baseline_m is not None:
            depth_hw = disparity_to_metric_depth(disparity_hw, float(K_33[0, 0]), baseline_m)
        return StereoDepthPrediction(disparity=disparity_hw, depth_meters=depth_hw, K_33=K_33, baseline_m=baseline_m)
