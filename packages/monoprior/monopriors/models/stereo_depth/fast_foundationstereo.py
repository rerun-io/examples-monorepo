"""Fast-FoundationStereo as a ``BaseStereoPredictor``."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from jaxtyping import Float32, UInt8
from omegaconf import DictConfig, OmegaConf
from torch import nn

from monopriors.models.stereo_depth.base_stereo_depth import (
    BaseStereoPredictor,
    BaseStereoPredictorConfig,
    StereoDepthPrediction,
    disparity_to_metric_depth,
)
from monopriors.third_party.fast_foundationstereo.foundation_stereo import FastFoundationStereo
from monopriors.third_party.fast_foundationstereo.utils import InputPadder

FAST_FOUNDATIONSTEREO_HF_REPO: str = "nvidia/c-fast-foundationstereo"
FAST_FOUNDATIONSTEREO_HF_REVISION: str = "9b446878c81ddb27593036767b29b2859d46103e"
FAST_FOUNDATIONSTEREO_CHECKPOINT: str = "model_best_bp2_serialize.pth"
FAST_FOUNDATIONSTEREO_CONFIG: str = "cfg.yaml"


class _FastFoundationStereoUnpickler(pickle.Unpickler):
    """Remap upstream ``core.*`` pickle globals onto the vendored package."""

    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("core."):
            module = f"monopriors.third_party.fast_foundationstereo.{module.removeprefix('core.')}"
        elif module.startswith("foundation_stereo_ori."):
            module = f"monopriors.third_party.fast_foundationstereo.{module.removeprefix('foundation_stereo_ori.')}"
        return super().find_class(module, name)


class _FastFoundationStereoPickleModule:
    """Module-like adapter accepted by ``torch.load(pickle_module=...)``."""

    Unpickler = _FastFoundationStereoUnpickler
    load = staticmethod(pickle.load)


def download_fast_foundationstereo_checkpoint() -> Path:
    """Download the released config and pickled module from the pinned Hugging Face revision.

    Returns:
        Local path to ``model_best_bp2_serialize.pth``; ``cfg.yaml`` is downloaded beside it.
    """
    hf_hub_download(repo_id=FAST_FOUNDATIONSTEREO_HF_REPO, filename=FAST_FOUNDATIONSTEREO_CONFIG, revision=FAST_FOUNDATIONSTEREO_HF_REVISION)
    return Path(hf_hub_download(repo_id=FAST_FOUNDATIONSTEREO_HF_REPO, filename=FAST_FOUNDATIONSTEREO_CHECKPOINT, revision=FAST_FOUNDATIONSTEREO_HF_REVISION))


def load_fast_foundationstereo(checkpoint: Path, valid_iters: int = 8, max_disp: int = 416) -> nn.Module:
    """Remap the released NAS architecture, clone it, and strictly load its state dict.

    Args:
        checkpoint: Pickled upstream ``FastFoundationStereo`` module with ``cfg.yaml`` beside it.
        valid_iters: Number of recurrent disparity updates used at inference.
        max_disp: Maximum modeled disparity in pixels.

    Returns:
        A fresh evaluation-ready architecture on CPU with the released state loaded strictly.

    Raises:
        TypeError: If the checkpoint is not a module or the config is not a mapping.
        FileNotFoundError: If ``cfg.yaml`` is not beside the checkpoint.
        RuntimeError: If the released state does not strictly match its remapped NAS architecture.
    """
    serialized: object = torch.load(checkpoint, map_location="cpu", pickle_module=_FastFoundationStereoPickleModule, weights_only=False)
    if not isinstance(serialized, nn.Module):
        raise TypeError(f"Expected a pickled nn.Module in {checkpoint}, got {type(serialized).__name__}")
    serialized_model: FastFoundationStereo = cast(FastFoundationStereo, serialized)
    serialized_args: Any = serialized_model.args

    config_path: Path = checkpoint.with_name(FAST_FOUNDATIONSTEREO_CONFIG)
    if not config_path.is_file():
        raise FileNotFoundError(f"Fast-FoundationStereo config must be beside the checkpoint: {config_path}")
    loaded_config: Any = OmegaConf.load(config_path)
    if not isinstance(loaded_config, DictConfig):
        raise TypeError(f"Expected a mapping config in {config_path}, got {type(loaded_config).__name__}")
    for key, value in loaded_config.items():
        serialized_args[key] = value
    serialized_args.valid_iters = valid_iters
    serialized_args.max_disp = max_disp
    serialized_args.normalize = True

    model: nn.Module = copy.deepcopy(serialized_model)
    model.load_state_dict(serialized_model.state_dict(), strict=True)
    return model


@dataclass
class FastFoundationStereoConfig(BaseStereoPredictorConfig):
    """Configuration for Fast-FoundationStereo."""

    valid_iters: int = 8
    """Number of recurrent disparity updates used at inference."""
    max_disp: int = 416
    """Maximum modeled disparity in pixels."""
    checkpoint: Path | None = None
    """Local released checkpoint, or None to download the pinned release."""

    def setup(self, device: Literal["cpu", "cuda"]) -> FastFoundationStereoPredictor:
        """Build the configured Fast-FoundationStereo predictor on one device."""
        return FastFoundationStereoPredictor(device=device, checkpoint=self.checkpoint, valid_iters=self.valid_iters, max_disp=self.max_disp)


class FastFoundationStereoPredictor(BaseStereoPredictor[nn.Module]):
    """Fast-FoundationStereo predictor using the upstream PyTorch cost-volume fallback."""

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        checkpoint: Path | None = None,
        valid_iters: int = 8,
        max_disp: int = 416,
    ) -> None:
        """Load Fast-FoundationStereo on one device.

        Args:
            device: Where the network runs.
            checkpoint: Local released checkpoint; downloaded from the pinned HF revision when None.
            valid_iters: Number of recurrent disparity updates.
            max_disp: Maximum modeled disparity in pixels.
        """
        self.valid_iters: int = valid_iters
        self.max_disp: int = max_disp
        checkpoint_path: Path = checkpoint or download_fast_foundationstereo_checkpoint()
        self.model: nn.Module = load_fast_foundationstereo(checkpoint_path, valid_iters=valid_iters, max_disp=max_disp).to(device).eval()

    def infer_disparity(self, left_rgb: UInt8[np.ndarray, "h w 3"], right_rgb: UInt8[np.ndarray, "h w 3"]) -> Float32[np.ndarray, "h w"]:
        """Predict left-view disparity in pixels at the input resolution.

        Args:
            left_rgb: Rectified left RGB image, ``UInt8[ndarray, "h w 3"]``.
            right_rgb: Rectified right RGB image, ``UInt8[ndarray, "h w 3"]``.

        Returns:
            Non-negative disparity, ``Float32[ndarray, "h w"]``.
        """
        device: torch.device = next(self.model.parameters()).device
        left_13hw: Float32[torch.Tensor, "1 3 h w"] = rearrange(torch.from_numpy(left_rgb).to(device=device, dtype=torch.float32), "h w c -> 1 c h w")
        right_13hw: Float32[torch.Tensor, "1 3 h w"] = rearrange(torch.from_numpy(right_rgb).to(device=device, dtype=torch.float32), "h w c -> 1 c h w")
        padder: InputPadder = InputPadder(left_13hw.shape, divis_by=32, force_square=False)
        padded: list[Float32[torch.Tensor, "1 3 hp wp"]] = padder.pad(left_13hw, right_13hw)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            disparity_11hw: Float32[torch.Tensor, "1 1 hp wp"] = self.model(
                padded[0],
                padded[1],
                iters=self.valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )
        disparity_hw: Float32[torch.Tensor, "h w"] = rearrange(padder.unpad(disparity_11hw.float()).clamp_min(0.0), "1 1 h w -> h w")
        return disparity_hw.cpu().numpy()

    def __call__(
        self,
        left_rgb: UInt8[np.ndarray, "h w 3"],
        right_rgb: UInt8[np.ndarray, "h w 3"],
        K_33: Float32[np.ndarray, "3 3"] | None = None,
        baseline_m: float | None = None,
    ) -> StereoDepthPrediction:
        """Predict disparity and optional metric depth for a rectified RGB pair.

        Args:
            left_rgb: Rectified left RGB image, ``UInt8[ndarray, "h w 3"]``.
            right_rgb: Rectified right RGB image, ``UInt8[ndarray, "h w 3"]``.
            K_33: Shared rectified intrinsics, ``Float32[ndarray, "3 3"]``, when metric depth is wanted.
            baseline_m: Stereo baseline in metres when metric depth is wanted.

        Returns:
            Left-view disparity and optional metric depth in a ``StereoDepthPrediction``.
        """
        disparity_hw: Float32[np.ndarray, "h w"] = self.infer_disparity(left_rgb, right_rgb)
        depth_hw: Float32[np.ndarray, "h w"] | None = None
        if K_33 is not None and baseline_m is not None:
            depth_hw = disparity_to_metric_depth(disparity_hw, float(K_33[0, 0]), baseline_m)
        return StereoDepthPrediction(disparity=disparity_hw, depth_meters=depth_hw, K_33=K_33, baseline_m=baseline_m)
