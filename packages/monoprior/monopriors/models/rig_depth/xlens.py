"""X-Lens predictor adapter for calibrated multi-camera rigs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray

from monopriors.models.rig_depth.base_rig_depth import BaseRigDepthPredictor, BaseRigDepthPredictorConfig, RigDepthPrediction
from monopriors.third_party.xlens.inference.pipeline import XLensInference, XLensOutput
from monopriors.third_party.xlens.inference.preprocess import AssembledBatch, assemble_batch

XLENS_HF_REPO: str = "henryzhou998/X-Lens"
XLENS_HF_REVISION: str = "1d0c96353b69464addad12389fadbb816e3978ae"
XLENS_CHECKPOINT: str = "model.safetensors"


def download_xlens_checkpoint() -> Path:
    """Download the gated non-commercial release from its pinned revision.

    Returns:
        Local path to ``model.safetensors``.

    Raises:
        RuntimeError: If the user's Hugging Face login lacks access to the gated
            ``henryzhou998/X-Lens`` repository.
    """
    try:
        return Path(hf_hub_download(repo_id=XLENS_HF_REPO, filename=XLENS_CHECKPOINT, revision=XLENS_HF_REVISION))
    except HfHubHTTPError as error:
        status_code: int | None = error.response.status_code if error.response is not None else None
        if status_code in (401, 403):
            raise RuntimeError(
                f"X-Lens weights are gated at {XLENS_HF_REPO}; log in with `hf auth login`, accept the repository terms, and retry"
            ) from error
        raise


@dataclass
class XLensConfig(BaseRigDepthPredictorConfig):
    """Configuration for the released X-Lens ViT-S model."""

    checkpoint: Path | None = None
    """Local safetensors state dict, or None to download the pinned gated release."""
    amp: Literal["bf16", "fp16"] = "bf16"
    """CUDA autocast dtype; BF16 is the upstream default."""

    def setup(self, device: Literal["cpu", "cuda"]) -> "XLensPredictor":
        """Build the configured X-Lens predictor on one device."""
        return XLensPredictor(device=device, checkpoint=self.checkpoint, amp=self.amp)


class XLensPredictor(BaseRigDepthPredictor):
    """Released X-Lens model with upstream preprocessing and output handling."""

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        checkpoint: Path | None = None,
        amp: Literal["bf16", "fp16"] = "bf16",
    ) -> None:
        """Load X-Lens and its frozen ViT-S architecture.

        Args:
            device: Where model inference runs.
            checkpoint: Local released state dict; downloaded with the user's
                Hugging Face login when None.
            amp: CUDA autocast dtype.
        """
        checkpoint_path: Path = checkpoint or download_xlens_checkpoint()
        config_path: Path = Path(__file__).parents[2] / "third_party" / "xlens" / "xlens_vits.yaml"
        self.pipeline: XLensInference = XLensInference(
            checkpoint_path=str(checkpoint_path),
            device=device,
            amp_dtype=amp,
            config=str(config_path),
        )
        self.model = self.pipeline.model

    def __call__(
        self,
        images: UInt8[ndarray, "s h w 3"],
        rays: Float32[ndarray, "s h w 3"],
        cam_types: Int64[ndarray, "s"],
        cam_T_ref: Float64[ndarray, "s 4 4"] | None,
    ) -> RigDepthPrediction:
        """Predict per-view camera-frame metric z-depth.

        Args:
            images: Shared-resolution RGB views, ``UInt8[ndarray, "s h w 3"]``.
            rays: Camera-frame unit rays, ``Float32[ndarray, "s h w 3"]``.
            cam_types: X-Lens camera ids, ``Int64[ndarray, "s"]``.
            cam_T_ref: Optional camera-to-reference poses,
                ``Float64[ndarray, "s 4 4"]``. X-Lens canonicalizes view 0.

        Returns:
            Metric depth, confidence, mask, and scale at input resolution.

        Raises:
            ValueError: If fewer than two views are supplied or the resolution
                is smaller than 28 pixels or not divisible by 14.
        """
        views: int = images.shape[0]
        height: int = images.shape[1]
        width: int = images.shape[2]
        if views < 2:
            raise ValueError(f"X-Lens needs at least two views, got {views}")
        if height < 28 or width < 28 or height % 14 != 0 or width % 14 != 0:
            raise ValueError(f"X-Lens height and width must be multiples of 14 and at least 28, got {height}x{width}")
        if rays.shape != (views, height, width, 3) or cam_types.shape != (views,):
            raise ValueError("X-Lens images, rays, and camera types must have matching view and image dimensions")

        device: torch.device = next(self.model.parameters()).device
        batch: AssembledBatch = assemble_batch(list(images), list(rays), cam_types.tolist(), c2w=cam_T_ref, device=device)
        output: XLensOutput = self.pipeline(batch)
        depth: Float32[torch.Tensor, "1 s h w"] = output["depth_metric"]
        confidence: Float32[torch.Tensor, "1 s h w"] = output["depth_conf"]
        mask: Float32[torch.Tensor, "1 s h w"] = output["mask"]
        scale: Float32[torch.Tensor, "1"] = output["metric_scaling_factor"]
        return RigDepthPrediction(depth_m=depth[0], confidence=confidence[0], mask=mask[0], scale=float(scale[0]))
