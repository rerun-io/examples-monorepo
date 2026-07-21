"""MammaNet behind posekit's ``TopDownDenseLandmarks2d`` role.

The adapter that lets mamma's dense-landmark net slot into any posekit
pipeline (posekit docs/design.md §6): detections with masks in — the tracker's
``BoxDetections`` carries both — dense landmarks with log-variance /
visibility / contact heads out, all GPU tensors. Crop math, normalization, and
precision match ``LandmarkEstimator.estimate`` exactly (same mamma ops).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Float32, Float64, UInt8
from numpy import ndarray
from posekit.models import TopDownDenseLandmarks2d
from posekit.predictions import BoxDetections, DenseLandmarks2d, validate_frames_rgb
from torch import Tensor

from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.landmarks.crops import box_geometry, gpu_crop_batch, gpu_mask_crop_batch, unproject_joints2d
from mamma.landmarks.mammanet import MammaNet, load_mammanet

MAMMANET_WEIGHTS_REPO: str = "pablovela5620/mamma-streaming-data"
MAMMANET_WEIGHTS_FILE: str = "weights/ma_2d/mamma_mask_full_cvpr.safetensors"


@dataclass(frozen=True, slots=True)
class MammaNetLandmarksConfig:
    """MammaNet dense-landmark estimator configuration (torch backend)."""

    weights_path: Path | None = None
    """Local safetensors checkpoint; ``None`` downloads mamma's HF weights."""
    device: str = "cuda"
    """Inference device."""

    def setup(self) -> "MammaNetLandmarks":
        """Load weights and return a ready estimator."""
        return MammaNetLandmarks(self)


class MammaNetLandmarks(TopDownDenseLandmarks2d):
    """Batched GPU MammaNet dense-landmark estimator (mask-conditioned)."""

    def __init__(self, config: MammaNetLandmarksConfig) -> None:
        """Load the checkpoint.

        Args:
            config: Weights source and device.
        """
        from huggingface_hub import hf_hub_download

        self.config: MammaNetLandmarksConfig = config
        weights_path: Path = (
            config.weights_path
            if config.weights_path is not None
            else Path(hf_hub_download(repo_id=MAMMANET_WEIGHTS_REPO, repo_type="dataset", filename=MAMMANET_WEIGHTS_FILE))
        )
        self.mamma_config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG
        self.model: MammaNet = load_mammanet(weights_path, device=config.device, config=self.mamma_config)
        self.num_landmarks = int(self.mamma_config.num_landmarks)

    @torch.no_grad()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> DenseLandmarks2d:
        """Estimate dense landmarks for every detection.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index;
                ``masks`` must be populated (MammaNet is mask-conditioned).

        Returns:
            Dense landmarks, one instance per detection row.

        Raises:
            ValueError: If ``detections.masks`` is ``None``.
        """
        if detections.masks is None:
            raise ValueError("MammaNetLandmarks requires detections.masks (mask-conditioned crops).")
        validate_frames_rgb(frames_rgb)
        device: torch.device = frames_rgb.device
        num_instances: int = detections.num_detections
        if num_instances == 0:
            empty_xy: Float32[Tensor, "0 p 2"] = torch.empty((0, self.num_landmarks, 2), dtype=torch.float32, device=device)
            empty_p: Float32[Tensor, "0 p"] = torch.empty((0, self.num_landmarks), dtype=torch.float32, device=device)
            return DenseLandmarks2d(empty_xy, empty_p, empty_p, empty_p, empty_p, detections.frame_indices)

        boxes_np: Float32[ndarray, "n 4"] = detections.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        geometry: list[tuple[Float64[ndarray, "2"], Float64[ndarray, "2"]]] = [
            box_geometry(boxes_np[row], self.mamma_config) for row in range(num_instances)
        ]
        centers: Float64[ndarray, "n 2"] = np.stack([center for center, _ in geometry], axis=0)
        sizes: Float64[ndarray, "n 2"] = np.stack([size for _, size in geometry], axis=0)
        # Gather per-detection frames while still uint8, then convert only the
        # gathered subset to float (4x lighter than float-converting all views).
        gathered: Float[Tensor, "n 3 h w"] = frames_rgb.permute(0, 3, 1, 2)[detections.frame_indices].float()
        masks_batch: Float[Tensor, "n 1 h w"] = detections.masks.unsqueeze(1).float() * 255.0
        img_crops, _ = gpu_crop_batch(gathered, centers, sizes, None, self.mamma_config)
        mask_crops = gpu_mask_crop_batch(masks_batch, centers, sizes, self.mamma_config)
        with torch.autocast("cuda", dtype=torch.float16, enabled="cuda" in self.config.device):
            out: dict[str, Tensor | None] = self.model(img_crops, mask_crops)
        joints2d_raw = out["joints2d"]
        visibility_raw = out["visibility"]
        contact_raw = out["contact"]
        floor_raw = out["floor_contact"]
        assert joints2d_raw is not None and visibility_raw is not None and contact_raw is not None and floor_raw is not None
        joints2d_px: Float32[Tensor, "n p 3"] = unproject_joints2d(joints2d_raw.float(), centers, sizes, self.mamma_config)
        return DenseLandmarks2d(
            xy=joints2d_px[:, :, 0:2].contiguous(),
            log_variance=joints2d_px[:, :, 2].contiguous(),
            visibility=torch.sigmoid(visibility_raw.squeeze(-1).float()),
            contact=torch.sigmoid(contact_raw.squeeze(-1).float()),
            floor_contact=torch.sigmoid(floor_raw.squeeze(-1).float()),
            frame_indices=detections.frame_indices,
        )


__all__ = ("MammaNetLandmarks", "MammaNetLandmarksConfig")
