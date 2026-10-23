"""Streaming MammaNet: one batch over every tracked (view, person) crop of a tick.

Runs the static-batch TensorRT engine (``.trt_cache``, CUDA graph; 3.9 ms vs 15.8 ms eager per 4-crop call) when the batch
fits it, else eager fp16. The RGB crops may sample hi-res frames (``rgb_scale`` = their size over the engine size) while
masks, un-projection and outputs stay in engine pixels: sharper MammaNet input without changing anything downstream.
posekit's ``MammaNetLandmarks`` role (dynamic engine, one resolution) serves posekit's own pipelines.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from mamma.models.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.models.landmarks.crops import box_geometry, gpu_crop_batch, gpu_mask_crop_batch, unproject_joints2d
from mamma.models.landmarks.mammanet import MammaNet, load_mammanet
from mamma.models.landmarks.tensorrt_backend import ENGINE_BATCH, INPUT_NAMES, OUTPUT_NAMES


@dataclass(frozen=True, slots=True)
class DenseLandmarks:
    """One row per input crop, engine pixels; owned tensors."""

    xy: Float32[Tensor, "k n 2"]
    log_variance: Float32[Tensor, "k n"]
    visibility: Float32[Tensor, "k n"]
    floor_contact: Float32[Tensor, "k n"]


class LandmarkEstimator:
    def __init__(self, weights_path: Path | None = None, device: str = "cuda", engine_path: Path | None = None, config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG) -> None:
        self.config, self.device = config, device
        self.model: MammaNet = load_mammanet(weights_path, device=device, config=config)
        self.runner = None
        if engine_path is not None:
            from trtkit.tensorrt_runtime import TensorRtRuntime

            self.runner = TensorRtRuntime(engine_path, use_cuda_graph=True)

    def __call__(
        self, frames: UInt8[Tensor, "v 3 hh ww"], view: Int64[ndarray, " k"], masks: Bool[Tensor, "k h w"], boxes_xyxy: Float32[ndarray, "k 4"], rgb_scale: float = 1.0
    ) -> DenseLandmarks:
        """Landmarks of each crop ``boxes_xyxy[i]`` (engine pixels, finite) of view ``view[i]``, conditioned on ``masks[i]``."""
        geometry: list[tuple[Float64[ndarray, "2"], Float64[ndarray, "2"]]] = [box_geometry(box, self.config) for box in boxes_xyxy]
        centers: Float64[ndarray, "k 2"] = np.asarray([g[0] for g in geometry], dtype=np.float64).reshape(-1, 2)
        sizes: Float64[ndarray, "k 2"] = np.asarray([g[1] for g in geometry], dtype=np.float64).reshape(-1, 2)
        rgb: Float32[Tensor, "k 3 hh ww"] = frames[torch.as_tensor(view, device=frames.device)].float()
        img_crops, _ = gpu_crop_batch(rgb, centers * rgb_scale, sizes * rgb_scale, None, self.config)
        mask_crops = gpu_mask_crop_batch(masks.unsqueeze(1).float() * 255.0, centers, sizes, self.config)
        if self.runner is not None and img_crops.shape[0] <= ENGINE_BATCH:
            out: dict[str, Tensor | None] = dict(self.runner({INPUT_NAMES[0]: img_crops, INPUT_NAMES[1]: mask_crops}))
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled="cuda" in self.device):
                out = self.model(img_crops, mask_crops)
        heads: list[Tensor] = [head for name in OUTPUT_NAMES if (head := out[name]) is not None]
        assert len(heads) == len(OUTPUT_NAMES), "MammaNet returned every head"
        joints: Float32[Tensor, "k n 3"] = unproject_joints2d(heads[0].float(), centers, sizes, self.config)  # a new tensor: the engine's buffers are reused
        visibility, _, floor_contact = (torch.sigmoid(head.squeeze(-1).float()) for head in heads[1:])
        return DenseLandmarks(joints[..., :2].contiguous(), joints[..., 2].contiguous(), visibility, floor_contact)
