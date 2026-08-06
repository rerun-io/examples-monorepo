"""Per-tick dense landmark estimation over all cameras x tracked persons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, Float64
from numpy import ndarray

from mamma.engine.types import CameraTracks
from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG, MammaNetConfig
from mamma.landmarks.crops import box_geometry, gpu_crop_batch, gpu_mask_crop_batch, unproject_joints2d
from mamma.landmarks.mammanet import MammaNet, load_mammanet


@dataclass(slots=True)
class LandmarkResult:
    """Dense 2D landmarks for one person in one camera at one tick."""

    obj_id: int
    """Person id, consistent with the tracker."""
    joints2d: Float32[torch.Tensor, "j 3"]
    """``[x_px, y_px, log_variance]`` in engine-resolution pixel coords."""
    visibility: Float32[torch.Tensor, "j"]
    """Per-landmark visibility probability (sigmoid applied)."""
    contact: Float32[torch.Tensor, "j"]
    """Per-landmark self-contact probability."""
    floor_contact: Float32[torch.Tensor, "j"]
    """Per-landmark floor-contact probability."""


CameraLandmarks = dict[int, LandmarkResult]
"""Per-person landmark results in one camera, keyed by ``obj_id``."""


class LandmarkEstimator:
    """Batches MammaNet over every (camera, person) crop of a tick."""

    def __init__(
        self,
        weights_path: Path,
        device: str = "cuda",
        config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG,
        compile_model: bool = False,
        engine_path: Path | None = None,
    ) -> None:
        self.config: MammaNetConfig = config
        self.device: str = device
        self.model: MammaNet = load_mammanet(weights_path, device=device, config=config)
        self.runner = None
        if engine_path is not None:
            from trtkit.tensorrt_runtime import TensorRtRuntime

            # FP16 TRT engine: 3.88ms vs 15.8ms eager per 4-crop call; joints
            # p99 diff 0.008 normalized units vs eager fp16 (gate re-verified).
            self.runner = TensorRtRuntime(engine_path, use_cuda_graph=True)
        if compile_model:
            import torch._dynamo

            # 15.8 -> 10.2 ms per 4-crop forward standalone, but inductor's
            # cudagraphs fight the fitter's manual CUDA graph in-pipeline
            # (landmarks 17.6 -> 35 ms/tick observed) — off by default until
            # the two share a pool or MammaNet moves to TRT.
            torch._dynamo.config.cache_size_limit = 16
            from typing import Any, cast

            self.model = cast(Any, torch.compile(self.model, mode="reduce-overhead"))

    def estimate(
        self,
        frames: list[Float32[torch.Tensor, "3 h w"]] | list[torch.Tensor],
        tracks: list[CameraTracks],
        frames_hires: list[torch.Tensor] | None = None,
        hires_scale: float = 1.0,
    ) -> list[CameraLandmarks]:
        """One synchronized tick: frames (uint8 or float RGB CHW) + tracks -> landmarks.

        With ``frames_hires`` the RGB crops are sampled from those frames
        (geometry scaled by ``hires_scale``) while masks, un-projection, and
        all outputs stay in the engine (``frames``) pixel space — sharper
        MammaNet inputs without touching anything downstream.
        """
        # Gather every (cam, person) entry with a usable box into one batch.
        entries: list[tuple[int, int]] = []
        centers_list: list[Float64[ndarray, "2"]] = []
        sizes_list: list[Float64[ndarray, "2"]] = []
        crop_frames: list[torch.Tensor] = []
        crop_masks: list[torch.Tensor] = []
        rgb_source: list[Float32[torch.Tensor, "3 h w"]] | list[torch.Tensor] = frames_hires if frames_hires is not None else frames
        for cam_idx, cam_tracks in enumerate(tracks):
            frame_f32: torch.Tensor = rgb_source[cam_idx].float()
            for obj_id, track in sorted(cam_tracks.items()):
                if track.bbox_xyxy is None:
                    continue
                center, bbox_size = box_geometry(track.bbox_xyxy, self.config)
                entries.append((cam_idx, obj_id))
                centers_list.append(center)
                sizes_list.append(bbox_size)
                crop_frames.append(frame_f32)
                crop_masks.append(track.mask.unsqueeze(0).float() * 255.0)

        results: list[CameraLandmarks] = [{} for _ in tracks]
        if not entries:
            return results

        frames_batch: Float32[torch.Tensor, "n 3 h w"] = torch.stack(crop_frames, dim=0)
        masks_batch: Float32[torch.Tensor, "n 1 h2 w2"] = torch.stack(crop_masks, dim=0)
        centers: Float64[ndarray, "n 2"] = np.stack(centers_list, axis=0)
        sizes: Float64[ndarray, "n 2"] = np.stack(sizes_list, axis=0)

        # RGB sampled from the (possibly hi-res) source; masks sampled from the
        # engine-res tracker output. Same normalized region either way.
        s: float = hires_scale if frames_hires is not None else 1.0
        img_crops, _ = gpu_crop_batch(frames_batch, centers * s, sizes * s, None, self.config)
        mask_crops = gpu_mask_crop_batch(masks_batch, centers, sizes, self.config)
        if self.runner is not None and img_crops.shape[0] <= 4:
            out: dict[str, torch.Tensor | None] = dict(self.runner({"crops": img_crops, "masks": mask_crops}))
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled="cuda" in self.device):
                out = self.model(img_crops, mask_crops)

        joints2d_raw = out["joints2d"]
        visibility_raw = out["visibility"]
        contact_raw = out["contact"]
        floor_raw = out["floor_contact"]
        assert joints2d_raw is not None and visibility_raw is not None and contact_raw is not None and floor_raw is not None
        joints2d_px: Float32[torch.Tensor, "n j 3"] = unproject_joints2d(joints2d_raw.float(), centers, sizes, self.config)
        visibility: Float32[torch.Tensor, "n j"] = torch.sigmoid(visibility_raw.squeeze(-1).float())
        contact: Float32[torch.Tensor, "n j"] = torch.sigmoid(contact_raw.squeeze(-1).float())
        floor_contact: Float32[torch.Tensor, "n j"] = torch.sigmoid(floor_raw.squeeze(-1).float())

        for row, (cam_idx, obj_id) in enumerate(entries):
            results[cam_idx][obj_id] = LandmarkResult(
                obj_id=obj_id,
                joints2d=joints2d_px[row],
                visibility=visibility[row],
                contact=contact[row],
                floor_contact=floor_contact[row],
            )
        return results
