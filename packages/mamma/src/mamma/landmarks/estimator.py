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
from mamma.landmarks.crops import box_geometry, gpu_crop_frame, gpu_mask_crop_batch, to_device_f32, unproject_joints2d
from mamma.landmarks.mammanet import MammaNet, load_mammanet
from mamma.landmarks.tensorrt_backend import INPUT_NAMES, OUTPUT_NAMES


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
        weights_path: Path | None = None,
        device: str = "cuda",
        config: MammaNetConfig = DEFAULT_MAMMANET_CONFIG,
        compile_model: bool = False,
        engine_path: Path | None = None,
    ) -> None:
        self.config: MammaNetConfig = config
        self.device: str = device
        # Eager MammaNet only without an engine: the engine runs every crop count (in chunks past its batch).
        self.model: MammaNet | None = load_mammanet(weights_path, device=device, config=config) if engine_path is None else None
        self.runner = None
        if engine_path is not None:
            from trtkit.tensorrt_runtime import TensorRtRuntime

            # FP16 TRT engine: 3.88ms vs 15.8ms eager per 4-crop call; joints
            # p99 diff 0.008 normalized units vs eager fp16 (gate re-verified).
            self.runner = TensorRtRuntime(engine_path, use_cuda_graph=True)
        if compile_model and self.model is not None:
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
        crop_masks: list[torch.Tensor] = []
        for cam_idx, cam_tracks in enumerate(tracks):
            for obj_id, track in sorted(cam_tracks.items()):
                # No box, or a degenerate one (a one-pixel-wide mask sliver): nothing to crop.
                if track.bbox_xyxy is None or track.bbox_xyxy[2] <= track.bbox_xyxy[0] or track.bbox_xyxy[3] <= track.bbox_xyxy[1]:
                    continue
                center, bbox_size = box_geometry(track.bbox_xyxy, self.config)
                entries.append((cam_idx, obj_id))
                centers_list.append(center)
                sizes_list.append(bbox_size)
                crop_masks.append(track.mask.unsqueeze(0).float() * 255.0)

        results: list[CameraLandmarks] = [{} for _ in tracks]
        if not entries:
            return results

        masks_batch: Float32[torch.Tensor, "n 1 h2 w2"] = torch.stack(crop_masks, dim=0)
        centers: Float64[ndarray, "n 2"] = np.stack(centers_list, axis=0)
        sizes: Float64[ndarray, "n 2"] = np.stack(sizes_list, axis=0)

        # RGB sampled from the (possibly hi-res) source; masks sampled from the
        # engine-res tracker output. Same normalized region either way. Each
        # camera's crop window is converted to float once and expanded (a view,
        # no copy) over its people. Entries are camera-major, so the per-camera
        # crops concatenate in entry order.
        rgb_source: list[Float32[torch.Tensor, "3 h w"]] | list[torch.Tensor] = frames_hires if frames_hires is not None else frames
        s: float = hires_scale if frames_hires is not None else 1.0
        geom: Float32[torch.Tensor, "n 4"] = to_device_f32(np.concatenate([centers, sizes], axis=1), masks_batch.device)
        centers_t, sizes_t = geom[:, 0:2], geom[:, 2:4]
        # Entries are camera-major: each camera's rows are one contiguous slice.
        cam_crops: list[Float32[torch.Tensor, "k 3 ch cw"]] = []
        for cam_idx in sorted({cam for cam, _ in entries}):
            rows: list[int] = [row for row, (cam, _) in enumerate(entries) if cam == cam_idx]
            lo, hi = rows[0], rows[-1] + 1
            cam_crops.append(gpu_crop_frame(rgb_source[cam_idx], centers[lo:hi] * s, sizes[lo:hi] * s, self.config))
        img_crops: Float32[torch.Tensor, "n 3 ch cw"] = torch.cat(cam_crops, dim=0)
        mask_crops = gpu_mask_crop_batch(masks_batch, centers_t, sizes_t, self.config)
        out: dict[str, torch.Tensor | None]
        if self.runner is None:
            assert self.model is not None, "eager MammaNet is loaded whenever no engine is"
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled="cuda" in self.device):
                out = self.model(img_crops, mask_crops)
        else:
            from trtkit.base import run_chunked

            inputs: dict[str, torch.Tensor] = {INPUT_NAMES[0]: img_crops, INPUT_NAMES[1]: mask_crops}
            # Past the engine's batch, run_chunked runs every chunk and copies each out of the runner's buffers
            # (an eager batch of 28+ crops ran out of memory); one batch is read straight from them.
            within: bool = img_crops.shape[0] <= self.runner.spec.max_batch_size
            out = dict(self.runner(inputs) if within else run_chunked(self.runner, inputs))

        joints2d_raw = out[OUTPUT_NAMES[0]]
        visibility_raw = out[OUTPUT_NAMES[1]]
        contact_raw = out[OUTPUT_NAMES[2]]
        floor_raw = out[OUTPUT_NAMES[3]]
        assert joints2d_raw is not None and visibility_raw is not None and contact_raw is not None and floor_raw is not None
        joints2d_px: Float32[torch.Tensor, "n j 3"] = unproject_joints2d(joints2d_raw.float(), centers_t, sizes_t, self.config)
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
