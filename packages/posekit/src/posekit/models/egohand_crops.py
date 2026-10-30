"""Calibrated GPU crops with upstream perspective and mirroring geometry."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from egohand_inference.camera.camera_models import create_perspective_remap_gpu
from egohand_inference.camera.perspective_crop import perspective_crop_geometry
from egohand_inference.preprocessing.mirroring import mirror_full_image_camera_horizontal
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.models.egohand_geometry import HandSide, OAKCamera
from posekit.ops.crops import remap_frames


def prepare_gpu_crops(frames: UInt8[Tensor, "b h w 3"], boxes: dict[str, dict[HandSide, Float32[ndarray, "3"]]], camera: OAKCamera) -> list[dict]:
    """Prepare model crops while retaining source-camera metadata for restoration.

    Small perimeter inversions stay on CPU. Full raster projection, sampling,
    mirroring and normalization remain on the frame's CUDA device.
    """
    jobs: list[tuple[str, HandSide, Float32[ndarray, "3"]]] = [
        (image_id, side, bbox) for image_id, per_hand in boxes.items() for side, bbox in per_hand.items()
    ]
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=8) as pool:
        geometry: list[tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"], Float64[ndarray, "3 3"]]] = list(
            pool.map(lambda job: perspective_crop_geometry(job[2], camera.model, 256), jobs)
        )
    maps: Float32[Tensor, "n 256 256 2"] = torch.empty((len(jobs), 256, 256, 2), device=frames.device, dtype=torch.float32)
    for index, (intrinsics, _, rotation) in enumerate(geometry):
        create_perspective_remap_gpu(camera.model, rotation, float(intrinsics[0, 0]), 256, maps[index].view(-1, 2))
    mirrored: Bool[Tensor, "n"] = torch.tensor([side == "right" for _, side, _ in jobs], device=frames.device)
    maps = torch.where(mirrored[:, None, None, None], maps.flip(2), maps)
    indices: Int64[Tensor, "n"] = torch.tensor([int(image_id) for image_id, _, _ in jobs], dtype=torch.int64, device=frames.device)
    # Development retains every runtime/local check; production fuses the same math.
    normalize = _normalize_crops if os.environ.get("PIXI_DEV_MODE") == "1" else _compiled_normalize_crops
    crops: Float32[Tensor, "n 3 256 256"] = normalize(frames, indices, maps)
    prepared: list[dict] = []
    for index, ((image_id, side, _), (intrinsics, bbox, rotation)) in enumerate(zip(jobs, geometry, strict=True)):
        model_bbox, model_intrinsics = mirror_full_image_camera_horizontal(bbox, intrinsics, 256) if side == "right" else (bbox, intrinsics)
        prepared.append(
            {
                "image_id": image_id,
                "real_hand_side": side,
                "crop_tensor": crops[index],
                "model_bbx_xys": model_bbox,
                "model_K": model_intrinsics,
                "model_img_size": np.array([256, 256], dtype=np.float32),
                "output_bbx_xys": bbox,
                "output_K": intrinsics,
                "crop_to_source_camera_rotation": rotation,
            }
        )
    return prepared


def _normalize_crops(
    frames: UInt8[Tensor, "b h w 3"], indices: Int64[Tensor, "n"], maps: Float32[Tensor, "n 256 256 2"]
) -> Float32[Tensor, "n 3 256 256"]:
    """Fuse uint8 gathering, interpolation and ImageNet normalization."""
    crops = remap_frames(frames, indices, maps) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=frames.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=frames.device)[None, :, None, None]
    return (crops - mean) / std


_compiled_normalize_crops = torch.compile(_normalize_crops, fullgraph=True, dynamic=True)
