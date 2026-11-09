"""Rerun evidence for complete hand outputs and the model's calibrated crops."""

import numpy as np
import rerun as rr
from egohand_inference.camera.perspective_crop import perspective_crop
from jaxtyping import Float32, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.cameras import CalibratedCamera, camera_key
from posekit.models.egohand import EgoHandPredictor, EgoHandReconstruction
from posekit.models.egohand_crops import IMAGENET_MEAN, IMAGENET_STD
from posekit.models.egohand_cuda_crops import PreparedHandCrops
from posekit.models.egohand_fast import FastEgoHandPredictor
from posekit.models.egohand_geometry import lens_model
from posekit.ops.crops import channel_vector
from posekit.predictions import HandDetections


def log_hand_intermediates(
    frames: UInt8[Tensor, "1 h w 3"],
    detections: HandDetections,
    result: EgoHandReconstruction,
    camera: CalibratedCamera,
    predictor: EgoHandPredictor | FastEgoHandPredictor,
) -> None:
    """Log full meshes and parameters, and regenerate the same crops outside timing."""
    crops: list[UInt8[ndarray, "256 256 3"]] = []
    if isinstance(predictor, FastEgoHandPredictor) and detections.num_detections:
        prepared: PreparedHandCrops = predictor.croppers[(camera_key(camera), str(frames.device))].prepare(frames, detections, predictor.config.padding)
        mean: Float32[Tensor, "1 3 1 1"] = channel_vector(IMAGENET_MEAN, frames.device)
        std: Float32[Tensor, "1 3 1 1"] = channel_vector(IMAGENET_STD, frames.device)
        images: UInt8[Tensor, "n 256 256 3"] = ((prepared.images * std + mean) * 255.0).round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        crops = list(images.cpu().numpy())
    elif detections.num_detections:
        image: UInt8[ndarray, "h w 3"] = np.ascontiguousarray(frames[0].cpu().numpy()[:, :, ::-1])
        for box, side in zip(detections.boxes.xyxy_numpy(), detections.sides, strict=True):
            square: Float32[ndarray, "3"] = np.array(
                [*((box[:2] + box[2:]) / 2.0), float(np.max(box[2:] - box[:2])) * (1.0 + predictor.config.padding)], dtype=np.float32
            )
            crop, _, _, _ = perspective_crop(image, square, lens_model(camera), 256)
            if side == "right":
                crop = crop[:, ::-1]
            crops.append((crop * 255.0).round().clip(0, 255).astype(np.uint8))
    for side in ("left", "right"):
        if side in detections.sides:
            rr.log(f"crops/{side}", rr.Image(crops[detections.sides.index(side)]))
        else:
            rr.log(f"crops/{side}", rr.Clear(recursive=True))
        if side not in result.sides:
            rr.log(f"camera/mesh/{side}", rr.Clear(recursive=True))
            rr.log(f"parameters/{side}", rr.Clear(recursive=True))
            continue
        row: int = result.sides.index(side)
        assert result.vertices is not None and result.faces is not None and result.scale_parameters is not None
        rr.log(
            f"camera/mesh/{side}",
            rr.Mesh3D(
                vertex_positions=result.vertices[row].cpu().numpy(),
                triangle_indices=result.faces[side].cpu().numpy(),
                albedo_factor=[80, 220, 160, 160] if side == "left" else [230, 120, 180, 160],
            ),
        )
        for name, tensor in (
            ("rotations", result.rotations),
            ("identity", result.identity),
            ("translation", result.translation),
            ("scales", result.scale_parameters),
        ):
            rr.log(f"parameters/{side}/{name}", rr.Tensor(tensor[row].cpu().numpy()))
        rr.log(f"parameters/{side}/global_scale", rr.Scalars(float(result.scale[row])))
