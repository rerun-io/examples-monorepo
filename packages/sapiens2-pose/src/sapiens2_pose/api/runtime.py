"""Model loading, person detection, and frame-level Sapiens2 pose inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

from sapiens2_pose.api.metadata import get_sapiens_metainfo
from sapiens2_pose.sapiens_lite.pose import estimate_pose, init_pose_model, nms

ModelSize = Literal["0.4B", "0.8B", "1B", "5B"]
DeviceChoice = Literal["auto", "cpu", "cuda"]

POSE_MODELS: dict[ModelSize, dict[str, str]] = {
    "0.4B": {
        "repo": "facebook/sapiens2-pose-0.4b",
        "filename": "sapiens2_0.4b_pose.safetensors",
    },
    "0.8B": {
        "repo": "facebook/sapiens2-pose-0.8b",
        "filename": "sapiens2_0.8b_pose.safetensors",
    },
    "1B": {
        "repo": "facebook/sapiens2-pose-1b",
        "filename": "sapiens2_1b_pose.safetensors",
    },
    "5B": {
        "repo": "facebook/sapiens2-pose-5b",
        "filename": "sapiens2_5b_pose.safetensors",
    },
}
DEFAULT_MODEL_SIZE: ModelSize = "0.4B"
DETECTOR_MODEL_ID: str = "facebook/detr-resnet-50"
DEFAULT_BBOX_THR: float = 0.3
DEFAULT_NMS_THR: float = 0.3


@dataclass(frozen=True, slots=True)
class PoseEstimation:
    """Frame-level pose-estimation result in native Sapiens/Goliath order."""

    bboxes: Float32[ndarray, "n 4"]
    """Detected person bounding boxes in XYXY image coordinates."""
    keypoints: list[Float32[ndarray, "sapiens_k 2"]]
    """Per-person native Sapiens/Goliath keypoints in image coordinates."""
    scores: list[Float32[ndarray, "sapiens_k"]]
    """Per-person native Sapiens/Goliath keypoint confidence scores."""


_pose_model_cache: dict[tuple[ModelSize, str], Any] = {}
_detector_cache: dict[str, tuple[Any, Any]] = {}


def clear_pose_model_cache() -> None:
    """Clear cached Sapiens pose models held by this runtime module."""
    _pose_model_cache.clear()


def resolve_device(device: DeviceChoice = "auto") -> str:
    """Resolve a user device choice into a concrete torch device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def get_detector(device: DeviceChoice = "auto") -> tuple[Any, Any]:
    """Return the cached DETR person detector processor and model."""
    resolved_device: str = resolve_device(device)
    if resolved_device not in _detector_cache:
        processor: DetrImageProcessor = DetrImageProcessor.from_pretrained(DETECTOR_MODEL_ID)
        model: DetrForObjectDetection = DetrForObjectDetection.from_pretrained(DETECTOR_MODEL_ID).eval().to(resolved_device)
        _detector_cache[resolved_device] = (processor, model)
    return _detector_cache[resolved_device]


def get_pose_model(size: ModelSize = DEFAULT_MODEL_SIZE, device: DeviceChoice = "auto") -> Any:
    """Return a cached Sapiens2 pose model."""
    resolved_device: str = resolve_device(device)
    cache_key: tuple[ModelSize, str] = (size, resolved_device)
    if cache_key not in _pose_model_cache:
        spec: dict[str, str] = POSE_MODELS[size]
        checkpoint_path: str = hf_hub_download(repo_id=spec["repo"], filename=spec["filename"])
        model: Any = init_pose_model(size, checkpoint_path, device=resolved_device)
        model.pose_metainfo = get_sapiens_metainfo()
        _pose_model_cache[cache_key] = model
    return _pose_model_cache[cache_key]


def detect_persons(
    image_rgb: UInt8[ndarray, "h w 3"],
    *,
    bbox_thr: float = DEFAULT_BBOX_THR,
    nms_thr: float = DEFAULT_NMS_THR,
    device: DeviceChoice = "auto",
) -> Float32[ndarray, "n 4"]:
    """Detect person bounding boxes in an RGB image using DETR."""
    resolved_device: str = resolve_device(device)
    processor: Any
    model: Any
    processor, model = get_detector(device)
    pil_image: Image.Image = Image.fromarray(image_rgb)
    inputs: Any = processor(images=pil_image, return_tensors="pt").to(resolved_device)

    with torch.no_grad():
        outputs: Any = model(**inputs)

    target_sizes: torch.Tensor = torch.tensor([image_rgb.shape[:2]], device=resolved_device)
    results: dict[str, torch.Tensor] = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=bbox_thr,
    )[0]
    person_mask: torch.Tensor = results["labels"] == 1
    boxes: Float32[ndarray, "n 4"] = results["boxes"][person_mask].detach().cpu().numpy().astype(np.float32, copy=False)
    scores: Float32[ndarray, "n 1"] = results["scores"][person_mask].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 1)

    if boxes.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32)

    bboxes_with_scores: Float32[ndarray, "n 5"] = np.concatenate([boxes, scores], axis=1).astype(np.float32, copy=False)
    keep_indices: list[int] = nms(bboxes_with_scores, nms_thr)
    bboxes: Float32[ndarray, "n 4"] = bboxes_with_scores[keep_indices, :4].astype(np.float32, copy=False)
    return bboxes


def estimate_frame_pose_from_bboxes(
    image_rgb: UInt8[ndarray, "h w 3"],
    bboxes: Float32[ndarray, "n 4"],
    *,
    model_size: ModelSize = DEFAULT_MODEL_SIZE,
    device: DeviceChoice = "auto",
) -> PoseEstimation:
    """Estimate native Sapiens2 pose for externally supplied person boxes."""
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if bboxes_f32.shape[0] == 0:
        return PoseEstimation(bboxes=bboxes_f32, keypoints=[], scores=[])

    image_bgr: UInt8[ndarray, "h w 3"] = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    model: Any = get_pose_model(model_size, device=device)
    keypoints_raw: list[ndarray]
    scores_raw: list[ndarray]
    keypoints_raw, scores_raw = estimate_pose(image_bgr, bboxes_f32, model)

    keypoints: list[Float32[ndarray, "sapiens_k 2"]] = [np.asarray(kpts, dtype=np.float32) for kpts in keypoints_raw]
    scores: list[Float32[ndarray, "sapiens_k"]] = [np.asarray(scr, dtype=np.float32).reshape(-1) for scr in scores_raw]
    return PoseEstimation(bboxes=bboxes_f32, keypoints=keypoints, scores=scores)


def estimate_frame_pose(
    image_rgb: UInt8[ndarray, "h w 3"],
    *,
    model_size: ModelSize = DEFAULT_MODEL_SIZE,
    bbox_thr: float = DEFAULT_BBOX_THR,
    nms_thr: float = DEFAULT_NMS_THR,
    device: DeviceChoice = "auto",
) -> PoseEstimation:
    """Detect people and estimate native Sapiens2 pose for one RGB frame."""
    bboxes: Float32[ndarray, "n 4"] = detect_persons(
        image_rgb,
        bbox_thr=bbox_thr,
        nms_thr=nms_thr,
        device=device,
    )
    if bboxes.shape[0] == 0:
        return PoseEstimation(bboxes=bboxes, keypoints=[], scores=[])

    return estimate_frame_pose_from_bboxes(
        image_rgb,
        bboxes,
        model_size=model_size,
        device=device,
    )
