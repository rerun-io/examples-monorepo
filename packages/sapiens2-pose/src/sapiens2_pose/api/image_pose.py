"""Single-image Sapiens2 pose inference, RRD logging, and baseline artifact writing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from huggingface_hub import hf_hub_download
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

from sapiens2_pose.api.pose_artifact import PosePredictionArtifact, save_pose_prediction_artifact
from sapiens2_pose.sapiens_lite.pose import estimate_pose, init_pose_model, nms, parse_pose_metainfo

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
CONFIGS_DIR: Path = PACKAGE_DIR / "assets" / "configs"
KEYPOINTS_308_PATH: Path = CONFIGS_DIR / "_base_" / "keypoints308.py"

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
DEFAULT_KPT_THR: float = 0.3


@dataclass(frozen=True, slots=True)
class ImagePoseConfig:
    """Configuration for writing one Sapiens2 pose prediction to Rerun and NumPy artifacts."""

    image_path: Path
    """Path to the input image."""
    rrd_path: Path
    """Path to the output Rerun recording."""
    artifact_path: Path
    """Path to the output numeric `.npz` pose artifact."""
    model_size: ModelSize = DEFAULT_MODEL_SIZE
    """Sapiens2 pose model size."""
    bbox_thr: float = DEFAULT_BBOX_THR
    """DETR person-detection confidence threshold."""
    nms_thr: float = DEFAULT_NMS_THR
    """Person-box NMS IoU threshold."""
    kpt_thr: float = DEFAULT_KPT_THR
    """Pose keypoint visibility threshold for Rerun rendering."""
    device: DeviceChoice = "auto"
    """Compute device selection; auto prefers CUDA when available."""


@dataclass(frozen=True, slots=True)
class ImagePoseSummary:
    """Summary returned after writing a single-image pose baseline."""

    rrd_path: Path
    """Path to the written Rerun recording."""
    artifact_path: Path
    """Path to the written numeric pose artifact."""
    person_count: int
    """Number of person instances written to the artifacts."""
    model_size: ModelSize
    """Sapiens2 pose model size used for inference."""


DetectPersonsFn = Callable[..., Float32[ndarray, "n 4"]]
EstimatePoseFn = Callable[..., PosePredictionArtifact]

_detector_cache: dict[str, tuple[Any, Any]] = {}
_pose_model_cache: dict[tuple[ModelSize, str], Any] = {}
_metainfo_cache: dict[str, Any] | None = None


def resolve_device(device: DeviceChoice = "auto") -> str:
    """Resolve a user device choice into a concrete torch device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def get_sapiens_metainfo() -> dict[str, Any]:
    """Return parsed Sapiens/Goliath 308-keypoint metadata."""
    global _metainfo_cache
    if _metainfo_cache is None:
        _metainfo_cache = parse_pose_metainfo({"from_file": str(KEYPOINTS_308_PATH)})
    return _metainfo_cache


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
    results: dict[str, torch.Tensor] = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=bbox_thr)[0]
    person_mask: torch.Tensor = results["labels"] == 1
    boxes: Float32[ndarray, "n 4"] = results["boxes"][person_mask].detach().cpu().numpy().astype(np.float32, copy=False)
    scores: Float32[ndarray, "n 1"] = results["scores"][person_mask].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 1)
    if boxes.shape[0] == 0:
        height: int = int(image_rgb.shape[0])
        width: int = int(image_rgb.shape[1])
        return np.asarray([[0.0, 0.0, float(width - 1), float(height - 1)]], dtype=np.float32)
    bboxes_with_scores: Float32[ndarray, "n 5"] = np.concatenate([boxes, scores], axis=1).astype(np.float32, copy=False)
    keep_indices: list[int] = nms(bboxes_with_scores, nms_thr)
    bboxes: Float32[ndarray, "n 4"] = bboxes_with_scores[keep_indices, :4].astype(np.float32, copy=False)
    return bboxes


def estimate_sapiens_pose(
    image_rgb: UInt8[ndarray, "h w 3"],
    bboxes: Float32[ndarray, "n 4"],
    *,
    model_size: ModelSize = DEFAULT_MODEL_SIZE,
    device: DeviceChoice = "auto",
) -> PosePredictionArtifact:
    """Estimate Sapiens2 pose for supplied RGB image and person boxes."""
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if bboxes_f32.shape[0] == 0:
        empty_keypoints: Float32[ndarray, "0 308 2"] = np.empty((0, 308, 2), dtype=np.float32)
        empty_scores: Float32[ndarray, "0 308"] = np.empty((0, 308), dtype=np.float32)
        return PosePredictionArtifact(bboxes=bboxes_f32, keypoints=empty_keypoints, scores=empty_scores)
    image_bgr: UInt8[ndarray, "h w 3"] = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    model: Any = get_pose_model(model_size, device=device)
    keypoints_raw: list[ndarray]
    scores_raw: list[ndarray]
    keypoints_raw, scores_raw = estimate_pose(image_bgr, bboxes_f32, model)
    keypoints: Float32[ndarray, "n k 2"] = np.stack([np.asarray(kpts, dtype=np.float32) for kpts in keypoints_raw], axis=0)
    scores: Float32[ndarray, "n k"] = np.stack([np.asarray(scr, dtype=np.float32).reshape(-1) for scr in scores_raw], axis=0)
    return PosePredictionArtifact(bboxes=bboxes_f32, keypoints=keypoints, scores=scores)


def _log_annotation_context(keypoint_count: int) -> None:
    """Log the Sapiens2 keypoint annotation context for an artifact."""
    meta: dict[str, Any] = get_sapiens_metainfo()
    keypoint_annotations: list[rr.AnnotationInfo] = [
        rr.AnnotationInfo(id=idx, label=str(meta["keypoint_id2name"][idx]))
        for idx in range(keypoint_count)
        if idx in meta["keypoint_id2name"]
    ]
    keypoint_connections: list[tuple[int, int]] = [
        (int(a), int(b))
        for a, b in meta["skeleton_links"]
        if int(a) < keypoint_count and int(b) < keypoint_count
    ]
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Sapiens2 308", color=(0, 255, 0)),
                    keypoint_annotations=keypoint_annotations,
                    keypoint_connections=keypoint_connections,
                )
            ]
        ),
        static=True,
    )


def _log_pose_recording(
    image_rgb: UInt8[ndarray, "h w 3"],
    artifact: PosePredictionArtifact,
    *,
    kpt_thr: float,
) -> None:
    """Log one image pose artifact into the active Rerun recording."""
    keypoint_count: int = int(artifact.keypoints.shape[1]) if artifact.keypoints.ndim == 3 else 0
    meta: dict[str, Any] = get_sapiens_metainfo()
    keypoint_ids: list[int] = list(range(keypoint_count))
    keypoint_colors: UInt8[ndarray, "k 3"] = np.asarray(meta["keypoint_colors"], dtype=np.uint8)[:keypoint_count]

    rr.send_blueprint(rrb.Blueprint(rrb.Spatial2DView(name="Pose", origin="image"), collapse_panels=True))
    rr.set_time("iteration", sequence=0)
    _log_annotation_context(keypoint_count)
    rr.log("image", rr.Image(image_rgb, color_model=rr.ColorModel.RGB))

    for idx, bbox in enumerate(artifact.bboxes):
        rr.log(
            f"image/person_{idx}/bbox",
            rr.Boxes2D(
                array=np.asarray(bbox, dtype=np.float32).reshape(1, 4),
                array_format=rr.Box2DFormat.XYXY,
                class_ids=0,
                labels=f"person_{idx}",
                colors=(0, 255, 0),
            ),
        )
        keypoints: Float32[ndarray, "k 2"] = np.asarray(artifact.keypoints[idx], dtype=np.float32).copy()
        scores: Float32[ndarray, "k"] = np.asarray(artifact.scores[idx], dtype=np.float32).reshape(-1)
        low_confidence_indices: ndarray = np.flatnonzero(scores < kpt_thr)
        keypoints[low_confidence_indices] = np.nan
        rr.log(
            f"image/person_{idx}/keypoints",
            rr.Points2D(
                positions=keypoints,
                class_ids=0,
                keypoint_ids=keypoint_ids,
                colors=keypoint_colors,
                show_labels=False,
            ),
        )


def run_image_pose(
    config: ImagePoseConfig,
    *,
    detect_persons_fn: DetectPersonsFn = detect_persons,
    estimate_pose_fn: EstimatePoseFn = estimate_sapiens_pose,
) -> ImagePoseSummary:
    """Run single-image pose inference and write RRD plus numeric artifacts."""
    image_pil: Image.Image = Image.open(config.image_path).convert("RGB")
    image_rgb: UInt8[ndarray, "h w 3"] = np.asarray(image_pil, dtype=np.uint8)
    bboxes: Float32[ndarray, "n 4"] = detect_persons_fn(
        image_rgb,
        bbox_thr=config.bbox_thr,
        nms_thr=config.nms_thr,
        device=config.device,
    )
    artifact: PosePredictionArtifact = estimate_pose_fn(
        image_rgb,
        bboxes,
        model_size=config.model_size,
        device=config.device,
    )
    save_pose_prediction_artifact(artifact, config.artifact_path)

    recording: rr.RecordingStream = rr.RecordingStream(application_id="sapiens2_pose_baseline")
    config.rrd_path.parent.mkdir(parents=True, exist_ok=True)
    with recording:
        _log_pose_recording(image_rgb, artifact, kpt_thr=config.kpt_thr)
    recording.save(str(config.rrd_path))

    return ImagePoseSummary(
        rrd_path=config.rrd_path,
        artifact_path=config.artifact_path,
        person_count=int(artifact.bboxes.shape[0]),
        model_size=config.model_size,
    )
