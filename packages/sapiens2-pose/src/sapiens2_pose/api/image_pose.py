"""Single-image Sapiens2 pose inference, RRD logging, and baseline artifact writing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, UInt8
from numpy import ndarray
from PIL import Image
from simplecv.rerun_custom_types import Points2DWithConfidence, confidence_scores_to_rgb

from sapiens2_pose.api.metadata import get_sapiens_metainfo
from sapiens2_pose.api.pose_artifact import PosePredictionArtifact, save_pose_prediction_artifact
from sapiens2_pose.api.runtime import (
    DEFAULT_BBOX_THR,
    DEFAULT_MODEL_SIZE,
    DEFAULT_NMS_THR,
    DeviceChoice,
    ModelSize,
    PoseEstimation,
    detect_persons,
    estimate_frame_pose_from_bboxes,
)

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

    pose_estimation: PoseEstimation = estimate_frame_pose_from_bboxes(
        image_rgb,
        bboxes_f32,
        model_size=model_size,
        device=device,
    )
    keypoints: Float32[ndarray, "n k 2"] = np.stack(
        [np.asarray(kpts, dtype=np.float32) for kpts in pose_estimation.keypoints],
        axis=0,
    )
    scores: Float32[ndarray, "n k"] = np.stack(
        [np.asarray(scr, dtype=np.float32).reshape(-1) for scr in pose_estimation.scores],
        axis=0,
    )
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
    keypoint_ids: list[int] = list(range(keypoint_count))

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
        confidence_rgb: UInt8[ndarray, "k 3"] = confidence_scores_to_rgb(scores[None, :, None])[0]
        rr.log(
            f"image/person_{idx}/keypoints",
            Points2DWithConfidence(
                positions=keypoints,
                confidences=scores,
                class_ids=0,
                keypoint_ids=keypoint_ids,
                colors=confidence_rgb,
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
