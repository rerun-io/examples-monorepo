"""Per-camera inference of the multiview body demo: SAM3 video segmentation of the text prompt, then SAM 3D Body
on the first detected person of each camera.

One SAM3 video session serves every camera; the body role (posekit ``TopDownBodyPose3d``) returns camera-space
rows that :func:`to_final_pose_predictions` turns into the upstream per-person records the multiview optimizer reads.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

from posekit.apis.body3d import pinhole_camera
from posekit.body_rerun import SEG_CLASS_OFFSET
from posekit.models.sam3d_body import Sam3dBodyPredictor, Sam3dBodyReconstruction
from posekit.predictions import BoxDetections
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction


@dataclass(slots=True)
class Sam3MVBodyModelConfig:
    """Settings for the SAM3 video checkpoint and compute placement."""

    checkpoint: str = "facebook/sam3"
    """Model identifier passed to ``Sam3VideoModel.from_pretrained``."""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    """Compute device selection; ``auto`` prefers CUDA when available."""
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    """Torch dtype used for model weights and inference."""
    processing_device: Literal["cpu", "cuda"] = "cpu"
    """Device used by the processor for per-frame preprocessing."""
    video_storage_device: Literal["cpu", "cuda"] = "cpu"
    """Device that caches video tensors within the inference session."""
    score_threshold_detection: float = 0.85
    """Minimum detection score before tracking."""
    new_det_thresh: float | None = None
    """Optional stricter threshold for new detections."""


@dataclass(frozen=True, slots=True)
class CameraSegmentation:
    """SAM3 output of one camera frame."""

    seg_map: UInt8[ndarray, "h w"]
    """Class id per pixel: ``SEG_CLASS_OFFSET`` for background, ``SEG_CLASS_OFFSET + i + 1`` for instance ``i``."""
    boxes: Float32[ndarray, "n 4"] | None
    """Instance boxes, ``xyxy``."""
    scores: Float32[ndarray, "n"] | None
    """Detection confidences."""
    masks: Float32[ndarray, "n h w"] | None
    """Raw instance probability masks."""


def to_final_pose_predictions(people: Sam3dBodyReconstruction) -> list[FinalPosePrediction]:
    """The role's camera-space rows as upstream per-person records (root-relative joints and mesh plus ``pred_cam_t``),
    the form the multiview optimizer and validation read."""
    assert people.keypoints.xyz_camera is not None
    translation: np.ndarray = people.translation.cpu().numpy()
    return [
        FinalPosePrediction(
            bbox=people.boxes[row].cpu().numpy(),
            focal_length=people.focal_length[row].cpu().numpy(),
            pred_keypoints_3d=people.keypoints.xyz_camera[row].cpu().numpy() - translation[row],
            pred_keypoints_2d=people.keypoints.xy[row].cpu().numpy(),
            pred_vertices=people.vertices[row].cpu().numpy() - translation[row],
            pred_cam_t=translation[row],
            pred_pose_raw=np.zeros(266, dtype=np.float32),
            global_rot=people.global_rotation[row].cpu().numpy(),
            body_pose_params=people.body_pose[row].cpu().numpy(),
            hand_pose_params=people.hand_pose[row].cpu().numpy(),
            scale_params=people.scale[row].cpu().numpy(),
            shape_params=people.shape[row].cpu().numpy(),
            expr_params=people.expression[row].cpu().numpy(),
            pred_joint_coords=people.joint_coordinates[row].cpu().numpy() - translation[row],
            pred_global_rots=people.joint_rotations[row].cpu().numpy(),
        )
        for row in range(people.num_people)
    ]


def _to_numpy(array_like: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """Convert tensors/lists to numpy for downstream processing."""
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _prepare_segmentation_assets(frame_rgb: UInt8[ndarray, "h w 3"], processed_outputs: dict) -> CameraSegmentation:
    """Convert SAM3 outputs into segmentation IDs, boxes, scores and raw masks."""
    raw_masks = processed_outputs.get("masks")
    raw_boxes = processed_outputs.get("boxes")
    raw_scores = processed_outputs.get("scores")

    if raw_masks is None or len(raw_masks) == 0:
        h: int = int(frame_rgb.shape[0])
        w: int = int(frame_rgb.shape[1])
        return CameraSegmentation(np.full((h, w), SEG_CLASS_OFFSET, dtype=np.uint8), None, None, None)

    masks_np: Float32[ndarray, "n h w"] = _to_numpy(raw_masks).astype(np.float32, copy=False)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    h = int(frame_rgb.shape[0])
    w = int(frame_rgb.shape[1])
    seg_map: UInt8[ndarray, "h w"] = np.full((h, w), SEG_CLASS_OFFSET, dtype=np.uint8)
    num_instances: int = int(masks_np.shape[0])

    for idx in range(num_instances):
        mask: Float32[ndarray, "h w"] = np.asarray(masks_np[idx], dtype=np.float32)
        mask_bool: Bool[ndarray, "h w"] = mask >= 0.5
        class_id: int = SEG_CLASS_OFFSET + idx + 1
        seg_map = np.where(mask_bool, np.uint8(class_id), seg_map)

    boxes_np: Float32[ndarray, "n 4"] | None = None
    scores_np: Float32[ndarray, "n"] | None = None
    if raw_boxes is not None:
        boxes_np = _to_numpy(raw_boxes).astype(np.float32, copy=False)
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None, :]
    if raw_scores is not None:
        scores_np = _to_numpy(raw_scores).astype(np.float32, copy=False).reshape(-1)

    return CameraSegmentation(seg_map, boxes_np, scores_np, masks_np)


class Sam3VideoSegmenter:
    """SAM3 video model with one text-prompted inference session shared by every camera."""

    def __init__(self, config: Sam3MVBodyModelConfig, prompt: str, device: torch.device, dtype: torch.dtype) -> None:
        # Load SAM3 video model and processor
        config_kwargs: dict = {
            "score_threshold_detection": config.score_threshold_detection,
        }
        if config.new_det_thresh is not None:
            config_kwargs["new_det_thresh"] = config.new_det_thresh
        else:
            config_kwargs["new_det_thresh"] = config.score_threshold_detection

        model_cfg = Sam3VideoConfig.from_pretrained(config.checkpoint, **config_kwargs)
        self.device: torch.device = device
        self.model = Sam3VideoModel.from_pretrained(config.checkpoint, config=model_cfg).to(device=device, dtype=dtype)  # pyrefly: ignore[missing-argument]  # transformers' decorated .to() stub
        self.processor = Sam3VideoProcessor.from_pretrained(config.checkpoint)

        # Initialize a single video inference session for all cameras
        self.inference_session = self.processor.init_video_session(
            video=None,
            inference_device=device,
            processing_device=config.processing_device,
            video_storage_device=config.video_storage_device,
            dtype=dtype,
        )
        self.processor.add_text_prompt(inference_session=self.inference_session, text=prompt)

    def segment(self, frame_rgb: UInt8[ndarray, "h w 3"]) -> CameraSegmentation:
        """Run SAM3 detection and tracking on the next frame of one camera."""
        inputs = self.processor(images=frame_rgb, device=self.device, return_tensors="pt")  # type: ignore[misc]
        model_outputs = self.model(
            inference_session=self.inference_session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        processed_outputs = self.processor.postprocess_outputs(
            self.inference_session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )
        return _prepare_segmentation_assets(frame_rgb, processed_outputs)


def first_person_body(
    body: Sam3dBodyPredictor,
    frame_rgb: UInt8[ndarray, "h w 3"],
    segmentation: CameraSegmentation,
    k_matrix: ndarray | None,
) -> FinalPosePrediction | None:
    """SAM 3D Body on the first SAM3 detection (single person assumption); None without a detection or intrinsics."""
    boxes: Float32[ndarray, "n 4"] | None = segmentation.boxes
    scores: Float32[ndarray, "n"] | None = segmentation.scores
    masks_np: Float32[ndarray, "n h w"] | None = segmentation.masks
    if boxes is None or masks_np is None or len(boxes) == 0 or k_matrix is None:
        return None
    h, w = frame_rgb.shape[:2]
    people: Sam3dBodyReconstruction = body(
        torch.from_numpy(frame_rgb).cuda()[None],
        BoxDetections(  # Take only first person
            torch.from_numpy(boxes[:1]).float().cuda(),
            torch.from_numpy(scores[:1]).float().cuda() if scores is not None else torch.ones(1, device="cuda"),
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            masks=torch.from_numpy(masks_np[:1] > 0.5).cuda(),
        ),
        pinhole_camera(k_matrix.astype(np.float32), w, h),
    )
    pred_list: list[FinalPosePrediction] = to_final_pose_predictions(people)
    return pred_list[0] if pred_list else None
