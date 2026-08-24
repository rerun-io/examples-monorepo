from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import torch
from jaxtyping import Bool, Float, Float32, Float64, Int, UInt8
from numpy import ndarray
from posekit.models import PersonDetector, RtmPoseConfig, TopDownPose2d, YoloxDetectorConfig
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.runtimes import TensorRtBackendConfig
from posekit.zoo import PRESETS, PosePreset
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco133_layers import COCO133_PREDICTION_LAYER_TO_PATH, Coco133AnnotationLayer
from simplecv.data.skeleton.coco_133 import COCO_133_IDS, LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.ops.triangulate import batch_triangulate, projectN3
from simplecv.rerun_custom_types import Points2DWithConfidence
from torch import Tensor
from wilor_nano.hand_keypoints import (
    FinalWilorPred,
    HandKeypointDetectorConfig,
    KeypointResults,
    WilorHandKeypointDetector,
)

WILOR_CONFIDENCE_THRESHOLD: float = 0.5
WILOR_BBOX_EXPANSION_RATIO: float = 0.25


def compute_square_bbox_from_uv(
    *,
    hand_uv: Float32[ndarray, "hand 2"],
    image_shape: tuple[int, int],
    expansion_ratio: float,
) -> Float32[ndarray, "4"] | None:
    """Return an expanded square ``XYXY`` box from finite hand keypoints."""
    valid_mask: Bool[ndarray, "hand"] = np.isfinite(hand_uv[:, 0]) & np.isfinite(hand_uv[:, 1])
    if not bool(np.any(valid_mask)):
        return None

    valid_uv: Float32[ndarray, "valid 2"] = hand_uv[valid_mask, :]
    min_xy: Float32[ndarray, "2"] = np.nanmin(valid_uv, axis=0).astype(np.float32, copy=False)
    max_xy: Float32[ndarray, "2"] = np.nanmax(valid_uv, axis=0).astype(np.float32, copy=False)
    side_length: float = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1]))
    if not np.isfinite(side_length) or side_length <= 0.0:
        return None

    center_xy: Float32[ndarray, "2"] = ((min_xy + max_xy) * 0.5).astype(np.float32, copy=False)
    half_side: float = 0.5 * side_length * (1.0 + expansion_ratio)
    if half_side <= 0.0:
        return None

    x1: float = float(center_xy[0] - half_side)
    y1: float = float(center_xy[1] - half_side)
    x2: float = float(center_xy[0] + half_side)
    y2: float = float(center_xy[1] + half_side)

    height: int
    width: int
    height, width = image_shape
    x1 = float(np.clip(x1, 0.0, float(width - 1)))
    x2 = float(np.clip(x2, 0.0, float(width - 1)))
    y1 = float(np.clip(y1, 0.0, float(height - 1)))
    y2 = float(np.clip(y2, 0.0, float(height - 1)))
    if x2 <= x1 or y2 <= y1:
        return None

    bbox: Float32[ndarray, "4"] = np.array([x1, y1, x2, y2], dtype=np.float32)
    return bbox


@dataclass
class MVHistory:
    """Temporal state for multiview keypoint tracking."""

    xyzc_t: Float32[ndarray, "n_kpts=133 4"] | None = None
    """Current frame 3D keypoints with confidence."""
    xyzc_t1: Float32[ndarray, "n_kpts=133 4"] | None = None
    """Previous frame 3D keypoints with confidence."""
    uvc_t: Float32[ndarray, "n_views n_kpts=133 3"] | None = None
    """Current per-view 2D detections with confidence."""
    uvc_extrap: Float32[ndarray, "n_views n_kpts=133 3"] | None = None
    """Per-view 2D projections from temporally extrapolated 3D keypoints."""


# mv-api mode -> posekit zoo preset (same OpenMMLab weights rtmlib served,
# maintained in ONE place: posekit.zoo). "lightweight" is a local pairing —
# yolox-tiny with the balanced rtmpose-m net (posekit carries no rtmpose-s).
MODE_PRESETS: dict[str, PosePreset] = {
    "performance": PRESETS["body-performance"],
    "balanced": PRESETS["body"],
    "wholebody": PRESETS["wholebody"],
    "lightweight": PosePreset(
        YoloxDetectorConfig(variant="yolox-tiny-humanart"),
        RtmPoseConfig(variant="rtmpose-m-coco17"),
        "Smallest detector with the balanced pose net.",
    ),
}


@dataclass
class MultiviewBodyTrackerConfig:
    """Configuration options for the multiview body tracker runtime."""

    mode: Literal["lightweight", "balanced", "performance", "wholebody"] = "wholebody"
    """Preset selecting detector and pose assets tuned for latency versus accuracy."""
    backend: Literal["onnxruntime", "tensorrt"] = "tensorrt"
    """Inference backend for the posekit models. tensorrt (default) loads cached dynamic-batch
    engines (~13s deserialize per process; first run on a new GPU builds them, taking minutes);
    onnxruntime skips that fixed cost — better for quick short-clip experiments."""
    device: Literal["cpu", "cuda"] = "cuda"
    """Device the frame tensors are staged on before the models. The posekit
    models themselves are CUDA-resident; ``cpu`` only makes sense with injected
    CPU-capable test doubles."""
    keypoint_threshold: float = 0.7
    """Minimum 2D keypoint confidence required for a detection to be kept."""
    cams_for_detection_idx: list[int] | None = None
    """Subset of camera indices to run detection on; None evaluates every view."""
    use_wilor: bool = False
    """Whether to use WiLor-Nano for hand keypoints instead of RTMPose."""
    perform_tracking: bool = True
    """Whether to extrapolate historical poses to assist detection."""
    verbose: bool = False
    """Enables additional debug logging when True."""


class MultiviewBodyTracker:
    def __init__(
        self,
        config: MultiviewBodyTrackerConfig,
        filter_body_idxes: Int[ndarray, "idx"] | None = None,
        *,
        detector: PersonDetector | None = None,
        pose: TopDownPose2d | None = None,
    ) -> None:
        """Create detector and pose models for single-person multiview tracking.

        Args:
            config: Runtime options; ``config.mode`` picks the default posekit
                detector/pose pairing (same OpenMMLab weights rtmlib served).
            filter_body_idxes: Optional keypoint subset kept in the outputs.
            detector: Override detector implementing the posekit role; any
                :class:`posekit.models.PersonDetector` slots in (RT-DETRv2,
                a tracker adapter, ...).
            pose: Override top-down pose estimator; any
                :class:`posekit.models.TopDownPose2d` slots in (ViTPose,
                Sapiens2, a TensorRT-backed RTMW, ...).
        """
        self.config: MultiviewBodyTrackerConfig = config
        preset: PosePreset = MODE_PRESETS[config.mode]
        det_config = preset.detector
        pose_config = preset.pose
        # All mv-api mode presets pair the ONNX-artifact models, whose configs
        # carry the swappable onnx/tensorrt backend field.
        assert isinstance(det_config, YoloxDetectorConfig) and isinstance(pose_config, RtmPoseConfig)
        if config.backend == "tensorrt":
            trt_backend = TensorRtBackendConfig()
            det_config = replace(det_config, backend=trt_backend)
            pose_config = replace(pose_config, backend=trt_backend)
        self.det_model: PersonDetector = detector if detector is not None else det_config.setup()
        self.pose_model: TopDownPose2d = pose if pose is not None else pose_config.setup()
        self.num_keypoints: int = self.pose_model.skeleton.num_keypoints
        self.filter_body_idxes: Int[ndarray, "idx"] = (
            filter_body_idxes if filter_body_idxes is not None else np.arange(self.num_keypoints, dtype=np.intp)
        )
        if self.filter_body_idxes.size and int(self.filter_body_idxes.max()) >= self.num_keypoints:
            raise ValueError(
                f"filter_body_idxes max index {int(self.filter_body_idxes.max())} exceeds the "
                f"{self.pose_model.skeleton.name} skeleton's {self.num_keypoints} keypoints."
            )
        self.hand_keypoint_engine: WilorHandKeypointDetector | None = None
        if self.config.use_wilor:
            self.hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))

    def __call__(
        self,
        *,
        frames_rgb: UInt8[Tensor, "n_total_views H W 3"],
        pinhole_list: list[PinholeParameters],
        pred_state: MVHistory,
        pinhole_log_paths: list[Path] | None = None,
        recording: rr.RecordingStream | None = None,
    ) -> MVHistory:
        """Estimate and triangulate one COCO-133 person across camera views.

        Args:
            frames_rgb: One uint8 RGB NHWC tensor holding every camera view
                (torchcodec CUDA decode path) — frames never touch the host on
                the way to the models.
            pinhole_list: Calibrated camera per view, aligned with the frames.
            pred_state: Temporal tracking state, updated in place.
            pinhole_log_paths: Pre-resolved pinhole entity path per view
                (rig layout) for the verbose 2D layers, aligned with
                ``pinhole_list``; ``None`` falls back to the legacy
                ``/world/exo/<name>`` layout.
            recording: Optional explicit Rerun recording stream.
        """
        num_total_views: int = int(frames_rgb.shape[0])
        selected_view_indices: list[int] = (
            [idx for idx in self.config.cams_for_detection_idx]
            if self.config.cams_for_detection_idx is not None
            else list(range(num_total_views))
        )
        if not selected_view_indices:
            raise ValueError("At least one camera must be selected for multiview body tracking.")
        for selected_view_idx in selected_view_indices:
            if selected_view_idx < 0 or selected_view_idx >= num_total_views or selected_view_idx >= len(pinhole_list):
                msg: str = f"Selected camera index {selected_view_idx} is outside the available view range."
                raise IndexError(msg)

        device: torch.device = torch.device(self.config.device)
        selected_frames_rgb: UInt8[Tensor, "n_views H W 3"] = (
            frames_rgb.to(device)
            if self.config.cams_for_detection_idx is None
            else frames_rgb[torch.tensor(selected_view_indices, dtype=torch.long, device=frames_rgb.device)].to(device)
        )
        selected_pinhole_list: list[PinholeParameters] = [pinhole_list[idx] for idx in selected_view_indices]
        pall: Float32[ndarray, "n_views 3 4"] = np.array(
            [pinhole.projection_matrix for pinhole in selected_pinhole_list], dtype=np.float32
        )

        xyzc_extrap: Float32[ndarray, "n_kpts 4"] | None = None
        bboxes_extrap: Float32[ndarray, "n_views 4"] | None = None
        tracked_confidences: Float32[ndarray, "n_kpts"] | None = None
        pred_state.uvc_extrap = None
        if (pred_state.xyzc_t1 is not None and pred_state.xyzc_t is not None) and self.config.perform_tracking:
            xyzc_extrap = self.extrapolate_3d_keypoints(xyzc_t=pred_state.xyzc_t, xyzc_t1=pred_state.xyzc_t1)
            uvc_extrap_float: Float[ndarray, "n_views n_kpts 3"] = projectN3(xyzc_extrap, pall)
            uvc_extrap: Float32[ndarray, "n_views n_kpts 3"] = np.asarray(uvc_extrap_float, dtype=np.float32)
            pred_state.uvc_extrap = uvc_extrap
            tracked_confidences = np.clip(xyzc_extrap[:, 3].astype(np.float32, copy=True), 0.0, 1.0)
            uv_max: Float[ndarray, "n_views 2"] = np.nanmax(uvc_extrap[:, :, 0:2], axis=1)
            uv_min: Float[ndarray, "n_views 2"] = np.nanmin(uvc_extrap[:, :, 0:2], axis=1)
            bboxes_extrap = np.concatenate([uv_min, uv_max], axis=1).astype(np.float32)

        # One detector call and one pose call over all selected views, on the
        # GPU-resident uint8 RGB tensor (the posekit contract).
        num_selected: int = len(selected_view_indices)
        uvc_by_selected: dict[int, Float32[ndarray, "n_kpts 3"]] = {}
        if bboxes_extrap is not None:
            detections: BoxDetections = BoxDetections(
                xyxy=torch.from_numpy(bboxes_extrap).to(device=device, dtype=torch.float32),
                scores=torch.ones((num_selected,), dtype=torch.float32, device=device),
                frame_indices=torch.arange(num_selected, dtype=torch.long, device=device),
            )
        else:
            all_detections: BoxDetections = self.det_model(selected_frames_rgb)
            # Pick the best-scoring detection per view on the host: two small
            # transfers instead of a GPU sync per view.
            det_views: list[int] = all_detections.frame_indices.cpu().tolist()
            det_scores: list[float] = all_detections.scores.cpu().tolist()
            best_by_view: dict[int, int] = {}
            for row, (view, score) in enumerate(zip(det_views, det_scores, strict=True)):
                if view not in best_by_view or score > det_scores[best_by_view[view]]:
                    best_by_view[view] = row
            rows: Tensor = torch.tensor([best_by_view[view] for view in sorted(best_by_view)], dtype=torch.long, device=device)
            detections = BoxDetections(
                xyxy=all_detections.xyxy[rows], scores=all_detections.scores[rows], frame_indices=all_detections.frame_indices[rows]
            )
        keypoints2d: Keypoints2d = self.pose_model(selected_frames_rgb, detections)
        xy: Float32[ndarray, "n_dets n_kpts 2"] = keypoints2d.xy_numpy()
        kpt_scores: Float32[ndarray, "n_dets n_kpts"] = keypoints2d.scores_numpy()
        for row, local_idx in enumerate(detections.frame_indices.cpu().tolist()):
            uvc_by_selected[local_idx] = np.concatenate([xy[row], kpt_scores[row][:, None]], axis=1).astype(np.float32)

        uvc_list: list[Float32[ndarray, "n_kpts 3"]] = []
        for selected_idx, (original_view_idx, pinhole) in enumerate(zip(selected_view_indices, selected_pinhole_list, strict=True)):
            view_uvc: Float32[ndarray, "n_kpts 3"] | None = uvc_by_selected.get(selected_idx)
            if view_uvc is None:
                uvc_list.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))
                continue
            filtered_keypoints: Float32[ndarray, "n_kpts 2"] = view_uvc[:, 0:2]
            filtered_scores: Float32[ndarray, "n_kpts"] = view_uvc[:, 2]

            if self.config.use_wilor:
                # WiLoR is a numpy/CPU engine; materialize the BGR view lazily
                # so the GPU frames only pay this when hands are refined.
                view_bgr: UInt8[ndarray, "H W 3"] = np.ascontiguousarray(selected_frames_rgb[selected_idx].cpu().numpy()[..., ::-1])
                filtered_keypoints = self._refine_hand_keypoints_with_wilor(
                    bgr=view_bgr,
                    keypoints=filtered_keypoints,
                    confidences=filtered_scores,
                )

            if self.config.verbose:
                view_pinhole_path: Path | None = pinhole_log_paths[original_view_idx] if pinhole_log_paths is not None else None
                self._log_uvc_layer(
                    view_idx=original_view_idx,
                    pinhole=pinhole,
                    keypoints=filtered_keypoints,
                    confidences=filtered_scores,
                    layer=Coco133AnnotationLayer.RAW_2D,
                    mask_below_threshold=True,
                    pinhole_log_path=view_pinhole_path,
                    recording=recording,
                )
                if tracked_confidences is not None and pred_state.uvc_extrap is not None:
                    tracked_uv: Float32[ndarray, "n_kpts 2"] = pred_state.uvc_extrap[selected_idx, :, 0:2]
                    self._log_uvc_layer(
                        view_idx=original_view_idx,
                        pinhole=pinhole,
                        keypoints=tracked_uv,
                        confidences=tracked_confidences,
                        layer=Coco133AnnotationLayer.TRACKED_2D,
                        mask_below_threshold=False,
                        pinhole_log_path=view_pinhole_path,
                        recording=recording,
                    )

            uvc: Float32[ndarray, "n_kpts 3"] = np.concatenate([filtered_keypoints, filtered_scores[:, None]], axis=1)
            uvc_list.append(uvc)

        multiview_uvc: Float32[ndarray, "n_views n_kpts 3"] = np.stack(uvc_list).astype(np.float32)
        pred_state.uvc_t = multiview_uvc
        xyzc: Float64[ndarray, "n_kpts 4"] = batch_triangulate(
            keypoints_2d=multiview_uvc,
            projection_matrices=pall,
            min_views=2,
        )
        pred_state.xyzc_t1 = pred_state.xyzc_t
        pred_state.xyzc_t = xyzc.astype(np.float32)
        return pred_state

    def _log_uvc_layer(
        self,
        *,
        view_idx: int,
        pinhole: PinholeParameters,
        keypoints: Float32[ndarray, "n_kpts 2"],
        confidences: Float32[ndarray, "n_kpts"],
        layer: Coco133AnnotationLayer,
        mask_below_threshold: bool,
        pinhole_log_path: Path | None,
        recording: rr.RecordingStream | None,
    ) -> None:
        """Log a COCO-133 2D layer for a single view."""
        view_name: str = pinhole.name if pinhole.name else f"view_{view_idx}"
        pinhole_path: str = str(pinhole_log_path) if pinhole_log_path is not None else f"/world/exo/{view_name}/pinhole"
        if mask_below_threshold:
            visibility_mask: Bool[ndarray, "n_kpts"] = confidences >= float(self.config.keypoint_threshold)
            filtered_keypoints: Float32[ndarray, "n_kpts 2"] = np.where(visibility_mask[:, None], keypoints, np.nan)
            filtered_confidences: Float32[ndarray, "n_kpts"] = np.where(visibility_mask, confidences, 0.0)
        else:
            filtered_keypoints = keypoints.astype(np.float32, copy=False)
            filtered_confidences = confidences.astype(np.float32, copy=False)

        finite_mask: Bool[ndarray, "n_kpts"] = np.isfinite(filtered_keypoints[:, 0]) & np.isfinite(
            filtered_keypoints[:, 1]
        )
        filtered_keypoints = np.where(finite_mask[:, None], filtered_keypoints, np.nan).astype(np.float32, copy=False)
        filtered_confidences = np.where(finite_mask, filtered_confidences, 0.0).astype(np.float32, copy=False)
        layer_segment: str = COCO133_PREDICTION_LAYER_TO_PATH[layer]
        rr.log(
            f"{pinhole_path}/pred/coco133_uv/{layer_segment}",
            Points2DWithConfidence(
                positions=filtered_keypoints,
                confidences=filtered_confidences,
                class_ids=int(layer),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
            recording=recording,
        )

    def _refine_hand_keypoints_with_wilor(
        self,
        *,
        bgr: UInt8[ndarray, "H W 3"],
        keypoints: Float32[ndarray, "n_kpts 2"],
        confidences: Float32[ndarray, "n_kpts"],
    ) -> Float32[ndarray, "n_kpts 2"]:
        engine: WilorHandKeypointDetector | None = self.hand_keypoint_engine
        if engine is None:
            return keypoints

        refined_keypoints: Float32[ndarray, "n_kpts 2"] = keypoints.copy()
        height: int = int(bgr.shape[0])
        width: int = int(bgr.shape[1])
        rgb_hw3: UInt8[ndarray, "H W 3"] = bgr[..., ::-1]
        score_threshold: float = max(self.config.keypoint_threshold, WILOR_CONFIDENCE_THRESHOLD)

        for hand_indices_raw, handedness in ((LEFT_HAND_IDX, "left"), (RIGHT_HAND_IDX, "right")):
            hand_indices: Int[ndarray, "hand"] = np.asarray(hand_indices_raw, dtype=np.intp)
            hand_uv: Float32[ndarray, "hand 2"] = refined_keypoints[hand_indices, :].copy()
            hand_scores: Float32[ndarray, "hand"] = confidences[hand_indices]
            low_conf_mask: Bool[ndarray, "hand"] = np.asarray(hand_scores < score_threshold, dtype=bool)
            hand_uv[low_conf_mask, :] = np.nan

            bbox: Float32[ndarray, "4"] | None = self._compute_hand_bbox(
                hand_uv=hand_uv,
                image_shape=(height, width),
                expansion_ratio=WILOR_BBOX_EXPANSION_RATIO,
            )
            if bbox is None:
                continue

            xyxy: Float32[ndarray, "1 4"] = bbox[np.newaxis, :]
            wilor_pred: FinalWilorPred | KeypointResults = engine(rgb_hw3=rgb_hw3, xyxy=xyxy, handedness=handedness)
            raw_pred_uv: Float[ndarray, "1 21 2"] = (
                wilor_pred.keypoints_2d if isinstance(wilor_pred, KeypointResults) else wilor_pred.pred_keypoints_2d
            )
            pred_uv: Float32[ndarray, "1 21 2"] = np.asarray(raw_pred_uv, dtype=np.float32)
            refined_hand_uv: Float32[ndarray, "21 2"] = pred_uv[0]
            refined_hand_uv[:, 0] = np.clip(refined_hand_uv[:, 0], 0.0, float(width - 1))
            refined_hand_uv[:, 1] = np.clip(refined_hand_uv[:, 1], 0.0, float(height - 1))
            refined_keypoints[hand_indices, :] = refined_hand_uv

        return refined_keypoints

    @staticmethod
    def _compute_hand_bbox(
        *,
        hand_uv: Float32[ndarray, "hand 2"],
        image_shape: tuple[int, int],
        expansion_ratio: float,
    ) -> Float32[ndarray, "4"] | None:
        return compute_square_bbox_from_uv(
            hand_uv=hand_uv,
            image_shape=image_shape,
            expansion_ratio=expansion_ratio,
        )

    def extrapolate_3d_keypoints(
        self,
        xyzc_t: Float32[ndarray, "n_kpts 4"],
        xyzc_t1: Float32[ndarray, "n_kpts 4"],
    ) -> Float32[ndarray, "n_kpts 4"]:
        """Linearly extrapolate keypoints from the previous two frames."""
        xyzc_extrap: Float32[ndarray, "n_kpts 4"] = 2 * xyzc_t - xyzc_t1
        return xyzc_extrap
