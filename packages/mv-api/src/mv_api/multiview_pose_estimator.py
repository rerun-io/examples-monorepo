from dataclasses import dataclass
from typing import Literal

import numpy as np
import rerun as rr
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Float64, Int, UInt8
from numpy import ndarray
from rtmlib import YOLOX, RTMPose
from simplecv.camera_parameters import PinholeParameters
from simplecv.data.skeleton.coco133_layers import COCO133_PREDICTION_LAYER_TO_PATH, Coco133AnnotationLayer
from simplecv.data.skeleton.coco_133 import COCO_133_IDS, LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.ops.triangulate import batch_triangulate, projectN3
from simplecv.rerun_custom_types import Points2DWithConfidence, confidence_scores_to_rgb
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


@dataclass(frozen=True, slots=True)
class ModelAssets:
    """ONNX detector and pose model assets."""

    det: str
    """Download URL or local path to the YOLOX ONNX model."""
    det_input_size: tuple[int, int]
    """Input width-height pair expected by YOLOX."""
    pose: str
    """Download URL or local path to the RTMPose ONNX model."""
    pose_input_size: tuple[int, int]
    """Input width-height pair expected by RTMPose."""


MODE: dict[str, ModelAssets] = {
    "performance": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
        pose_input_size=(288, 384),
    ),
    "lightweight": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        det_input_size=(416, 416),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        pose_input_size=(192, 256),
    ),
    "balanced": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
        pose_input_size=(192, 256),
    ),
    "wholebody": ModelAssets(
        det="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        det_input_size=(640, 640),
        pose="https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
        pose_input_size=(192, 256),
    ),
}


@dataclass
class MultiviewBodyTrackerConfig:
    """Configuration options for the multiview body tracker runtime."""

    mode: Literal["lightweight", "balanced", "performance", "wholebody"] = "wholebody"
    """Preset selecting detector and pose assets tuned for latency versus accuracy."""
    backend: Literal["onnxruntime"] = "onnxruntime"
    """Inference backend used to execute ONNX models."""
    device: Literal["cpu", "cuda"] = "cuda"
    """Hardware accelerator requested by the ONNX runtime backend."""
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
    ) -> None:
        """Create detector and pose models for single-person multiview tracking."""
        self.config: MultiviewBodyTrackerConfig = config
        self.num_keypoints: int = 133
        self.filter_body_idxes: Int[ndarray, "idx"] = (
            filter_body_idxes if filter_body_idxes is not None else np.arange(self.num_keypoints, dtype=np.intp)
        )

        assets: ModelAssets = MODE[config.mode]
        self.det_model = YOLOX(
            assets.det,
            model_input_size=assets.det_input_size,
            backend=config.backend,
            device=config.device,
        )
        self.pose_model = RTMPose(
            assets.pose,
            model_input_size=assets.pose_input_size,
            to_openpose=False,
            backend=config.backend,
            device=config.device,
        )
        self.hand_keypoint_engine: WilorHandKeypointDetector | None = None
        if self.config.use_wilor:
            self.hand_keypoint_engine = WilorHandKeypointDetector(HandKeypointDetectorConfig(verbose=False))

    def __call__(
        self,
        *,
        bgr_list: list[UInt8[ndarray, "H W 3"]],
        pinhole_list: list[PinholeParameters],
        pred_state: MVHistory,
        recording: rr.RecordingStream | None = None,
    ) -> MVHistory:
        """Estimate and triangulate one COCO-133 person across camera views."""
        selected_view_indices: list[int] = (
            [idx for idx in self.config.cams_for_detection_idx]
            if self.config.cams_for_detection_idx is not None
            else list(range(len(bgr_list)))
        )
        if not selected_view_indices:
            raise ValueError("At least one camera must be selected for multiview body tracking.")
        for selected_view_idx in selected_view_indices:
            if selected_view_idx < 0 or selected_view_idx >= len(bgr_list) or selected_view_idx >= len(pinhole_list):
                msg: str = f"Selected camera index {selected_view_idx} is outside the available view range."
                raise IndexError(msg)

        selected_bgr_list: list[UInt8[ndarray, "H W 3"]] = [bgr_list[idx] for idx in selected_view_indices]
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

        uvc_list: list[Float32[ndarray, "n_kpts 3"]] = []
        for selected_idx, (original_view_idx, bgr, pinhole) in enumerate(
            zip(selected_view_indices, selected_bgr_list, selected_pinhole_list, strict=True)
        ):
            if xyzc_extrap is not None and bboxes_extrap is not None:
                bboxes: Float32[ndarray, "n_dets 4"] = rearrange(bboxes_extrap[selected_idx], "b -> 1 b")
            else:
                det_output: np.ndarray | tuple[np.ndarray, ...] = self.det_model(bgr)
                det_bboxes_np: np.ndarray = det_output[0] if isinstance(det_output, tuple) else det_output
                bboxes = np.asarray(det_bboxes_np, dtype=np.float32)

            if bboxes.shape[0] == 0:
                uvc_list.append(np.zeros((self.num_keypoints, 3), dtype=np.float32))
                continue

            selected_bboxes: Float32[ndarray, "1 4"] = bboxes[0:1]
            bbox_list: list[list[float]] = selected_bboxes.astype(np.float32).tolist()
            pose_output: tuple[Float64[ndarray, "n_dets n_kpts=133 2"], Float32[ndarray, "n_dets n_kpts=133"]] = (
                self.pose_model(bgr, bboxes=bbox_list)
            )
            keypoints: Float32[ndarray, "n_dets n_kpts=133 2"] = pose_output[0].astype(np.float32)
            scores: Float32[ndarray, "n_dets n_kpts=133"] = pose_output[1]
            filtered_keypoints, filtered_scores = self.filter_kpt_outputs(keypoints, scores)

            if self.config.use_wilor:
                filtered_keypoints = self._refine_hand_keypoints_with_wilor(
                    bgr=bgr,
                    keypoints=filtered_keypoints,
                    confidences=filtered_scores,
                )

            if self.config.verbose:
                self._log_uvc_layer(
                    view_idx=original_view_idx,
                    pinhole=pinhole,
                    keypoints=filtered_keypoints,
                    confidences=filtered_scores,
                    layer=Coco133AnnotationLayer.RAW_2D,
                    mask_below_threshold=True,
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
        recording: rr.RecordingStream | None,
    ) -> None:
        """Log a COCO-133 2D layer for a single view."""
        view_name: str = pinhole.name if pinhole.name else f"view_{view_idx}"
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
        confidence_rgb: UInt8[ndarray, "n_kpts 3"] = confidence_scores_to_rgb(
            filtered_confidences[np.newaxis, :, np.newaxis]
        )[0]
        layer_segment: str = COCO133_PREDICTION_LAYER_TO_PATH[layer]
        rr.log(
            f"/world/exo/{view_name}/pinhole/pred/coco133_uv/{layer_segment}",
            Points2DWithConfidence(
                positions=filtered_keypoints,
                confidences=filtered_confidences,
                class_ids=int(layer),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=confidence_rgb,
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

    def filter_kpt_outputs(
        self,
        keypoints: Float32[ndarray, "n_dets n_kpts 2"],
        scores: Float32[ndarray, "n_dets n_kpts"],
    ) -> tuple[Float32[ndarray, "n_kpts 2"], Float32[ndarray, "n_kpts"]]:
        """Select the detection with the highest maximum keypoint score."""
        max_scores: Float32[ndarray, "n_dets"] = scores.max(axis=1)
        max_idx: int = int(max_scores.argmax())
        return keypoints[max_idx], scores[max_idx]
