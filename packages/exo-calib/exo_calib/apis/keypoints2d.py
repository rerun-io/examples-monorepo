"""Stage B: per-view 2D whole-body keypoints on the exo videos.

All exo cameras are processed in lockstep over the whole capture: frames stream
from the catalog (NVDEC), the person box per view comes from projecting the
skeleton tracked in 3D (YOLOX detects when a projection is missing or clipped,
and re-anchors every view at least every ``DETECTOR_REANCHOR_INTERVAL_S`` of
capture time), and RTMW-x (TensorRT) produces COCO-133 keypoints with
confidences. Everything downstream needs — raw keypoints, confidences, boxes,
and the crop rectangles the AssemblyHands-X margin rule operates on — is logged
into the ``exocalib_kp2d`` layer's ``…_raw`` entities, which the refine stage
queries back: the catalog is the stage store, not a side channel.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import rerun as rr
import torch
import tyro
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from posekit.models.base import PersonDetector
from posekit.models.rtmpose import RtmPose2d
from posekit.predictions import BoxDetections
from simplecv.rerun_custom_types import Points2DWithConfidence
from torch import Tensor

from exo_calib.boxes import boxes_from_projected_keypoints, boxes_needing_detection, crop_rectangles, extrapolate_keypoints
from exo_calib.cameras import RigCameras
from exo_calib.catalog_io import StageConfig, StageContext, stage_context
from exo_calib.keypoints import (
    AHX_CONF_TAU,
    AHX_MARGIN_PX,
    AHX_MEDIAN_WINDOW,
    STAGE_B_FRAME_BUDGET,
    BoxSource,
    CameraKeypoints,
    GatedKeypoints,
    postprocess_confidences,
    sampled_frame_indices,
)
from exo_calib.kp2d_layer import KP2D_LABEL, RAW_COLOR, kp2d_entity, log_keypoints_record
from exo_calib.layer_io import (
    KP2D_LAYER,
    TIMELINE,
    CalibrationVariant,
    ClassSpec,
    log_coco133_class_context,
    new_layer_recording,
    read_rig_cameras,
    register_layer,
)
from exo_calib.triangulation import triangulate_frame_keypoints
from exo_calib.video_io import ExoVideoStreams, open_exo_streams

DETECTION_SCORE_THRESHOLD: float = 0.1
"""Minimum YOLOX person score."""
DETECTOR_REANCHOR_INTERVAL_S: float = 2.0
"""Run YOLOX on every view at least this often in capture time, so a drifting tracked box cannot survive indefinitely."""
BOX_PADDING: float = 1.25
"""Padding multiplier applied about projected-keypoint box centers."""
BOX_MIN_JOINTS: int = 4
"""Minimum confident projected joints inside a view before its tracked box is used."""
BOX_CONFIDENCE_TAU: float = 0.15
"""Minimum confidence for joints used by tracking triangulation and box projection."""
TRACKING_REPROJECTION_THRESHOLD_PX: float = 100.0
"""Robust-triangulation reprojection threshold used by the online tracker."""
OVERLAY_LOG_STRIDE: int = 5
"""Every N-th processed frame gets the gated / rejected keypoint overlay; the raw record covers every frame."""
OVERLAY_COLOR: tuple[int, int, int] = (240, 140, 60)


@dataclass
class Keypoints2dConfig:
    """Config for the Stage B 2D keypoint tool."""

    stage: tyro.conf.OmitArgPrefixes[StageConfig] = field(default_factory=StageConfig)
    """Shared stage flags (catalog, dataset, segment, rigs, output, register); the CLI shows them unprefixed."""
    batch_size: int = 32
    """Largest inference batch the TensorRT engines are built for (at least the number of exo views)."""


class TrackedFrame(NamedTuple):
    """The online tracker's memory of one processed frame."""

    sample_idx: int
    points_xyz: Float64[ndarray, "133 3"]
    conf: Float64[ndarray, "133"]


def _best_box_per_frame(detections: BoxDetections, num_frames: int) -> Float32[ndarray, "t 4"]:
    """Pick the highest-scoring box per frame; NaN rows where a frame has none."""
    xyxy: Float32[ndarray, "n 4"] = detections.xyxy.cpu().numpy()
    scores: Float32[ndarray, "n"] = detections.scores.cpu().numpy()
    frame_indices: Int64[ndarray, "n"] = detections.frame_indices.cpu().numpy()
    best_xyxy: Float32[ndarray, "t 4"] = np.full((num_frames, 4), np.nan, dtype=np.float32)
    for frame in range(num_frames):
        rows: Int64[ndarray, " m"] = np.nonzero(frame_indices == frame)[0]
        if rows.size:
            best_xyxy[frame] = xyxy[int(rows[np.argmax(scores[rows])])]
    return best_xyxy


def track_multiview_keypoints(streams: ExoVideoStreams, detector: PersonDetector, pose: RtmPose2d, cameras: RigCameras) -> dict[str, CameraKeypoints]:
    """Run lockstep multi-view pose with reprojection-driven person boxes.

    Args:
        streams: Open NVDEC decoders over the segment.
        detector: Person detector used for initialization and re-anchoring.
        pose: RTMW-x top-down pose model (its crop spec sizes the crop rectangles).
        cameras: Pipeline cameras used for triangulation and projection.

    Returns:
        Raw per-camera keypoints, boxes, crop rectangles, and box provenance.

    Raises:
        ValueError: If the exo cameras do not share one video resolution.
    """
    # One index set for every view, spread over the shortest stream so each index exists in all of them.
    sample_indices: Int64[ndarray, "t"] = sampled_frame_indices(min(len(times) for times in streams.times_ns), STAGE_B_FRAME_BUDGET)
    frame_times_ns: Int64[ndarray, "t"] = streams.times_ns[0][sample_indices].astype("timedelta64[ns]").astype(np.int64)
    num_frames: int = sample_indices.size
    num_views: int = len(streams.names)
    kp_xy: Float32[ndarray, "v t 133 2"] = np.full((num_views, num_frames, 133, 2), np.nan, dtype=np.float32)
    conf: Float32[ndarray, "v t 133"] = np.zeros((num_views, num_frames, 133), dtype=np.float32)
    bbox_xyxy: Float32[ndarray, "v t 4"] = np.full((num_views, num_frames, 4), np.nan, dtype=np.float32)
    box_sources: list[list[BoxSource]] = [[] for _ in range(num_views)]
    video_wh: Int64[ndarray, "2"] | None = None
    history: list[TrackedFrame] = []
    last_anchor_ns: int | None = None
    reanchor_interval_ns: int = int(DETECTOR_REANCHOR_INTERVAL_S * 1e9)

    for frame_idx, sample_idx_value in enumerate(sample_indices):
        sample_idx: int = int(sample_idx_value)
        frames: list[UInt8[Tensor, "3 h w"]] = [streams.decoders[view_idx].get_frames_at([sample_idx]).data[0] for view_idx in range(num_views)]
        if len({tuple(frame.shape) for frame in frames}) != 1:
            raise ValueError(f"exo cameras must share one video resolution; got {[tuple(frame.shape[1:]) for frame in frames]}")
        frames_hwc: UInt8[Tensor, "v h w 3"] = torch.stack(frames).permute(0, 2, 3, 1).contiguous()
        frame_wh: Int64[ndarray, "2"] = np.array([frames_hwc.shape[2], frames_hwc.shape[1]], dtype=np.int64)
        video_wh = frame_wh
        image_wh: tuple[int, int] = (int(frame_wh[0]), int(frame_wh[1]))
        boxes: Float64[ndarray, "v 4"] = np.full((num_views, 4), np.nan, dtype=np.float64)
        frame_box_sources: list[BoxSource] = ["none"] * num_views

        if history:
            latest: TrackedFrame = history[-1]
            predicted_xyz: Float64[ndarray, "133 3"] = latest.points_xyz
            if len(history) >= 2:
                previous: TrackedFrame = history[-2]
                history_interval: int = latest.sample_idx - previous.sample_idx
                step_ratio: float = (sample_idx - latest.sample_idx) / float(history_interval) if history_interval > 0 else 1.0
                predicted_xyz = extrapolate_keypoints(latest.points_xyz, previous.points_xyz, step_ratio=step_ratio)
            for view_idx in range(num_views):
                boxes[view_idx] = boxes_from_projected_keypoints(
                    predicted_xyz,
                    latest.conf,
                    cameras.intrinsics[view_idx],
                    cameras.cam_T_world[view_idx],
                    image_wh,
                    pad=BOX_PADDING,
                    min_joints=BOX_MIN_JOINTS,
                    min_confidence=BOX_CONFIDENCE_TAU,
                )
                if np.isfinite(boxes[view_idx]).all():
                    frame_box_sources[view_idx] = "tracked"

        detector_view_mask: Bool[ndarray, "v"] = boxes_needing_detection(boxes, image_wh)
        frame_time_ns: int = int(frame_times_ns[frame_idx])
        if last_anchor_ns is None or frame_time_ns - last_anchor_ns >= reanchor_interval_ns:
            detector_view_mask[:] = True
            last_anchor_ns = frame_time_ns
        detector_view_idx: Int64[ndarray, "d"] = np.flatnonzero(detector_view_mask).astype(np.int64)
        if detector_view_idx.size:
            detector_frames_hwc: UInt8[Tensor, "d h w 3"] = frames_hwc[torch.as_tensor(detector_view_idx, dtype=torch.long, device=frames_hwc.device)]
            detected_boxes: Float32[ndarray, "d 4"] = _best_box_per_frame(detector(detector_frames_hwc), detector_view_idx.size)
            for detector_row, view_idx_value in enumerate(detector_view_idx):
                view_idx: int = int(view_idx_value)
                if np.isfinite(detected_boxes[detector_row]).all():
                    boxes[view_idx] = detected_boxes[detector_row]
                    frame_box_sources[view_idx] = "yolox"

        usable_view_idx: Int64[ndarray, "u"] = np.flatnonzero(np.isfinite(boxes).all(axis=1)).astype(np.int64)
        if usable_view_idx.size:
            pose_boxes: BoxDetections = BoxDetections(
                xyxy=torch.as_tensor(boxes[usable_view_idx], dtype=torch.float32, device=frames_hwc.device),
                scores=torch.ones(usable_view_idx.size, dtype=torch.float32, device=frames_hwc.device),
                frame_indices=torch.as_tensor(usable_view_idx, dtype=torch.long, device=frames_hwc.device),
            )
            keypoints = pose(frames_hwc, pose_boxes)
            keypoint_view_idx: Int64[ndarray, "u"] = keypoints.frame_indices.cpu().numpy().astype(np.int64, copy=False)
            kp_xy[keypoint_view_idx, frame_idx] = keypoints.xy.cpu().numpy().astype(np.float32, copy=False)
            conf[keypoint_view_idx, frame_idx] = keypoints.scores.cpu().numpy().astype(np.float32, copy=False)

        bbox_xyxy[:, frame_idx] = boxes.astype(np.float32)
        for view_idx in range(num_views):
            box_sources[view_idx].append(frame_box_sources[view_idx])

        tracked_xyz, tracked_conf = triangulate_frame_keypoints(
            kp_xy[:, frame_idx].astype(np.float64),
            conf[:, frame_idx].astype(np.float64),
            cameras.intrinsics,
            cameras.cam_T_world,
            min_confidence=BOX_CONFIDENCE_TAU,
            reproj_threshold_px=TRACKING_REPROJECTION_THRESHOLD_PX,
        )
        if int(np.isfinite(tracked_xyz).all(axis=1).sum()) >= BOX_MIN_JOINTS:
            history.append(TrackedFrame(sample_idx=sample_idx, points_xyz=tracked_xyz, conf=tracked_conf))
            history = history[-2:]

    assert video_wh is not None  # num_frames >= 1: sampled_frame_indices rejects empty captures
    per_camera: dict[str, CameraKeypoints] = {}
    for view_idx, name in enumerate(streams.names):
        crop_origin_xy, crop_size_wh = crop_rectangles(bbox_xyxy[view_idx], pose.crop_spec)
        per_camera[name] = CameraKeypoints(
            sample_indices=sample_indices,
            times_ns=streams.times_ns[view_idx][sample_indices].astype("timedelta64[ns]").astype(np.int64),
            kp_xy=kp_xy[view_idx],
            conf=conf[view_idx],
            bbox_xyxy=bbox_xyxy[view_idx],
            box_source=tuple(box_sources[view_idx]),
            crop_origin_xy=crop_origin_xy,
            crop_size_wh=crop_size_wh,
            crop_input_wh=np.asarray(pose.crop_spec.input_size, dtype=np.int64),
            video_wh=video_wh,
        )
    return per_camera


def log_keypoints_overlay(per_camera: dict[str, CameraKeypoints], recording: rr.RecordingStream) -> None:
    """Draw the AssemblyHands-X-gated skeleton on every ``OVERLAY_LOG_STRIDE``-th frame, with the rejected joints beside it.

    Visualization only: the refine stage recomputes the same gate from the raw
    record. Joints the gate zeroes are NaN'd on the skeleton entity (the only
    per-keypoint mechanism that also suppresses their edges) and shown raw,
    connectionless, on the ``…_rejected`` sibling.
    """
    for name, cam in per_camera.items():
        entity: str = kp2d_entity(name)
        rejected_entity: str = f"{entity}_rejected"
        log_coco133_class_context(recording, entity, (ClassSpec(0, KP2D_LABEL, OVERLAY_COLOR),))
        log_coco133_class_context(recording, rejected_entity, (ClassSpec(0, f"{KP2D_LABEL} rejected", RAW_COLOR),), connections=False)
        gated: GatedKeypoints = postprocess_confidences(cam, AHX_MEDIAN_WINDOW, AHX_MARGIN_PX, AHX_CONF_TAU)
        for t in range(0, len(cam.sample_indices), OVERLAY_LOG_STRIDE):
            if not np.isfinite(cam.kp_xy[t]).any():
                continue
            keep_mask: Bool[ndarray, "133"] = gated.conf[t] > 0.0
            gated_xy: Float64[ndarray, "133 2"] = gated.xy[t].copy()
            gated_xy[~keep_mask] = np.nan
            rejected_mask: Bool[ndarray, "133"] = ~keep_mask & np.isfinite(cam.kp_xy[t]).all(axis=1)
            recording.set_time(TIMELINE, duration=1e-9 * float(cam.times_ns[t]))
            recording.log(entity, Points2DWithConfidence(positions=gated_xy, confidences=gated.conf[t], class_ids=0, keypoint_ids=np.arange(133), radii=2.0))
            recording.log(
                rejected_entity,
                Points2DWithConfidence(
                    positions=cam.kp_xy[t][rejected_mask],
                    confidences=cam.conf[t][rejected_mask],
                    class_ids=0,
                    keypoint_ids=np.flatnonzero(rejected_mask),
                    radii=1.5,
                ),
            )


def main(config: Keypoints2dConfig, camera_variant: CalibrationVariant = "init") -> None:
    """Run Stage B over all exo cameras and (optionally) register the layer.

    ``camera_variant`` selects the rig the tracker projects through: the Stage A
    estimate on the first pass, the refined rig on the second (the pipeline
    driver runs both).
    """
    from posekit.models.rtmpose import RtmPoseConfig
    from posekit.models.yolox import YoloxDetectorConfig
    from posekit.runtimes import TensorRtBackendConfig

    context: StageContext = stage_context(config.stage)
    names: tuple[str, ...] = context.layout.exo_camera_names
    streams: ExoVideoStreams = open_exo_streams(context.dataset, context.segment_id, names)
    detector: PersonDetector = YoloxDetectorConfig(score_thr=DETECTION_SCORE_THRESHOLD, backend=TensorRtBackendConfig(max_batch_size=config.batch_size)).setup()
    pose: RtmPose2d = RtmPoseConfig(variant="rtmw-x-coco133", backend=TensorRtBackendConfig(max_batch_size=config.batch_size)).setup()
    cameras: RigCameras = read_rig_cameras(context.dataset, context.segment_id, names, source=camera_variant)

    per_camera: dict[str, CameraKeypoints] = track_multiview_keypoints(streams, detector, pose, cameras)
    for name, cam in per_camera.items():
        detected: int = int(np.isfinite(cam.bbox_xyxy).all(axis=1).sum())
        tracked: int = sum(source == "tracked" for source in cam.box_source)
        positive_conf: Float32[ndarray, " n"] = cam.conf[cam.conf > 0]
        mean_conf: float = float(positive_conf.mean()) if positive_conf.size else 0.0
        print(f"{name}: {detected}/{len(cam.sample_indices)} frames with a box (tracked {tracked}, yolox {detected - tracked}), mean conf {mean_conf:.3f}")

    recording, rrd_path = new_layer_recording(context.segment_id, context.segment_dir / f"{KP2D_LAYER}.rrd")
    log_keypoints_record(per_camera, recording)
    log_keypoints_overlay(per_camera, recording)
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.stage.register:
        register_layer(context.dataset, rrd_path, KP2D_LAYER)
        print(f"registered layer {KP2D_LAYER}")
