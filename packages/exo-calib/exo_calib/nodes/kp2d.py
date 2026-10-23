"""Stage B: lockstep multi-view COCO-133 keypoints of one person, with boxes tracked through a camera rig.

The box per view is the projection of the skeleton tracked in 3D. YOLOX detects where that projection is missing or clipped,
on every view every ``DETECTOR_REANCHOR_INTERVAL_S`` of capture time, and again at the next frameset on a view whose projected
box disagrees with its detection (``TRACK_AGREE_IOU``) or whose keypoints do not see the body (``sees_body``): a camera Stage A
placed badly projects the person beside them. Every detection is the tracked person's (``nearest_candidates`` of its centre,
which ``vote_person`` sets once from the detections). RTMW-x
(TensorRT) produces the keypoints. Models come from the graph resources ``detector`` (posekit ``PersonDetector``) and
``pose`` (posekit ``RtmPose2d``); the tracking rig arrives as a STATIC AsOf value (Stage A on pass 1, the pass-1 refined rig
on pass 2).
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import torch
from dagkit.node import Emit, Fenced, PortType, RunContext, Tick
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from posekit.predictions import BoxDetections
from simplecv.gpu import consume
from simplecv.types.frames import MultiViewFrames
from simplecv.types.rig import RigCameras
from torch import Tensor

from exo_calib.ops.boxes import (
    box_iou,
    boxes_from_projected_keypoints,
    boxes_needing_detection,
    crop_rectangles,
    extrapolate_keypoints,
    nearest_candidates,
    skeleton_centre,
    vote_person,
)
from exo_calib.ops.confidence import BODY_JOINTS, sees_body
from exo_calib.ops.lens import undistort
from exo_calib.ops.triangulation import triangulate_frame_keypoints
from exo_calib.types import BoxSource, MultiViewKeypoints, RigLayout

if TYPE_CHECKING:
    from posekit.models.base import PersonDetector
    from posekit.models.rtmpose import RtmPose2d

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
TRACK_AGREE_IOU: float = 0.5
"""A view whose detection overlaps its projected box less than this detects again at the next frameset."""
"""COCO-133 lists the COCO-17 body joints first."""
TRACKING_VIEW_GATE: float = 0.03
"""The tracker drops a view whose joints reproject farther than this fraction of the image diagonal on average (another person)."""


class TrackedFrame(NamedTuple):
    """The online tracker's memory of one processed frame."""

    sample_idx: int
    points_xyz: Float64[ndarray, "133 3"]
    conf: Float64[ndarray, "133"]


class Kp2dNode:
    """Find 2D body keypoints in every view."""

    inputs: Mapping[str, PortType] = {"frames": Fenced[MultiViewFrames], "tracking_rig": RigCameras}
    outputs: Mapping[str, PortType] = {"kp2d": MultiViewKeypoints}

    def __init__(self, layout: RigLayout) -> None:
        self.layout: RigLayout = layout
        """The cameras' lenses: detections are undistorted before they are matched to the tracked person through the rig."""

    def reset(self, ctx: RunContext) -> None:
        self.detector, self.pose = cast("PersonDetector", ctx.resources["detector"]), cast("RtmPose2d", ctx.resources["pose"])
        self.history: list[TrackedFrame] = []
        self.last_anchor_ns: int | None = None
        self.person_xyz: Float64[ndarray, "3"] | None = None
        """The tracked person's centre: voted once from the detections, then moved by every frameset with body joints."""
        self.redetect: Bool[ndarray, " v"] | None = None
        """Views to detect at the next frameset (None before the first, which detects in every view)."""

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, Fenced) and isinstance(tick.value.value, MultiViewFrames)
        frames: MultiViewFrames = consume(tick.value)
        tracking = tick.asof["tracking_rig"]
        assert isinstance(tracking, RigCameras), "kp2d needs the tracking rig (a STATIC AsOf value)"
        rig: RigCameras = tracking.select(frames.views)
        frames_hwc: UInt8[Tensor, "v h w 3"] = frames.rgb.permute(0, 2, 3, 1).contiguous()
        num_views, image_wh = len(frames.views), (int(frames_hwc.shape[2]), int(frames_hwc.shape[1]))
        boxes: Float64[ndarray, "v 4"] = np.full((num_views, 4), np.nan, dtype=np.float64)
        box_source: list[BoxSource] = ["none"] * num_views

        if self.history:
            latest: TrackedFrame = self.history[-1]
            predicted_xyz: Float64[ndarray, "133 3"] = latest.points_xyz
            if len(self.history) >= 2:
                interval: int = latest.sample_idx - self.history[-2].sample_idx
                step_ratio: float = (frames.frame_index - latest.sample_idx) / float(interval) if interval > 0 else 1.0
                predicted_xyz = extrapolate_keypoints(latest.points_xyz, self.history[-2].points_xyz, step_ratio=step_ratio)
            if (centre := skeleton_centre(predicted_xyz[:BODY_JOINTS], latest.conf[:BODY_JOINTS], BOX_CONFIDENCE_TAU)) is not None:
                self.person_xyz = centre
            for view in range(num_views):
                boxes[view] = boxes_from_projected_keypoints(
                    predicted_xyz, latest.conf, rig.intrinsics[view], rig.cam_T_world[view], image_wh,
                    pad=BOX_PADDING, min_joints=BOX_MIN_JOINTS, min_confidence=BOX_CONFIDENCE_TAU,
                )  # fmt: skip
                if np.isfinite(boxes[view]).all():
                    box_source[view] = "tracked"

        projected: Float64[ndarray, "v 4"] = boxes.copy()
        detect: Bool[ndarray, "v"] = boxes_needing_detection(boxes, image_wh)
        if self.redetect is not None:
            detect |= self.redetect
        if self.last_anchor_ns is None or tick.t_ns - self.last_anchor_ns >= int(DETECTOR_REANCHOR_INTERVAL_S * 1e9):
            detect[:] = True
            self.last_anchor_ns = tick.t_ns
        chosen: Int64[ndarray, " v"] = np.full(num_views, -1, dtype=np.int64)
        detect_idx: Int64[ndarray, "d"] = np.flatnonzero(detect).astype(np.int64)
        if detect_idx.size:
            detections: BoxDetections = self.detector(frames_hwc[torch.as_tensor(detect_idx, device=frames_hwc.device)])
            view_of: Int64[ndarray, "n"] = detect_idx[detections.frame_indices.cpu().numpy()]
            xyxy: Float64[ndarray, "n 4"] = detections.xyxy_numpy().astype(np.float64)
            pinhole: Float64[ndarray, "n 4"] = xyxy.copy()
            for view in np.unique(view_of).tolist():
                mine: Bool[ndarray, "n"] = view_of == view
                pinhole[mine] = undistort(self.layout.lens(frames.views[view]), xyxy[mine].reshape(-1, 2), rig.intrinsics[view]).reshape(-1, 4)
            scores: Float64[ndarray, "n"] = detections.scores.cpu().numpy().astype(np.float64)
            projection: Float64[ndarray, "v 3 4"] = rig.intrinsics @ rig.cam_T_world[:, :3]
            if self.person_xyz is None:
                self.person_xyz = vote_person(pinhole, scores, view_of, projection)
            if self.person_xyz is not None:
                chosen = nearest_candidates(self.person_xyz[None], pinhole, view_of, projection)[0][0]
            for view in np.flatnonzero(chosen >= 0).tolist():
                boxes[view], box_source[view] = xyxy[chosen[view]], "yolox"
        agrees: Bool[ndarray, "v"] = (chosen >= 0) & (box_iou(projected, boxes) >= TRACK_AGREE_IOU)

        kp_xy: Float32[ndarray, "v 133 2"] = np.full((num_views, 133, 2), np.nan, dtype=np.float32)
        conf: Float32[ndarray, "v 133"] = np.zeros((num_views, 133), dtype=np.float32)
        usable: Int64[ndarray, "u"] = np.flatnonzero(np.isfinite(boxes).all(axis=1)).astype(np.int64)
        if usable.size:
            device: torch.device = frames_hwc.device
            pose_boxes = BoxDetections(
                xyxy=torch.as_tensor(boxes[usable], dtype=torch.float32, device=device),
                scores=torch.ones(usable.size, dtype=torch.float32, device=device),
                frame_indices=torch.as_tensor(usable, dtype=torch.long, device=device),
            )
            keypoints = self.pose(frames_hwc, pose_boxes)
            pose_rows: Int64[ndarray, "u"] = keypoints.frame_indices.cpu().numpy().astype(np.int64, copy=False)
            kp_xy[pose_rows] = keypoints.xy.cpu().numpy().astype(np.float32, copy=False)
            conf[pose_rows] = keypoints.scores.cpu().numpy().astype(np.float32, copy=False)

        self.redetect = (detect & ~agrees) | ~sees_body(conf.astype(np.float64))
        tracked_xyz, tracked_conf = triangulate_frame_keypoints(
            kp_xy.astype(np.float64), conf.astype(np.float64), rig.intrinsics, rig.cam_T_world,
            min_confidence=BOX_CONFIDENCE_TAU, reproj_threshold_px=TRACKING_REPROJECTION_THRESHOLD_PX, view_gate_px=TRACKING_VIEW_GATE * float(np.hypot(*image_wh)),
        )  # fmt: skip
        if int(np.isfinite(tracked_xyz).all(axis=1).sum()) >= BOX_MIN_JOINTS:
            self.history = [*self.history[-1:], TrackedFrame(frames.frame_index, tracked_xyz, tracked_conf)]
        bbox_xyxy: Float32[ndarray, "v 4"] = boxes.astype(np.float32)
        crop_origin_xy, crop_size_wh = crop_rectangles(bbox_xyxy, self.pose.crop_spec)
        out.put("kp2d", tick.t_ns, MultiViewKeypoints(
            frames.frame_index, frames.views, frames.cam_t_ns.cpu().numpy().astype(np.int64), kp_xy, conf, bbox_xyxy, tuple(box_source),
            crop_origin_xy, crop_size_wh, self.pose.crop_spec.input_size, image_wh,
        ))  # fmt: skip

    def flush(self, out: Emit) -> None:
        pass
