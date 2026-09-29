"""The reference ladder on one decode shared by DetNet and every crop mode."""
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from handtrack.eval.segment import position_score
from handtrack.models.detnet import Detections
from handtrack.pipeline import SegmentData, net_frames
from handtrack.reference.catalog import ReferenceLabels
from handtrack.reference.results import CameraStatistics, Mode, ReferenceFrames, ReferenceMetrics, circle_statistics, position_statistics
from handtrack.reference.state import TrackEnd, TrackState
from handtrack.reference.upstream import Crops, Pose, PoseStage
from handtrack.tracker import DetNetDetector


@dataclass(frozen=True, slots=True)
class LadderResult:
    metrics: list[ReferenceMetrics]
    """Segment metrics, in the stream's mode-axis order."""
    streams: ReferenceFrames
    """Frame diagnostics shared by all modes."""


def native_images(images: UInt8[Tensor, "4 480 640"], data: SegmentData) -> list[UInt8[ndarray, "h w"]]:
    """Losslessly undo UmeTrack's identity resize and horizontal padding."""
    native: list[UInt8[ndarray, "h w"]] = []
    for index, letterbox in enumerate(data.letterboxes):
        if letterbox.quarter_turn_cw or letterbox.scale != 1.0:
            raise ValueError("Reference eval only supports native UmeTrack letterboxes")
        start: int = int(letterbox.pad_x)
        native.append(np.ascontiguousarray(images[index, :, start:start + letterbox.source_width].cpu().numpy()))
    return native


def run_ladder(data: SegmentData, labels: ReferenceLabels, stages: dict[tuple[Mode, TrackEnd], PoseStage], detector: DetNetDetector | None,
               frames: int, scale: float, miss_frames: int, identity: str, device: torch.device) -> LadderResult:
    """Each mode owns a tracker/network so temporal memory cannot leak across modes."""
    for video in data.videos:
        if not np.array_equal(video.t_ns, labels.times):
            raise ValueError(f"{data.info.segment_id}: video times differ from A6 label times")
    if not np.array_equal(data.timeline.video_time_ns, labels.times):
        raise ValueError("Shared decode and A6 adapter timeline differ")
    predictions: dict[tuple[Mode, TrackEnd], Float32[ndarray, "f 2 21 3"]] = {key: np.full((frames, 2, 21, 3), np.nan, dtype=np.float32) for key in stages}
    previous: dict[tuple[Mode, TrackEnd], dict[int, Pose]] = {key: {} for key in stages}
    states: dict[tuple[Mode, TrackEnd], list[TrackState]] = {key: [TrackState(key[1], miss_frames) for _ in range(2)] for key in stages}
    target: Float32[ndarray, "f 2 21 3"] = np.full((frames, 2, 21, 3), np.nan, dtype=np.float32)
    detected_circles: Float64[ndarray, "f 2 4 3"] = np.full((frames, 2, 4, 3), np.nan)
    presence: Float64[ndarray, "f 2 4"] = np.full((frames, 2, 4), np.nan)
    truth_circles: Float64[ndarray, "f 2 4 3"] = np.full((frames, 2, 4, 3), np.nan)
    visibility: Int64[ndarray, "f 2 4"] = np.full((frames, 2, 4), -1, dtype=np.int64)
    selected: Bool[ndarray, "f m 2 4"] = np.zeros((frames, len(stages), 2, 4), dtype=bool)
    row: int = 0
    for chunk in net_frames(data, frames, device):
        for images in chunk:
            native: list[UInt8[ndarray, "h w"]] = native_images(images, data)
            circles: Float64[ndarray, "4 2 3"] = np.full((4, 2, 3), np.nan)
            probability: Float64[ndarray, "4 2"] = np.zeros((4, 2))
            if detector is not None:
                detection: Detections = detector(images, row, torch.arange(4))
                probability = detection.probability.numpy().astype(np.float64)
                for index, letterbox in enumerate(data.letterboxes):
                    circles[index, :, :2] = letterbox.from_net(detection.circle[index, :, :2]).numpy()
                    circles[index, :, 2] = detection.circle[index, :, 2].numpy() / letterbox.scale
            base: PoseStage = next(iter(stages.values()))
            base.set_frame(row)
            gt: dict[int, Pose] = base.ground_truth(row)
            gt_circles: Float64[ndarray, "4 2 3"] = np.full((4, 2, 3), np.nan)
            visible: Float64[ndarray, "4 2"] = np.zeros((4, 2))
            if labels.tracked[row]:
                gt_circles, visible = base.gt_circles(gt)
                for hand, pose in gt.items():
                    target[row, hand] = base.landmarks(pose, hand) / 1000.0
            detected_circles[row] = circles.transpose(1, 0, 2)
            if detector is not None:
                presence[row] = probability.T
            truth_circles[row] = gt_circles.transpose(1, 0, 2)
            if labels.tracked[row]:
                visibility[row] = visible.T.astype(np.int64)
            for mode_index, (key, stage) in enumerate(stages.items()):
                mode: Mode = key[0]
                stage.set_frame(row)
                crops: Crops = {}
                if labels.tracked[row]:
                    if mode == "gt_pose":
                        crops = stage.pose_crops(gt)
                    elif mode == "gt_circle":
                        gt_scores: Float64[ndarray, "4 2"] = visible.copy()
                        for hand, pose in gt.items():
                            if pose.hand_confidence < 0.5:
                                gt_scores[:, hand] = 0.0
                        crops = stage.circle_crops(gt_circles, gt_scores, scale, gt=True)
                    else:
                        acquired: Crops = stage.circle_crops(circles, probability, scale)
                        crops = acquired
                        if mode == "track":
                            predicted: Crops = stage.pose_crops(previous[key])
                            crops = {}
                            for hand in range(2):
                                was_tracked: bool = states[key][hand].tracked
                                views: list[int] = states[key][hand].choose(list(acquired.get(hand, {})), list(predicted.get(hand, {})), probability[:, hand].tolist())
                                if views:
                                    source = predicted[hand] if was_tracked else acquired[hand]
                                    crops[hand] = {view: source[view] for view in views}
                for hand, cameras in crops.items():
                    selected[row, mode_index, hand, list(cameras)] = True
                poses: dict[int, Pose] = stage.track(native, crops)
                previous[key] = poses
                for hand in range(2):
                    states[key][hand].accept(hand in poses)
                    if hand in poses:
                        predictions[key][row, hand] = stage.landmarks(poses[hand], hand) / 1000.0
            row += 1
    if row != frames:
        raise ValueError(f"Decoded {row} frames, expected {frames}")
    valid: Bool[ndarray, "f 2"] = labels.confidence[:frames] > 0
    metrics: list[ReferenceMetrics] = []
    errors: Float64[ndarray, "f m 2"] = np.full((frames, len(stages), 2), np.nan)
    all_posed: Bool[ndarray, "f m 2"] = np.zeros((frames, len(stages), 2), dtype=bool)
    for mode_index, (key, predicted_points) in enumerate(predictions.items()):
        posed: Bool[ndarray, "f 2"] = np.isfinite(predicted_points).all(axis=(2, 3))
        count: int = int((valid & posed).sum())
        total: int = int(valid.sum())
        errors[:, mode_index] = np.where(valid & posed,
            np.linalg.norm(predicted_points.astype(np.float64) - target.astype(np.float64), axis=-1).mean(axis=-1) * 1000.0, np.nan)
        all_posed[:, mode_index] = posed
        camera_stats: list[CameraStatistics] = circle_statistics(detected_circles, presence, truth_circles, visibility,
            selected[:, mode_index] if key[0] == "detnet" else np.zeros((frames, 2, 4), dtype=bool)) if detector is not None and key[0] != "gt_pose" else []
        samples: int = sum(item.error_pairs for item in camera_stats)
        metrics.append(ReferenceMetrics(data.info.segment_id, key[0], key[1] if key[0] == "track" else "none", identity, frames,
            position_score(torch.from_numpy(predicted_points), torch.from_numpy(target), torch.from_numpy(valid & posed)),
            total, count, int((posed & ~valid).sum()), count / total if total else None,
            samples, sum((item.centre_mean_px or 0.0) * item.error_pairs for item in camera_stats) / samples if samples else None,
            sum((item.radius_mean_px or 0.0) * item.error_pairs for item in camera_stats) / samples if samples else None,
            position_statistics(errors[:, mode_index], posed, valid), camera_stats))
    return LadderResult(metrics, ReferenceFrames(labels.times[:frames], errors, all_posed, valid,
        detected_circles, presence, truth_circles, visibility, selected))
