"""Scores of one segment's ``SegmentTrack`` against its ground truth, and their exact aggregation over many segments.

Protocol (our choices where the paper is silent):

- **Ground-truth hand:** labelled (UmeTrack confidence 1) with a pose, on a frame with a tracked headset.
- **MKPE / MKA / MKA GT:** over the frames where the hand is both a ground-truth hand and tracked (the paper averages
  "over all frames"; a frame we do not track has no estimate, so it is left out here and counted by the tracking
  coverage instead). MKA uses only triples of consecutive such frames (``eval.metrics.pipeline_metrics``).
- **Visible hand** (tracking statistics): a ground-truth hand present in at least one camera, i.e. >= 17 of its 21
  keypoints in front and inside that image (DetNet's present rule). ``tracked_absent`` counts tracked frames where no
  ground-truth keypoint is inside any image (or the hand is unlabelled).
- **DetNet precision/recall (§5.4), per camera, both hands pooled, net frame:** with tracking, every box the pipeline put
  in a camera (DetNet's or the tracked projection) is a positive prediction; DetNet alone, DetNet's box where its presence
  exceeds 0.5. A prediction on a partly visible hand (1-16 keypoints inside) counts as a false positive, as the rule reads.
  That literal rule (``eval.metrics.detection_metrics``) is the headline. A diagnostic variant (``*_crop`` fields) tests
  containment against the box enlarged x1.2 about its centre, the region KeyNet actually sees; its width test stays on
  the unenlarged box. The literal containment test fails at a ~2 px offset of a tight box.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64
from serde import serde
from torch import Tensor

from handtrack.data.segment_labels import SegmentLabels
from handtrack.eval.metrics import Counts, DetectionMetrics, PipelineMetrics, TrackingMetrics, detection_metrics, pipeline_metrics, tracking_metrics
from handtrack.geometry.letterbox import Letterbox
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.crops import BOX_ENLARGE
from handtrack.labels.validity import MIN_VISIBLE_KEYPOINTS
from handtrack.pipeline import net_circles
from handtrack.results import SegmentTrack


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PositionScore:
    """MKPE, MKA and MKA GT with the counts they average over, so segments aggregate exactly."""

    mkpe_mm: float | None
    mka_mm: float | None
    """mm / frame²."""
    mka_gt_mm: float | None
    keypoints: int
    """Scored keypoint positions (21 per hand and frame)."""
    accelerations: int
    """Scored keypoint accelerations (21 per hand and triple of frames)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrackingScore:
    """One hand's tracking statistics (``eval.metrics.tracking_metrics``)."""

    gt_frames: int
    visible_frames: int
    tracked_frames: int
    visible_tracked_frames: int
    visible_tracked_fraction: float | None
    acquire_frames: list[int | None]
    """Delay from each appearance to the first tracked frame; None: never acquired in that appearance."""
    drop_frames: list[int | None]
    """Delay from each disappearance to the first untracked frame; None: not dropped before it reappeared or the sequence ended."""
    tracked_without_hand: int
    """Tracked frames where the hand is not visible (present in no camera)."""
    tracked_absent: int
    """Tracked frames with no ground-truth keypoint inside any image."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class HandScore:
    side: str
    position: PositionScore
    tracking: TrackingScore


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetectionScore:
    """§5.4 counts of one camera, both hands pooled."""

    camera: int
    true_positive: int
    predicted: int
    ground_truth: int
    precision: float | None
    recall: float | None


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    """``<segment>.metrics.json``: one segment, one hand mode."""

    segment: str
    domain: str
    interaction: str
    hand_mode: str
    hand_scale: float
    frames: int
    detector: str
    keypoints: str
    detnet_sha256: str
    keynet_sha256: str
    position: PositionScore
    """Both hands."""
    hands: list[HandScore]
    """Left, right."""
    detnet_with_tracking: list[DetectionScore]
    """Per camera, §5.4 as written (the headline)."""
    detnet_with_tracking_crop: list[DetectionScore]
    """Per camera, the diagnostic variant: containment in the x1.2 crop box."""
    keynet_views: int
    """KeyNet crops run."""
    detnet_runs: int
    """Frames on which the tracker ran DetNet (on one camera)."""
    track_sha256: str
    """sha256 of the ``SegmentTrack`` npz these scores were computed from."""
    timings_s: dict[str, float]
    calibration_blocks: int = 0
    """Unknown hand: the stereo (hand, frame) observations ϕ was solved on."""
    calibration_note: str = "profile"
    """Where ϕ came from: ``profile``, ``calibrated``, ``clamped from <ϕ>`` or ``generic fallback: <reason>``."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetNetAloneMetrics:
    """``<segment>.metrics.json`` of the DetNet-alone record."""

    segment: str
    domain: str
    interaction: str
    frames: int
    detnet_sha256: str
    per_camera: list[DetectionScore]
    """§5.4 as written (the headline)."""
    per_camera_crop: list[DetectionScore]
    """The diagnostic variant: containment in the x1.2 crop box."""
    track_sha256: str


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The ground truth of the first f frames of a segment, as the scores need it (c cameras, slot 0 = left)."""

    landmarks: Float32[Tensor, "f 2 21 3"]
    valid: Bool[Tensor, "f 2"]
    """A ground-truth hand: labelled, with a pose, on a frame with a headset pose."""
    net_xy: Float32[Tensor, "f c 2 21 2"]
    in_front: Bool[Tensor, "f c 2 21"]
    """In front of the camera and a ground-truth hand."""
    inside: Int64[Tensor, "f c 2"]
    """Keypoints in front and inside the image, for ground-truth hands (0 otherwise)."""
    image_valid: Bool[Tensor, "f c"]

    @staticmethod
    def from_labels(labels: SegmentLabels, frames: int) -> "GroundTruth":
        has_pose: Bool[Tensor, "f 2"] = torch.isfinite(labels.landmarks[:frames]).all(dim=-1).all(dim=-1)
        valid: Bool[Tensor, "f 2"] = labels.labelled[:frames, 0] & has_pose & labels.image_valid[:frames, 0, None]
        return GroundTruth(
            landmarks=labels.landmarks[:frames],
            valid=valid,
            net_xy=labels.projection.net_xy[:frames],
            in_front=labels.projection.in_front[:frames] & valid[:, None, :, None],
            inside=torch.where(valid[:, None, :], labels.projection.visible[:frames], 0),
            image_valid=labels.image_valid[:frames],
        )


def position_score(predicted: Float32[Tensor, "f h 21 3"], target: Float32[Tensor, "f h 21 3"], scored: Bool[Tensor, "f h"]) -> PositionScore:
    """``pipeline_metrics`` with the counts behind each mean."""
    metrics: PipelineMetrics = pipeline_metrics(predicted, target, scored)
    triples: int = int((scored[:-2] & scored[1:-1] & scored[2:]).sum()) if scored.shape[0] >= 3 else 0
    return PositionScore(metrics.mkpe_mm, metrics.mka_mm, metrics.mka_gt_mm, int(scored.sum()) * 21, triples * 21)


def detection_metrics_in_box(
    boxes: Float32[Tensor, "n 4"],
    probability: Float32[Tensor, "n"],
    points: Float32[Tensor, "n 21 2"],
    in_front: Bool[Tensor, "n 21"],
    camera: Int64[Tensor, "n"],
    hand: Int64[Tensor, "n"],
    containment_scale: float,
) -> DetectionMetrics:
    """``eval.metrics.detection_metrics`` with its containment test against the box scaled by ``containment_scale`` about its centre.

    Every other criterion is the same (presence > 0.5, width within 20% of the enclosing-circle diameter, > 16 keypoints
    inside the 640x480 frame); at ``containment_scale = 1`` this is exactly ``detection_metrics``.
    """
    circles: Float32[Tensor, "n 3"] = torch.from_numpy(enclosing_circles(points.detach().cpu().numpy(), in_front.cpu().numpy())).to(points.device)
    inside: Bool[Tensor, "n 21"] = in_front & (points >= 0).all(-1) & (points[..., 0] < 640) & (points[..., 1] < 480)
    eligible: Bool[Tensor, "n"] = inside.sum(-1) > 16
    width: Float32[Tensor, "n"] = boxes[:, 2] - boxes[:, 0]
    diameter: Float32[Tensor, "n"] = 2 * circles[:, 2]
    centre: Float32[Tensor, "n 2"] = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    half: Float32[Tensor, "n 2"] = (boxes[:, 2:] - boxes[:, :2]) * (0.5 * containment_scale)
    low: Float32[Tensor, "n 2"] = centre - half
    high: Float32[Tensor, "n 2"] = centre + half
    contains: Bool[Tensor, "n"] = (((points >= low[:, None]) & (points <= high[:, None])).all(-1) | ~in_front).all(-1)
    positive: Bool[Tensor, "n"] = probability > 0.5
    tp: Bool[Tensor, "n"] = positive & eligible & contains & (diameter > 0) & ((width - diameter).abs() <= diameter * 0.2)
    tp &= torch.isfinite(boxes).all(-1) & torch.isclose(width, boxes[:, 3] - boxes[:, 1])
    groups: dict[tuple[int, int], Counts] = {}
    for key in {(int(c), int(h)) for c, h in zip(camera.tolist(), hand.tolist(), strict=True)}:
        select: Bool[Tensor, "n"] = (camera == key[0]) & (hand == key[1])
        groups[key] = Counts(int(tp[select].sum()), int(positive[select].sum()), int(eligible[select].sum()))
    return DetectionMetrics(Counts(int(tp.sum()), int(positive.sum()), int(eligible.sum())), groups)


def detection_scores(
    truth: GroundTruth, circle: Float32[Tensor, "f c 2 3"], probability: Float32[Tensor, "f c 2"], cameras: int, containment_scale: float = 1.0
) -> list[DetectionScore]:
    """§5.4 per camera over the frames with a valid image; ``circle`` is the predicted hand circle in the net frame.

    ``containment_scale = 1`` is the rule as written (``eval.metrics.detection_metrics``); anything else is the diagnostic variant.
    """
    keep: Bool[Tensor, "f c 2"] = truth.image_valid[..., None].expand_as(probability)
    boxes: Float32[Tensor, "n 4"] = torch.cat([circle[..., :2] - circle[..., 2:], circle[..., :2] + circle[..., 2:]], dim=-1)[keep]
    camera: Int64[Tensor, "n"] = torch.arange(cameras)[None, :, None].expand_as(probability)[keep]
    hand: Int64[Tensor, "n"] = torch.arange(2)[None, None, :].expand_as(probability)[keep]
    points: Float32[Tensor, "n 21 2"] = torch.nan_to_num(truth.net_xy[keep], nan=0.0)
    metrics: DetectionMetrics = (
        detection_metrics(boxes, probability[keep], points, truth.in_front[keep], camera, hand)
        if containment_scale == 1.0
        else detection_metrics_in_box(boxes, probability[keep], points, truth.in_front[keep], camera, hand, containment_scale)
    )
    scores: list[DetectionScore] = []
    for index in range(cameras):
        counts: Counts = sum((value for key, value in metrics.by_camera_hand.items() if key[0] == index), Counts())
        scores.append(DetectionScore(index, counts.true_positive, counts.predicted, counts.ground_truth, counts.precision, counts.recall))
    return scores


def score_track(
    track: SegmentTrack, labels: SegmentLabels, letterboxes: tuple[Letterbox, ...]
) -> tuple[PositionScore, list[HandScore], list[DetectionScore], list[DetectionScore]]:
    """MKPE/MKA for both hands and per hand, the tracking statistics per hand, and DetNet P/R with tracking per camera (as written, and in the crop)."""
    frames: int = track.tracked.shape[0]
    truth: GroundTruth = GroundTruth.from_labels(labels, frames)
    tracked: Bool[Tensor, "f 2"] = torch.from_numpy(track.tracked)
    predicted: Float32[Tensor, "f 2 21 3"] = torch.from_numpy(track.landmarks)
    scored: Bool[Tensor, "f 2"] = truth.valid & tracked
    visible: Bool[Tensor, "f 2"] = (truth.inside >= MIN_VISIBLE_KEYPOINTS).any(dim=1)
    anywhere: Bool[Tensor, "f 2"] = (truth.inside > 0).any(dim=1)
    hands: list[HandScore] = []
    for side, name in enumerate(("left", "right")):
        stats: TrackingMetrics = tracking_metrics(visible[:, side], tracked[:, side])
        hands.append(
            HandScore(
                side=name,
                position=position_score(predicted[:, side : side + 1], truth.landmarks[:, side : side + 1], scored[:, side : side + 1]),
                tracking=TrackingScore(
                    gt_frames=int(truth.valid[:, side].sum()),
                    visible_frames=int(visible[:, side].sum()),
                    tracked_frames=int(tracked[:, side].sum()),
                    visible_tracked_frames=int((visible[:, side] & tracked[:, side]).sum()),
                    visible_tracked_fraction=stats.visible_tracked_fraction,
                    acquire_frames=list(stats.acquire_frames),
                    drop_frames=list(stats.drop_frames),
                    tracked_without_hand=stats.tracked_without_hand,
                    tracked_absent=int((tracked[:, side] & ~anywhere[:, side]).sum()),
                ),
            )
        )
    circle: Float32[Tensor, "f c 2 3"] = net_circles(letterboxes, torch.from_numpy(track.box))
    boxed: Float32[Tensor, "f c 2"] = torch.from_numpy(track.box_source > 0).float()
    finite_circle: Float32[Tensor, "f c 2 3"] = torch.nan_to_num(circle, nan=0.0)
    return (
        position_score(predicted, truth.landmarks, scored),
        hands,
        detection_scores(truth, finite_circle, boxed, len(letterboxes)),
        detection_scores(truth, finite_circle, boxed, len(letterboxes), BOX_ENLARGE),
    )


def score_detnet_alone(
    track: SegmentTrack, labels: SegmentLabels, letterboxes: tuple[Letterbox, ...]
) -> tuple[list[DetectionScore], list[DetectionScore]]:
    """DetNet-alone §5.4 per camera, as written and in the crop; its probability is ``track.presence``."""
    frames: int = track.presence.shape[0]
    truth: GroundTruth = GroundTruth.from_labels(labels, frames)
    circle: Float32[Tensor, "f c 2 3"] = torch.nan_to_num(net_circles(letterboxes, torch.from_numpy(track.box)), nan=0.0)
    probability: Float32[Tensor, "f c 2"] = torch.from_numpy(np.where(np.isfinite(track.box).all(axis=-1), track.presence, 0.0).astype(np.float32))
    return (
        detection_scores(truth, circle, probability, len(letterboxes)),
        detection_scores(truth, circle, probability, len(letterboxes), BOX_ENLARGE),
    )


def combine_positions(scores: list[PositionScore]) -> PositionScore:
    """Exact pooled means over segments (keypoint-weighted)."""
    keypoints: int = sum(score.keypoints for score in scores)
    accelerations: int = sum(score.accelerations for score in scores)

    def pooled(values: list[tuple[float | None, int]], total: int) -> float | None:
        return sum(value * count for value, count in values if value is not None) / total if total else None

    return PositionScore(
        mkpe_mm=pooled([(score.mkpe_mm, score.keypoints) for score in scores], keypoints),
        mka_mm=pooled([(score.mka_mm, score.accelerations) for score in scores], accelerations),
        mka_gt_mm=pooled([(score.mka_gt_mm, score.accelerations) for score in scores], accelerations),
        keypoints=keypoints,
        accelerations=accelerations,
    )


def combine_detections(scores: list[list[DetectionScore]]) -> list[DetectionScore]:
    """Sum the per-camera counts of many segments."""
    cameras: int = max((len(score) for score in scores), default=0)
    combined: list[DetectionScore] = []
    for camera in range(cameras):
        counts: Counts = sum(
            (Counts(s[camera].true_positive, s[camera].predicted, s[camera].ground_truth) for s in scores if len(s) > camera), Counts()
        )
        combined.append(DetectionScore(camera, counts.true_positive, counts.predicted, counts.ground_truth, counts.precision, counts.recall))
    return combined
