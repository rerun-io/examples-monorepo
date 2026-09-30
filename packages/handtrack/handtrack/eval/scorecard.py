"""The fixed scorecard: per-frame errors of a tracked segment against its ground truth, and tail-aware KPIs over whole segments.

Per frame and hand (``FrameScores``): the mean 3D keypoint error, the mean mesh-vertex error (both hands skinned with the same
subject model), whether the track sits closer to the other hand (a swap), whether the hand is required (it has a pose and at
least one camera shows >= 17 of its keypoints) and the share of its keypoints hidden behind a hand surface in its least hidden
camera (``labels.visibility``).

Per segment and hand (``ScoreRow``) and pooled over any set of rows (``pool``): coverage, the error distribution of tracked
frames (median, p90, p95), the share of tracked frames above 50 and 100 mm, catastrophic episodes (runs above 100 mm) per minute,
good frames (required, tracked and under 30 mm) per required frame, restarts and swaps per minute, and jitter. A tracker is judged
first by its tails (p90, >100 mm, episodes) at a coverage no lower than before.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from serde import serde

from handtrack.data.catalog import HandTimeline, SegmentInfo, layout_for, read_hand_timeline, read_rig, read_statics
from handtrack.data.segment_labels import SegmentLabels, segment_labels
from handtrack.hand.pose import HandPose, Side, mesh_vertices
from handtrack.labels.validity import HandLabel

GOOD_MM: float = 30.0
"""A tracked frame this close to the truth counts as good."""
BAD_MM: float = 50.0
CATASTROPHE_MM: float = 100.0
"""A run of tracked frames above this is one catastrophic episode."""


@dataclass(frozen=True, slots=True)
class Truth:
    """What scoring needs from the catalog: the timeline and the per-frame labels (no video)."""

    timeline: HandTimeline
    labels: SegmentLabels
    fps: float


def read_truth(entry: DatasetEntry, info: SegmentInfo) -> Truth:
    """The segment's hand timeline and labels, without reading its video."""
    statics = read_statics(entry, info)
    rig, letterboxes = read_rig(statics, info)
    timeline = read_hand_timeline(entry, info, statics)
    rows: Int64[ndarray, "f"] = np.arange(len(timeline.video_time_ns), dtype=np.int64)
    labels: SegmentLabels = segment_labels(timeline, rig, letterboxes, rows, layout_for(info.dataset).pose_gated)
    return Truth(timeline=timeline, labels=labels, fps=float(info.fps))


@dataclass(frozen=True, slots=True)
class FrameScores:
    """Per frame f and hand (slot 0 = left)."""

    keypoint_mm: Float32[ndarray, "f 2"]
    """Mean 3D keypoint error; NaN where untracked or without truth."""
    vertex_mm: Float32[ndarray, "f 2"]
    """Mean mesh-vertex error; NaN where untracked or without truth."""
    tracked: Bool[ndarray, "f 2"]
    required: Bool[ndarray, "f 2"]
    """Has a pose, is labelled, and at least one camera shows >= 17 of its keypoints."""
    swapped: Bool[ndarray, "f 2"]
    """Tracked and closer to the other hand's truth than to its own."""
    hidden_share: Float32[ndarray, "f 2"]
    """Share of the hand's keypoints hidden behind a hand surface in its least hidden camera that shows it; NaN if none shows it."""
    predicted: Float32[ndarray, "f 2 21 3"]
    """Tracked landmarks, world metres (NaN when untracked)."""
    truth: Float32[ndarray, "f 2 21 3"]


def frame_scores(track: dict[str, ndarray], truth: Truth) -> FrameScores:
    """Score a ``SegmentTrack`` (its npz arrays) frame by frame."""
    frames: int = len(track["tracked"])
    timeline = truth.timeline
    model = timeline.hand_model
    has_pose: Bool[ndarray, "f 2"] = timeline.has_pose[:frames].numpy()
    tracked: Bool[ndarray, "f 2"] = track["tracked"][:frames].astype(bool)
    predicted: Float32[ndarray, "f 2 21 3"] = track["landmarks"][:frames].astype(np.float32)
    labels: SegmentLabels = truth.labels
    truth_points: Float32[ndarray, "f 2 21 3"] = labels.landmarks[:frames].numpy()
    keypoint: Float32[ndarray, "f 2"] = np.linalg.norm(predicted - truth_points, axis=-1).mean(-1) * 1000.0
    other: Float32[ndarray, "f 2"] = np.linalg.norm(predicted - truth_points[:, ::-1], axis=-1).mean(-1) * 1000.0
    vertex: Float32[ndarray, "f 2"] = np.full((frames, 2), np.nan, dtype=np.float32)
    for side in Side:
        rows: Int64[ndarray, "k"] = np.flatnonzero(tracked[:, side] & has_pose[:, side])
        if not len(rows):
            continue
        index = torch.from_numpy(rows)
        fitted = HandPose(torch.from_numpy(track["rotation"][rows, side]), torch.from_numpy(track["translation"][rows, side]),
                          torch.from_numpy(track["joint_angles"][rows, side]))
        pose = timeline.poses[side]
        reference = HandPose(pose.rotation[index], pose.translation[index], pose.joint_angles[index])
        with torch.inference_mode():
            difference = mesh_vertices(model, fitted, side) - mesh_vertices(model, reference, side)
        vertex[rows, side] = difference.norm(dim=-1).mean(dim=-1).numpy() * 1000.0
    scored: Bool[ndarray, "f 2"] = tracked & has_pose & np.isfinite(keypoint)
    present: Bool[ndarray, "f c 2"] = labels.hand_label[:frames].numpy() == int(HandLabel.PRESENT)
    required: Bool[ndarray, "f 2"] = has_pose & labels.labelled[:frames].numpy().any(axis=1) & present.any(axis=1)
    shows: Bool[ndarray, "f c 2"] = labels.projection.visible[:frames].numpy() >= 10
    hidden: Float32[ndarray, "f c 2"] = (labels.projection.hidden[:frames].numpy().mean(-1) if labels.projection.hidden is not None
                                         else np.zeros(shows.shape, dtype=np.float32))
    least: Float32[ndarray, "f 2"] = np.where(shows, hidden, np.inf).min(axis=1)
    return FrameScores(
        keypoint_mm=np.where(scored, keypoint, np.nan).astype(np.float32),
        vertex_mm=np.where(scored, vertex, np.nan).astype(np.float32),
        tracked=tracked,
        required=required,
        swapped=scored & has_pose[:, ::-1] & (other < keypoint),
        hidden_share=np.where(np.isfinite(least), least, np.nan).astype(np.float32),
        predicted=np.where(tracked[..., None, None], predicted, np.nan).astype(np.float32),
        truth=truth_points.astype(np.float32),
    )


def _runs(mask: Bool[ndarray, "f"]) -> Int64[ndarray, "r"]:
    """Lengths of the maximal runs of True."""
    edges: Int64[ndarray, "e"] = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int64), 0]))
    return edges[1::2] - edges[::2]


@serde
@dataclass(frozen=True, slots=True)
class ScoreRow:
    """KPIs of one segment and hand (or pooled: ``segment`` names the pool, ``hand`` is 'both')."""

    dataset: str
    segment: str
    hand: str
    minutes: float
    required_frames: int
    scored_frames: int
    coverage: float
    """Required frames that are tracked."""
    good30: float
    """Required frames that are tracked and under 30 mm."""
    mkpe: float | None
    median: float | None
    p90: float | None
    p95: float | None
    over50: float | None
    """Share of scored frames above 50 mm."""
    over100: float | None
    catastrophes_per_min: float
    """Runs of scored frames above 100 mm, per minute of segment."""
    longest_bad_run: int
    """Longest run of consecutive frames above 50 mm."""
    restarts_per_min: float
    swaps_per_min: float
    mpvpe: float | None
    """Mean mesh-vertex error of scored frames."""
    mpvpe_p90: float | None
    mka: float | None
    """Mean keypoint acceleration over runs of 3 tracked frames, mm/frame^2."""
    mka_truth: float | None
    hidden_error_share: float | None
    """Share of the summed error carried by frames where >= half of the hand's keypoints are hidden in every camera that shows it."""


def _acceleration(points: Float32[ndarray, "f 21 3"], valid: Bool[ndarray, "f"]) -> Float32[ndarray, "a"]:
    triple: Bool[ndarray, "g"] = valid[:-2] & valid[1:-1] & valid[2:]
    if not triple.any():
        return np.zeros(0, dtype=np.float32)
    acc: Float32[ndarray, "g 21 3"] = points[:-2] + points[2:] - 2.0 * points[1:-1]
    return np.linalg.norm(acc[triple], axis=-1).mean(-1) * 1000.0


def score_hand(scores: FrameScores, side: int, dataset: str, segment: str, fps: float) -> ScoreRow:
    """The KPIs of one hand of one segment."""
    frames: int = len(scores.tracked)
    minutes: float = frames / fps / 60.0
    error: Float32[ndarray, "f"] = scores.keypoint_mm[:, side]
    scored: Bool[ndarray, "f"] = np.isfinite(error)
    values: Float32[ndarray, "s"] = error[scored]
    required: Bool[ndarray, "f"] = scores.required[:, side]
    tracked: Bool[ndarray, "f"] = scores.tracked[:, side]
    starts: int = int((tracked & ~np.r_[False, tracked[:-1]]).sum())
    catastrophes: int = len(_runs(scored & (np.nan_to_num(error) > CATASTROPHE_MM)))
    bad_runs: Int64[ndarray, "r"] = _runs(scored & (np.nan_to_num(error) > BAD_MM))
    vertex: Float32[ndarray, "s"] = scores.vertex_mm[scored, side]
    hidden: Bool[ndarray, "f"] = scored & (np.nan_to_num(scores.hidden_share[:, side]) >= 0.5)
    acc_pred = _acceleration(scores.predicted[:, side], scored)
    acc_truth = _acceleration(scores.truth[:, side], scored)
    empty: bool = not len(values)
    return ScoreRow(
        dataset=dataset, segment=segment, hand=("left", "right")[side], minutes=minutes,
        required_frames=int(required.sum()), scored_frames=int(scored.sum()),
        coverage=float((required & tracked).sum() / max(required.sum(), 1)),
        good30=float((required & scored & (np.nan_to_num(error, nan=np.inf) < GOOD_MM)).sum() / max(required.sum(), 1)),
        mkpe=None if empty else float(values.mean()), median=None if empty else float(np.median(values)),
        p90=None if empty else float(np.percentile(values, 90)), p95=None if empty else float(np.percentile(values, 95)),
        over50=None if empty else float((values > BAD_MM).mean()), over100=None if empty else float((values > CATASTROPHE_MM).mean()),
        catastrophes_per_min=catastrophes / minutes, longest_bad_run=int(bad_runs.max()) if len(bad_runs) else 0,
        restarts_per_min=starts / minutes, swaps_per_min=float(scores.swapped[:, side].sum()) / minutes,
        mpvpe=None if empty else float(vertex.mean()), mpvpe_p90=None if empty else float(np.percentile(vertex, 90)),
        mka=float(acc_pred.mean()) if len(acc_pred) else None, mka_truth=float(acc_truth.mean()) if len(acc_truth) else None,
        hidden_error_share=None if empty else float(error[hidden].sum() / max(values.sum(), 1e-9)),
    )


def pool(rows: list[ScoreRow], frames: list[tuple[Float32[ndarray, "s"], Float32[ndarray, "s"]]], name: str, dataset: str) -> ScoreRow:
    """Pool rows: rates weighted by minutes or required frames; the distribution over all scored frames (``frames`` gives
    each row's scored keypoint and vertex errors, in the same order)."""
    minutes: float = sum(r.minutes for r in rows)
    required: int = sum(r.required_frames for r in rows)
    values: Float32[ndarray, "s"] = np.concatenate([f[0] for f in frames]) if frames else np.zeros(0, dtype=np.float32)
    vertex: Float32[ndarray, "s"] = np.concatenate([f[1] for f in frames]) if frames else np.zeros(0, dtype=np.float32)
    empty: bool = not len(values)

    def weighted(field: str, weight: str) -> float | None:
        pairs = [(getattr(r, field), getattr(r, weight)) for r in rows if getattr(r, field) is not None]
        total = sum(w for _, w in pairs)
        return None if not pairs or total == 0 else float(sum(v * w for v, w in pairs) / total)

    return ScoreRow(
        dataset=dataset, segment=name, hand="both", minutes=minutes, required_frames=required, scored_frames=int(len(values)),
        coverage=float(sum(r.coverage * r.required_frames for r in rows) / max(required, 1)),
        good30=float(sum(r.good30 * r.required_frames for r in rows) / max(required, 1)),
        mkpe=None if empty else float(values.mean()), median=None if empty else float(np.median(values)),
        p90=None if empty else float(np.percentile(values, 90)), p95=None if empty else float(np.percentile(values, 95)),
        over50=None if empty else float((values > BAD_MM).mean()), over100=None if empty else float((values > CATASTROPHE_MM).mean()),
        catastrophes_per_min=sum(r.catastrophes_per_min * r.minutes for r in rows) / max(minutes, 1e-9),
        longest_bad_run=max((r.longest_bad_run for r in rows), default=0),
        restarts_per_min=sum(r.restarts_per_min * r.minutes for r in rows) / max(minutes, 1e-9),
        swaps_per_min=sum(r.swaps_per_min * r.minutes for r in rows) / max(minutes, 1e-9),
        mpvpe=None if empty else float(vertex.mean()), mpvpe_p90=None if empty else float(np.percentile(vertex, 90)),
        mka=weighted("mka", "scored_frames"), mka_truth=weighted("mka_truth", "scored_frames"),
        hidden_error_share=weighted("hidden_error_share", "scored_frames"),
    )
