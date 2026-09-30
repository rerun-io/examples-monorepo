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
from handtrack.pinch import pinch_state as runtime_pinch_state

GOOD_MM: float = 30.0
"""A tracked frame this close to the truth counts as good."""
BAD_MM: float = 50.0
CATASTROPHE_MM: float = 100.0
"""A run of tracked frames above this is one catastrophic episode."""
PINCH_CLOSED_MM: float = 10.0
"""Ground truth: thumb-tip to index-tip distance below this is a pinch (UmeTrack's epsilon_1)."""
PINCH_OPEN_MM: float = 20.0
"""Ground truth: above this the hand is not pinching (UmeTrack's epsilon_2); in between is ambiguous and not scored."""
PINCH_DETECT_MM: float = 15.0
"""Predicted pinch: the fitted thumb-tip to index-tip distance below this."""
NEAR_PINCH_MM: float = 40.0
"""Frames whose true fingertip distance is below this score the thumb-index distance error."""
THUMB_TIP, INDEX_TIP = 0, 1
THUMB_DISTAL_BONE: int = 4
"""Skinning frame of the thumb's distal phalanx (the thumb-tip landmark's bone)."""
INDEX_CONTACT_BONES: tuple[int, int] = (6, 7)
"""The index finger's middle and distal phalanges (the index-tip landmark's bone is 7)."""
CONTACT_CLOSED_MM: float = 5.0
"""Mesh contact: the thumb's distal pad within this of the index finger's last two phalanges is a pinch (pad pinches count; the tip
landmarks sit 22-25 mm past the DIP joint, so a tip-to-tip rule misses them)."""
CONTACT_OPEN_MM: float = 15.0
DETECT_ENTER_MM: float = 10.0
"""State-machine detector on the fitted mesh's contact distance: enter a pinch after ``DETECT_FRAMES`` frames under this..."""
DETECT_EXIT_MM: float = 16.0
"""...and release after ``DETECT_FRAMES`` frames over this; untracked frames hold the state for up to ``HOLD_FRAMES`` frames."""
DETECT_FRAMES: int = 2
HOLD_FRAMES: int = 3
EVENT_TOLERANCE: int = 3
"""A detected pinch onset matches a true onset within this many frames."""


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
    pinch_pred_mm: Float32[ndarray, "f 2"]
    """Fitted thumb-tip to index-tip distance (NaN untracked)."""
    pinch_true_mm: Float32[ndarray, "f 2"]
    contact_pred_mm: Float32[ndarray, "f 2"]
    """Fitted mesh: minimum distance from the thumb's distal-phalanx vertices to the index finger's middle and distal phalanx vertices."""
    contact_true_mm: Float32[ndarray, "f 2"]


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
    contact_pred: Float32[ndarray, "f 2"] = np.full((frames, 2), np.nan, dtype=np.float32)
    contact_true: Float32[ndarray, "f 2"] = np.full((frames, 2), np.nan, dtype=np.float32)
    bones: Int64[ndarray, "v"] = model.dense_bone_weights.argmax(dim=1).numpy()
    thumb_vertices: Int64[ndarray, "a"] = np.flatnonzero(bones == THUMB_DISTAL_BONE)
    index_vertices: Int64[ndarray, "b"] = np.flatnonzero(np.isin(bones, INDEX_CONTACT_BONES))
    for side in Side:
        truth_rows: Int64[ndarray, "k"] = np.flatnonzero(has_pose[:, side])
        if len(truth_rows):
            pose = timeline.poses[side]
            index = torch.from_numpy(truth_rows)
            with torch.inference_mode():
                mesh = mesh_vertices(model, HandPose(pose.rotation[index], pose.translation[index], pose.joint_angles[index]), side)
            contact_true[truth_rows, side] = _contact_mm(mesh, thumb_vertices, index_vertices)
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
            fitted_mesh = mesh_vertices(model, fitted, side)
            difference = fitted_mesh - mesh_vertices(model, reference, side)
        vertex[rows, side] = difference.norm(dim=-1).mean(dim=-1).numpy() * 1000.0
        contact_pred[rows, side] = _contact_mm(fitted_mesh, thumb_vertices, index_vertices)
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
        pinch_pred_mm=np.where(tracked, np.linalg.norm(predicted[:, :, THUMB_TIP] - predicted[:, :, INDEX_TIP], axis=-1) * 1000.0, np.nan).astype(np.float32),
        pinch_true_mm=(np.linalg.norm(truth_points[:, :, THUMB_TIP] - truth_points[:, :, INDEX_TIP], axis=-1) * 1000.0).astype(np.float32),
        contact_pred_mm=contact_pred,
        contact_true_mm=contact_true,
    )


def _contact_mm(mesh: torch.Tensor, thumb: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Per frame: the minimum thumb-pad to index vertex distance, mm."""
    distances = torch.cdist(mesh[:, torch.from_numpy(thumb)], mesh[:, torch.from_numpy(index)])
    return (distances.amin(dim=(-1, -2)) * 1000.0).numpy().astype(np.float32)


def pinch_state(distance: np.ndarray, enter: float = DETECT_ENTER_MM, leave: float = DETECT_EXIT_MM, frames: int = DETECT_FRAMES,
                hold: int = HOLD_FRAMES) -> np.ndarray:
    """``handtrack.pinch.pinch_state`` with the scorecard's detector thresholds as defaults (the runtime detector and the KPI are one machine)."""
    return runtime_pinch_state(distance, enter, leave, frames, hold)


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
    pinch_tp: int = 0
    """Scored frames with a true pinch (< 10 mm) that the fitted pose calls a pinch (< 15 mm)."""
    pinch_fp: int = 0
    """Scored frames with an open hand (> 20 mm) that the fitted pose calls a pinch."""
    pinch_fn: int = 0
    pinch_precision: float | None = None
    pinch_recall: float | None = None
    pinch_f1: float | None = None
    false_pinches_per_min: float = 0.0
    """Onsets of a predicted pinch (open -> pinch) that start while the true hand is open (> 20 mm), per minute."""
    pinch_distance_mae: float | None = None
    """Mean |fitted - true| thumb-index distance over scored frames with the true distance under 40 mm."""
    near_pinch_frames: int = 0
    contact_tp: int = 0
    """Frames in mesh contact (< 5 mm) that the state machine calls a pinch."""
    contact_fp: int = 0
    """Open frames (> 15 mm) that the state machine calls a pinch."""
    contact_fn: int = 0
    contact_precision: float | None = None
    contact_recall: float | None = None
    contact_f1: float | None = None
    true_onsets: int = 0
    """Pinch onsets of the state machine run on the true contact distance (thresholds 5 / 15 mm)."""
    matched_onsets: int = 0
    """True onsets with a detected onset within 3 frames."""
    detected_onsets: int = 0
    onset_recall: float | None = None
    onset_precision: float | None = None
    false_onsets_per_min: float = 0.0
    """Detected onsets with no true onset within 3 frames, per minute."""
    onset_latency_frames: float | None = None
    """Median frames from a true onset to its matched detected onset."""


def _acceleration(points: Float32[ndarray, "f 21 3"], valid: Bool[ndarray, "f"]) -> Float32[ndarray, "a"]:
    triple: Bool[ndarray, "g"] = valid[:-2] & valid[1:-1] & valid[2:]
    if not triple.any():
        return np.zeros(0, dtype=np.float32)
    acc: Float32[ndarray, "g 21 3"] = points[:-2] + points[2:] - 2.0 * points[1:-1]
    return np.linalg.norm(acc[triple], axis=-1).mean(-1) * 1000.0


def pinch_counts(predicted: Float32[ndarray, "f"], truth: Float32[ndarray, "f"], scored: Bool[ndarray, "f"], minutes: float) -> dict:
    """The pinch KPIs of one hand: frame-level detection against the true state, false onsets, and the fingertip distance error."""
    closed: Bool[ndarray, "f"] = scored & (truth < PINCH_CLOSED_MM)
    opened: Bool[ndarray, "f"] = scored & (truth > PINCH_OPEN_MM)
    called: Bool[ndarray, "f"] = scored & (np.nan_to_num(predicted, nan=np.inf) < PINCH_DETECT_MM)
    tp, fp, fn = int((closed & called).sum()), int((opened & called).sum()), int((closed & ~called).sum())
    onsets: Bool[ndarray, "f"] = called & ~np.r_[False, called[:-1]]
    near: Bool[ndarray, "f"] = scored & (truth < NEAR_PINCH_MM) & np.isfinite(predicted)
    return {
        "pinch_tp": tp, "pinch_fp": fp, "pinch_fn": fn,
        "pinch_precision": tp / (tp + fp) if tp + fp else None, "pinch_recall": tp / (tp + fn) if tp + fn else None,
        "pinch_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "false_pinches_per_min": float((onsets & opened).sum()) / max(minutes, 1e-9),
        "pinch_distance_mae": float(np.abs(predicted[near] - truth[near]).mean()) if near.any() else None, "near_pinch_frames": int(near.sum()),
    }


def contact_counts(predicted: np.ndarray, truth: np.ndarray, scored: np.ndarray, minutes: float) -> dict:
    """Mesh-contact pinch KPIs: frame-level state machine against the true contact state, and onset events."""
    detected: np.ndarray = pinch_state(np.where(scored, predicted, np.nan))
    true_state: np.ndarray = pinch_state(truth, CONTACT_CLOSED_MM, CONTACT_OPEN_MM, 1, 0)
    closed: np.ndarray = scored & (truth < CONTACT_CLOSED_MM)
    opened: np.ndarray = scored & (truth > CONTACT_OPEN_MM)
    tp, fp, fn = int((closed & detected).sum()), int((opened & detected).sum()), int((closed & ~detected).sum())
    true_on: np.ndarray = np.flatnonzero(true_state & ~np.r_[False, true_state[:-1]])
    det_on: np.ndarray = np.flatnonzero(detected & ~np.r_[False, detected[:-1]])
    matched, latencies, used = 0, [], set()
    for onset in true_on:
        near = [d for d in det_on if abs(d - onset) <= EVENT_TOLERANCE and d not in used]
        if near:
            best = min(near, key=lambda d: abs(d - onset))
            used.add(best)
            matched += 1
            latencies.append(best - onset)
    false: int = len(det_on) - len(used)
    return {
        "contact_tp": tp, "contact_fp": fp, "contact_fn": fn,
        "contact_precision": tp / (tp + fp) if tp + fp else None, "contact_recall": tp / (tp + fn) if tp + fn else None,
        "contact_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        "true_onsets": int(len(true_on)), "matched_onsets": matched, "detected_onsets": int(len(det_on)),
        "onset_recall": matched / len(true_on) if len(true_on) else None, "onset_precision": len(used) / len(det_on) if len(det_on) else None,
        "false_onsets_per_min": false / max(minutes, 1e-9), "onset_latency_frames": float(np.median(latencies)) if latencies else None,
    }


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
        **pinch_counts(scores.pinch_pred_mm[:, side], scores.pinch_true_mm[:, side], scored, minutes),
        **contact_counts(scores.contact_pred_mm[:, side], scores.contact_true_mm[:, side], scored, minutes),
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
        pinch_tp=sum(r.pinch_tp for r in rows), pinch_fp=sum(r.pinch_fp for r in rows), pinch_fn=sum(r.pinch_fn for r in rows),
        pinch_precision=(sum(r.pinch_tp for r in rows)) / (sum(r.pinch_tp for r in rows) + sum(r.pinch_fp for r in rows)) if sum(r.pinch_tp for r in rows) + sum(r.pinch_fp for r in rows) else None,
        pinch_recall=(sum(r.pinch_tp for r in rows)) / (sum(r.pinch_tp for r in rows) + sum(r.pinch_fn for r in rows)) if sum(r.pinch_tp for r in rows) + sum(r.pinch_fn for r in rows) else None,
        pinch_f1=2 * (sum(r.pinch_tp for r in rows)) / (2 * sum(r.pinch_tp for r in rows) + sum(r.pinch_fp for r in rows) + sum(r.pinch_fn for r in rows)) if sum(r.pinch_tp for r in rows) + sum(r.pinch_fp for r in rows) + sum(r.pinch_fn for r in rows) else None,
        false_pinches_per_min=sum(r.false_pinches_per_min * r.minutes for r in rows) / max(minutes, 1e-9),
        pinch_distance_mae=weighted("pinch_distance_mae", "near_pinch_frames"), near_pinch_frames=sum(r.near_pinch_frames for r in rows),
        contact_tp=sum(r.contact_tp for r in rows), contact_fp=sum(r.contact_fp for r in rows), contact_fn=sum(r.contact_fn for r in rows),
        contact_precision=(sum(r.contact_tp for r in rows)) / (sum(r.contact_tp for r in rows) + sum(r.contact_fp for r in rows)) if sum(r.contact_tp for r in rows) + sum(r.contact_fp for r in rows) else None,
        contact_recall=(sum(r.contact_tp for r in rows)) / (sum(r.contact_tp for r in rows) + sum(r.contact_fn for r in rows)) if sum(r.contact_tp for r in rows) + sum(r.contact_fn for r in rows) else None,
        contact_f1=2 * (sum(r.contact_tp for r in rows)) / (2 * sum(r.contact_tp for r in rows) + sum(r.contact_fp for r in rows) + sum(r.contact_fn for r in rows)) if sum(r.contact_tp for r in rows) + sum(r.contact_fp for r in rows) + sum(r.contact_fn for r in rows) else None,
        true_onsets=sum(r.true_onsets for r in rows), matched_onsets=sum(r.matched_onsets for r in rows), detected_onsets=sum(r.detected_onsets for r in rows),
        onset_recall=(sum(r.matched_onsets for r in rows)) / (sum(r.true_onsets for r in rows)) if sum(r.true_onsets for r in rows) else None,
        onset_precision=sum(r.detected_onsets - r.false_onsets_per_min * r.minutes for r in rows) / (sum(r.detected_onsets for r in rows)) if sum(r.detected_onsets for r in rows) else None,
        false_onsets_per_min=sum(r.false_onsets_per_min * r.minutes for r in rows) / max(minutes, 1e-9),
        onset_latency_frames=weighted("onset_latency_frames", "matched_onsets"),
    )
