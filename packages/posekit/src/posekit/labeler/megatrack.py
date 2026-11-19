"""MEgATrack-style box propagation (section 4.2): the reviewer's box seeds a 3D hand track per side, and every view's box is
the box around the track's keypoints projected into that view (``--propagator megatrack``).

Per grid time and hand side:

1. **Seed** (a reviewer box for that side at this time: any label with a box whose source is not ``tracker``/``range``):
   KeyNet runs on a crop of every reviewer box; the reviewer views are fused (``rig.fuse_views``) or, with one view, lifted
   to 3D (``rig.lift_view``: KeyNet's relative distances scaled to the generic hand). The old track of that side is dropped.
2. **Crop every view** from the 3D hand: the seed's points, or the previous time's points extrapolated with MEgATrack's
   constant velocity damped by ``EXTRAPOLATION_GAIN`` (handtrack ``ROBUST_TRACKER_CONFIG``). MEgATrack tracks every frame;
   a hand moves too far in one 200 ms grid step, so a live track first runs steps 2-4 on ``substeps - 1`` frames evenly
   spaced between the previous grid time and this one (no boxes come from them; they only carry the 3D hand). A view gets a crop when at
   least ``MIN_CROP_KEYPOINTS`` keypoints project into it; views with a ``no_hand``/``rejected`` label at this time and
   reviewer views are not cropped again.
3. **KeyNet** per crop (``KeypointEstimator``); a view sees the hand when KeyNet's presence reaches ``PRESENCE``.
4. **Fuse** the seeing views (KB4 rays, view-robust triangulation). No smoothing over time.
5. **Box** in every seeing view: ``rig.keypoint_box`` (minimum enclosing circle of the 21 projected keypoints -> the square
   around it, clipped to the image; MEgATrack's DetNet label). Origin ``fused``; origin ``disagree`` when the views do not
   agree: fewer than two views support the 3D points (then one view's lift stands in) or a seeing view is an outlier
   to them. The score is that view's KeyNet presence.
6. The track ends when no view sees the hand. A side without a track falls back to the chain tracker's ``detected`` rule
   (the best EgoHand proposal of that side with score >= ``DETECT_SCORE``, not overlapping a box already placed in the camera).

Stored labels win at their (time, cam, side) as in ``posekit.labeler.tracker`` (origin ``reviewed``); ``tracker``/``range``
labels (accepted machine boxes) do not touch the 3D track. ``MegaTrackPropagator`` is deterministic given the frames and
the estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from jaxtyping import Bool, Float64, UInt8
from numpy import ndarray

from posekit.labeler.proposals import iou
from posekit.labeler.rig import Fusion, Kb4Rig, ViewKeypoints, fuse_views, keypoint_box, lift_view
from posekit.labeler.schema import HAND_SIDES, Box, HandSide, Proposal, TrackBox, TrackFrame, ViewLabel
from posekit.labeler.tracker import CLAIM_IOU, DETECT_SCORE, MACHINE_SOURCES, MachineStep

PRESENCE: float = 0.5
"""KeyNet presence from which a view sees the hand."""
EXTRAPOLATION_GAIN: float = 0.5
"""Damped constant velocity of the crop prediction: p(t-1) + gain (p(t-1) - p(t-2)) (handtrack ``ROBUST_TRACKER_CONFIG``)."""
MIN_CROP_KEYPOINTS: int = 6
"""Keypoints of the predicted hand that must project in front of a view and inside its image for that view to get a crop."""
SUBSTEPS: int = 3
"""Tracking steps per 200 ms grid step (3 = every 67 ms, every second frame at 30 fps)."""
MAX_SUBSTEP_GAP_NS: int = 400_000_000
"""Grid steps longer than this (a gap in the recording) are not subdivided."""


@dataclass(frozen=True, slots=True)
class CropRequest:
    """One KeyNet crop: aimed at a box (a seed) or at camera-frame keypoints (a tracked hand)."""

    cam: int
    """Camera index."""
    side: HandSide
    """Hand side (right hands are mirrored)."""
    box: Box | None = None
    """Native-px box to aim at."""
    points_cam: Float64[ndarray, "21 3"] | None = None
    """Keypoints in the camera frame, metres, to aim at (they win over ``box``)."""


@runtime_checkable
class KeypointEstimator(Protocol):
    """A 2D hand keypoint network on crops (``posekit.labeler.keynet.KeyNetEstimator``)."""

    def estimate(self, frames_rgb: Mapping[int, UInt8[ndarray, "h w 3"]], requests: Sequence[CropRequest]) -> list[ViewKeypoints | None]:
        """Keypoints per request, in order; None where the crop is unusable."""
        ...


@runtime_checkable
class FrameReader(Protocol):
    """Native frames of one segment (``posekit.labeler.frames.FrameSource``)."""

    def frame(self, cam: int, t_ns: int) -> UInt8[ndarray, "h w 3"]:
        """RGB frame of ``cam`` at grid time ``t_ns``; KeyError when there is none."""
        ...


@dataclass(frozen=True, slots=True)
class HandTrack:
    """The 3D hand of one side, carried to the next grid time."""

    points_rig: Float64[ndarray, "21 3"]
    """Keypoints at this time, rig frame, metres."""
    previous_rig: Float64[ndarray, "21 3"] | None
    """Keypoints one grid time earlier (None right after a seed)."""

    def predicted(self) -> Float64[ndarray, "21 3"]:
        """The crop prediction for the next grid time (damped constant velocity)."""
        if self.previous_rig is None:
            return self.points_rig
        return self.points_rig + EXTRAPOLATION_GAIN * (self.points_rig - self.previous_rig)


@dataclass(frozen=True, slots=True)
class SideResult:
    """What the propagator saw for one side at one time (for overlays and evaluation)."""

    views: dict[int, ViewKeypoints]
    """KeyNet estimate per camera that sees the hand."""
    fusion: Fusion | None
    """The agreed 3D points; None when fewer than two views agree."""
    points_rig: Float64[ndarray, "21 3"]
    """The 3D keypoints the boxes come from (fused, or one view's lift)."""
    seeded: bool
    """This time re-seeded the side from reviewer boxes."""
    disagree: bool
    """The views do not agree (the boxes carry origin ``disagree``)."""


@dataclass(frozen=True, slots=True)
class MegaTrackStep:
    """The machine boxes of one grid time and the 3D hands they leave for the next."""

    frame: TrackFrame
    """Boxes of every (cam, side)."""
    hands: dict[HandSide, HandTrack]
    """Live 3D tracks after this time."""
    results: dict[HandSide, SideResult] = field(default_factory=dict)
    """Per side that had a 3D hand at this time."""


def projected_count(rig: Kb4Rig, cam: int, points_rig: Float64[ndarray, "21 3"]) -> int:
    """Keypoints in front of camera ``cam`` that project inside its image."""
    points_cam: Float64[ndarray, "21 3"] = rig.to_cam(cam, np.nan_to_num(points_rig))
    pixels: Float64[ndarray, "21 2"] = rig.project(cam, points_cam)
    width, height = (float(value) for value in rig.size_wh[cam])
    inside: Bool[ndarray, "21"] = (
        np.isfinite(points_rig).all(axis=-1)
        & (points_cam[:, 2] > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return int(inside.sum())


class MegaTrackPropagator:
    """The ``Propagator`` of the module docstring for one segment."""

    def __init__(self, rig: Kb4Rig, frames: FrameReader, estimator: KeypointEstimator, *, substeps: int = SUBSTEPS) -> None:
        """Bind the propagator to a segment.

        Args:
            rig: The segment's calibrated cameras.
            frames: Its decoded frames (``frame`` must also answer times between grid times).
            estimator: The 2D keypoint network.
            substeps: Tracking steps per grid step (1 = grid frames only).
        """
        if substeps < 1:
            raise ValueError("substeps must be at least 1")
        self.rig: Kb4Rig = rig
        self.frames: FrameReader = frames
        self.estimator: KeypointEstimator = estimator
        self.substeps: int = substeps

    def _estimate(self, t_ns: int, requests: list[CropRequest]) -> list[ViewKeypoints | None]:
        """Run the estimator on the frames the requests need (a camera without a frame at ``t_ns`` gets None)."""
        frames: dict[int, UInt8[ndarray, "h w 3"]] = {}
        for cam in sorted({request.cam for request in requests}):
            try:
                frames[cam] = self.frames.frame(cam, t_ns)
            except KeyError:
                continue
        return self.estimator.estimate(frames, requests) if requests else []

    def track_side(self, t_ns: int, cams: list[int], side: HandSide, labels: dict[int, ViewLabel], track: HandTrack | None) -> SideResult | None:
        """The 3D hand of one side at ``t_ns`` (steps 1-4 of the module docstring); None when there is none."""
        seeds: dict[int, Box] = {
            cam: label.box for cam, label in labels.items() if label.box is not None and label.source not in MACHINE_SOURCES and cam in cams
        }
        excluded: set[int] = {cam for cam, label in labels.items() if label.box is None}
        views: dict[int, ViewKeypoints] = {}
        aim: Float64[ndarray, "21 3"] | None = None
        if seeds:
            seed_cams: list[int] = sorted(seeds)
            estimates: list[ViewKeypoints | None] = self._estimate(t_ns, [CropRequest(cam, side, box=seeds[cam]) for cam in seed_cams])
            views = {cam: view for cam, view in zip(seed_cams, estimates, strict=True) if view is not None}
            if not views:
                return None
            seeded: Fusion | None = fuse_views(self.rig, list(views.values()))
            aim = seeded.points_rig if seeded is not None else lift_view(self.rig, max(views.values(), key=lambda view: view.presence))
        elif track is not None:
            aim = track.predicted()
        if aim is None:
            return None
        targets: list[int] = [
            cam for cam in cams if cam not in excluded and cam not in seeds and projected_count(self.rig, cam, aim) >= MIN_CROP_KEYPOINTS
        ]
        requests: list[CropRequest] = [CropRequest(cam, side, points_cam=self.rig.to_cam(cam, aim)) for cam in targets]
        for cam, view in zip(targets, self._estimate(t_ns, requests), strict=True):
            if view is not None and view.presence >= PRESENCE:
                views[cam] = view
        if not views:
            return None
        fusion: Fusion | None = fuse_views(self.rig, list(views.values()))
        if fusion is not None:
            # A joint no two inlier views solved keeps its predicted position.
            points: Float64[ndarray, "21 3"] = np.where(np.isfinite(fusion.points_rig), fusion.points_rig, aim)
        else:
            points = lift_view(self.rig, max(views.values(), key=lambda view: view.presence))
        disagree: bool = fusion is None or bool(fusion.outliers)
        return SideResult(views, fusion, points, bool(seeds), disagree)

    def step(self, t_ns: int, cams: list[int], proposals: list[Proposal], labels: list[ViewLabel], previous: MachineStep | None) -> MegaTrackStep:
        """The machine boxes at ``t_ns``; see the module docstring."""
        if previous is not None and not isinstance(previous, MegaTrackStep):
            raise TypeError(f"MegaTrackPropagator continues a MegaTrackStep, not {type(previous).__name__}")
        hands: dict[HandSide, HandTrack] = {} if previous is None else previous.hands
        stored: dict[tuple[int, HandSide], ViewLabel] = {(label.cam, label.side): label for label in labels}
        results: dict[HandSide, SideResult] = {}
        next_hands: dict[HandSide, HandTrack] = {}
        for side in HAND_SIDES:
            side_labels: dict[int, ViewLabel] = {cam: label for (cam, label_side), label in stored.items() if label_side == side}
            track: HandTrack | None = hands.get(side)
            seeded: bool = any(label.box is not None and label.source not in MACHINE_SOURCES for label in side_labels.values())
            if track is not None and previous is not None and not seeded and 0 < t_ns - previous.frame.t_ns <= MAX_SUBSTEP_GAP_NS:
                for index in range(1, self.substeps):
                    between: int = previous.frame.t_ns + (t_ns - previous.frame.t_ns) * index // self.substeps
                    carried: SideResult | None = self.track_side(between, cams, side, {}, track)
                    if carried is None:
                        track = None
                        break
                    track = HandTrack(carried.points_rig, track.points_rig)
            result: SideResult | None = self.track_side(t_ns, cams, side, side_labels, track)
            if result is None:
                continue
            results[side] = result
            continues: bool = track is not None and not result.seeded
            next_hands[side] = HandTrack(result.points_rig, track.points_rig if continues and track is not None else None)
        decided: dict[tuple[int, HandSide], TrackBox] = {}
        for cam in cams:
            for side in HAND_SIDES:
                label: ViewLabel | None = stored.get((cam, side))
                if label is not None:
                    decided[(cam, side)] = TrackBox(cam, side, label.box, "derived" if label.source == "rig" else "reviewed", None)
                    continue
                found: SideResult | None = results.get(side)
                if found is None or cam not in found.views:
                    continue
                box: Box | None = keypoint_box(self.rig, cam, found.points_rig)[0]
                if box is not None:
                    decided[(cam, side)] = TrackBox(cam, side, box, "disagree" if found.disagree else "fused", round(found.views[cam].presence, 4))
            for side in HAND_SIDES:
                if (cam, side) in decided or side in results:
                    continue
                placed: list[Box] = [box.box for (other_cam, _), box in decided.items() if other_cam == cam and box.box is not None]
                candidates: list[Proposal] = [
                    p
                    for p in proposals
                    if p.model == "egohand" and p.cam == cam and p.side == side and p.score >= DETECT_SCORE and all(iou(p.box, b) < CLAIM_IOU for b in placed)
                ]
                if candidates:
                    best: Proposal = max(candidates, key=lambda p: p.score)
                    decided[(cam, side)] = TrackBox(cam, side, best.box, "detected", best.score)
        boxes: list[TrackBox] = [decided.get((cam, side), TrackBox(cam, side, None, "none", None)) for cam in cams for side in HAND_SIDES]
        return MegaTrackStep(TrackFrame(t_ns, boxes), next_hands, results)
