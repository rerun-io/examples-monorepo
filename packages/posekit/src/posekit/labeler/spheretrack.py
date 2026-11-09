"""The machine track as one 3D hand sphere per side, shared by every camera (closer to MEgATrack's multi-view fit than one
2D chain per camera).

At each grid time, per hand side:

1. **Labels:** the stored boxes of that side at this time (a person's, or ones derived and saved with them) give the
   sphere: two or more views are triangulated, one view is lifted with the other views' proposals
   (``posekit.labeler.sphere.solve_views``). A labeled view shows its label (origin ``reviewed``, or ``derived`` for
   source ``rig``); every other view shows the sphere's box (origin ``sphere``).
2. **Track:** otherwise the previous time's sphere is projected into every camera. In each camera the EgoHand proposal of
   either side (EgoHand flips sides) whose centre is nearest the projected box, within one box size, is that hand there.
   The new centre is the least-squares point of those boxes' centre rays plus a weak pull (``PRIOR_WEIGHT``) towards the
   previous centre; the radius stays the one the labels gave. EgoHand boxes take in the forearm, so their sizes would bias
   depth, and at 5 fps a velocity model overshoots: the trace on s13 lost the hand within 0.6 s with both. One view slides
   the centre onto its ray; with no view the sphere is held for ``MAX_HELD`` steps (origin ``held``), then the track ends.
3. **No track:** the best EgoHand proposal of that side (score at least ``DETECT_SCORE``) per camera, origin ``detected``.

Not modelled: occlusion (a camera whose view of the hand is blocked still gets the sphere's box), re-acquisition of a lost
hand from detections in several views, and a person's ``no_hand`` holding a track off in later frames.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from posekit.labeler.proposals import iou
from posekit.labeler.rig import Kb4Rig
from posekit.labeler.schema import HAND_SIDES, Box, HandSide, Proposal, TrackBox, TrackFrame, ViewLabel
from posekit.labeler.sphere import HandSphere, solve_views, sphere_box
from posekit.labeler.tracker import CLAIM_IOU, DETECT_SCORE, MachineStep

MAX_HELD: int = 1
"""Grid steps a sphere may hold without a matching proposal in any view."""
MATCH_REACH: float = 1.0
"""A proposal belongs to the hand in a view when its centre is within this many projected box sizes."""
PRIOR_WEIGHT: float = 0.05
"""Weight of the previous centre against one view's ray (each ray has weight 1) in the centre update."""


@dataclass(frozen=True, slots=True)
class SphereTrack:
    """One hand's sphere and what the next time needs."""

    sphere: HandSphere
    held: int = 0
    """Consecutive held steps."""


@dataclass(frozen=True, slots=True)
class SphereStep:
    """One grid time of ``SpherePropagator``."""

    frame: TrackFrame
    hands: dict[HandSide, SphereTrack] = field(default_factory=dict)
    """Live hand tracks after this time."""


def box_size(box: Box) -> float:
    return max(box.x1 - box.x0, box.y1 - box.y0)


class SpherePropagator:
    """Carries a person's 1-2 view labels through time as one sphere per hand (see the module docstring)."""

    def __init__(self, rig: Kb4Rig) -> None:
        self.rig: Kb4Rig = rig

    def assign(
        self, guesses: Mapping[HandSide, HandSphere], cams: Sequence[int], proposals: Sequence[Proposal]
    ) -> dict[HandSide, dict[int, Float64[ndarray, "3"]]]:
        """Per hand, the cameras where an EgoHand box is that hand, as rig-frame rays through the hand's centre.

        In each camera an EgoHand box (of either side: EgoHand flips sides) goes to the hand whose projected box centre is
        nearest, within ``MATCH_REACH`` box sizes, and to one hand only, so two hands that meet do not merge. The box
        centre is moved by the offset between the projected box's centre and the projected sphere centre (in a fisheye
        they differ off-axis), so the ray passes through the hand's centre and not beside it.
        """
        rays: dict[HandSide, dict[int, Float64[ndarray, "3"]]] = {side: {} for side in guesses}
        for cam in cams:
            boxes: list[Box] = [p.box for p in proposals if p.cam == cam and p.model == "egohand"]
            pairs: list[tuple[float, HandSide, int, Float64[ndarray, "2"]]] = []
            for side, guess in guesses.items():
                projected: Box | None = sphere_box(self.rig, cam, guess)
                if projected is None:
                    continue
                centre_px: Float64[ndarray, "2"] = self.rig.project(cam, self.rig.to_cam(cam, guess.centre[None]))[0]
                offset: Float64[ndarray, "2"] = centre_px - np.array([(projected.x0 + projected.x1) / 2.0, (projected.y0 + projected.y1) / 2.0])
                for index, box in enumerate(boxes):
                    centre: Float64[ndarray, "2"] = np.array([(box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0])
                    distance: float = float(np.linalg.norm(centre + offset - centre_px))
                    if distance <= MATCH_REACH * box_size(projected):
                        pairs.append((distance, side, index, centre + offset))
            taken: set[int] = set()
            for _, side, index, pixel in sorted(pairs, key=lambda pair: pair[0]):
                if index in taken or cam in rays[side]:
                    continue
                taken.add(index)
                rays[side][cam] = self.rig.rays_to_rig(cam, self.rig.unproject(cam, pixel[None]))[0]
        return rays

    def follow(self, track: SphereTrack, rays: Mapping[int, Float64[ndarray, "3"]]) -> SphereTrack | None:
        """The track one grid step later from its matched rays; None when it ends."""
        if not rays:
            return None if track.held >= MAX_HELD else SphereTrack(track.sphere, track.held + 1)
        normal: Float64[ndarray, "3 3"] = PRIOR_WEIGHT * np.eye(3)
        rhs: Float64[ndarray, "3"] = PRIOR_WEIGHT * track.sphere.centre
        for cam, ray in rays.items():
            projector: Float64[ndarray, "3 3"] = np.eye(3) - np.outer(ray, ray)
            normal += projector
            rhs += projector @ self.rig.centre(cam)
        return SphereTrack(HandSphere(np.linalg.solve(normal, rhs), track.sphere.radius), 0)

    def step(self, t_ns: int, cams: list[int], proposals: list[Proposal], labels: list[ViewLabel], previous: MachineStep | None) -> SphereStep:
        """The machine boxes at ``t_ns``; see the module docstring."""
        if previous is not None and not isinstance(previous, SphereStep):
            raise TypeError(f"SpherePropagator continues a SphereStep, not {type(previous).__name__}")
        stored: dict[tuple[int, HandSide], ViewLabel] = {(label.cam, label.side): label for label in labels}
        hands: Mapping[HandSide, SphereTrack] = {} if previous is None else previous.hands
        next_hands: dict[HandSide, SphereTrack] = {}
        moving: dict[HandSide, SphereTrack] = {}
        for side in HAND_SIDES:
            labeled: dict[int, Box] = {cam: label.box for (cam, label_side), label in stored.items() if label_side == side and label.box is not None}
            if labeled:
                others: dict[int, list[Box]] = {}
                for proposal in proposals:
                    if proposal.model == "egohand" and proposal.cam not in labeled:
                        others.setdefault(proposal.cam, []).append(proposal.box)
                next_hands[side] = SphereTrack(solve_views(self.rig, labeled, others).sphere, 0)
            elif side in hands:
                moving[side] = hands[side]
        rays: dict[HandSide, dict[int, Float64[ndarray, "3"]]] = self.assign({side: track.sphere for side, track in moving.items()}, cams, proposals)
        for side, track in moving.items():
            followed: SphereTrack | None = self.follow(track, rays[side])
            if followed is not None:
                next_hands[side] = followed
        boxes: list[TrackBox] = []
        for cam in cams:
            placed: list[Box] = []
            decided: dict[HandSide, TrackBox] = {}
            for side in HAND_SIDES:
                label: ViewLabel | None = stored.get((cam, side))
                if label is not None:
                    decided[side] = TrackBox(cam, side, label.box, "derived" if label.source == "rig" else "reviewed", None)
                elif side in next_hands:
                    track: SphereTrack = next_hands[side]
                    decided[side] = TrackBox(cam, side, sphere_box(self.rig, cam, track.sphere), "held" if track.held else "sphere", None)
                box: Box | None = decided[side].box if side in decided else None
                if box is not None:
                    placed.append(box)
            for side in HAND_SIDES:
                if side in decided:
                    continue
                candidates: list[Proposal] = [
                    p
                    for p in proposals
                    if p.model == "egohand"
                    and p.cam == cam
                    and p.side == side
                    and p.score >= DETECT_SCORE
                    and all(iou(p.box, b) < CLAIM_IOU for b in placed)
                ]
                if candidates:
                    best: Proposal = max(candidates, key=lambda p: p.score)
                    decided[side] = TrackBox(cam, side, best.box, "detected", best.score)
            boxes += [decided.get(side, TrackBox(cam, side, None, "none", None)) for side in HAND_SIDES]
        return SphereStep(TrackFrame(t_ns, boxes), next_hands)
