"""Records the hand labeler passes between the catalog, the backend, the browser and the labeling agents.

One label is a decision about one hand side in one camera at one time: the hand is there (with a box),
the hand is not there, or the frame cannot be judged. Boxes are in native image pixels, ``x0 y0 x1 y1``.

Everything here crosses a boundary (HTTP JSON, the catalog layer), so it is a pyserde record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from serde import serde

HandSide: TypeAlias = Literal["left", "right"]
"""Anatomical side of the hand (the wearer's left or right)."""

LabelStatus: TypeAlias = Literal["accepted", "corrected", "no_hand", "rejected"]
"""What the reviewer decided.

- ``accepted``: the proposal box is right as it is.
- ``corrected``: the reviewer moved, resized, drew or re-sided the box.
- ``no_hand``: that hand side is not visible in this camera.
- ``rejected``: this camera/time cannot be judged (blur, exposure, decode error); excluded from training and eval.
"""

LabelSource: TypeAlias = Literal["egohand", "hands_layer", "person", "agent", "range", "tracker", "rig"]
"""Where the box geometry came from.

- ``egohand``: the posekit EgoHand proposal, accepted unchanged.
- ``hands_layer``: the RoboCap ``hands`` layer (DetNet/KeyNet/fit) proposal, accepted unchanged.
- ``person``: drawn or edited by a person.
- ``agent``: drawn or edited by a labeling agent (for example Astra).
- ``range``: filled between two reviewed keyframes from consistent proposals (MEgATrack "accept the tracked run").
- ``tracker``: the machine track's box (or its "no hand"), accepted by a reviewer who watched it (``/api/accept_range``).
- ``rig``: derived from the reviewer's boxes in other cameras at the same time (the hand sphere through the KB4 rig,
  ``posekit.labeler.sphere``), saved with them.
"""

TrackOrigin: TypeAlias = Literal["reviewed", "derived", "chained", "held", "tracked", "detected", "fused", "disagree", "sphere", "none"]
"""Why the machine track has (or has no) box at one (time, cam, side); see ``posekit.labeler.tracker``. ``fused`` and
``disagree`` come only from ``posekit.labeler.megatrack`` (the box around a 3D hand's projected keypoints; ``disagree``:
the views do not agree on that hand, so the reviewer should look). ``derived`` is a stored label with source ``rig``;
``sphere`` is the box of the 3D hand sphere that ``posekit.labeler.spheretrack`` carries through time."""

ActionKind: TypeAlias = Literal["accept_range", "draw_box", "pick_proposal", "set_side", "move_box", "no_hand", "reject", "accept_frame"]
"""One reviewer action of the MEgATrack loop; every kind except ``accept_range``/``accept_frame`` is an intervention."""
PASSIVE_ACTIONS: frozenset[ActionKind] = frozenset({"accept_range", "accept_frame"})
"""Action kinds that are not interventions (the reviewer only confirmed the machine)."""

SegmentSplit: TypeAlias = Literal["train", "eval"]
"""Held-out eval segments are never trained on."""

HAND_SIDES: tuple[HandSide, HandSide] = ("left", "right")
LABEL_LAYER: str = "hand_boxes"
"""Catalog layer that holds the reviewed labels."""
PROPOSAL_LAYER: str = "hand_proposals"
"""Catalog layer that holds the model proposals the labeler shows."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in native image pixels."""

    x0: float
    """Left edge, px."""
    y0: float
    """Top edge, px."""
    x1: float
    """Right edge, px."""
    y1: float
    """Bottom edge, px."""

    def __post_init__(self) -> None:
        if not (self.x1 > self.x0 and self.y1 > self.y0):
            raise ValueError(f"empty box {self}")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Proposal:
    """One model's box for one hand side in one camera at one time."""

    cam: int
    """Camera index in the rig (0..5 on RoboCap)."""
    side: HandSide
    """Hand side the model assigned."""
    box: Box
    """Proposed box."""
    score: float
    """Model confidence in [0, 1]."""
    model: Literal["egohand", "hands_layer"]
    """Which proposer made it."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ViewLabel:
    """The reviewed decision for one hand side in one camera at one time."""

    cam: int
    """Camera index in the rig."""
    side: HandSide
    """Hand side."""
    status: LabelStatus
    """Decision."""
    box: Box | None
    """Box for ``accepted``/``corrected``; None for ``no_hand``/``rejected``."""
    source: LabelSource
    """Where the box came from."""
    reviewer: str
    """Who decided (a person's name, or ``astra``)."""

    def __post_init__(self) -> None:
        has_box: bool = self.status in ("accepted", "corrected")
        if has_box != (self.box is not None):
            raise ValueError(f"status {self.status!r} {'needs' if has_box else 'must not have'} a box")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FrameLabels:
    """All decisions a reviewer posts for one time of one segment (any subset of cams x sides)."""

    segment_id: str
    """Catalog segment id."""
    t_ns: int
    """``video_time`` of the frame set, ns."""
    labels: list[ViewLabel]
    """Decisions; at most one per (cam, side)."""

    def __post_init__(self) -> None:
        keys: list[tuple[int, HandSide]] = [(label.cam, label.side) for label in self.labels]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate (cam, side) in one frame")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CameraInfo:
    """One camera of the segment's rig."""

    cam: int
    """Camera index."""
    name: str
    """Rig name (left_front, right_front, left_eye, right_eye, left, right on RoboCap)."""
    width: int
    """Native image width, px."""
    height: int
    """Native image height, px."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SegmentInfo:
    """What the browser needs to lay out one segment."""

    segment_id: str
    """Catalog segment id."""
    split: SegmentSplit
    """train or held-out eval."""
    cams: list[CameraInfo]
    """Cameras in rig order."""
    times_ns: list[int]
    """The 5 fps label grid on ``video_time``, ns (each a real frame time of the reference camera)."""
    keyframes_ns: list[int]
    """The 1 fps review keyframes plus the last grid time, a subset of ``times_ns``."""
    viewer_url: str
    """Segment URL the embedded Rerun web viewer opens."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ViewState:
    """Proposals and current labels of one camera at one time."""

    cam: int
    """Camera index."""
    proposals: list[Proposal]
    """Model proposals for this camera and time."""
    labels: list[ViewLabel]
    """Labels already stored for this camera and time."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FrameState:
    """Everything shown for one time of one segment."""

    segment_id: str
    """Catalog segment id."""
    t_ns: int
    """``video_time``, ns."""
    views: list[ViewState]
    """One entry per camera, rig order."""
    flags: list[str]
    """Why this frame is in the review queue (empty when it is not)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RangeRequest:
    """Fill the 5 fps frames between two reviewed keyframes from consistent proposals."""

    segment_id: str
    """Catalog segment id."""
    t0_ns: int
    """Reviewed keyframe at the start."""
    t1_ns: int
    """Reviewed keyframe at the end."""
    reviewer: str
    """Who asked for the fill."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RangeResult:
    """Outcome of a range fill."""

    filled: int
    """(time, cam, side) labels written with source ``range``."""
    queued: list[int]
    """Times (ns) left for review because the proposals were not consistent with both keyframes."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class QueueItem:
    """One frame set that needs a reviewer."""

    segment_id: str
    """Catalog segment id."""
    t_ns: int
    """``video_time``, ns."""
    reasons: list[str]
    """Short reasons (for example ``keyframe``, ``side_flip``, ``views_disagree``, ``range_gap``)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SegmentSummary:
    """One row of the segment list."""

    segment_id: str
    """Catalog segment id."""
    split: SegmentSplit
    """train or held-out eval."""
    duration_s: float
    """Length of the segment on ``video_time``, s."""
    n_keyframes: int
    """1 fps review keyframes."""
    n_reviewed_keyframes: int
    """Keyframes with a decision for every (cam, side)."""
    n_labels: int
    """Stored (time, cam, side) labels, range fills included."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrackBox:
    """The machine track of one (cam, side) at one time."""

    cam: int
    """Camera index."""
    side: HandSide
    """Hand side of the track."""
    box: Box | None
    """Box in native px; None when the machine says there is no hand (or the reviewer said no hand / rejected)."""
    origin: TrackOrigin
    """Where the box comes from."""
    score: float | None
    """EgoHand score of the proposal for ``chained``/``detected``; KeyNet presence in this view for ``fused``/``disagree``;
    None otherwise (``tracked`` included)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrackFrame:
    """The machine track of every (cam, side) at one grid time."""

    t_ns: int
    """``video_time``, ns."""
    boxes: list[TrackBox]
    """One per cam x side, rig order, left then right."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class AcceptRange:
    """Accept the machine track of every grid time in [t0_ns, t1_ns] (a reviewer watched it and saw no failure)."""

    segment_id: str
    """Catalog segment id."""
    t0_ns: int
    """First grid time, inclusive."""
    t1_ns: int
    """Last grid time, inclusive."""
    reviewer: str
    """Who watched."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class AcceptResult:
    """Outcome of ``/api/accept_range``."""

    accepted: int
    """(time, cam, side) labels written from the machine track (source ``tracker``)."""
    kept: int
    """(time, cam, side) that already had a label, left untouched."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ActionEvent:
    """One logged reviewer action, with its wall-clock time (effort is measured from these)."""

    wall_ns: int
    """Wall-clock time of the action, ns since the Unix epoch (unique and increasing within a segment)."""
    segment_id: str
    """Catalog segment id."""
    t_ns: int
    """Grid time the action applies to (the first time of an ``accept_range``), ns."""
    t1_ns: int | None
    """Last grid time of an ``accept_range``; None for every other kind."""
    cam: int | None
    """Camera of a per-view action; None for ``accept_range``/``accept_frame``."""
    side: HandSide | None
    """Hand side of a per-view action; None for ``accept_range``/``accept_frame``."""
    kind: ActionKind
    """What the reviewer did."""
    reviewer: str
    """Who did it."""
    n_labels: int
    """Labels the action wrote."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EffortSummary:
    """Reviewer effort of one segment, MEgATrack style (``GET /api/effort/<sid>``); see ``posekit.labeler.actions``."""

    video_s: float
    """Video covered by labels: grid times with at least one label x the grid period, s."""
    n_actions: int
    """Logged actions."""
    n_interventions: int
    """Actions other than ``accept_range``/``accept_frame``."""
    interventions_per_min: float | None
    """Interventions per minute of ``video_s``; None when no video is covered."""
    frames: int
    """Grid times with at least one label."""
    frames_corrected: int
    """Grid times with at least one intervention."""
    share_corrected: float | None
    """``frames_corrected / frames``; None when ``frames`` is 0."""
    boxes_accepted_by_watching: int
    """Stored labels with a box and source ``tracker``."""
    reviewer_wall_s: float
    """Wall-clock time between consecutive actions, each gap capped at ``actions.IDLE_CAP_S``, s."""
    boxes_per_reviewer_s: float | None
    """Stored labels with a box per ``reviewer_wall_s``; None when no wall time was measured."""


SolveMode: TypeAlias = Literal["triangulated", "proposals", "prior"]
"""How a hand sphere was placed: from two or more labeled views, from one view plus the other views' proposals, or from
one view and the hand-size prior (``posekit.labeler.sphere``)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LabeledBox:
    """A reviewer's box for one hand side in one camera."""

    cam: int
    box: Box


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SolveRequest:
    """Derive one hand side's box in every camera from the reviewer's boxes in one or more cameras (``/api/solve``)."""

    segment_id: str
    t_ns: int
    """Grid time; with one labeled view, that time's proposals choose the depth."""
    side: HandSide
    labeled: list[LabeledBox]
    """At least one."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SolvedBox:
    """One camera's box from a solve: the reviewer's in a labeled view, the sphere's elsewhere (None = not in view)."""

    cam: int
    box: Box | None


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SolveResult:
    """Answer of ``/api/solve``."""

    side: HandSide
    mode: SolveMode
    confidence: float
    """Depth confidence in [0, 1]; ask for another view when it is low (``posekit.labeler.sphere.ViewSolve``)."""
    radius_m: float
    """Sphere radius, m."""
    labeled: list[int]
    """Cameras the reviewer boxed."""
    boxes: list[SolvedBox]
    """Every camera, rig order."""
