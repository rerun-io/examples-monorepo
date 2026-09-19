"""Typed third-party SHOW3D hand, profile, and caption records."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

from jaxtyping import Float32
from numpy import ndarray
from serde import serde
from simplecv.umetrack_temp.generic_hand_model_numpy import NUM_JOINTS_PER_HAND, NUM_LANDMARKS_PER_HAND

from dataforge.datasets.show3d_source import CAMERAS, FrameClock, FrameInfo, agrees_with_frame, read_json


@serde
@dataclass(frozen=True, slots=True)
class HandPose:
    """One hand; each optional measurement has its own availability."""

    confidence: float
    """Tracking confidence, including zero on lost frames."""
    joint_angles: Float32[ndarray, "22"] | None
    """UmeTrack joint angles in radians."""
    wrist_rotation: Float32[ndarray, "3 3"] | None
    """World-from-wrist rotation."""
    wrist_translation: Float32[ndarray, "3"] | None
    """World wrist position in millimetres."""
    landmarks_3d_mm: Float32[ndarray, "21 3"] | None
    """World landmarks in millimetres."""
    landmarks_3d_mm_local: Float32[ndarray, "21 3"] | None
    """Wrist-local landmarks in millimetres."""
    landmarks_2d: dict[str, list[list[float] | None]] | None
    """Headset pixels, with null entries outside the image."""

    def __post_init__(self) -> None:
        if not isfinite(self.confidence):
            raise ValueError("hand confidence must be finite")
        if (self.wrist_rotation is None) != (self.wrist_translation is None):
            raise ValueError("wrist rotation and translation must be present together")
        for camera, landmarks in (self.landmarks_2d or {}).items():
            if camera not in {c.source_name for c in CAMERAS if c.rig == 1} or len(landmarks) != NUM_LANDMARKS_PER_HAND:
                raise ValueError("UV landmarks require a headset camera and 21 entries")
            if any(point is not None and (len(point) != 2 or not all(isfinite(value) for value in point)) for point in landmarks):
                raise ValueError("UV landmarks must be finite pixel pairs or null")


@serde
@dataclass(frozen=True, slots=True)
class HandFrame(FrameInfo):
    """A source frame with left (0) and right (1) hand measurements."""

    hand_poses: dict[Literal["0", "1"], HandPose]
    """Both hands, including confidence-zero records."""

    def __post_init__(self) -> None:
        if set(self.hand_poses) != {"0", "1"}:
            raise ValueError("hand frame must contain hands 0 and 1")


HAND_SIDES: tuple[tuple[Literal["0", "1"], str], ...] = (("0", "left"), ("1", "right"))
"""Upstream hand IDs and schema side names."""


def read_hand_frames(hand_path: Path, clock: FrameClock) -> list[HandFrame]:
    """Decode once and align hand records with the metadata frame clock."""
    frames: dict[str, HandFrame] = read_json(hand_path, dict[str, HandFrame])
    if len(frames) != clock.info.num_frames:
        raise ValueError(f"{hand_path}: {len(frames)} hand frames != scene census {clock.info.num_frames}")
    for key, frame in frames.items():
        if key != str(frame.index):
            raise ValueError(f"{hand_path}: frame key {key} disagrees with index {frame.index}")
    selected: list[HandFrame] = []
    for base in clock.frames:
        hand: HandFrame | None = frames.get(str(base.index))
        if hand is None or not agrees_with_frame(hand, base):
            raise ValueError(f"{hand_path}: hand frame {base.index} disagrees with base sidecars")
        selected.append(hand)
    return selected


@serde
@dataclass(frozen=True, slots=True)
class ProfileModel:
    """Partial profile schema; the full JSON is retained as a TextDocument."""

    landmark_rest_positions: Float32[ndarray, "21 3"]
    """Rest landmarks in millimetres."""
    joint_rotation_axes: Float32[ndarray, "22 3"]
    """Joint axes in the local hand model."""

    def __post_init__(self) -> None:
        if len(self.landmark_rest_positions) != NUM_LANDMARKS_PER_HAND:
            raise ValueError("hand profile requires 21 rest landmarks")
        if len(self.joint_rotation_axes) != NUM_JOINTS_PER_HAND:
            raise ValueError("hand profile requires 22 joint rotation axes")


@serde
@dataclass(frozen=True, slots=True)
class HandProfile:
    """Subject profile envelope."""

    hand_model: ProfileModel
    """UmeTrack model; unknown fields remain in the verbatim document."""


@serde
@dataclass(frozen=True, slots=True)
class Caption:
    """All ten released caption fields."""

    object_alias: str
    """Object name used by the annotator."""
    action_hint: str
    """Short action label."""
    hand: str
    """Hand or hands used."""
    interaction_description: str
    """Detailed interaction."""
    start_state: str
    """Initial state."""
    end_state: str
    """Final state."""
    intent: str
    """Annotated intent."""
    scene_description: str
    """Capture setting."""
    additional_observations: str
    """Other observations, possibly empty."""
    overall_caption: str
    """Summary shown first in the instruction pane."""

    def markdown(self) -> str:
        """Summary followed by a Markdown definition list of structured fields."""
        pairs: tuple[tuple[str, str], ...] = (
            ("Object alias", self.object_alias),
            ("Action hint", self.action_hint),
            ("Hand", self.hand),
            ("Interaction description", self.interaction_description),
            ("Start state", self.start_state),
            ("End state", self.end_state),
            ("Intent", self.intent),
            ("Scene description", self.scene_description),
            ("Additional observations", self.additional_observations),
        )
        definitions: list[str] = [f"{label}\n:   " + value.replace("\n", "\n    ") for label, value in pairs]
        return self.overall_caption + "\n\n" + "\n\n".join(definitions)
