"""Typed SHOW3D sidecars and rigid headset calibration; source distances are mm."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import GenericAlias
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray
from serde import SerdeError, serde
from serde.json import from_json

from dataforge.datasets.show3d_calibration import HeadsetCalibration, HeadsetPose
from dataforge.logging_toolkit import frame_index_column, time_column

Split: TypeAlias = Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class Show3dCamera:
    """Stable source-to-rig mapping, including cameras absent from a scene."""

    source_name: str
    """Upstream camera basename."""
    rig: int
    """Rerun rig index."""
    cam: int
    """Camera index within the rig."""


HEADSET_CAMERAS: tuple[Show3dCamera, Show3dCamera] = (Show3dCamera("headset0", 1, 0), Show3dCamera("headset1", 1, 1))
"""The two moving headset cameras, in stereo order."""

CAMERAS: tuple[Show3dCamera, ...] = (
    *HEADSET_CAMERAS,
    *(Show3dCamera(f"rig{i}", 0, i) for i in range(8)),
)
"""Dataset-wide camera identities; missing cameras never shift their peers."""


OBJECTS: dict[str, str | None] = {
    "dumbbell": "dumbbell_5lb",
    "mouse": "mouse",
    "keyboard": "keyboard",
    "mug": "mug_white",
    "mug2": "mug_patterned",
    "balandabowl": "bowl",
    "vase": "vase",
    "brushholder": "holder_black",
    "birdhousetoy": "birdhouse_toy",
    "dinotoy": "dino_toy",
    "whiteboardmarker": "whiteboard_marker",
    "milk": "carton_milk",
    "orangejuice": "carton_oj",
    "mustard": "bottle_mustard",
    "ranch": "bottle_ranch",
    "bbq": "bottle_bbq",
    "cansoup": "can_soup",
    "canparmesan": "can_parmesan",
    "cantomatosauce": "can_tomato_sauce",
    "waffles": "food_waffles",
    "vegetables": "food_vegetables",
    "aria": "aria_small",
    "keyboard2": None,
    "cancoke": None,
    "windex": None,
    "clock": None,
    "mug3": None,
    "none": None,
}


def mesh_name(alias: str) -> str | None:
    """Resolve a known alias; only listed unmapped aliases return None."""
    try:
        return OBJECTS[alias]
    except KeyError as error:
        raise ValueError(f"unknown SHOW3D object alias: {alias}") from error


def calibration_file(key: str, camera: Show3dCamera) -> str:
    """Repository-relative camera calibration."""
    return f"scenes/{key}/camera_calibration/{camera.source_name}.json"


HAND_POSE_VERSION: str = "v2"
"""Released hand annotation version."""
OBJECT_POSE_VERSION: str = "v1"
"""Released object annotation version."""
CAPTIONS_VERSION: str = "v1"
"""Released caption version."""


def hand_pose_file(key: str) -> str:
    """Repository-relative hand measurements."""
    return f"hand_pose/{HAND_POSE_VERSION}/scenes/{key}/hand_pose.json"


def object_pose_file(key: str) -> str:
    """Repository-relative object annotation."""
    return f"object_pose/{OBJECT_POSE_VERSION}/scenes/{key}/object_pose.json"


def hand_profile_file(subject: str) -> str:
    """Repository-relative subject profile."""
    return f"hand_pose/hand_profiles/{subject}/profile_umetrack.json"


def caption_file(key: str) -> str:
    """Repository-relative caption record."""
    return f"captions/{CAPTIONS_VERSION}/scenes/{key}/caption.json"


def scene_id_parts(scene_id: str) -> tuple[str, str]:
    """Validate the object_action_suffix grammar at the index boundary."""
    parts: list[str] = scene_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"invalid SHOW3D scene id: {scene_id!r}")
    return parts[0], "_".join(parts[1:-1])


def validate_component(part: str) -> None:
    """Reject empty, relative and compound SHOW3D path components."""
    if not part or part in (".", "..") or Path(part).name != part:
        raise ValueError(f"invalid SHOW3D path component: {part!r}")


@serde
@dataclass(frozen=True, slots=True)
class IndexRow:
    """Partial index schema; scene files supersede the index at conversion."""

    subject_id: str
    """Subject directory."""
    scene_id: str
    """Scene directory."""
    num_frames: int
    """Index census, used only to reject degenerate rows."""
    has_object_pose: bool
    """Whether to prioritize this scene."""
    split: Split
    """Split attached when reading the parquet."""

    has_hand_pose: bool
    """Whether the index lists hand_pose."""
    has_caption: bool
    """Whether the index lists caption."""

    has_headset0: bool
    """Whether the index lists camera headset0."""
    has_headset1: bool
    """Whether the index lists camera headset1."""
    has_rig0: bool
    """Whether the index lists camera rig0."""
    has_rig1: bool
    """Whether the index lists camera rig1."""
    has_rig2: bool
    """Whether the index lists camera rig2."""
    has_rig3: bool
    """Whether the index lists camera rig3."""
    has_rig4: bool
    """Whether the index lists camera rig4."""
    has_rig5: bool
    """Whether the index lists camera rig5."""
    has_rig6: bool
    """Whether the index lists camera rig6."""
    has_rig7: bool
    """Whether the index lists camera rig7."""

    def __post_init__(self) -> None:
        for part in (self.subject_id, self.scene_id):
            validate_component(part)
        scene_id_parts(self.scene_id)

    @property
    def cameras(self) -> list[Show3dCamera]:
        """Index camera availability used to plan a single raw-input fetch."""
        available: tuple[bool, ...] = (
            self.has_headset0,
            self.has_headset1,
            self.has_rig0,
            self.has_rig1,
            self.has_rig2,
            self.has_rig3,
            self.has_rig4,
            self.has_rig5,
            self.has_rig6,
            self.has_rig7,
        )
        return [camera for camera, enabled in zip(CAMERAS, available, strict=True) if enabled]

    @property
    def object_alias(self) -> str:
        """Object token validated at the index boundary."""
        return scene_id_parts(self.scene_id)[0]

    @property
    def action(self) -> str:
        """Action tokens between object and suffix."""
        return scene_id_parts(self.scene_id)[1]


@serde
@dataclass(frozen=True, slots=True)
class RecordingInfo:
    """Scene-local recording census."""

    start_frame_id: int
    """Original recording's starting frame."""
    num_frames: int
    """Released frames per video."""
    fps: float
    """Nominal container frame rate."""
    resolution: dict[str, list[int]]
    """Present cameras and their height/width."""

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or not self.fps > 0:
            raise ValueError("SHOW3D requires a nonempty recording with positive fps")
        for name, size in self.resolution.items():
            if name not in {camera.source_name for camera in CAMERAS} or len(size) != 2 or min(size) <= 0:
                raise ValueError(f"invalid camera resolution: {name} {size}")


@serde
@dataclass(frozen=True, slots=True)
class FrameInfo:
    """One entry in frame_info.json, in presentation order."""

    index: int
    """Upstream sequence index."""
    agt_frame_id: int
    """Original recording frame id."""
    timestamp: float
    """Source clock in seconds."""
    missing_cameras: list[str]
    """Camera names missing at this frame."""


def agrees_with_frame(record: FrameInfo | HeadsetPose, frame: FrameInfo) -> bool:
    """Match identity and source time within one microsecond."""
    return record.index == frame.index and record.agt_frame_id == frame.agt_frame_id and abs(record.timestamp - frame.timestamp) <= 1e-6


@dataclass(frozen=True, slots=True)
class FrameClock:
    """Validated metadata census and recording clocks, independent of camera sidecars."""

    info: RecordingInfo
    """Full source census."""
    frames: list[FrameInfo]
    """Selected frame rows."""
    times_ns: Int64[ndarray, "n"]
    """Shifted duration clock."""
    frame_indices: Int64[ndarray, "n"]
    """Selected upstream indices."""

    def indexes(self, positions: Int64[ndarray, "n"] | list[int] | slice) -> list[rr.TimeColumn]:
        """Both recording clocks at the selected frame positions."""
        return [time_column(self.times_ns[positions]), frame_index_column(self.frame_indices[positions])]


    def send_sparse(self, recording: rr.RecordingStream, path: str, positions: list[int], columns: Iterable[rr.ComponentColumn]) -> None:
        """Send available rows on both recording clocks; omit absent measurements."""
        if positions:
            rr.send_columns(path, indexes=self.indexes(positions), columns=columns, recording=recording)


T = TypeVar("T")
RowT = TypeVar("RowT")


def sparse_rows(poses: Sequence[RowT], getter: Callable[[RowT], T | None]) -> tuple[list[int], list[T]]:  # noqa: UP047
    """Select sparse measurements and their positions on the shared clock."""
    positions: list[int] = []
    values: list[T] = []
    for position, pose in enumerate(poses):
        value: T | None = getter(pose)
        if value is not None:
            positions.append(position)
            values.append(value)
    return positions, values



def read_frame_clock(scene_dir: Path, scene_key: str) -> FrameClock:
    """Read and cross-check only metadata/recording_info.json and frame_info.json."""
    for name in ("recording_info", "frame_info"):
        if not (scene_dir / f"metadata/{name}.json").is_file():
            raise ValueError(f"{scene_key}: missing metadata/{name}.json")
    info: RecordingInfo = read_json(scene_dir / "metadata/recording_info.json", RecordingInfo)
    frames: list[FrameInfo] = read_json(scene_dir / "metadata/frame_info.json", list[FrameInfo])
    if len(frames) != info.num_frames:
        raise ValueError(f"{scene_key}: frame_info has {len(frames)} rows, recording_info declares {info.num_frames}")
    if frames[0].agt_frame_id != info.start_frame_id:
        raise ValueError(f"{scene_key}: recording start_frame_id disagrees with frame_info")
    indices: Int64[ndarray, "n"] = np.array([frame.index for frame in frames], dtype=np.int64)
    timestamps: Float64[ndarray, "n"] = np.array([frame.timestamp for frame in frames], dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0) or np.any(np.diff(indices) <= 0):
        raise ValueError(f"{scene_key}: source frame indices and timestamps must be finite and strictly increasing")
    times: Int64[ndarray, "n"] = np.rint((timestamps - timestamps[0]) * 1e9).astype(np.int64)
    return FrameClock(info, frames, times, indices)


@serde
@dataclass(frozen=True, slots=True)
class BlurInfo:
    """Face boxes already applied to the released pixels."""

    blur_boxes: dict[str, list[list[float]]]
    """Source index to xyxy boxes in pixels; empty dictionaries are valid."""

    num_normalized_boxes: int = field(init=False, default=0)
    """Boxes whose interpolated corners needed reordering at load time."""

    def __post_init__(self) -> None:
        normalized: int = 0
        for key, boxes in self.blur_boxes.items():
            if int(key) < 0:
                raise ValueError(f"negative blur index {key}")
            for box in boxes:
                if len(box) != 4 or not np.isfinite(box).all():
                    raise ValueError(f"invalid blur box at {key}: {box}")
                if box[2] < box[0] or box[3] < box[1]:
                    box[:] = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
                    normalized += 1
        object.__setattr__(self, "num_normalized_boxes", normalized)


SourceT = TypeVar("SourceT")


def read_json(path: Path, cls: type[SourceT] | GenericAlias, *, text: str | None = None) -> SourceT:  # noqa: UP047 — beartype requires legacy generics
    """Decode a third-party record, naming the file on schema/parser errors."""
    try:
        return from_json(cls, path.read_text() if text is None else text)
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def read_headset_calibrations(scene_dir: Path, clock: FrameClock) -> dict[str, HeadsetCalibration]:
    """Read each headset once and check every pose against the full metadata clock."""
    frames: dict[int, FrameInfo] = {frame.index: frame for frame in clock.frames}
    headsets: dict[str, HeadsetCalibration] = {}
    for camera in HEADSET_CAMERAS:
        path: Path = scene_dir / f"camera_calibration/{camera.source_name}.json"
        headset: HeadsetCalibration = read_json(path, HeadsetCalibration)
        for pose in headset.T_WorldFromCamera_by_index.values():
            frame: FrameInfo | None = frames.get(pose.index)
            if frame is None:
                raise ValueError(f"{path}: headset pose index absent from frame_info")
            if not agrees_with_frame(pose, frame):
                raise ValueError(f"{path}: headset pose {pose.index} disagrees with frame_info")
        headsets[camera.source_name] = headset
    return headsets
