"""Typed SHOW3D sidecars and rigid headset calibration; source distances are mm."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import GenericAlias
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import SerdeError, serde
from serde.json import from_json

from dataforge.logging_toolkit import frame_index_column, time_column

Split: TypeAlias = Literal["train", "test"]
PoseSource: TypeAlias = Literal["mocap", "vio", "endpoint_interpolation", "legacy_unspecified"]


@dataclass(frozen=True, slots=True)
class Show3dCamera:
    """Stable source-to-rig mapping, including cameras absent from a scene."""

    source_name: str
    """Upstream camera basename."""
    rig: int
    """Rerun rig index."""
    cam: int
    """Camera index within the rig."""


CAMERAS: tuple[Show3dCamera, ...] = (
    Show3dCamera("headset0", 1, 0),
    Show3dCamera("headset1", 1, 1),
    *(Show3dCamera(f"rig{i}", 0, i) for i in range(8)),
)
"""Dataset-wide camera identities; missing cameras never shift their peers."""


HAND_POSE_VERSION: str = "v2"
"""Released hand annotation version."""
CAPTIONS_VERSION: str = "v1"
"""Released caption version."""


def hand_pose_file(key: str) -> str:
    """Repository-relative hand measurements."""
    return f"hand_pose/{HAND_POSE_VERSION}/scenes/{key}/hand_pose.json"


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


def validate_transform(transform: Float64[ndarray, "4 4"]) -> None:
    """Reject non-rigid Float64[ndarray, '4 4'] transforms at the boundary."""
    rotation: Float64[ndarray, "3 3"] = transform[:3, :3]
    if (
        not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("T_WorldFromCamera must be a finite proper rigid transform (det=1)")


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
class HeadsetPose:
    """Legacy and version-1 headset pose entries."""

    index: int
    """Released video index."""
    agt_frame_id: int
    """Original frame id."""
    timestamp: float
    """Source seconds."""
    T_WorldFromCamera: Float64[ndarray, "4 4"] | None
    """Camera-to-back-rig transform in mm; absent transforms produce no pose row."""
    is_synthesized: bool
    """Legacy interpolation flag, independent of validity."""
    pose_source: PoseSource | None = None
    """New contract provenance; absent in legacy files."""
    is_pose_valid: bool | None = None
    """New contract validity; absent in legacy files."""

    def __post_init__(self) -> None:
        if self.T_WorldFromCamera is not None:
            validate_transform(self.T_WorldFromCamera)


@serde
@dataclass(frozen=True, slots=True)
class Intrinsics:
    """Released undistorted pinhole intrinsics."""

    ImageSizeX: int
    """Width in pixels."""
    ImageSizeY: int
    """Height in pixels."""
    fx: float
    """Horizontal focal length in pixels."""
    fy: float
    """Vertical focal length in pixels."""
    cx: float
    """Principal point x in pixels."""
    cy: float
    """Principal point y in pixels."""
    DistortionModel: Literal["PinholePlane"]
    """The released images are already undistorted."""


@serde
@dataclass(frozen=True, slots=True)
class RigCalibration(Intrinsics):
    """One fixed back-rig camera."""

    T_WorldFromCamera: Float64[ndarray, "4 4"]
    """Camera-to-back-rig transform in mm."""

    def __post_init__(self) -> None:
        validate_transform(self.T_WorldFromCamera)


@serde
@dataclass(frozen=True, slots=True)
class HeadsetCalibration(Intrinsics):
    """One headset camera, including sparse pose rows."""

    T_WorldFromCamera_by_index: dict[str, HeadsetPose]
    """Pose records keyed by the decimal source index."""
    pose_contract_version: int | None = None
    """Optional new contract version."""

    def __post_init__(self) -> None:
        for key, pose in self.T_WorldFromCamera_by_index.items():
            if key != str(pose.index):
                raise ValueError(f"headset pose key {key} disagrees with index {pose.index}")


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


@dataclass(frozen=True, slots=True)
class HeadsetRig:
    """Measured fixed stereo transform with its rigidity evidence."""

    cam0_T_cam1: Float64[ndarray, "4 4"]
    """Camera 1 in camera 0's frame, in metres."""
    translation_std_m: float
    """Maximum coordinate standard deviation over measured pairs."""
    rotation_max_deg: float
    """Maximum angular deviation from the mean rotation."""


def headset_rig(left: HeadsetCalibration, right: HeadsetCalibration, *, scene: str) -> HeadsetRig:
    """Fit non-synthesized valid stereo pairs and reject a non-rigid scene."""
    relatives: list[Float64[ndarray, "4 4"]] = []
    for key, pose in left.T_WorldFromCamera_by_index.items():
        peer: HeadsetPose | None = right.T_WorldFromCamera_by_index.get(key)
        if (
            peer is None
            or pose.is_synthesized
            or peer.is_synthesized
            or pose.is_pose_valid is False
            or peer.is_pose_valid is False
            or pose.T_WorldFromCamera is None
            or peer.T_WorldFromCamera is None
        ):
            continue
        relatives.append(np.linalg.inv(pose.T_WorldFromCamera) @ peer.T_WorldFromCamera)
    if not relatives:
        raise ValueError(f"{scene}: no non-synthesized valid headset pairs for rigidity")
    transforms: Float64[ndarray, "n 4 4"] = np.stack(relatives)
    mean: Float64[ndarray, "4 4"] = transforms.mean(axis=0)
    mean[:3, :3] = Rotation.from_matrix(mean[:3, :3]).as_matrix()
    translation_std: float = float(transforms[:, :3, 3].std(axis=0).max()) * 0.001
    rotation_max: float = float(np.degrees(Rotation.from_matrix(transforms[:, :3, :3] @ mean[:3, :3].T).magnitude()).max())
    if translation_std >= 0.001 or rotation_max >= 0.5:
        raise ValueError(f"{scene}: headset is not rigid: translation std {translation_std * 1000:.6g} mm, rotation deviation {rotation_max:.6g} deg")
    mean[:3, 3] *= 0.001
    return HeadsetRig(mean, translation_std, rotation_max)
