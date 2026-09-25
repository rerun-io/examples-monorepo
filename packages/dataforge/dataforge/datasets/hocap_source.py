"""Typed HO-Cap archive boundary; no extracted tree is read or written."""

from dataclasses import dataclass
from io import BytesIO
from typing import Literal, TypeAlias, TypeVar
from zipfile import ZipFile

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import SerdeError, coerce, serde
from serde.yaml import from_yaml
from yaml import YAMLError

from dataforge.hands import coco133_from_coco_hands

EGO_RIG: int = 8
EXO_RIGS: tuple[int, ...] = tuple(range(8))

FRAME_RATE: int = 30
"""Shared frame-index rate; origin recorded in CLOCK_SOURCE."""
CLOCK_SOURCE: str = (
    "no timestamps shipped; 30 Hz inferred from HO-Cap paper §5.1 (10 FPS subsample = mean frame step 3.0 in the released hpe benchmark GT)"
)
SOURCE_REVISION: str = "d9a562638bc4eda48451ce1211385371bbd17be0"

HandSide: TypeAlias = Literal["left", "right"]
"""MANO hand side as meta.yaml spells it."""
MANO_SIDES: tuple[HandSide, HandSide] = ("right", "left")
"""Hand order of poses_m.npy and of every shipped label array."""


@serde
@dataclass(frozen=True, slots=True)
class RealSense:
    """Release image geometry and camera identities."""

    serials: list[str]
    """Source ordering; consumers sort by serial."""
    width: int
    """Color width."""
    height: int
    """Color height."""


@serde
@dataclass(frozen=True, slots=True)
class HoloLens:
    """Release PV image stream."""

    serial: str
    """PV camera name."""
    pv_width: int
    """PV width."""
    pv_height: int
    """PV height."""


@serde
@dataclass(frozen=True, slots=True)
class Metadata:
    """Per-sequence meta.yaml; no inferred task names."""

    realsense: RealSense
    """Eight fixed cameras."""
    hololens: HoloLens
    """One moving PV camera."""
    extrinsics: str
    """Calibration filename chosen by this sequence."""
    subject_id: str
    """Subject owning the MANO shape."""
    object_ids: list[str]
    """Object pose slot order."""
    mano_sides: list[HandSide]
    """Present hands."""
    num_frames: int
    """Full shared index length."""
    task_id: int
    """Uninterpreted upstream task ID."""

    def __post_init__(self) -> None:
        if self.num_frames <= 0 or len(set(self.realsense.serials)) != len(EXO_RIGS) or len(self.object_ids) != 4:
            raise ValueError("HOCap requires positive frame count, eight distinct RealSense cameras and four objects")


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Color:
    """Undistorted color pinhole; coefficients are source metadata only."""

    width: int
    """Image width."""
    height: int
    """Image height."""
    fx: float
    """Horizontal focal length."""
    fy: float
    """Vertical focal length."""
    ppx: float
    """Principal x."""
    ppy: float
    """Principal y."""

    @property
    def matrix(self) -> Float64[ndarray, "3 3"]:
        return np.array([[self.fx, 0.0, self.ppx], [0.0, self.fy, self.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)


@serde
@dataclass(frozen=True, slots=True)
class CalibrationIntrinsics:
    """Only color is ingested in this release."""

    color: Color
    """Color pinhole."""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class CalibrationExtrinsics:
    """Master-frame 3x4 transforms, in metres."""

    extrinsics: dict[str, list[float]]
    """Camera and tag transforms."""

    def __post_init__(self) -> None:
        if "tag_1" not in self.extrinsics:
            raise ValueError("extrinsics requires tag_1")
        if any(len(row) != 12 for row in self.extrinsics.values()):
            raise ValueError("extrinsics rows must contain 12 floats")

    def world_T_cam(self, serial: str) -> Float64[ndarray, "4 4"]:
        tag: Float64[ndarray, "4 4"] = np.eye(4)
        camera: Float64[ndarray, "4 4"] = np.eye(4)
        tag[:3] = np.asarray(self.extrinsics["tag_1"]).reshape(3, 4)
        camera[:3] = np.asarray(self.extrinsics[serial]).reshape(3, 4)
        return np.linalg.inv(tag) @ camera


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Shape:
    """Subject shape shared by both hands."""

    betas: list[float]
    """Ten MANO shape coefficients."""

    def __post_init__(self) -> None:
        if len(self.betas) != 10:
            raise ValueError("MANO requires ten betas")


T = TypeVar("T")


def read_yaml(archive: ZipFile, member: str, cls: type[T]) -> T:  # noqa: UP047 — keep Python 3.10-compatible annotations
    """Decode one third-party record, retaining the archive member in errors."""
    try:
        return from_yaml(cls, archive.read(member).decode())
    except (SerdeError, YAMLError) as error:
        raise ValueError(f"{archive.filename}:{member}: {error}") from error


def frame_times(count: int) -> Int64[ndarray, "n"]:
    """Nearest nanosecond on the inferred shared frame-index clock."""
    return np.rint(np.arange(count, dtype=np.float64) * (1e9 / FRAME_RATE)).astype(np.int64)


def present_rows(rows: Float32[ndarray, "n d"]) -> Bool[ndarray, "n"]:
    """HO-Cap marks a missing joint, pose or parameter row with NaN or with -1 in every entry."""
    return np.isfinite(rows).all(axis=-1) & ~np.all(rows == -1, axis=-1)


def posed_rows(poses: Float32[ndarray, "n 7"]) -> Bool[ndarray, "n"]:
    """Select finite Float32[n,7] pose rows with a nonzero quaternion."""
    return present_rows(poses) & (np.linalg.norm(poses[:, :4], axis=-1) > 0)


def pose_matrices(poses: Float32[ndarray, "n 7"]) -> Float64[ndarray, "n 4 4"]:
    """Decode xyzw + metres, preserving invalid rows as NaN, never carry forward."""
    valid: Bool[ndarray, "n"] = posed_rows(poses)
    matrices: Float64[ndarray, "n 4 4"] = np.full((len(poses), 4, 4), np.nan, dtype=np.float64)
    matrices[valid] = np.eye(4)
    matrices[valid, :3, :3] = Rotation.from_quat(poses[valid, :4]).as_matrix()
    matrices[valid, :3, 3] = poses[valid, 4:]
    return matrices


@dataclass(frozen=True, slots=True)
class HocapCamera:
    """Source camera and its stable recording identity."""

    serial: str
    """Archive camera directory."""
    rig: int
    """Recording rig index."""
    kind: Literal["exo", "ego"]
    """Fixed RealSense or moving HoloLens."""


@dataclass(frozen=True, slots=True)
class SequenceData:
    """Validated source metadata and full pose tables."""

    meta: Metadata
    """Sequence metadata."""
    cameras: tuple[HocapCamera, ...]
    """Sorted RealSense serials, then PV."""
    intrinsics: tuple[Color, ...]
    """Color pinholes in camera order."""
    world_T_cam: Float64[ndarray, "e 4 4"]
    """Static RealSense transforms."""
    mano: Float32[ndarray, "2 n 51"]
    """Right then left MANO PCA parameters."""
    objects: Float32[ndarray, "4 n 7"]
    """World object xyzw and translation."""
    pv: Float32[ndarray, "n 7"]
    """World PV xyzw and translation."""
    betas: Float32[ndarray, "10"]
    """Shared subject shape."""
    times_ns: Int64[ndarray, "n"]
    """Inferred video_time of each selected frame."""
    frame_indices: Int64[ndarray, "n"]
    """Source index of each selected frame."""

    @property
    def exo(self) -> tuple[HocapCamera, ...]:
        """Fixed cameras in rig order."""
        return tuple(camera for camera in self.cameras if camera.kind == "exo")

    @property
    def ego(self) -> HocapCamera:
        """The single HoloLens camera."""
        return next(camera for camera in self.cameras if camera.kind == "ego")

    @property
    def count(self) -> int:
        """Selected prefix length."""
        return len(self.frame_indices)


def read_sequence(
    subject: ZipFile, calibration: ZipFile, poses: ZipFile, key: str, frame_limit: int | None = None, *, members: frozenset[str]
) -> SequenceData:
    """Validate full stream counts before selecting a test prefix."""
    meta: Metadata = read_yaml(subject, f"{key}/meta.yaml", Metadata)
    if meta.subject_id != key.split("/")[0]:
        raise ValueError(f"{key}: subject_id disagrees with archive path")
    if frame_limit is not None and frame_limit <= 0:
        raise ValueError("frame_limit must be positive")
    count: int = meta.num_frames if frame_limit is None else min(frame_limit, meta.num_frames)
    cameras: tuple[HocapCamera, ...] = (
        *[HocapCamera(serial, rig, "exo") for rig, serial in zip(EXO_RIGS, sorted(meta.realsense.serials), strict=True)],
        HocapCamera(meta.hololens.serial, EGO_RIG, "ego"),
    )
    for camera in cameras:
        serial: str = camera.serial
        expected: set[str] = {f"{key}/{serial}/color_{index:06d}.jpg" for index in range(meta.num_frames)}
        actual: set[str] = {member for member in members if member.startswith(f"{key}/{serial}/color_") and member.endswith(".jpg")}
        if actual != expected:
            raise ValueError(f"{key}/{serial}: color frame indices disagree with num_frames={meta.num_frames}")
    intrinsics: tuple[Color, ...] = tuple(
        read_yaml(calibration, f"calibration/intrinsics/{camera.serial}.yaml", CalibrationIntrinsics).color for camera in cameras
    )
    for rig, camera in enumerate(intrinsics):
        expected_size: tuple[int, int] = (
            (meta.realsense.width, meta.realsense.height) if cameras[rig].kind == "exo" else (meta.hololens.pv_width, meta.hololens.pv_height)
        )
        if (camera.width, camera.height) != expected_size:
            raise ValueError(f"{key}/{cameras[rig].serial}: calibration resolution differs from meta.yaml")
    extrinsics: CalibrationExtrinsics = read_yaml(calibration, f"calibration/extrinsics/{meta.extrinsics}", CalibrationExtrinsics)
    for camera in cameras:
        serial = camera.serial
        if camera.kind == "exo" and serial not in extrinsics.extrinsics:
            raise ValueError(f"{key}: extrinsics {meta.extrinsics} lacks {serial}")
    mano: Float32[ndarray, "2 n 51"] = np.load(BytesIO(poses.read(f"{key}/poses_m.npy")), allow_pickle=False)
    objects: Float32[ndarray, "4 n 7"] = np.load(BytesIO(poses.read(f"{key}/poses_o.npy")), allow_pickle=False)
    pv: Float32[ndarray, "n 7"] = np.load(BytesIO(poses.read(f"{key}/poses_pv.npy")), allow_pickle=False)
    if mano.shape != (2, meta.num_frames, 51) or objects.shape != (4, meta.num_frames, 7) or pv.shape != (meta.num_frames, 7):
        raise ValueError(f"{key}: pose counts disagree with num_frames={meta.num_frames}")
    shape: Shape = read_yaml(calibration, f"calibration/mano/{meta.subject_id}.yaml", Shape)
    return SequenceData(
        meta,
        cameras,
        intrinsics,
        np.stack([extrinsics.world_T_cam(camera.serial) for camera in cameras if camera.kind == "exo"]),
        mano[:, :count],
        objects[:, :count],
        pv[:count],
        np.asarray(shape.betas, dtype=np.float32),
        frame_times(count),
        np.arange(count, dtype=np.int64),
    )


@dataclass(frozen=True, slots=True)
class HandLabels:
    """Shipped labels assembled on the shared clock."""

    xyz: Float32[ndarray, "n 133 3"]
    """World joints, choosing an available camera per frame."""
    uv: Float32[ndarray, "e n 133 2"]
    """Each camera's own pixels; missing files remain NaN."""


def read_labels(archive: ZipFile, key: str, scene: SequenceData, *, members: frozenset[str]) -> HandLabels:
    """Recover 3D from any available camera, preserving the known master-camera gap."""
    xyz: Float32[ndarray, "n 133 3"] = np.full((scene.count, 133, 3), np.nan, dtype=np.float32)
    uv: Float32[ndarray, "e n 133 2"] = np.full((len(scene.exo), scene.count, 133, 2), np.nan, dtype=np.float32)
    selected: set[int] = set()
    for camera in scene.exo:
        rig: int = camera.rig
        serial: str = camera.serial
        transform: Float64[ndarray, "4 4"] = scene.world_T_cam[rig]
        for index in range(scene.count):
            member: str = f"{key}/{serial}/label_{index:06d}.npz"
            if member not in members:
                continue
            with np.load(BytesIO(archive.read(member)), allow_pickle=False) as label:
                pixels: Float32[ndarray, "2 21 2"] = label["hand_joints_2d"].astype(np.float32)
                pixels = np.where(present_rows(pixels.reshape(-1, 2)).reshape(2, 21, 1), pixels, np.float32(np.nan))
                uv[rig, index] = coco133_from_coco_hands(pixels[::-1])
                if index not in selected:
                    joints: Float32[ndarray, "2 21 3"] = label["hand_joints_3d"]
                    joints = np.where(present_rows(joints.reshape(-1, 3)).reshape(2, 21, 1), joints, np.float32(np.nan))
                    camera_joints: Float32[ndarray, "133 3"] = coco133_from_coco_hands(joints[::-1])
                    xyz[index] = (camera_joints @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
                    selected.add(index)
    return HandLabels(xyz, uv)
