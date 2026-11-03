"""The OAK rig hangs on the Quest's left-eye frame by one mount: a nominal prior in the base layer, replaced by the
fitted mount in the calibration layer (see ``PRIOR_LEAD_NS``). This module turns that into OAK world poses for code
that needs them per frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetSelection
from scipy.spatial.transform import Rotation
from serde import serde

from dataforge.selfcap_schema import HeadConvention, rdf_head

MOUNT_SOURCE = "oak_mount_source"
"""AnyValues field on the OAK rig: ``nominal`` (base prior) or ``fitted`` (calibration)."""
PRIOR_LEAD_NS = 1_000_000
"""The base logs the nominal prior this long before the first Quest pose; calibration logs its fit at that pose. The
viewer resolves the latest timed transform (ahead of a static one), so the fit wins whichever layer was written last."""


def _first(table: pa.Table, name: str):  # noqa: ANN202
    if name not in table.column_names:
        return None
    for value in table[name].to_pylist():
        while isinstance(value, list) and len(value) == 1:
            value = value[0]
        if value is not None:
            return value
    return None


def _rigid(column_major_rotation: list[float] | Float64[ndarray, "9"], translation: list[float] | Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """A 4x4 pose from a Transform3D's stored values; Rerun's mat3x3 is column-major."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = np.asarray(column_major_rotation, dtype=np.float64).reshape(3, 3).T
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return pose


def quaternion_matrices(xyzw: Float64[ndarray, "n 4"]) -> Float64[ndarray, "n 3 3"]:
    """Rotation matrices from xyzw quaternions (normalised first)."""
    return Rotation.from_quat(xyzw).as_matrix()


def compose_track(quest_t: Float64[ndarray, "n 3"], quest_xyzw: Float64[ndarray, "n 4"], mount: Float64[ndarray, "4 4"]) -> Float64[ndarray, "n 4 4"]:
    """``world_T_oak = world_T_quest @ quest_T_oak`` for every Quest sample."""
    world_T_quest = np.tile(np.eye(4), (len(quest_t), 1, 1))
    world_T_quest[:, :3, :3] = quaternion_matrices(quest_xyzw)
    world_T_quest[:, :3, 3] = quest_t
    return world_T_quest @ mount


def read_fitted_mount(view: DatasetSelection, oak: str) -> Float64[ndarray, "4 4"] | None:
    """The fitted Quest-left-eye → OAK-RGB mount of one segment's view, or None when only the prior (or nothing) exists."""
    static = view.filter_contents([oak]).reader(index=None).to_arrow_table()
    rotation, translation = _first(static, f"{oak}:Transform3D:mat3x3"), _first(static, f"{oak}:Transform3D:translation")
    if _first(static, f"{oak}:{MOUNT_SOURCE}") != "fitted" or rotation is None or translation is None:
        return None
    return _rigid(rotation, translation)


@dataclass(frozen=True, slots=True)
class RigTrack:
    """World poses and the basis needed to attach stored camera extrinsics."""

    times_s: Float64[ndarray, "n"]
    """Pose timestamps on video_time, in seconds."""
    world_T_rig: Float64[ndarray, "n 4 4"]
    """Rig poses, with RDF axes when normalization was requested."""
    recorded_T_rig: Float64[ndarray, "4 4"]
    """Read-side rig axes expressed in the unchanged recorded parent frame."""
    head_convention: HeadConvention = "RDF"
    """Source head axes retained when the track was normalized."""


def read_rig_track(
    view: DatasetSelection, rig: str, *, start_ns: int | None = None, end_ns: int | None = None, normalize_head: bool = False
) -> RigTrack | None:
    """A rig's world poses on ``video_time`` (seconds, 4x4), or None when it has no pose in the window."""
    frame = view.filter_contents([rig]).reader(index="video_time")
    if start_ns is not None:
        frame = frame.filter(col("video_time").cast(pa.int64()) >= lit(start_ns))
    if end_ns is not None:
        frame = frame.filter(col("video_time").cast(pa.int64()) <= lit(end_ns))
    rows: pa.Table = frame.select("video_time", f"{rig}:Transform3D:translation", f"{rig}:Transform3D:quaternion").to_arrow_table()
    rows = rows.filter(rows[f"{rig}:Transform3D:quaternion"].is_valid())
    if not len(rows):
        return None
    times_s = rows["video_time"].combine_chunks().cast(pa.int64()).to_numpy() * 1e-9
    translations = np.asarray(rows[f"{rig}:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    quaternions = np.asarray(rows[f"{rig}:Transform3D:quaternion"].to_pylist(), dtype=np.float64).reshape(-1, 4)
    recorded_T_rig: Float64[ndarray, "4 4"] = np.eye(4)
    convention: HeadConvention = "RDF"
    if normalize_head:
        axes = rdf_head(quaternions)
        convention = axes.convention
        quaternions, recorded_T_rig = axes.quaternions, axes.recorded_T_rdf
    return RigTrack(times_s, compose_track(translations, quaternions, np.eye(4)), recorded_T_rig, convention)


def read_oak_track(
    view: DatasetSelection, oak: str, quest: str, *, start_ns: int | None = None, end_ns: int | None = None, poses: DatasetSelection | None = None
) -> tuple[Float64[ndarray, "n"], Float64[ndarray, "n 4 4"]] | None:
    """OAK world poses on ``video_time`` (seconds, 4x4), or None when the segment has no fitted mount.

    The mount comes from ``view``; the Quest track from ``poses`` when the mount lives in a calibration-only layer.
    """
    mount = read_fitted_mount(view, oak)
    if mount is None:
        return None
    track = read_rig_track(view if poses is None else poses, quest, start_ns=start_ns, end_ns=end_ns, normalize_head=True)
    if track is None:
        return None
    return track.times_s, track.world_T_rig @ np.linalg.inv(track.recorded_T_rig) @ mount


@dataclass(frozen=True, slots=True)
class StaticCamera:
    """What a segment's layers hold statically for one camera; a field is None when no layer holds it."""

    rig_T_cam: Float64[ndarray, "4 4"] | None
    """Camera in its rig frame, RDF metres."""
    K: Float64[ndarray, "3 3"] | None
    """Encoded-image pinhole matrix."""
    resolution: tuple[int, int] | None
    """Image width, height."""
    distortion: Float64[ndarray, "c"] | None
    """OpenCV coefficients, k1 first."""


def read_static_cameras(calibration: DatasetSelection, base: DatasetSelection, cameras: list[str]) -> dict[str, StaticCamera]:
    """The static calibration of one segment's camera entities, calibration layer over base layer, component by component.

    The base holds factory facts (OAK and healthy-Quest stereo, lenses); the calibration layer holds fitted lenses and a
    borrowed camera-50 transform. A faulty headset's base has no right-eye transform, so an unresolved one stays None.
    """
    entities: list[str] = [path for camera in cameras for path in (camera, f"{camera}/pinhole")]
    layers: list[pa.Table] = [layer.filter_contents(entities).reader(index=None).to_arrow_table() for layer in (calibration, base)]

    def value(name: str) -> Float64[ndarray, "k"] | None:
        found = next((item for item in (_first(layer, name) for layer in layers) if item is not None), None)
        return None if found is None else np.asarray(found, dtype=np.float64).reshape(-1)

    result: dict[str, StaticCamera] = {}
    for camera in cameras:
        rotation, translation = value(f"{camera}:Transform3D:mat3x3"), value(f"{camera}:Transform3D:translation")
        matrix, size = value(f"{camera}/pinhole:Pinhole:image_from_camera"), value(f"{camera}/pinhole:Pinhole:resolution")
        result[camera] = StaticCamera(
            None if rotation is None or translation is None else _rigid(rotation, translation),
            None if matrix is None else matrix.reshape(3, 3).T,  # Rerun matrices are column-major
            None if size is None else (int(size[0]), int(size[1])),
            value(f"{camera}/pinhole:simplecv.components.DistortionCoefficients"),
        )
    return result


@serde
@dataclass(frozen=True, slots=True)
class Size:
    width: int
    """Array/capture width in pixels."""
    height: int
    """Array/capture height in pixels."""


@serde
@dataclass(frozen=True, slots=True)
class Sensor:
    pixel_array_size: Size
    """Full sensor array before centre crop."""


@serde
@dataclass(frozen=True, slots=True)
class Lens:
    focal_length_x: float | None
    """Horizontal focal in pixels."""
    focal_length_y: float | None
    """Vertical focal in pixels."""
    principal_point_x: float | None
    """Horizontal principal point in sensor pixels."""
    principal_point_y: float | None
    """Vertical principal point in sensor pixels."""
    available: bool
    """Whether factory intrinsics are valid."""


@serde
@dataclass(frozen=True, slots=True)
class Distortion:
    coefficients: tuple[float, ...] | None = None
    """OAK Brown-Conrady vector; absent for Quest."""


@serde
@dataclass(frozen=True, slots=True)
class Intrinsic:
    camera_id: str
    """Physical camera id, including Quest 50/51."""
    positional_layout: str
    """Factory slot name."""
    lens_intrinsics: Lens
    """Raw factory lens values."""
    sensor_info: Sensor
    """Sensor crop metadata."""
    capture_resolution: Size
    """Captured image dimensions."""
    distortion: Distortion
    """Full OAK distortion; Quest has none."""


@serde
@dataclass(frozen=True, slots=True)
class FactoryExtrinsic:
    type: str
    """Factory direction label; matrices follow scratch's device-specific convention."""
    R: Float64[ndarray, "3 3"]
    """Raw factory rotation."""
    T: Float64[ndarray, "3"]
    """Translation: OAK centimetres, Quest metres."""
    to: str = ""
    """Physical destination camera id."""


@serde
@dataclass(frozen=True, slots=True)
class FactoryCalibration:
    device_id: str
    """Full factory device identity."""
    intrinsics: tuple[Intrinsic, ...]
    """Available and unavailable camera slots."""
    extrinsics: tuple[FactoryExtrinsic, ...] = ()
    """Factory transforms, never estimated session calibration."""


def factory_transform(rotation: Float64[ndarray, "3 3"], translation: Float64[ndarray, "3"]) -> Float64[ndarray, "4 4"]:
    """Rigid homogeneous transform from a factory rotation and translation."""
    matrix: Float64[ndarray, "4 4"] = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


# Prior for the OAK mount on the headset: the RGB camera (RDF) in the Quest left-eye frame, as (healthy, camera-50) pairs.
# The mount belongs to the OAK's bracket, so the prior is keyed by OAK device id. Medians of the fitted mounts in
# selfcap-post-ep1-v2 (2026-10-30, 678 sessions); spread about each prior is p50 0.4-1.0 cm / 0.5-1.5 deg. The camera-50
# column is for headsets whose broken left camera borrows camera 50: their eye track's frame sits about 175 deg about x.
OAK_MOUNT_OTHER: tuple[Float64[ndarray, "4 4"], Float64[ndarray, "4 4"]] = (
    np.array([[0.9996, -0.0131, -0.0230, 0.0509], [0.0212, 0.9161, 0.4004, 0.0408], [0.0158, -0.4008, 0.9160, -0.0513], [0, 0, 0, 1.0]]),
    np.array([[0.9996, -0.0036, -0.0271, 0.0510], [-0.0168, -0.8630, -0.5048, -0.0479], [-0.0216, 0.5051, -0.8628, -0.0223], [0, 0, 0, 1.0]]),
)
"""380 healthy / 59 camera-50 sessions on OAKs C1F3CFD2 and F1FBBCD6, and the fallback for an OAK no fit has seen."""
OAK_MOUNTS: dict[str, tuple[Float64[ndarray, "4 4"], Float64[ndarray, "4 4"]]] = {
    "OAK - 14442C10D1FCD7D200": (  # 214 healthy / 25 camera-50 sessions; sits about 4 cm from the others, tilted 7 deg more
        np.array([[0.9989, -0.0328, 0.0346, 0.0306], [0.0108, 0.8627, 0.5056, 0.0671], [-0.0464, -0.5047, 0.8621, -0.0144], [0, 0, 0, 1.0]]),
        np.array([[0.9995, -0.0069, 0.0324, 0.0333], [0.0123, -0.8297, -0.5581, -0.0728], [0.0307, 0.5581, -0.8292, -0.0414], [0, 0, 0, 1.0]]),
    ),
}


def nominal_oak_mount(oak: FactoryCalibration, quest: FactoryCalibration) -> Float64[ndarray, "4 4"] | None:
    """Prior Quest-left-eye → OAK-RGB mount for this OAK and headset layout, or None for a layout no fit has seen."""
    healthy, camera_50 = OAK_MOUNTS.get(oak.device_id, OAK_MOUNT_OTHER)
    layout = headset_layout(quest)
    return None if layout is None else healthy if layout == "healthy" else camera_50


def quest_mesh_frame(oak: FactoryCalibration, quest: FactoryCalibration) -> Float64[ndarray, "4 4"] | None:
    """Where the Quest mesh (built in a healthy left-eye frame) sits in this headset's eye-track frame.

    Identity for a healthy headset. A camera-50 headset's eye frame is turned about 175 deg; both OAK priors describe the
    same physical mount, so ``camera50_T_healthy = camera_50 @ inv(healthy)``. None for a layout no fit has seen.
    """
    healthy, camera_50 = OAK_MOUNTS.get(oak.device_id, OAK_MOUNT_OTHER)
    layout = headset_layout(quest)
    if layout is None:
        return None
    return np.eye(4) if layout == "healthy" else camera_50 @ np.linalg.inv(healthy)


def quest_left_T_right(calibration: FactoryCalibration) -> Float64[ndarray, "4 4"] | None:
    """The right camera in the left camera's frame, when the headset is healthy: both cameras measured and both
    ``device_to_*`` extrinsics present. Faulty headsets return None; their fix borrows from another session."""
    ext = {e.type: e for e in calibration.extrinsics}
    measured = {c.positional_layout for c in calibration.intrinsics if c.lens_intrinsics.available}
    if not ({"device_to_left", "device_to_right"} <= ext.keys() and {"left", "right"} <= measured):
        return None
    left, right = ext["device_to_left"], ext["device_to_right"]
    return np.linalg.inv(factory_transform(left.R.T, left.T)) @ factory_transform(right.R.T, right.T)


def headset_layout(quest: FactoryCalibration) -> Literal["healthy", "camera_50"] | None:
    """Identify the measured eye frame once for both mount and mesh placement."""
    if quest_left_T_right(quest) is not None:
        return "healthy"
    return "camera_50" if any(c.camera_id == "50" for c in quest.intrinsics) else None
