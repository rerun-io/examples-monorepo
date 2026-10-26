"""Gen-HumanWholeBody (GenRobot DAS-Ego) source: Hub paths, one MCAP's typed records, its clock and its lens.

GenRobot publishes no description of the DAS-Ego MCAP beyond its protobuf schemas (the public
``genrobot-ai/das-datakit`` documents the DAS gripper, whose protos have no ``T_b_c``). Every
convention below was therefore solved from the data and checked with pixel overlays on episodes
from two device and calibration versions:

- Quaternions are ``xyzw``.
- ``/robot0/vio/eef_pose`` is ``world_T_rig``: the VIO body frame (forward-left-up) in a Z-up world, metres.
- ``camera_info.T_b_c`` = ``[tx, ty, tz, qx, qy, qz, qw]`` is ``calib_T_cam`` in a camera-like calibration
  body frame; ``RIG_FROM_CALIB`` turns that frame into the VIO body frame. The IMU's gyro agrees with the
  VIO's angular rate under the same rotation (0.4-0.7 degrees on device V6 and V7, zero time offset), and
  its accelerometer reads gravity 0.6 degrees from +z through it, so the calibration body frame is the
  IMU's: the IMU sits at the rig origin, turned by ``RIG_FROM_CALIB``.
- Every topic is stamped on one clock: MCAP ``log_time`` = ``publish_time`` = ``header.timestamp``.
  ``frame_validity`` names camera2 as the reference camera and carries its exact stamps.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun.chunk as rrc
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray
from serde import serde

from dataforge.records import SourceT, decode_values

REPO_ID: str = "genrobot2025/Gen-HumanWholeBody"
"""Hugging Face dataset repository."""
REVISION: str = "22a037591598f07992b2488e5eacc62b06a1036d"
"""Pinned Hub commit. The publisher rewrote the history after the port: the first pin (79fa81bf) is gone; this one serves the same files."""

NUM_CAMERAS: int = 6
"""Cameras on every DAS-Ego: a horizontal fan at 0, +-22 and +-44 degrees."""
REFERENCE_CAMERA: int = 2
"""The camera ``frame_validity`` names as its ``ref_topic``; its frames define ``frame_index``."""
SOURCE_WIDTH: int = 1600
"""Encoded width of every camera, in pixels."""
SOURCE_HEIGHT: int = 1300
"""Encoded height of every camera, in pixels."""
FRAME_RATE_HZ: float = 30.0
"""Nominal camera rate; a stamp off the reference frames is placed at this rate from the nearest one."""
STANDARD_GRAVITY_M_S2: float = 9.80665
"""The accelerometer reports in g; dataforge stores m/s^2."""

RIG_FROM_CALIB: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""Rotation from the calibration (IMU) body frame into the VIO body frame the rig pose is given in."""

DOMAIN_LEVELS: int = 4
"""Most domain directories a Hub path holds; the shallow layout has two, the deep one four."""

Side: TypeAlias = Literal["left", "right"]
"""Hand side, as the handtracking topics name it."""

EEF_POSE_TOPIC: str = "/robot0/vio/eef_pose"
"""World pose of the VIO body frame, 30 Hz."""
BODY_TOPIC: str = "/robotx/motioncapture/human_pose_fuse_smooth_concat"
"""SMPL-X body22 joint positions in world, on the reference camera's stamps (the MCAP topic has no leading ``/``)."""
IMU_TOPIC: str = "/robot0/sensor/imu"
"""Gyro (rad/s) and accelerometer (g), 200 Hz."""
ANNOTATION_TOPIC: str = "/robot0/annotation_v2"
"""One message: task text, domain path, segments and sub-segments."""
FRAME_VALIDITY_TOPIC: str = "/robot0/frame_validity"
"""Per reference frame: validity and failures."""
TIME_RANGE_VALIDITY_TOPIC: str = "/robot0/time_range_validity"
"""One message: valid and invalid ranges with reason codes."""
SYSTEM_INFO_TOPIC: str = "/robot0/system_info"
"""About 1 Hz: firmware, device name and uuid; only the CPU load varies."""
ROBOT_INFO_TOPIC: str = "/robot0/sim/robot_info"
"""Device name plus a ``base_link`` that duplicates the VIO pose bit for bit."""
DUPLICATE_TOPICS: tuple[str, ...] = ("/robot0/vio/relative_eef_pose", ROBOT_INFO_TOPIC)
"""Topics with nothing of their own: an identity pose, and a copy of the VIO pose (its device name is a property)."""


def camera_topic(cam: int) -> str:
    """H.264 packets of one camera, as ``foxglove.CompressedImage``."""
    return f"/robot0/sensor/camera{cam}/compressed"


def calibration_topic(cam: int) -> str:
    """One ``foxglove.CameraCalibration`` per camera, sent once."""
    return f"/robot0/sensor/camera{cam}/camera_info"


def resize_calibration_topic(cam: int) -> str:
    """A 960x780 calibration for an image the file does not contain; dropped."""
    return f"/robot0/sensor/camera{cam}/camera_info_resize"


def hand_topic(side: Side) -> str:
    """One hand's 21 bones and MANO parameters, on the reference camera's stamps."""
    return f"/robot0/handtracking/{side}"


HUB_PATH: re.Pattern[str] = re.compile(r"data/((?:[a-z0-9_]+/){2}(?:(?:[a-z0-9_]+/){2})?)([0-9a-f]{2})/([0-9a-f]{32})\.mcap")
"""``data/<2 or 4 domain levels>/<shard>/<uuid>.mcap``: the only two layouts at the pinned commit."""


def parse_hub_path(path: str) -> tuple[tuple[str, ...], str]:
    """Split a Hub path into its domain levels and the episode uuid.

    Args:
        path: Repository-relative file path.

    Returns:
        The domain directories, outermost first, and the 32-hex uuid that names the episode.

    Raises:
        ValueError: The path is not one of the two layouts, or its shard is not the uuid's prefix.
    """
    match: re.Match[str] | None = HUB_PATH.fullmatch(path)
    if match is None:
        raise ValueError(f"not a Gen-HumanWholeBody episode path: {path}")
    shard: str = match.group(2)
    uuid: str = match.group(3)
    if not uuid.startswith(shard):
        raise ValueError(f"{path}: shard {shard} is not the uuid's prefix")
    return tuple(part for part in match.group(1).split("/") if part), uuid


@dataclass(frozen=True, slots=True)
class DoubleSphere:
    """The double-sphere camera model (Usenko, Demmel, Cremers, 3DV 2018), the source's ``"ds"``.

    ``camera_info.D`` holds its six parameters in this order: ``[fx, fy, cx, cy, xi, alpha]``.
    """

    fx: float
    """Focal length along x, pixels."""
    fy: float
    """Focal length along y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    xi: float
    """Distance between the two unit spheres' centres."""
    alpha: float
    """Weight of the second sphere against the pinhole, in [0, 1]."""

    @classmethod
    def from_coefficients(cls, coefficients: Sequence[float]) -> DoubleSphere:
        """Read ``camera_info.D``."""
        if len(coefficients) != 6:
            raise ValueError(f"a double-sphere calibration has 6 coefficients, got {len(coefficients)}")
        fx, fy, cx, cy, xi, alpha = (float(value) for value in coefficients)
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, xi=xi, alpha=alpha)

    def project(self, points_cam: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
        """Pixels of camera-frame points; NaN where the model has no image of the point (paper eq. 40-45).

        Args:
            points_cam: Points in the camera frame (x right, y down, z forward), metres.

        Returns:
            Image coordinates in pixels, NaN for points outside the model's valid region.
        """
        x: Float64[ndarray, "n"] = points_cam[:, 0]
        y: Float64[ndarray, "n"] = points_cam[:, 1]
        z: Float64[ndarray, "n"] = points_cam[:, 2]
        d1: Float64[ndarray, "n"] = np.sqrt(x * x + y * y + z * z)
        zz: Float64[ndarray, "n"] = self.xi * d1 + z
        d2: Float64[ndarray, "n"] = np.sqrt(x * x + y * y + zz * zz)
        w1: float = self.alpha / (1.0 - self.alpha) if self.alpha <= 0.5 else (1.0 - self.alpha) / self.alpha
        w2: float = (w1 + self.xi) / np.sqrt(2.0 * w1 * self.xi + self.xi * self.xi + 1.0)
        denominator: Float64[ndarray, "n"] = self.alpha * d2 + (1.0 - self.alpha) * zz
        valid: Bool[ndarray, "n"] = np.isfinite(d1) & (z > -w2 * d1) & (denominator > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels: Float64[ndarray, "n 2"] = np.stack([self.fx * x / denominator + self.cx, self.fy * y / denominator + self.cy], axis=1)
        return np.where(valid[:, None], pixels, np.nan)


def struct_field(values: pa.StructArray, name: str) -> pa.Array:
    """One field of a struct column, null wherever the struct itself is null.

    ``StructArray.field`` returns the child as stored, and Arrow lets a null struct keep any child
    values; ``flatten`` applies the parent's validity.
    """
    return values.flatten()[values.type.get_field_index(name)]


def struct_rows(values: pa.StructArray, fields: str = "xyz") -> Float64[ndarray, "n k"]:
    """The named float fields of a struct column, one row per struct (``"xyz"``, ``"xyzw"``); NaN where a struct or field is null."""
    return np.stack([struct_field(values, name).to_numpy(zero_copy_only=False) for name in fields], axis=1).astype(np.float64)


def quaternion_xyzw_to_matrix(quaternion: Float64[ndarray, "4"]) -> Float64[ndarray, "3 3"]:
    """Rotation of a scalar-last quaternion; it need not be normalized."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_matrices(quaternions: Float64[ndarray, "n 4"]) -> Float64[ndarray, "n 3 3"]:
    """Rotations of scalar-last quaternions; an empty batch gives an empty batch."""
    return np.array([quaternion_xyzw_to_matrix(quaternion) for quaternion in quaternions], dtype=np.float64).reshape(-1, 3, 3)


def rig_T_cam(t_b_c: Sequence[float]) -> Float64[ndarray, "4 4"]:
    """A camera's pose in the rig (VIO body) frame from its ``T_b_c`` = ``[tx, ty, tz, qx, qy, qz, qw]``."""
    if len(t_b_c) != 7:
        raise ValueError(f"T_b_c has 7 values, got {len(t_b_c)}")
    values: Float64[ndarray, "7"] = np.asarray(t_b_c, dtype=np.float64)
    transform: Float64[ndarray, "4 4"] = np.eye(4)
    transform[:3, :3] = RIG_FROM_CALIB @ quaternion_xyzw_to_matrix(values[3:])
    transform[:3, 3] = RIG_FROM_CALIB @ values[:3]
    return transform


def rig_T_imu() -> Float64[ndarray, "4 4"]:
    """The IMU's pose in the rig frame: at the origin, turned by ``RIG_FROM_CALIB`` (module docstring)."""
    transform: Float64[ndarray, "4 4"] = np.eye(4)
    transform[:3, :3] = RIG_FROM_CALIB
    return transform


SPS_CONSTRAINT_SET3: int = 0x10
"""``constraint_set3_flag`` in the byte after an H.264 SPS's ``profile_idc``."""


def clear_sps_constraint_set3(samples: pa.ListArray) -> pa.ListArray:
    """Clear ``constraint_set3_flag`` in every H.264 SPS of a column of Annex-B samples.

    The DAS-Ego encoder toggles the flag between keyframes (``avc1.64002A`` and ``avc1.64102A``), and the
    viewer refuses a stream whose codec string changes. The flag carries no meaning for the High profile
    the cameras use. Emulation prevention keeps ``00 00 01`` out of every NAL payload, so a start code
    followed by the SPS header byte ``0x67`` is always an SPS.

    Args:
        samples: One Annex-B access unit per row, as ``list<uint8>``.

    Returns:
        The same rows and offsets with that one bit cleared.
    """
    values: UInt8[ndarray, "n"] = samples.values.to_numpy(zero_copy_only=False).copy()
    is_sps: Bool[ndarray, "m"] = (values[:-3] == 0x00) & (values[1:-2] == 0x00) & (values[2:-1] == 0x01) & (values[3:] == 0x67)
    flags: Int64[ndarray, "k"] = np.flatnonzero(is_sps) + 5  # start code (3), NAL header, profile_idc
    flags = flags[flags < values.size]
    values[flags] &= np.uint8(~SPS_CONSTRAINT_SET3 & 0xFF)
    return pa.ListArray.from_arrays(samples.offsets, pa.array(values, type=samples.type.value_type))


def is_keyframe(sample: bytes) -> bool:
    """Whether an Annex-B access unit holds an IDR slice (NAL type 5), so a decoder can start at it."""
    start: int = sample.find(b"\x00\x00\x01")
    while 0 <= start < len(sample) - 3:
        if sample[start + 3] & 0x1F == 5:
            return True
        start = sample.find(b"\x00\x00\x01", start + 3)
    return False


@dataclass(frozen=True, slots=True)
class EpisodeClock:
    """The episode's one clock: the ``video_time`` origin and the reference camera's shipped frames."""

    t0_ns: int
    """First message stamp of the MCAP; ``video_time`` is measured from it."""
    frame_times_ns: Int64[ndarray, "f"]
    """Stamp of every reference frame ``frame_validity`` ships, ascending."""
    frame_numbers: Int64[ndarray, "f"]
    """Its shipped ``frame_index``, ascending; this is the ``frame_index`` timeline."""

    def __post_init__(self) -> None:
        if self.frame_times_ns.size == 0 or self.frame_times_ns.size != self.frame_numbers.size:
            raise ValueError("a clock needs one shipped frame number per reference stamp, and at least one")
        if np.any(np.diff(self.frame_times_ns) <= 0) or np.any(np.diff(self.frame_numbers) <= 0):
            raise ValueError("reference stamps and frame numbers must both increase")

    @property
    def duration_s(self) -> float:
        """Span of ``video_time`` the reference frames cover."""
        return float(self.frame_times_ns[-1] - self.frame_times_ns[0]) / 1e9

    def video_time_ns(self, stamps_ns: Int64[ndarray, "n"]) -> Int64[ndarray, "n"]:
        """``video_time`` of source stamps."""
        return stamps_ns - self.t0_ns

    def frame_indices(self, stamps_ns: Int64[ndarray, "n"]) -> Int64[ndarray, "n"]:
        """The shipped frame number of each stamp.

        Each stamp takes the number of the nearest reference frame plus the whole frames, at
        ``FRAME_RATE_HZ``, between them: a stamp on a reference frame (the six cameras fire within
        0.16 ms of each other) gets its number; one in a gap of the reference stream gets the number
        the gap skipped; one before the first frame gets ``first - 1``.
        """
        times: Int64[ndarray, "f"] = self.frame_times_ns
        right: Int64[ndarray, "n"] = np.clip(np.searchsorted(times, stamps_ns), 0, times.size - 1)
        left: Int64[ndarray, "n"] = np.maximum(right - 1, 0)
        nearest: Int64[ndarray, "n"] = np.where(np.abs(times[left] - stamps_ns) <= np.abs(times[right] - stamps_ns), left, right)
        frames_away: Int64[ndarray, "n"] = np.rint((stamps_ns - times[nearest]) * FRAME_RATE_HZ / 1e9).astype(np.int64)
        return self.frame_numbers[nearest] + frames_away

    def stamps_ns(self, frame_numbers: Int64[ndarray, "n"]) -> Int64[ndarray, "n"]:
        """Source stamps of frame numbers: the shipped stamp of a reference frame, else ``FRAME_RATE_HZ`` from the nearest one."""
        numbers: Int64[ndarray, "f"] = self.frame_numbers
        right: Int64[ndarray, "n"] = np.clip(np.searchsorted(numbers, frame_numbers), 0, numbers.size - 1)
        left: Int64[ndarray, "n"] = np.maximum(right - 1, 0)
        nearest: Int64[ndarray, "n"] = np.where(np.abs(numbers[left] - frame_numbers) <= np.abs(numbers[right] - frame_numbers), left, right)
        return self.frame_times_ns[nearest] + np.rint((frame_numbers - numbers[nearest]) * 1e9 / FRAME_RATE_HZ).astype(np.int64)


@serde
@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """``foxglove.CameraCalibration`` with the DAS-Ego's extra ``T_b_c``; fields dataforge does not read are allowed."""

    width: int
    """Image width the calibration describes, pixels."""
    height: int
    """Image height, pixels."""
    distortion_model: str
    """``"ds"``, the double-sphere model."""
    D: list[float]
    """``[fx, fy, cx, cy, xi, alpha]``."""
    T_b_c: list[float]
    """Camera pose in the calibration body frame, ``[tx, ty, tz, qx, qy, qz, qw]``."""
    frame_id: str
    """Calibration name and version, e.g. ``das_left_optical_frame_update2body_0.1.1.2``."""

    def __post_init__(self) -> None:
        if self.distortion_model != "ds":
            raise ValueError(f"unsupported camera model {self.distortion_model!r}: the DAS-Ego ships double-sphere ('ds') calibrations")

    @property
    def lens(self) -> DoubleSphere:
        """The double-sphere model these coefficients describe."""
        return DoubleSphere.from_coefficients(self.D)


@serde
@dataclass(frozen=True, slots=True)
class SubSegment:
    """One fine action of a segment; its frames are reference-camera frames."""

    segment_id: str
    """Id within its segment."""
    start_time_s: float
    """Start, seconds from the first reference frame."""
    end_time_s: float
    """End, same clock."""
    fine_label: str
    """What happens."""
    start_frame: int
    """First frame."""
    end_frame: int
    """Last frame; the annotation tool writes it inclusive of nothing in particular, so it is kept as shipped."""
    is_success: bool
    """Whether the action succeeded."""
    fine_label_detail: str = ""
    """Extra detail; empty in every episode seen."""


@serde
@dataclass(frozen=True, slots=True)
class Segment:
    """One coarse action."""

    segment_id: str
    """Id within the episode."""
    start_time_s: float
    """Start, seconds from the first reference frame."""
    end_time_s: float
    """End, same clock."""
    fine_label: str
    """What happens."""
    start_frame: int
    """First frame."""
    end_frame: int
    """Last frame, as shipped."""
    sub_segments_info: list[SubSegment] | None = None
    """Its fine actions; they may cross the segment's own bounds."""
    scene: str = ""
    """Scene name; empty in every episode seen."""
    scene_description: str = ""
    """Scene description; empty in every episode seen."""


@serde
@dataclass(frozen=True, slots=True)
class Annotation:
    """``/robot0/annotation_v2``: the episode's task and actions."""

    video_id: str
    """The device recording uuid (``system_info.uuid``), not the file name."""
    bold_mark: str
    """Task instruction."""
    sst: str
    """Domain path, as the Hub tree spells it."""
    segments_info: list[Segment] | None = None
    """Coarse actions in order."""
    environment_description: str = ""
    """Free text; empty in every episode seen."""

    @property
    def sub_segments(self) -> list[SubSegment]:
        """Every fine action in order."""
        return [sub for segment in self.segments_info or [] for sub in segment.sub_segments_info or []]


@serde
@dataclass(frozen=True, slots=True)
class StartTrigger:
    """How the recording was started."""

    name: str
    """Enum name, e.g. ``RECORDING_START_TRIGGER_BUTTON``."""
    value: int
    """Enum value."""


@serde
@dataclass(frozen=True, slots=True)
class SystemInfo:
    """``/robot0/system_info``: the device and firmware; dataforge reads the first message.

    Every field is optional: older firmware leaves some out.
    """

    version: str | None = None
    """Firmware version."""
    HardwareVersion: str | None = None
    """Board revision, e.g. ``V6``."""
    VersionName: str | None = None
    """Product name, e.g. ``DAS Ego V7``."""
    uuid: str | None = None
    """Recording uuid (``annotation.video_id``); absent on DAS Ego V5 firmware 2.1.10."""
    product_model_id: int | None = None
    """Product model number."""
    recording_start_trigger: StartTrigger | None = None
    """Present from firmware 2.2."""
    abnormal_reason: int | None = None
    """Present from firmware 2.2; 0 = normal."""


@serde
@dataclass(frozen=True, slots=True)
class InvalidRange:
    """One stretch of the recording the source marks invalid."""

    start_time_s: float
    """Start, seconds from the first message."""
    end_time_s: float
    """End, same clock."""
    invalid_code: str
    """Reason code, e.g. ``camera_overlap_range``."""
    invalid_message: str
    """Human-readable reason."""


@serde
@dataclass(frozen=True, slots=True)
class TimeRangeValidity:
    """``/robot0/time_range_validity``."""

    invalid_ranges: list[InvalidRange] | None = None
    """Invalid stretches, in order."""


@serde
@dataclass(frozen=True, slots=True)
class RobotInfo:
    """``/robot0/sim/robot_info``: only the device name is its own."""

    robot_name: str
    """``das_ego``."""


@dataclass(frozen=True, slots=True)
class Episode:
    """Everything a layer writer needs from one MCAP besides its streams."""

    clock: EpisodeClock
    """``video_time`` origin and the reference frames."""
    calibrations: tuple[CameraCalibration, ...]
    """One per camera, in camera order."""
    annotation: Annotation | None
    """Task and actions; None when the episode ships no annotation."""
    system: SystemInfo | None
    """Device and firmware; None when the episode ships no system info."""
    validity: TimeRangeValidity | None
    """Invalid stretches; None when the episode ships no validity ranges."""


def topic_pattern(entity_path: str) -> str:
    """The MCAP topic regex of an entity path.

    ``McapReader`` names each entity after its topic with a leading ``/``, which the body topic lacks
    (``robotx/motioncapture/...``); the pattern accepts either spelling and nothing else.
    """
    return f"^/?{re.escape(entity_path.lstrip('/'))}$"


@dataclass(frozen=True, slots=True)
class EpisodeFile:
    """An episode's raw messages: the source MCAP, or the sidecar rrd that keeps its non-camera topics.

    Both hand out the same chunks (protobuf-decoded ``<schema>:message`` columns on ``message_log_time``),
    so every reader below serves both the first conversion and a rebuild of a derived layer.
    """

    path: Path
    """``.mcap`` or ``.rrd``."""

    def stream(self, topics: Sequence[str], *, start_time_ns: int | None = None, end_time_ns: int | None = None) -> rrc.LazyChunkStream:
        """Temporal chunks of the named topics, optionally only those stamped in ``[start_time_ns, end_time_ns)``."""
        if self.path.suffix == ".mcap":
            reader: rrc.McapReader = rrc.McapReader(
                self.path, decoders=["protobuf"], include_topic_regex=[topic_pattern(topic) for topic in topics], start_time_ns=start_time_ns, end_time_ns=end_time_ns
            )
            return reader.stream().drop(is_static=True)
        if start_time_ns is not None or end_time_ns is not None:
            raise ValueError("a sidecar is read whole")
        return rrc.RrdReader(self.path).store().stream().filter(content=list(topics)).drop(is_static=True)

    def first_stamp_ns(self) -> int:
        """Stamp of the MCAP's first message, cameras included: the ``video_time`` origin.

        A sidecar keeps the MCAP's own statistics (``/__mcap_properties``), so it answers for the
        camera topics it no longer holds.
        """
        if self.path.suffix == ".mcap":
            return rrc.McapReader(self.path).time_bounds()[0]
        for chunk in rrc.RrdReader(self.path).store().stream().filter(content=MCAP_PROPERTIES, components=[MCAP_START_COMPONENT]):
            return int(chunk.to_record_batch().column(MCAP_START_COMPONENT).to_pylist()[0][0])
        raise ValueError(f"{self.path}: no {MCAP_PROPERTIES} statistics; it was not written by write_sidecar")


def messages(stream: rrc.LazyChunkStream) -> Iterator[tuple[Int64[ndarray, "n"], pa.Array]]:
    """Each chunk's stamps and its message column (one message per row, the list level removed)."""
    for chunk in stream:
        batch: pa.RecordBatch = chunk.to_record_batch()
        column: str = next(name for name in batch.schema.names if name.endswith(":message"))
        stamps: Int64[ndarray, "n"] = np.asarray(batch.column("message_log_time").cast(pa.int64()))
        values: pa.ListArray = batch.column(column)
        if not np.all(values.value_lengths().to_numpy(zero_copy_only=False) == 1):
            raise ValueError(f"{chunk.entity_path}: a row holds other than one message")
        yield stamps, values.flatten()


def topic_rows(file: EpisodeFile, topic: str) -> tuple[Int64[ndarray, "n"], pa.Array]:
    """One topic's stamps and messages, in stamp order."""
    parts: list[tuple[Int64[ndarray, "n"], pa.Array]] = list(messages(file.stream([topic])))
    if not parts:
        return np.zeros(0, dtype=np.int64), pa.array([])
    stamps: Int64[ndarray, "n"] = np.concatenate([stamps for stamps, _ in parts])
    values: pa.Array = pa.concat_arrays([values for _, values in parts])
    order: Int64[ndarray, "n"] = np.argsort(stamps, kind="stable")
    return stamps[order], values.take(pa.array(order))


def first_message(file: EpisodeFile, topic: str, cls: type[SourceT]) -> SourceT:  # noqa: UP047 — beartype requires legacy generics
    """The topic's first message as a typed record, naming the file and topic when it is absent or does not fit."""
    record: SourceT | None = optional_message(file, topic, cls)
    if record is None:
        raise ValueError(f"{file.path}: no {topic} message")
    return record


def optional_message(file: EpisodeFile, topic: str, cls: type[SourceT]) -> SourceT | None:  # noqa: UP047 — beartype requires legacy generics
    """The topic's first message as a typed record, or None when the topic has none."""
    _, values = topic_rows(file, topic)
    if len(values) == 0:
        return None
    return decode_values(cls, values[0].as_py(), source=f"{file.path}:{topic}")


MCAP_PROPERTIES: str = "/__mcap_properties"
"""Where ``McapReader`` puts the file's own statistics."""
MCAP_START_COMPONENT: str = "McapStatistics:message_start_time"
"""First message stamp of the whole MCAP."""


def read_episode(file: EpisodeFile) -> Episode:
    """Read the clock and the typed records of one episode, from the MCAP or the sidecar written from it.

    The reference frames and the six calibrations are required; the annotation, system info and validity
    ranges are read when the episode ships them.
    """
    frame_stamps, frames = topic_rows(file, FRAME_VALIDITY_TOPIC)
    if frame_stamps.size == 0:
        raise ValueError(f"{file.path}: no {FRAME_VALIDITY_TOPIC}, so no reference frames")
    numbers: Int64[ndarray, "f"] = frames.field("frame_index").to_numpy(zero_copy_only=False).astype(np.int64)
    return Episode(
        clock=EpisodeClock(t0_ns=file.first_stamp_ns(), frame_times_ns=frame_stamps, frame_numbers=numbers),
        calibrations=tuple(first_message(file, calibration_topic(cam), CameraCalibration) for cam in range(NUM_CAMERAS)),
        annotation=optional_message(file, ANNOTATION_TOPIC, Annotation),
        system=optional_message(file, SYSTEM_INFO_TOPIC, SystemInfo),
        validity=optional_message(file, TIME_RANGE_VALIDITY_TOPIC, TimeRangeValidity),
    )
