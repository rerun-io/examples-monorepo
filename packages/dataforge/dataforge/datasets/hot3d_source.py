"""Read HOT3D sidecars without resampling, rebasing, or carrying poses forward."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import SerdeError, field, from_dict, serde
from serde.json import from_json

from dataforge.datasets.show3d_source import read_json

Device: TypeAlias = Literal["aria", "quest3"]
"""The two source devices."""


URL_LIST_DATE: str = "20260924"
"""Download census revision shared by filenames and capture properties."""


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Shipped device facts used by discovery, readers and writers."""

    url_label: str
    """Device spelling in the URL-list filename."""
    camera_streams: tuple[tuple[str, str], ...]
    """VRS stream IDs and display names in camera order."""
    view_coordinates: rr.components.ViewCoordinates
    """Source world axes."""
    rig_forward: tuple[float, float, float]
    """Headset forward in the rig frame: camera 0's optical axis (measured from its shipped pose)."""
    rig_up: tuple[float, float, float]
    """Headset up in the rig frame: minus camera 0's image y axis (the RGB image is shown rotated)."""
    clock_source: str
    """Capture property describing native clock provenance."""
    has_imu: bool
    """Whether the VRS has IMU streams."""
    has_timecode_mapping: bool
    """Whether GT uses a shipped timecode-to-device mapping."""


DEVICES: dict[Device, DeviceSpec] = {
    "aria": DeviceSpec(
        "Aria",
        (("214-1", "RGB"), ("1201-1", "SLAM left"), ("1201-2", "SLAM right")),
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        (0.1, -0.62, 0.78),
        (-0.99, -0.09, 0.06),
        "Aria VRS DEVICE_TIME unshifted; labels joined by timecode to shipped devicetime_ns (usually capture minus 1 ns)",
        True,
        True,
    ),
    "quest3": DeviceSpec(
        "Quest",
        (("1201-1", "Left"), ("1201-2", "Right")),
        rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        (0.96, -0.28, 0.08),
        (0.28, 0.96, 0.0),
        "Quest 3 VRS capture timestamps, 30 Hz grid as shipped; unshifted",
        False,
        False,
    ),
}


@serde
@dataclass(frozen=True, slots=True)
class Metadata:
    """The source's GT availability and episode identities."""

    have_hand_object_pose_gt: bool
    """False publishes only the base layer."""
    participant_id: str
    """Source participant identifier."""
    object_uids: list[str]
    """Native HOT3D asset IDs, not BOP IDs."""


@dataclass(frozen=True, slots=True)
class Hot3dSource:
    """A verified sequence and the metadata decoded during discovery."""

    path: Path
    """Read-only sequence directory."""
    metadata: Metadata
    """GT availability and episode identities."""


@serde
@dataclass(frozen=True, slots=True)
class DownloadFile:
    """Only the size is needed for verification; URLs are never used."""

    file_size_bytes: int
    """Expected complete file size."""


@serde
@dataclass(frozen=True, slots=True)
class Release:
    """The files extracted from each release group."""

    data_groups: dict[str, list[str]]
    """Required relative filenames by group."""
    release: str
    """Upstream release identifier."""

    def __post_init__(self) -> None:
        if self.release != "v4.0.0":
            raise ValueError(f"unsupported HOT3D release {self.release}")


@serde
@dataclass(frozen=True, slots=True)
class Manifest:
    """Download census used as a read-only completeness contract."""

    sequences: dict[str, dict[str, DownloadFile]]
    """Per-sequence download sizes."""
    sequence_config: Release
    """Release and extracted file inventory."""


def complete_sequences(root: Path, device: Device, selections: tuple[str, ...] | None) -> list[Hot3dSource]:
    """Verify local folders; a partial sequence is skipped with an explicit reason."""
    label: str = DEVICES[device].url_label
    manifest: Manifest = read_json(root / f"Hot3D{label}_download_urls-{URL_LIST_DATE}.json", Manifest)
    result: list[Hot3dSource] = []
    for source in sorted((root / device).glob("*")):
        if not source.is_dir() or (selections is not None and source.name not in selections):
            continue
        reason: str | None = None
        metadata: Metadata | None = None
        entry: DownloadFile | None = manifest.sequences.get(source.name, {}).get("main_vrs")
        vrs: Path = source / "recording.vrs"
        if entry is None:
            reason = "absent from URL list"
        elif not vrs.is_file():
            reason = "missing recording.vrs"
        elif vrs.stat().st_size != entry.file_size_bytes:
            reason = f"recording.vrs size {vrs.stat().st_size} != {entry.file_size_bytes}"
        else:
            required: list[str] = ["metadata.json", "camera_models.json"]
            if (source / "metadata.json").is_file():
                try:
                    metadata = from_json(Metadata, (source / "metadata.json").read_text())
                except (SerdeError, json.JSONDecodeError) as error:
                    print(f"skip {source.name}: {source / 'metadata.json'}: {error}")
                    continue
                if metadata.have_hand_object_pose_gt:
                    required += manifest.sequence_config.data_groups["ground_truth"] + manifest.sequence_config.data_groups["hand_data"]
                    if DEVICES[device].has_timecode_mapping:
                        required.append("timecode_devicetime_mapping.csv")
            missing: list[str] = sorted({name for name in required if not (source / name).is_file()})
            if missing:
                reason = f"missing {', '.join(missing)}"
        if reason is not None:
            print(f"skip {source.name}: {reason}")
        elif metadata is not None:
            result.append(Hot3dSource(source, metadata))
    return result


@serde
@dataclass(frozen=True, slots=True)
class Wrist:
    """World-from-wrist in the source's metre units."""

    t_xyz: Float32[ndarray, "3"]
    """World translation, metres."""
    q_wxyz: Float64[ndarray, "4"]
    """Scalar-first rotation quaternion."""

    def matrix(self) -> Float32[ndarray, "4 4"]:
        """Return the source rigid transform in metres."""
        transform: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
        transform[:3, :3] = Rotation.from_quat(self.q_wxyz, scalar_first=True).as_matrix()
        transform[:3, 3] = self.t_xyz
        return transform


@serde
@dataclass(frozen=True, slots=True)
class UmePose:
    """A present hand, with no inferred confidence."""

    wrist_xform: Wrist
    """World wrist pose."""
    joint_angles: Float32[ndarray, "22"]
    """UmeTrack angles in radians."""
    hand_confidence: float
    """Shipped confidence."""


@serde
@dataclass(frozen=True, slots=True)
class UmeFrame:
    """Every source row, including empty hands and all three Aria stamps."""

    timestamp_ns: int
    """Source timecode (Aria) or device time (Quest)."""
    hand_poses: dict[Literal["0", "1"], UmePose]
    """Absent key means absent hand."""


@serde
@dataclass(frozen=True, slots=True)
class ManoPose:
    """MANO parameters are data only, without zero padding."""

    pose: Float32[ndarray, "15"]
    """PCA coefficients, not axis angles."""
    wrist_xform: Wrist
    """World wrist pose."""
    betas: Float32[ndarray, "10"]
    """Per-hand shape coefficients."""


@serde
@dataclass(frozen=True, slots=True)
class ManoFrame:
    """A MANO row indexed by its own source timestamp."""

    timestamp_ns: int
    """Source timecode or device time; zero is invalid."""
    hand_poses: dict[Literal["0", "1"], ManoPose]
    """Source handedness: 0 left, 1 right."""


@dataclass(frozen=True, slots=True)
class LabelClock:
    """An exact source-to-device join, never a positional or nearest join."""

    mapping: dict[int, int]
    """Timecode to device time, or the primary camera census as an identity mapping."""

    @classmethod
    def read(cls, source: Path) -> "LabelClock":
        """Validate both columns before any label is joined."""
        path: Path = source / "timecode_devicetime_mapping.csv"
        with path.open() as stream:
            rows: list[dict[str, str]] = list(csv.DictReader(stream))
        timecodes: list[int] = [int(row["timecode_ns"]) for row in rows]
        devices: list[int] = [int(row["devicetime_ns"]) for row in rows]
        if any(b <= a for values in (timecodes, devices) for a, b in zip(values, values[1:], strict=False)):
            raise ValueError(f"{path}: mapping columns must be strictly increasing")
        return cls(dict(zip(timecodes, devices, strict=True)))

    def device_time(self, stamp: int) -> int:
        """Require an exact timecode match; preserve the shipped 1 ns offset."""
        if stamp not in self.mapping:
            raise ValueError(f"label timestamp_ns={stamp} has no matching mapping.timecode_ns in the label census")
        return self.mapping[stamp]


def read_hands(source: Path, clock: LabelClock, stop_ns: int | None = None) -> dict[int, UmeFrame]:
    """Keep each label row at its own stamp, in source order."""
    frames: dict[int, UmeFrame] = {}
    path: Path = source / "umetrack_hand_pose_trajectory.jsonl"
    previous: int = -1
    with path.open() as stream:
        for line in stream:
            row: UmeFrame = from_json(UmeFrame, line)
            stamp: int = clock.device_time(row.timestamp_ns)
            if stamp <= previous:
                raise ValueError(f"{path}: hand timestamps must be strictly increasing")
            previous = stamp
            if stop_ns is None or stamp <= stop_ns:
                frames[stamp] = row
    return frames


def read_mano(source: Path, clock: LabelClock, stop_ns: int | None = None) -> dict[int, ManoFrame]:
    """Read MANO's own stamps; drop and count its invalid stamp-zero rows."""
    frames: dict[int, ManoFrame] = {}
    dropped: int = 0
    previous: int = -1
    with (source / "mano_hand_pose_trajectory.jsonl").open() as stream:
        for line in stream:
            row: ManoFrame = from_json(ManoFrame, line)
            if row.timestamp_ns == 0:
                dropped += 1
                continue
            stamp: int = clock.device_time(row.timestamp_ns)
            if stamp <= previous:
                raise ValueError(f"{source}: MANO timestamps must be strictly increasing: {stamp}")
            previous = stamp
            if stop_ns is None or stamp <= stop_ns:
                frames[stamp] = row
    print(f"{source.name}: dropped {dropped} MANO stamp-zero rows")
    return frames


@serde
@dataclass(frozen=True, slots=True)
class PoseRow:
    """One CSV world-from-object (or headset) pose."""

    object_uid: str
    """Native instance ID."""
    timestamp_ns: int = field(rename="timestamp[ns]", deserializer=int)
    """Source clock stamp."""
    tx: float = field(rename="t_wo_x[m]", deserializer=float)
    """World x, metres."""
    ty: float = field(rename="t_wo_y[m]", deserializer=float)
    """World y, metres."""
    tz: float = field(rename="t_wo_z[m]", deserializer=float)
    """World z, metres."""
    qw: float = field(rename="q_wo_w", deserializer=float)
    """Quaternion scalar."""
    qx: float = field(rename="q_wo_x", deserializer=float)
    """Quaternion x."""
    qy: float = field(rename="q_wo_y", deserializer=float)
    """Quaternion y."""
    qz: float = field(rename="q_wo_z", deserializer=float)
    """Quaternion z."""

    def matrix(self) -> Float64[ndarray, "4 4"]:
        """World-from-object in metres."""
        transform: Float64[ndarray, "4 4"] = np.eye(4)
        transform[:3, :3] = Rotation.from_quat([self.qx, self.qy, self.qz, self.qw]).as_matrix()
        transform[:3, 3] = [self.tx, self.ty, self.tz]
        return transform


def read_poses(path: Path, clock: LabelClock) -> dict[str, dict[int, PoseRow]]:
    """Group source rows by ID and exact device time; reject duplicate poses."""
    result: dict[str, dict[int, PoseRow]] = {}
    with path.open() as stream:
        for raw in csv.DictReader(stream):
            row: PoseRow = from_dict(PoseRow, raw)
            stamp: int = clock.device_time(row.timestamp_ns)
            poses: dict[int, PoseRow] = result.setdefault(row.object_uid, {})
            if poses and stamp <= next(reversed(poses)):
                raise ValueError(f"{path}: non-increasing or duplicate pose for {row.object_uid} at {stamp}")
            poses[stamp] = row
    return result


def pose_array(poses: dict[int, PoseRow], times_ns: Int64[ndarray, "n"]) -> Float64[ndarray, "n 4 4"]:
    """Dense exact lookup with NaN at every missing row; no hold or interpolation."""
    transforms: Float64[ndarray, "n 4 4"] = np.full((len(times_ns), 4, 4), np.nan)
    for index, stamp in enumerate(times_ns):
        if int(stamp) in poses:
            transforms[index] = poses[int(stamp)].matrix()
    return transforms


def parse_mask(value: str) -> bool:
    """CSV booleans have explicit spellings; bool("False") would be wrong."""
    if value not in ("True", "False"):
        raise ValueError(f"invalid quality flag {value!r}")
    return value == "True"


@serde
@dataclass(frozen=True, slots=True)
class MaskRow:
    """A quality value belongs to one stream, even when two stamps coincide."""

    timestamp_ns: int = field(rename="timestamp[ns]", deserializer=int)
    """Source clock stamp."""
    stream_id: str
    """VRS camera stream ID."""
    mask: bool = field(deserializer=parse_mask)
    """Shipped flag, not a derived confidence."""


MaskSamples: TypeAlias = tuple[Int64[ndarray, "n"], Bool[ndarray, "n"]]
"""Device timestamps and shipped booleans for one camera's quality flag."""


def read_masks(source: Path, clock: LabelClock, stop_ns: int | None = None) -> dict[str, dict[str, MaskSamples]]:
    """Read device-stamped flags; ordering is strict within each camera stream."""
    result: dict[str, dict[str, MaskSamples]] = {}
    dropped: int = 0
    for path in sorted((source / "masks").glob("mask_*.csv")):
        streams: dict[str, list[tuple[int, bool]]] = {}
        previous: dict[str, int] = {}
        with path.open() as stream:
            for raw in csv.DictReader(stream):
                row: MaskRow = from_dict(MaskRow, raw)
                if row.timestamp_ns == 0:
                    dropped += 1
                    continue
                stamp: int = clock.device_time(row.timestamp_ns)
                if stamp <= previous.get(row.stream_id, -1):
                    raise ValueError(f"{path}: {row.stream_id} quality timestamps must be strictly increasing: {stamp}")
                previous[row.stream_id] = stamp
                if stop_ns is None or stamp <= stop_ns:
                    streams.setdefault(row.stream_id, []).append((stamp, row.mask))
        result[path.stem.removeprefix("mask_")] = {
            stream: (np.array([stamp for stamp, _ in rows], dtype=np.int64), np.array([flag for _, flag in rows], dtype=np.bool_))
            for stream, rows in streams.items()
        }
    print(f"{source.name}: dropped {dropped} quality stamp-zero rows")
    return result


@dataclass(frozen=True, slots=True)
class Labels:
    """All annotations joined to a fixed device census before leaving the source reader."""

    times_ns: Int64[ndarray, "n"]
    """Sorted census, cut at the preview stop before reading labels."""
    hands: dict[int, UmeFrame]
    """UmeTrack rows keyed by device time."""
    mano: dict[int, ManoFrame]
    """MANO rows keyed by device time."""
    masks: dict[str, dict[str, MaskSamples]]
    """Quality arrays keyed by flag and VRS stream."""
    headset: dict[int, PoseRow]
    """Sparse GT headset track."""
    objects: dict[str, dict[int, PoseRow]]
    """Sparse GT object tracks."""


def read_labels(source: Hot3dSource, device: Device, primary: Int64[ndarray, "n"], stop_ns: int | None) -> Labels:
    """Fix the census before reading labels; an off-grid label is always an error."""
    clock: LabelClock = (
        LabelClock.read(source.path)
        if source.metadata.have_hand_object_pose_gt and DEVICES[device].has_timecode_mapping
        else LabelClock({int(stamp): int(stamp) for stamp in primary})
    )
    times_ns: Int64[ndarray, "n"] = np.array(sorted(clock.mapping.values()), dtype=np.int64)
    if stop_ns is not None:
        times_ns = times_ns[times_ns <= stop_ns]
    if not source.metadata.have_hand_object_pose_gt:
        return Labels(times_ns, {}, {}, {}, {}, {})
    tracks: dict[str, dict[int, PoseRow]] = read_poses(source.path / "headset_trajectory.csv", clock)
    if len(tracks) != 1:
        raise ValueError(f"{source.path}: expected exactly one headset trajectory")
    return Labels(
        times_ns,
        read_hands(source.path, clock, stop_ns),
        read_mano(source.path, clock, stop_ns),
        read_masks(source.path, clock, stop_ns),
        next(iter(tracks.values())),
        read_poses(source.path / "dynamic_objects.csv", clock),
    )


@serde
@dataclass(frozen=True, slots=True)
class AssetInfo:
    """Native HOT3D instance identity; geometry is assets/<instance_id>.glb."""

    instance_id: str
    """ID used by dynamic_objects.csv."""
    instance_name: str
    """Human source name."""
