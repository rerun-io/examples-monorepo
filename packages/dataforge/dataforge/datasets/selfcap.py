"""SelfCap: iPhones, an OAK ego rig and a Quest produce one base RRD per episode.

Raw layout on disk (already local; ``download`` only verifies it)::

    <root>/<subset>/<YYYY-MM-DD>/<session-uuid>/
        metadata.json                     task / collector / time
        calibration_manifest.json
        episodes/episode-<NNN>/
            ego/{rgb,left,right}.mp4      OAK, 1280x720
            ego/{rgb,left,right}.csv      ts_ns,frame_idx at the OAK's native rate
            ego/imu.csv                   ts_ns + accel/gyro/mag/rot, ~100 Hz
            ego/calibration.json          3 cameras, 14-coeff Brown-Conrady
            exo/<DeviceName>/<DeviceName>.mp4 + calibration.json
            quest/{left,right}.mp4 + head_pose.csv + body/hand poses + calibration.json
            episode-<NNN>.rrd             prior art; deliberately ignored

Clocks. Raw CSV timestamps are episode-relative. Video presentation PTS,
including each container's start, is the episode clock for every camera.
``Mp4Reader`` rebases files to zero; ``read_video_start_ns`` restores the start.
The cutting pipeline re-encoded streams onto a shared near-CFR grid. OAK MP4s
include duplicate packets, so their CSV rows cannot index video samples. The
CSVs describe the native-rate clock and provide ``num_native_frames`` and
``native_duration_ns`` on each OAK camera.

Both ``video_time`` and ``capture_time`` hold the as-cut clock at ingest.
The separate selfcap-curate package later shifts ``video_time`` while preserving
``capture_time``. No latency correction or alignment is applied here.

Rig topology. Exo devices sorted by name take ``rig_00..``, then the OAK ego rig,
then the Quest — nine cameras and ``rig_00..rig_05`` in the usual 4-iPhone case.
Only the Quest rig has a known pose (its left-eye track in the Quest's Y-up
world frame), so the base layer owns the root ``ViewCoordinates``. The base holds
only measured facts: the OAK and Quest factory stereo extrinsics become static
``rig_T_cam`` transforms, while the iPhones measured neither pose (empty
extrinsics) nor lens (``available: false``, a 700 px fallback). Their rigs carry no
transform and their cameras no Pinhole until a calibration layer estimates both.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion

from dataforge import paths, schema, transports, writing
from dataforge.datasets import selfcap_blueprint
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.selfcap_blueprint import CameraPane, build_blueprint, segment_rig_kinds
from dataforge.datasets.selfcap_raw import (
    QUEST_HAND_SIDES,
    FactoryCalibration,
    assumed_phone_lens,
    coco133_hands,
    oak_stereo,
    quest_left_T_right,
    read_quest_joint_hands,
)
from dataforge.datasets.selfcap_raw import CameraCalibration as CameraCalibration
from dataforge.datasets.selfcap_raw import HeadPoses as HeadPoses
from dataforge.datasets.selfcap_raw import QuestHands as QuestHands
from dataforge.datasets.selfcap_raw import read_calibration as read_calibration
from dataforge.datasets.selfcap_raw import read_head_poses as read_head_poses
from dataforge.datasets.selfcap_raw import read_imu_csv as read_imu_csv
from dataforge.datasets.selfcap_raw import read_quest_hands as read_quest_hands
from dataforge.datasets.selfcap_raw import read_session_task as read_session_task
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    ImuChannel,
    annotation_context,
    capture_column,
    log_audio_track,
    log_imu,
    log_pose_track,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.oak_mount import PRIOR_LEAD_NS, nominal_oak_mount, quest_mesh_frame
from dataforge.objects import strip_texture_transform, with_root_transform
from dataforge.quest_joint_poses import QuestJointPoses, joint_poses_from_table, quest_hand_path
from dataforge.quest_joint_poses import read_segment_hand_poses as read_segment_hand_poses
from dataforge.records import read_csv_columns, read_json
from dataforge.selfcap_schema import DEFAULT_EXO_CAMERAS, EGO_STREAM_LAYOUTS, IMU_DEVICE, QUEST_STREAM_LAYOUTS, rig_label_path
from dataforge.video_encoding import read_video_start_ns

REPO_ID: str = "pablovela5620/selfcap-hard26-raw"
"""Private Hub mirror of the raw tree; for now the 26-episode hard set, byte-identical to the NAS ``cut`` copies."""
REVISION: str = "84a01e79ae35aee923e6dd5dedc91cd695196445"
"""Pinned ``REPO_ID`` commit."""

DATE_DIR_RE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
"""Capture-day directories under the subset; the loose ``manifest.json`` siblings do not match."""
EPISODE_DIR_RE: re.Pattern[str] = re.compile(r"^episode-(?P<number>\d+)$")
"""Episode directory names, e.g. ``episode-009``."""
SESSION_KEY_LENGTH: int = 8
"""Prefix of the session uuid used in the sequence identity (uuid4 is unique well inside 8 hex chars)."""

IMAGE_PLANE_DISTANCE: float = 0.04
"""Frustum length in metres. The five head cameras sit 3-7 cm apart, so longer frustums overlap into one blob."""
PHONE_IMAGE_PLANE_DISTANCE: float = 0.15
"""The phones stand metres apart; the calibration layer draws them at the same length."""
HEAD_AXIS_LENGTH: float = 0.05
"""Axes on each head camera (OAK, Quest lens); the Quest left lens's frame is where the head track, and so the world, starts."""
PHONE_AXIS_LENGTH: float = 0.1
"""Axes on each phone camera, the length of its frustum's near half."""
RIG_LABEL_COLOR: tuple[int, int, int] = (255, 200, 0)
"""The small labelled point that names each phone rig."""
DEVICE_MESHES: Path = Path(__file__).parent / "selfcap_devices"
"""Display meshes of the head-worn devices, in their reference camera frames (see its NOTICE.md)."""

EGO_CORE_FILES: tuple[str, ...] = (
    "ego/rgb.mp4",
    "ego/left.mp4",
    "ego/right.mp4",
    "ego/rgb.csv",
    "ego/left.csv",
    "ego/right.csv",
    "ego/imu.csv",
    "ego/calibration.json",
)
"""Files an episode must expose readably for its ego rig to be convertible."""
QUEST_CORE_FILES: tuple[str, ...] = ("quest/left.mp4", "quest/right.mp4", "quest/head_pose.csv", "quest/calibration.json")
"""Files an episode must expose readably for its Quest rig to be convertible."""


def log_quest_hand_poses(recording: rr.RecordingStream, hands: dict[str, QuestJointPoses], *, shift_ns: int = 0) -> None:
    """Log each hand as recorded: ``InstancePoses3D`` with all 26 joints' world positions and xyzw orientations.

    One row per CSV sample; an untracked sample (an all-zero CSV row) is an empty batch, so the CSV round-trips.
    ``TransformAxes3D`` draws a small frame at every joint (the viewer instances it per pose). Position-only CSVs
    log translations alone. ``video_time`` is shifted by ``shift_ns``; ``capture_time`` keeps the raw stamps.
    """
    for side, hand in hands.items():
        entity = quest_hand_path(side)
        rr.log(
            entity,
            rr.TransformAxes3D(axis_length=0.01),
            rr.AnyValues(joint_names=list(hand.joint_names), frame="Quest world, metres; quaternions xyzw"),
            static=True,
            recording=recording,
        )
        if not hand.times_ns.size:
            continue
        tracked = np.isfinite(hand.positions).all(axis=(1, 2))
        lengths = np.where(tracked, 26, 0)
        columns = rr.InstancePoses3D.columns(
            translations=hand.positions[tracked].reshape(-1, 3),
            quaternions=None if hand.orientations is None else hand.orientations[tracked].reshape(-1, 4),
        ).partition(lengths)
        rr.send_columns(
            entity,
            indexes=[time_column(hand.times_ns + shift_ns), capture_column(hand.times_ns)],
            columns=columns,
            recording=recording,
        )


def read_hand_poses(base_rrd: Path) -> tuple[dict[str, QuestJointPoses], int]:
    """The joint poses a base layer holds (on ``capture_time``), and its hands' ``video_time`` shift."""
    # The footer index loads only the hand chunks; streaming the file would read every video sample too.
    store = RrdReader(base_rrd).store().stream().filter(content=[quest_hand_path(side) for side in QUEST_HAND_SIDES]).collect()
    hands: dict[str, QuestJointPoses] = {}
    shifts: set[int] = set()
    for side in QUEST_HAND_SIDES:
        entity = quest_hand_path(side)
        table = store.reader(index=schema.CAPTURE_TIMELINE, contents=entity).to_arrow_table()
        if not table.num_rows:
            continue
        names = store.reader(index=None, contents=entity).to_arrow_table()[f"{entity}:joint_names"].to_pylist()[0] or []
        names = names[0] if names and isinstance(names[0], list) else names  # one row of an AnyValues string list
        hands[side] = joint_poses_from_table(table, entity, schema.CAPTURE_TIMELINE, tuple(names))
        stamps = table.select([schema.CAPTURE_TIMELINE, schema.TIMELINE]).drop_null()
        shifts |= set((stamps[schema.TIMELINE].cast(pa.int64()).to_numpy() - stamps[schema.CAPTURE_TIMELINE].cast(pa.int64()).to_numpy()).tolist())
    if len(shifts) > 1:
        raise ValueError(f"{base_rrd}: Quest hands carry {len(shifts)} different video_time shifts")
    return hands, shifts.pop() if shifts else 0


def write_coco133_layer(hands: dict[str, QuestJointPoses], shift_ns: int, target: Path, recording_id: str) -> None:
    """The ``gt`` layer: the COCO-133 view (``/world/gt/coco133_xyz``) of the Quest hands a base holds.

    The base holds the hands as recorded; this layer is the dataset-agnostic keypoint view the shared exo/ego tools
    read. Ingest passes the hands it just logged; a retimed base (align) passes ``read_hand_poses`` of itself, so
    the layer never needs its own retime.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with writing.recording_to(target, recording_id=recording_id, send_properties=False) as recording:
        log_quest_hands(recording, coco133_hands(hands), shift_ns=shift_ns)


DEVICE_MESH_FILES: dict[str, str] = {"oak": "oak_d_pro_w", "quest": "quest3s", "phone": "iphone12"}
"""Device kind → mesh file in ``DEVICE_MESHES``."""
RIG_DEVICES: dict[str, str] = {"ego": "oak", "quest": "quest", "exo": "phone"}
"""Rig ``kind`` → the device whose model the catalog attaches there."""


def device_asset_id(device: str, rig: int) -> str:
    """``devices__<mesh>__rig_NN``: one asset per device and rig index (a 3-phone session puts the OAK at rig_03)."""
    return f"devices__{DEVICE_MESH_FILES[device]}__rig_{rig:02d}"


def write_device_asset(target: Path, device: str, rig: int) -> None:
    """A catalog asset: the device's mesh, static, in its reference camera frame.

    Head-worn devices sit at ``/world/rig_NN/device``. A phone's model sits at ``/world/rig_NN/cam_00/device``: the
    phone pose is the camera's (calibrate's ``cam_00`` transform), and the layout keeps it out of 3D until then.
    """
    node: str = schema.cam_path(rig, 0) if device == "phone" else schema.rig_path(rig)
    # The shipped model is one static Mesh3D or Asset3D at /device; the asset is that chunk, moved to the device's node.
    mesh = RrdReader(DEVICE_MESHES / f"{DEVICE_MESH_FILES[device]}.rrd").stream().filter(content="/device")
    mesh = mesh.map(lambda chunk: chunk.with_entity_path(f"{node}/device"))
    with writing.atomic_write(target) as staged:
        mesh.write_rrd(staged, application_id=writing.APPLICATION_ID, recording_id=device_asset_id(device, rig))


QUEST_MODEL_LEFT_RGB: tuple[float, float, float] = (-0.0318, -0.0138, -0.0805)
"""The Quest 3S model's left RGB lens (its wearer-left upper front aperture) in the GLB's frame, metres.

From the model package's dimensions.json: columns at x = +-31.8 mm (a 63.6 mm baseline, as d31's factory 63.55 mm),
upper apertures at 13.8 mm below the lens midpoint, lens glass on the front panel. glTF axes: x wearer-right, y up,
z toward the wearer (the front panel faces -z).
"""


def quest_model_camera_T_glb() -> Float64[ndarray, "4 4"]:
    """The GLB frame in the Quest's left RGB camera frame (RDF): a half turn about x, then the lens to the origin."""
    turn: Float64[ndarray, "3 3"] = np.diag([1.0, -1.0, -1.0])  # glTF up/back -> camera down/forward
    camera_T_glb: Float64[ndarray, "4 4"] = np.eye(4)
    camera_T_glb[:3, :3] = turn
    camera_T_glb[:3, 3] = -turn @ np.asarray(QUEST_MODEL_LEFT_RGB)
    return camera_T_glb


def build_quest_model(glb: Path, target: Path = DEVICE_MESHES / f"{DEVICE_MESH_FILES['quest']}.rrd") -> None:
    """Write ``quest3s.rrd``: the model package's GLB as a static ``Asset3D`` at ``/device``, in the left RGB camera frame.

    The GLB keeps its meshes, PBR materials and textures. Two structural edits make it a device asset: the scene is
    placed in the camera frame (``quest_model_camera_T_glb``), and the texture-transform extension Rerun refuses is
    dropped (it only scales the strap's weave texture).
    """
    blob: bytes = with_root_transform(strip_texture_transform(glb.read_bytes()), quest_model_camera_T_glb())
    chunk: Chunk = Chunk.from_columns("/device", indexes=[], columns=rr.Asset3D.columns(blob=[blob], media_type=["model/gltf-binary"]))
    with writing.atomic_write(target) as staged:
        LazyChunkStream.from_iter([chunk]).write_rrd(staged, application_id=writing.APPLICATION_ID, recording_id=f"devices__{DEVICE_MESH_FILES['quest']}")


def log_quest_hands(recording: rr.RecordingStream, hands: QuestHands, *, shift_ns: int = 0) -> None:
    """Log raw COCO-133 hands, shifting only video_time by ``shift_ns``.

    The static annotation context and point metadata are shared by ingestion and
    the one-off base migration. capture_time always keeps the raw CSV stamps.
    """
    rr.log("/world/gt", annotation_context(), static=True, recording=recording)
    rr.log(
        schema.coco133_xyz_path(),
        rr.Points3D.from_fields(class_ids=0, keypoint_ids=list(range(133)), radii=0.006),
        static=True,
        recording=recording,
    )
    if hands.times_ns.size:
        rr.send_columns(
            schema.coco133_xyz_path(),
            indexes=[time_column(hands.times_ns + shift_ns), capture_column(hands.times_ns)],
            columns=rr.Points3D.columns(positions=hands.positions_xyz.reshape(-1, 3)).partition([133] * len(hands.times_ns)),
            recording=recording,
        )


@dataclass(frozen=True, slots=True)
class CameraPlan:
    """One camera that *will* be logged: its slot, its media, and its metadata."""

    rig: int
    """Rig index owning the camera."""
    cam: int
    """Camera index within the rig."""
    video_path: Path
    """mp4 remuxed onto ``schema.video_path(rig, cam)``."""
    calibration: CameraCalibration
    """Intrinsics; a camera without them never reaches a plan."""
    name: str
    """``name`` AnyValue on the cam node (device name or stream label)."""
    kind: str
    """``kind`` AnyValue on the cam node: ``rgb`` or ``grayscale``."""
    native_frames: int | None = None
    """Rows in the ego stream's native-rate csv; None for cameras without one."""
    native_duration_ns: int | None = None
    """Last native-rate timestamp; None for cameras without a csv."""
    first_frame_idx: int | None = None
    """First session-global 24 fps counter; preserves the RGB drift origin."""
    rig_T_cam: Float64[ndarray, "4 4"] | None = None
    """Factory transform from the rig's reference camera (cam_00); None for the reference itself and the iPhones."""
    lens_source: str = "factory"
    """``lens_source`` AnyValue on the pinhole: ``factory`` (measured) or ``assumed`` (iPhones, until calibration)."""
    image_plane_distance: float = IMAGE_PLANE_DISTANCE
    """Frustum length in metres."""
    axis_length: float = HEAD_AXIS_LENGTH
    """The camera's own frame (``TransformAxes3D``) is drawn this long, in metres."""
    named: bool = False
    """Draw a small point labelled ``rig_NN`` at this camera (``rig_label_path``); the phones only."""


@dataclass(frozen=True, slots=True)
class RigPlan:
    """One rig node and the cameras that survived plan building."""

    rig: int
    """Rig index; the node is ``schema.rig_path(rig)``."""
    reference: str
    """Sensor child whose frame the rig frame coincides with."""
    name: str
    """Human device label (the exo device name, ``oak``, ``quest``)."""
    kind: str
    """Device role: ``exo`` / ``ego`` / ``quest``."""
    cameras: tuple[CameraPlan, ...]
    """Cameras actually logged under this rig; sets the node's ``num_cameras``."""


@dataclass(frozen=True, slots=True)
class EpisodePlan:
    """Everything convert needs, decided *before* the recording opens.

    Building the plan first is what keeps the embedded blueprint and the
    ``num_cameras`` property honest: a camera that lacks intrinsics is warned
    about and dropped here, so it can never appear as a pane describing an
    entity the converter then never logs.
    """

    episode_dir: Path
    """``episodes/episode-<NNN>`` directory the plan was built from."""
    rigs: tuple[RigPlan, ...]
    """Exo rigs in name order, then the OAK ego rig, then the Quest."""
    ego_rig: int
    """Rig index of the OAK, whose ``imu_00`` feeds the time-series views."""
    quest_rig: int
    """Rig index of the Quest, whose node carries the temporal ``world_T_rig``."""
    panes: tuple[CameraPane, ...]
    """Blueprint panes — exactly the cameras in ``rigs``, in rig-then-camera order."""


def raw_calibration_chunks(episode_dir: Path, devices: list[str], *, oak_rig: int) -> list[Chunk]:
    """Preserve opaque vendor JSON verbatim on each device rig as static raw-source facts.

    The phones take rig_00 onwards; the OAK sits at ``oak_rig`` and the Quest right after it, so fewer phones leave a gap.
    """
    folders: dict[int, Path] = {rig: episode_dir / "exo" / device for rig, device in enumerate(devices)}
    folders |= {oak_rig: episode_dir / "ego", oak_rig + 1: episode_dir / "quest"}
    return [
        Chunk.from_columns(
            schema.rig_path(rig),
            indexes=[],
            columns=rr.AnyValues.columns(raw_calibration_json=[(folder / "calibration.json").read_bytes().decode("utf-8")]),
        )
        for rig, folder in folders.items()
    ]


@dataclass
class SelfcapConfig(DataforgeDatasetConfig):
    """Self-collected exo/ego/Quest captures, cut into per-task episodes."""

    command: ClassVar[str] = "selfcap"
    """Registry key, catalog dataset name, and identity ``dataset`` part."""

    _target: type = field(default_factory=lambda: SelfcapDataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "selfcap")
    """Corpus root holding the ``cut`` subset; ``download`` fills it from the Hub."""
    subset: str = "cut"
    """Subset directory to convert; only ``cut`` holds per-episode captures."""
    repo_id: str = REPO_ID
    """Private Hub dataset mirroring the raw ``cut/<date>/<session>/...`` tree."""
    revision: str = REVISION
    """Pinned Hub commit; a deliberate override must be a full lowercase commit SHA."""

    def __post_init__(self) -> None:
        transports.require_commit_sha(self.revision)


def exo_devices(episode_dir: Path) -> list[str]:
    """Exo device names with a readable video *and* calibration, sorted by name.

    Five episodes ship a split-brain ``exo/`` tree (bare ``<Device>/`` holding
    only calibration next to ``multiCam-<Device>._multicam._tcp.local./`` holding
    only the mp4); requiring both files in one directory drops those halves.

    Args:
        episode_dir: An ``episodes/episode-<NNN>`` directory.

    Returns:
        Device directory names, which double as the video stems.
    """
    exo_dir: Path = episode_dir / "exo"
    if not exo_dir.is_dir():
        return []
    return sorted(
        device_dir.name
        for device_dir in exo_dir.iterdir()
        if device_dir.is_dir() and os.access(device_dir / f"{device_dir.name}.mp4", os.R_OK) and os.access(device_dir / "calibration.json", os.R_OK)
    )


def episode_is_readable(episode_dir: Path) -> bool:
    """Whether every file the converter needs is present and readable.

    Args:
        episode_dir: An ``episodes/episode-<NNN>`` directory.

    Returns:
        True when the ego and Quest cores plus at least one exo device are readable.
    """
    if not all(os.access(episode_dir / relative, os.R_OK) for relative in EGO_CORE_FILES + QUEST_CORE_FILES):
        return False
    return bool(exo_devices(episode_dir))


def build_episode_plan(episode_dir: Path) -> EpisodePlan:
    """Decide the whole recording's shape from the raw tree, before anything is logged.

    Every camera is vetted here — the device directory must be readable (already
    checked by discovery) *and* its ``calibration.json`` must actually describe
    the camera. A camera that fails is warned about and dropped, so the blueprint
    panes and the ``num_cameras`` property describe exactly what gets logged.

    Args:
        episode_dir: A readable ``episodes/episode-<NNN>`` directory.

    Returns:
        The rig/camera/pane plan for this episode.
    """
    devices: list[str] = exo_devices(episode_dir)
    # The head rigs never move with the phone count (rig_03 stays empty with three phones), so one catalog layout
    # serves every segment.
    if len(devices) > DEFAULT_EXO_CAMERAS:
        raise ValueError(f"{episode_dir}: {len(devices)} phones; the SelfCap layout has {DEFAULT_EXO_CAMERAS} phone rigs")
    ego_rig: int = DEFAULT_EXO_CAMERAS
    quest_rig: int = ego_rig + 1
    rigs: list[RigPlan] = []
    panes: list[CameraPane] = []

    for rig, device in enumerate(devices):
        device_dir: Path = episode_dir / "exo" / device
        cameras: dict[str, CameraCalibration] = read_calibration(device_dir / "calibration.json")
        # iPhone calibration ships one entry whose positional_layout is always "center";
        # fall back to the sole entry so a renamed layout does not lose the camera.
        calibration: CameraCalibration | None = cameras.get("center") or next(iter(cameras.values()), None)
        exo_cameras: list[CameraPlan] = []
        if calibration is None:
            print(f"  warning: no usable intrinsics in {device_dir / 'calibration.json'}, skipping {device}")
        else:
            measured: bool = calibration.available
            exo_cameras.append(
                CameraPlan(
                    rig=rig, cam=0, video_path=device_dir / f"{device}.mp4", name=device, kind="rgb",
                    calibration=calibration if measured else assumed_phone_lens(calibration.width, calibration.height),
                    lens_source="factory" if measured else "assumed", image_plane_distance=PHONE_IMAGE_PLANE_DISTANCE,
                    axis_length=PHONE_AXIS_LENGTH, named=True,
                )
            )
            panes.append(CameraPane(name=device.split("-")[0], rig=rig, cam=0, kind="exo"))
        # reference=schema.CAM0_REFERENCE: a single-camera rig's frame is its camera's frame by construction.
        rigs.append(RigPlan(rig=rig, reference=schema.CAM0_REFERENCE, name=device, kind="exo", cameras=tuple(exo_cameras)))

    ego_dir: Path = episode_dir / "ego"
    ego_calibration: dict[str, CameraCalibration] = read_calibration(ego_dir / "calibration.json")
    stereo: dict[str, Float64[ndarray, "4 4"]] = oak_stereo(read_json(ego_dir / "calibration.json", FactoryCalibration))
    ego_rig_T_cam: dict[str, Float64[ndarray, "4 4"] | None] = {"left": stereo.get("rgb_T_left"), "right": stereo.get("rgb_T_right")}
    ego_cameras: list[CameraPlan] = []
    for cam, (stream, layout) in enumerate(EGO_STREAM_LAYOUTS):
        calibration = ego_calibration.get(layout)
        if calibration is None:
            print(f"  warning: no '{layout}' intrinsics in {ego_dir / 'calibration.json'}, skipping ego {stream}")
            continue
        # The mp4 is resampled onto the shared episode grid, so the csv's native-rate
        # clock is surfaced as metadata rather than used to retime the samples.
        columns: pa.Table = read_csv_columns(ego_dir / f"{stream}.csv", dict.fromkeys(("ts_ns", "frame_idx"), pa.float64()))
        frame_times_ns: Int64[ndarray, "n_frames"] = np.rint(columns.column("ts_ns").to_numpy()).astype(np.int64)
        frame_indices: Float64[ndarray, "n_frames"] = columns.column("frame_idx").to_numpy()
        ego_cameras.append(
            CameraPlan(
                rig=ego_rig,
                cam=cam,
                video_path=ego_dir / f"{stream}.mp4",
                calibration=calibration,
                name=stream,
                kind="rgb" if stream == "rgb" else "grayscale",
                native_frames=frame_times_ns.size,
                native_duration_ns=int(frame_times_ns[-1]) if frame_times_ns.size else 0,
                first_frame_idx=int(frame_indices[0]) if frame_indices.size else None,
                rig_T_cam=ego_rig_T_cam.get(stream),
            )
        )
        panes.append(CameraPane(name=f"OAK {stream}", rig=ego_rig, cam=cam, kind="ego"))
    # reference=schema.CAM0_REFERENCE: the OAK's rgb camera is the rig frame; left/right carry rgb_T_left/right.
    rigs.append(RigPlan(rig=ego_rig, reference=schema.CAM0_REFERENCE, name="oak", kind="ego", cameras=tuple(ego_cameras)))

    quest_dir: Path = episode_dir / "quest"
    quest_calibration: dict[str, CameraCalibration] = read_calibration(quest_dir / "calibration.json")
    left_T_right: Float64[ndarray, "4 4"] | None = quest_left_T_right(read_json(quest_dir / "calibration.json", FactoryCalibration))
    quest_cameras: list[CameraPlan] = []
    for cam, (stream, layout) in enumerate(QUEST_STREAM_LAYOUTS):
        calibration = quest_calibration.get(layout)
        if calibration is None:
            print(f"  warning: no '{layout}' intrinsics in {quest_dir / 'calibration.json'}, skipping quest {stream}")
            continue
        quest_cameras.append(
            CameraPlan(
                rig=quest_rig,
                cam=cam,
                video_path=quest_dir / f"{stream}.mp4",
                calibration=calibration,
                name=stream,
                kind="grayscale",
                rig_T_cam=left_T_right if stream == "right" else None,
            )
        )
        panes.append(CameraPane(name=f"Quest {stream}", rig=quest_rig, cam=cam, kind="quest"))
    # reference=schema.CAM0_REFERENCE: the head_pose track logged on the rig node *is* the left-eye pose.
    rigs.append(RigPlan(rig=quest_rig, reference=schema.CAM0_REFERENCE, name="quest", kind="quest", cameras=tuple(quest_cameras)))

    return EpisodePlan(episode_dir=episode_dir, rigs=tuple(rigs), ego_rig=ego_rig, quest_rig=quest_rig, panes=tuple(panes))


class SelfcapDataset(DataforgeDataset[SelfcapConfig, Path]):
    """Converts SelfCap episodes into exoego:v2 base-layer recordings."""

    def default_blueprint(self, *, posed_phones: bool = True) -> rrb.Blueprint:
        """See ``selfcap_blueprint.default_blueprint``."""
        return selfcap_blueprint.default_blueprint(posed_phones=posed_phones)

    def table_blueprint(self, *, posed_phones: bool = True) -> rrb.Blueprint:
        """See ``selfcap_blueprint.table_blueprint``."""
        return selfcap_blueprint.table_blueprint(posed_phones=posed_phones)

    def preview_table_blueprint(self, columns: list[str]) -> rrb.table.TableBlueprint:
        """See ``selfcap_blueprint.preview_table_blueprint``."""
        return selfcap_blueprint.preview_table_blueprint(columns, self.table_fields())

    def catalog_blueprints(self, entry: DatasetEntry) -> tuple[rrb.Blueprint, rrb.Blueprint | rrb.table.TableBlueprint]:
        """See ``selfcap_blueprint.catalog_blueprints``."""
        return selfcap_blueprint.catalog_blueprints(entry, self.table_fields())

    def register_assets(self, entry: DatasetEntry, output_root: Path) -> None:
        """Attach the OAK, Quest and phone models to each segment at the rigs it holds them on (``device_asset_id``).

        A rig's role is its ``kind`` (``RIG_DEVICES``), read for every registered segment in one query; the rig index
        depends on the phone count. Each run writes fresh stamped files and replaces any asset of the same id.
        """
        segments: dict[tuple[str, int], list[str]] = {}
        for row, kinds in segment_rig_kinds(entry, range(DEFAULT_EXO_CAMERAS + 2)):
            for rig, kind in kinds.items():
                if kind in RIG_DEVICES:
                    segments.setdefault((RIG_DEVICES[kind], rig), []).append(str(row["rerun_segment_id"]))
        registered: set[str] = set(entry.assets())
        stamp: str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        for (device, rig), ids in sorted(segments.items()):
            asset_id: str = device_asset_id(device, rig)
            target: Path = paths.asset_path(output_root, asset_id, stamp=stamp).resolve()
            write_device_asset(target, device, rig)
            if asset_id in registered:
                entry.unregister_asset(asset_id)
            entry.register_asset(target.as_uri(), mode="opt_in", segments=sorted(ids))
        print(f"registered {len(segments)} device assets: {', '.join(device_asset_id(*key) for key in sorted(segments))}")

    def table_fields(self) -> writing.TableFields:
        """See ``selfcap_blueprint.table_fields``."""
        return selfcap_blueprint.table_fields()

    # ── raw-tree discovery ────────────────────────────────────────────────
    def subset_root(self) -> Path:
        """Root of the configured subset, e.g. ``<root>/cut``."""
        return self.config.root / self.config.subset

    def session_dirs(self) -> list[tuple[str, Path]]:
        """Every ``(capture_date, session_dir)`` pair holding an ``episodes/`` tree."""
        pairs: list[tuple[str, Path]] = []
        for date_dir in sorted(path for path in self.subset_root().glob("*") if path.is_dir() and DATE_DIR_RE.match(path.name)):
            pairs += [(date_dir.name, session_dir) for session_dir in sorted(date_dir.iterdir()) if (session_dir / "episodes").is_dir()]
        return pairs

    def download(self) -> None:
        """Fetch the raw ``cut`` tree from the pinned Hub mirror into ``root``, then verify it."""
        transports.hf_fetch(self.config.repo_id, allow_patterns=[f"{self.config.subset}/**"], local_dir=self.config.root, revision=self.config.revision)
        missing: list[str] = transports.local_verify(self.config.root, required=[f"{self.config.subset}/*/*/episodes"])
        if missing:
            raise FileNotFoundError(f"SelfCap corpus at {self.config.root} is missing: {', '.join(missing)}")
        sessions: list[tuple[str, Path]] = self.session_dirs()
        dates: set[str] = {date for date, _ in sessions}
        print(f"selfcap: {self.subset_root()} — {len(dates)} days, {len(sessions)} sessions, {len(self.sequences())} readable episodes")

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        """Pair every readable episode with its directory; unreadable ones are dropped, not raised on."""
        pairs: list[tuple[SequenceIdentity, Path]] = []
        skipped: int = 0
        for date, session_dir in self.session_dirs():
            for episode_dir in sorted((session_dir / "episodes").iterdir()):
                match: re.Match[str] | None = EPISODE_DIR_RE.match(episode_dir.name)
                if match is None or not episode_dir.is_dir():
                    continue
                if not episode_is_readable(episode_dir):
                    skipped += 1
                    continue
                identity: SequenceIdentity = SequenceIdentity(
                    dataset=self.config.name,
                    parts=(date, session_dir.name[:SESSION_KEY_LENGTH], f"ep{int(match['number']):03d}"),
                )
                pairs.append((identity, episode_dir))
        if skipped:
            print(f"  warning: skipped {skipped} episode(s) with unreadable files (corpus chmod pending)")
        return pairs

    # ── conversion ────────────────────────────────────────────────────────
    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write one episode's base-layer rrd (nine cameras, ego IMU, Quest pose and hands)."""
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        gt_target: Path = paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)
        if writing.should_skip(target, force=force) and gt_target.is_file():
            print(f"skip {identity.sequence_key} → {target}")
            return target

        # Plan first, log second: the blueprint panes and num_cameras below are the
        # plan's, so they can only ever describe cameras that really get logged.
        plan: EpisodePlan = build_episode_plan(source)
        hands: dict[str, QuestJointPoses] = read_quest_joint_hands(plan.episode_dir / "quest")

        # Both layers are staged and published together, from the same hands: the base logs them unshifted, and
        # read_hand_poses of that base gives them back exactly (test_quest_hand_poses_round_trip_through_the_base).
        with writing.atomic_write(gt_target) as staged_gt, writing.atomic_write(target) as staged_base:
            self._write_base(identity, plan, hands, staged_base)
            write_coco133_layer(hands, 0, staged_gt, identity.recording_id)
        print(f"done {identity.sequence_key} → {target} + {gt_target.parent.name} ({len(plan.panes)} cameras)")
        return target

    def _write_base(self, identity: SequenceIdentity, plan: EpisodePlan, hands: dict[str, QuestJointPoses], staged: Path) -> None:
        """The base layer: every raw stream of the episode, as recorded (see the module docstring)."""
        with writing.recording_to(
            staged,
            recording_id=identity.recording_id,
            default_blueprint=build_blueprint(plan.panes, ego_rig=plan.ego_rig),
        ) as recording:
            # The base layer owns the root ViewCoordinates because it already carries a real
            # moving-rig pose: the Quest world is right-handed Y-up.
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True, recording=recording)

            rr.send_chunks(
                raw_calibration_chunks(plan.episode_dir, [rig.name for rig in plan.rigs if rig.kind == "exo"], oak_rig=plan.ego_rig), recording=recording
            )
            num_frames: int = 0
            for rig_plan in plan.rigs:
                log_rig_node(
                    recording,
                    rig_plan.rig,
                    reference=rig_plan.reference,
                    num_cameras=len(rig_plan.cameras),
                    name=rig_plan.name,
                    kind=rig_plan.kind,
                )
                for camera in rig_plan.cameras:
                    num_frames = max(num_frames, self._log_camera(recording, camera))
                if rig_plan.kind == "exo" and rig_plan.cameras:
                    # The phone's own microphone, on the phone's clock; align moves it with the phone's video.
                    log_audio_track(recording, rig_plan.cameras[0].video_path, rig_plan.rig, 0, label=rig_plan.name.split("-")[0])
            self._log_ego_imu(recording, plan)
            first_pose_ns = self._log_quest_pose(recording, plan)
            if first_pose_ns is not None:
                self._log_oak_mount_prior(recording, plan, first_pose_ns)
            self._log_quest_mesh_frame(recording, plan)
            log_quest_hand_poses(recording, hands)

            # AnyValues drops None-valued kwargs, so an unreadable metadata.json simply
            # leaves task/collector off the property instead of writing empty strings.
            session: dict[str, str] = read_session_task(plan.episode_dir.parents[1] / "metadata.json")
            writing.send_capture_properties(
                recording,
                identity,
                num_cameras=len(plan.panes),
                num_frames=num_frames,
                session=plan.episode_dir.parents[1].name,
                task=session.get("task"),
                collector=session.get("collector"),
            )


    def _log_camera(self, recording: rr.RecordingStream, camera: CameraPlan) -> int:
        """Log one planned camera: node metadata, its factory ``rig_T_cam``, a static pinhole, and its video stream.

        Every camera gets a pinhole tagged with ``lens_source``. An iPhone's is assumed (see ``assumed_phone_lens``);
        the calibration layer writes the same components tagged ``fitted``, so the newer static write replaces it.
        Brown-Conrady coefficients ride along when the calibration declares them.

        Args:
            recording: Destination recording stream.
            camera: The camera's slot, media, and metadata as planned.

        Returns:
            Number of video samples written for this camera.
        """
        rr.log(
            schema.cam_path(camera.rig, camera.cam),
            rr.AnyValues(
                drop_untyped_nones=True,
                name=camera.name,
                kind=camera.kind,
                num_native_frames=camera.native_frames,
                native_duration_ns=camera.native_duration_ns,
                **({"first_frame_idx": camera.first_frame_idx} if camera.first_frame_idx is not None else {}),
            ),
            static=True,
            recording=recording,
        )
        if camera.rig_T_cam is not None:
            rr.log(
                schema.cam_path(camera.rig, camera.cam),
                rr.Transform3D(translation=camera.rig_T_cam[:3, 3], mat3x3=camera.rig_T_cam[:3, :3]),
                static=True,
                recording=recording,
            )
        rr.log(schema.cam_path(camera.rig, camera.cam), rr.TransformAxes3D(axis_length=camera.axis_length), static=True, recording=recording)
        if camera.named:
            rr.log(rig_label_path(camera.rig), rr.Points3D([[0.0, 0.0, 0.0]], radii=rr.Radius.ui_points(4.0), colors=RIG_LABEL_COLOR,
                                                       labels=f"rig_{camera.rig:02d}", show_labels=True), static=True, recording=recording)
        calibration: CameraCalibration = camera.calibration
        focal_length_xy: Float64[ndarray, "2"] = calibration.focal_length_xy
        principal_point_xy: Float64[ndarray, "2"] = calibration.principal_point_xy
        image_from_camera: Float32[ndarray, "3 3"] = np.array(
            [
                [focal_length_xy[0], 0.0, principal_point_xy[0]],
                [0.0, focal_length_xy[1], principal_point_xy[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        distortion: CameraDistortion | None = (
            None
            if calibration.distortion_coefficients is None
            else CameraDistortion(model="brown_conrady", coefficients=calibration.distortion_coefficients.astype(np.float32))
        )
        rr.log(
            schema.pinhole_path(camera.rig, camera.cam),
            PinholeWithDistortion(
                pinhole=rr.Pinhole(
                    image_from_camera=image_from_camera,
                    width=calibration.width,
                    height=calibration.height,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=camera.image_plane_distance,
                ),
                distortion=distortion,
            ),
            rr.AnyValues(lens_source=camera.lens_source),
            static=True,
            recording=recording,
        )
        # The presented PTS is the shared episode clock (see the module docstring).
        return log_video_stream(
            recording,
            camera.video_path,
            schema.video_path(camera.rig, camera.cam),
            shift_ns=read_video_start_ns(camera.video_path),
            capture_time=True,
        )

    def _log_ego_imu(self, recording: rr.RecordingStream, plan: EpisodePlan) -> None:
        """Log the OAK's single IMU on the ego rig, when its csv holds samples."""
        channels: tuple[ImuChannel, ImuChannel] = read_imu_csv(plan.episode_dir / "ego" / "imu.csv")
        if channels[0].times_ns.size:
            log_imu(recording, plan.ego_rig, IMU_DEVICE, gyro=channels[0], accel=channels[1], name="oak-imu", capture_time=True)

    def _log_quest_mesh_frame(self, recording: rr.RecordingStream, plan: EpisodePlan) -> None:
        """Turn the Quest model's frame on a camera-50 headset, whose eye-track frame sits about 175 deg off.

        The mesh itself is a catalog asset (``write_device_asset``), shared by every segment; this static transform
        on ``/world/rig_NN/device`` is the one per-episode fact it needs. Healthy headsets need none.
        """
        frame = quest_mesh_frame(
            read_json(plan.episode_dir / "ego" / "calibration.json", FactoryCalibration),
            read_json(plan.episode_dir / "quest" / "calibration.json", FactoryCalibration),
        )
        if frame is not None and not np.allclose(frame, np.eye(4)):
            rr.log(
                f"{schema.rig_path(plan.quest_rig)}/device",
                rr.Transform3D(translation=frame[:3, 3], mat3x3=frame[:3, :3]),
                static=True,
                recording=recording,
            )

    def _log_oak_mount_prior(self, recording: rr.RecordingStream, plan: EpisodePlan, first_pose_ns: int) -> None:
        """Hang the OAK rig on the Quest's left-eye frame with a nominal mount, just before the first Quest pose.

        Calibration logs its fitted mount at the first Quest pose (``PRIOR_LEAD_NS`` later), so the fit replaces the
        prior from the first drawable frame on, whichever layer was written last.
        """
        mount = nominal_oak_mount(
            read_json(plan.episode_dir / "ego" / "calibration.json", FactoryCalibration),
            read_json(plan.episode_dir / "quest" / "calibration.json", FactoryCalibration),
        )
        if mount is None:
            return
        rr.send_columns(
            schema.rig_path(plan.ego_rig),
            indexes=[time_column(np.array([first_pose_ns - PRIOR_LEAD_NS])), capture_column(np.array([first_pose_ns - PRIOR_LEAD_NS]))],
            columns=[
                *rr.Transform3D.columns(translation=[mount[:3, 3]], mat3x3=[mount[:3, :3]], parent_frame=[f"tf#{schema.rig_path(plan.quest_rig)}"]),
                *rr.AnyValues.columns(oak_mount_source=["nominal"]),
            ],
            recording=recording,
        )

    def _log_quest_pose(self, recording: rr.RecordingStream, plan: EpisodePlan) -> int | None:
        """Log the Quest's left-eye track as the temporal ``world_T_rig`` on its rig node; return its first time, if any."""
        head_pose_path: Path = plan.episode_dir / "quest" / "head_pose.csv"
        poses: HeadPoses = read_head_poses(head_pose_path)
        if not poses.times_ns.size:
            print(f"  warning: no head poses in {head_pose_path}; the quest rig stays unposed")
            return None
        log_pose_track(
            recording,
            schema.rig_path(plan.quest_rig),
            times_ns=poses.times_ns,
            translations_xyz=poses.translations_xyz,
            quaternions_xyzw=poses.quaternions_xyzw,
            capture_time=True,
        )
        return int(poses.times_ns[0])
