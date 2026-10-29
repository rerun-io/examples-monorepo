"""SelfCap: 4 iPhones + an OAK ego rig + a Quest → one exoego:v2 base rrd per episode.

Raw layout on disk (already local; ``download`` only verifies it)::

    <root>/<subset>/<YYYY-MM-DD>/<session-uuid>/
        metadata.json                     task / collector / time (usually mode 000)
        calibration_manifest.json         (usually mode 000)
        episodes/episode-<NNN>/
            ego/{rgb,left,right}.mp4      OAK, 1280x720
            ego/{rgb,left,right}.csv      ts_ns,frame_idx at the OAK's native rate
            ego/imu.csv                   ts_ns + accel/gyro/mag/rot, ~100 Hz
            ego/calibration.json          3 cameras, 14-coeff Brown-Conrady
            exo/<DeviceName>/<DeviceName>.mp4 + calibration.json
            quest/{left,right}.mp4 + head_pose.csv + body/hand poses + calibration.json
            episode-<NNN>.rrd             prior art; deliberately ignored

Clocks (verified on disk, 2026-08-20, over four days x one episode each plus
three episodes of ``2025-12-20/25aeb66a``). Every ``ts_ns`` column — ego frame
csvs, ``imu.csv``, ``head_pose.csv`` — is **episode-relative** and starts at 0,
so ``video_time`` is a duration timeline anchored at the episode start.

The nine videos are *not* the raw device streams: the cutting pipeline
re-encoded all of them to AV1 on one shared, near-CFR grid. Within an episode
every stream carries the same frame count, and the container duration matches
the ego csv span to ~40 ms; the ego mp4s additionally carry ~50% zero-cost
duplicate packets (3 bytes = ``show_existing_frame``) because the 24 fps OAK
was resampled onto the iPhone grid. Each mp4 presents its first frame at its
own container start: 0 in about half of the 4,168 cut episodes, otherwise about
one frame of that stream (16, 20, 33, 40 or 41 ms; measured 2026-10-26).
Consequences:

* The presented PTS **is** the episode clock, for exo *and* ego *and* quest.
  ``Mp4Reader`` rebases every file to start at 0, so each video is shifted back
  by its container start (``read_video_start_ns``); nothing else is retimed.
* The ego frame csvs cannot index the ego mp4 samples (e.g. 322 csv rows vs 671
  video samples). They stay the authoritative *native-rate* clock and are
  surfaced as ``num_native_frames`` / ``native_duration_ns`` on each ego camera.
  TODO(selfcap-curate): a lens that drops the duplicate packets and retimes the ego
  streams onto their native csv timestamps.

Both ``video_time`` and ``capture_time`` hold the as-cut clock at ingest.
The separate selfcap-curate package later shifts ``video_time`` while preserving
``capture_time``. No latency correction or alignment is applied here.

Permissions. Large parts of the corpus are mode 000 (a NAS sync artifact; a
corpus-wide chmod is pending). Discovery gates every required file behind
``os.access(..., os.R_OK)`` and silently drops episodes that fail; convert warns
and degrades instead of aborting. Session ``metadata.json`` is currently
unreadable for **all** 692 ``cut`` sessions, so ``task`` / ``collector`` are
absent from the capture property until the chmod lands.

Rig topology. Exo devices sorted by name take ``rig_00..``, then the OAK ego rig,
then the Quest — nine cameras and ``rig_00..rig_05`` in the usual 4-iPhone case.
Only the Quest rig has a known pose (its left-eye track in the Quest's Y-up
world frame), so the base layer owns the root ``ViewCoordinates``. Exo extrinsics
are empty in the raw calibration: those rigs are world-anchored-unknown and carry
no transform at all, waiting for a calibration/SLAM layer to localize them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from rerun.chunk import Chunk
from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
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
    log_imu,
    log_pose_track,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.records import read_csv_columns
from dataforge.video_encoding import read_video_start_ns

DATE_DIR_RE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
"""Capture-day directories under the subset; the loose ``manifest.json`` siblings do not match."""
EPISODE_DIR_RE: re.Pattern[str] = re.compile(r"^episode-(?P<number>\d+)$")
"""Episode directory names, e.g. ``episode-009``."""
SESSION_KEY_LENGTH: int = 8
"""Prefix of the session uuid used in the sequence identity (uuid4 is unique well inside 8 hex chars)."""

EGO_STREAM_LAYOUTS: tuple[tuple[str, str], ...] = (("rgb", "center"), ("left", "left"), ("right", "right"))
"""OAK stream name → ``positional_layout`` in ``ego/calibration.json``, in cam_00..cam_02 order.

Keying on ``positional_layout`` rather than ``camera_id`` because the OAK's
``CAM_A``/``CAM_B``/``CAM_C`` lettering is device-firmware detail.
"""
QUEST_STREAM_LAYOUTS: tuple[tuple[str, str], ...] = (("left", "left"), ("right", "right"))
"""Quest stream name → ``positional_layout``, in cam_00..cam_01 order."""

IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres; the Quest world is metric, so this reads as 10 cm."""
IMU_DEVICE: int = 0
"""The OAK ships one IMU, so the ego rig has a single ``imu_00``."""
DEFAULT_EXO_CAMERAS: int = 4
"""Exo device count in 2937 of the 3085 readable ``cut`` episodes; shapes the dataset-wide blueprint."""

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
class CameraPane:
    """One camera's identity in the blueprint: where it lives and what to call it."""

    name: str
    """Human label shown on the 2D view."""
    rig: int
    """Rig index owning the camera."""
    cam: int
    """Camera index within the rig."""
    kind: str
    """Device group the pane belongs to: ``exo``, ``ego``, or ``quest``.

    Blueprints cannot select entities by AnyValues, so the exo/ego grouping the
    layout communicates has to travel with the pane list itself."""


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


def raw_calibration_chunks(episode_dir: Path, devices: list[str]) -> list[Chunk]:
    """Preserve opaque vendor JSON verbatim on each device rig as static raw-source facts."""
    folders: list[Path] = [episode_dir / "exo" / device for device in devices] + [episode_dir / "ego", episode_dir / "quest"]
    return [
        Chunk.from_columns(
            schema.rig_path(rig),
            indexes=[],
            columns=rr.AnyValues.columns(raw_calibration_json=[(folder / "calibration.json").read_bytes().decode("utf-8")]),
        )
        for rig, folder in enumerate(folders)
    ]


@dataclass
class SelfcapConfig(DataforgeDatasetConfig):
    """Self-collected exo/ego/Quest captures, cut into per-task episodes."""

    command: ClassVar[str] = "selfcap"
    """Registry key, catalog dataset name, and identity ``dataset`` part."""

    _target: type = field(default_factory=lambda: SelfcapDataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = Path("/mnt/nas/datasets/exoego-self-collected/main")
    """Corpus root holding the ``cut`` / ``good`` / ``sample`` subsets."""
    subset: str = "cut"
    """Subset directory to convert; only ``cut`` holds per-episode captures."""


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
    ego_rig: int = len(devices)
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
            exo_cameras.append(CameraPlan(rig=rig, cam=0, video_path=device_dir / f"{device}.mp4", calibration=calibration, name=device, kind="rgb"))
            panes.append(CameraPane(name=device.split("-")[0], rig=rig, cam=0, kind="exo"))
        # reference=schema.CAM0_REFERENCE: a single-camera rig's frame is its camera's frame by construction.
        rigs.append(RigPlan(rig=rig, reference=schema.CAM0_REFERENCE, name=device, kind="exo", cameras=tuple(exo_cameras)))

    ego_dir: Path = episode_dir / "ego"
    ego_calibration: dict[str, CameraCalibration] = read_calibration(ego_dir / "calibration.json")
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
            )
        )
        panes.append(CameraPane(name=f"ego {stream}", rig=ego_rig, cam=cam, kind="ego"))
    # reference=schema.CAM0_REFERENCE: the OAK's rgb camera stands in for the rig frame while the
    # stereo extrinsics stay unlogged (see read_calibration's TODO on their units).
    rigs.append(RigPlan(rig=ego_rig, reference=schema.CAM0_REFERENCE, name="oak", kind="ego", cameras=tuple(ego_cameras)))

    quest_dir: Path = episode_dir / "quest"
    quest_calibration: dict[str, CameraCalibration] = read_calibration(quest_dir / "calibration.json")
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
            )
        )
        panes.append(CameraPane(name=f"quest {stream}", rig=quest_rig, cam=cam, kind="quest"))
    # reference=schema.CAM0_REFERENCE: the head_pose track logged on the rig node *is* the left-eye pose.
    rigs.append(RigPlan(rig=quest_rig, reference=schema.CAM0_REFERENCE, name="quest", kind="quest", cameras=tuple(quest_cameras)))

    return EpisodePlan(episode_dir=episode_dir, rigs=tuple(rigs), ego_rig=ego_rig, quest_rig=quest_rig, panes=tuple(panes))


def build_blueprint(panes: tuple[CameraPane, ...], ego_rig: int) -> rrb.Blueprint:
    """Default layout: the 3D scene plus a camera grid over the ego IMU plots.

    Args:
        panes: Camera labels and indices, in rig-then-camera order.
        ego_rig: Rig index of the OAK, whose ``imu_00`` feeds the time-series views.

    Returns:
        The blueprint embedded in every SelfCap base-layer rrd.
    """

    def views(kind: str) -> list[rrb.Spatial2DView]:
        return [blueprints.camera_view(pane.name, pane.rig, pane.cam) for pane in panes if pane.kind == kind]

    # The layout is grouped by device kind (blueprints cannot select on the rigs'
    # `kind` AnyValues, so the grouping travels through the pane list instead) and
    # mirrors simplecv's exoego ergonomics: a dominant 3D scene with the ego views
    # (OAK + Quest) in a two-column grid beside it, the exo cameras as a bottom
    # row, and the IMU strip beneath. Layout only — the logged schema is untouched.
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Scene", origin="/", line_grid=True),
                rrb.Grid(*views("ego"), *views("quest"), grid_columns=2, name="Ego"),
                column_shares=[3.0, 1.0],
            ),
            rrb.Horizontal(*views("exo"), name="Exo"),
            rrb.Horizontal(
                blueprints.sensor_plot("Gyroscope", schema.imu_path(ego_rig, IMU_DEVICE), schema.gyro_path(ego_rig, IMU_DEVICE)),
                blueprints.sensor_plot("Accelerometer", schema.imu_path(ego_rig, IMU_DEVICE), schema.accel_path(ego_rig, IMU_DEVICE)),
            ),
            row_shares=[4.0, 1.4, 1.0],
        ),
        collapse_panels=True,
    )


class SelfcapDataset(DataforgeDataset[SelfcapConfig, Path]):
    """Converts SelfCap episodes into exoego:v2 base-layer recordings."""

    def default_blueprint(self) -> rrb.Blueprint:
        """Corpus-wide calibration review layout for the modal four-iPhone rig.

        The calibrated Quest world in 3D beside the OAK rgb, the OAK left/right mono pair and both
        Quest cameras, the iPhones beneath; every camera pane also draws ``/world/gt`` (the Quest hands),
        so the calib layer's poses can be judged on the video. On the base layer alone
        the panes show the videos and the uncalibrated iPhones sit at the origin.
        Exo device names vary per session, so the dataset default labels them by rig;
        the per-recording blueprint embedded at convert keeps the real names.

        Sapiens2 detections only feed alignment, so they stay in each pane's tree but hidden. The ego panes
        also draw ``/world/pred`` (the triangulated keypoints) through their pose chain; the iPhones carry
        explicit lens-model projections instead.
        """
        ego_rig, quest_rig = DEFAULT_EXO_CAMERAS, DEFAULT_EXO_CAMERAS + 1

        def camera(name: str, rig: int, cam: int, *, ego: bool = False) -> rrb.Spatial2DView:
            pinhole: str = schema.pinhole_path(rig, cam)
            return blueprints.camera_view(
                name,
                rig,
                cam,
                contents=[f"{pinhole}/**", "/world/gt/**", *(["/world/pred/**"] if ego else [])],
                overrides={f"{pinhole}/pred/sapiens2": rrb.EntityBehavior(visible=False)},
            )

        ego = rrb.Vertical(
            camera("OAK rgb", ego_rig, 0, ego=True),
            rrb.Horizontal(camera("OAK left", ego_rig, 1, ego=True), camera("OAK right", ego_rig, 2, ego=True)),
            rrb.Horizontal(camera("Quest left", quest_rig, 0, ego=True), camera("Quest right", quest_rig, 1, ego=True)),
        )
        exo: list[rrb.Spatial2DView] = [camera(f"exo rig_{rig:02d}", rig, 0) for rig in range(DEFAULT_EXO_CAMERAS)]
        scene = rrb.Spatial3DView(name="World (Quest frame, calibrated)", origin="/world")
        return rrb.Blueprint(
            rrb.Vertical(rrb.Horizontal(scene, ego, column_shares=[3.0, 1.2]), rrb.Horizontal(*exo), row_shares=[3.0, 2.0]),
            collapse_panels=True,
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card: the ego rgb stream only, nothing else decoded.

        Every visible table row renders through this at once, so the card must
        decode exactly one video (see the base-class contract).
        """
        return rrb.Blueprint(blueprints.camera_view("ego rgb", DEFAULT_EXO_CAMERAS, 0), collapse_panels=True)

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
        """Verify the local corpus; SelfCap has no upstream fetch."""
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
        if writing.should_skip(target, force=force):
            print(f"skip {identity.sequence_key} → {target}")
            return target

        # Plan first, log second: the blueprint panes and num_cameras below are the
        # plan's, so they can only ever describe cameras that really get logged.
        plan: EpisodePlan = build_episode_plan(source)

        with writing.atomic_recording(
            target,
            recording_id=identity.recording_id,
            default_blueprint=build_blueprint(plan.panes, ego_rig=plan.ego_rig),
        ) as recording:
            # The base layer owns the root ViewCoordinates because it already carries a real
            # moving-rig pose: the Quest world is right-handed Y-up.
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True, recording=recording)

            rr.send_chunks(raw_calibration_chunks(plan.episode_dir, [rig.name for rig in plan.rigs if rig.kind == "exo"]), recording=recording)
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
            self._log_ego_imu(recording, plan)
            self._log_quest_pose(recording, plan)
            hands: QuestHands = read_quest_hands(plan.episode_dir / "quest")
            log_quest_hands(recording, hands)

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

        print(f"done {identity.sequence_key} → {target} ({len(plan.panes)} cameras, {num_frames} frames)")
        return target

    def _log_camera(self, recording: rr.RecordingStream, camera: CameraPlan) -> int:
        """Log one planned camera: node metadata, static pinhole, and its video stream.

        No ``rig_T_cam`` transform is written: the raw exo calibration ships empty
        extrinsics, and the ego/Quest ones are in unvalidated units. Brown-Conrady
        coefficients ride along when the calibration declares them.

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
        # TODO(dataforge): the Quest intrinsics describe the 1280x1280 sensor array while
        # capture_resolution (and the mp4) is 1280x960, so cy lands ~160 px below the image
        # centre. Confirm the crop origin against the device before shifting the principal point.
        rr.log(
            schema.pinhole_path(camera.rig, camera.cam),
            PinholeWithDistortion(
                pinhole=rr.Pinhole(
                    image_from_camera=image_from_camera,
                    width=calibration.width,
                    height=calibration.height,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                ),
                distortion=distortion,
            ),
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

    def _log_quest_pose(self, recording: rr.RecordingStream, plan: EpisodePlan) -> None:
        """Log the Quest's left-eye track as the temporal ``world_T_rig`` on its rig node."""
        head_pose_path: Path = plan.episode_dir / "quest" / "head_pose.csv"
        poses: HeadPoses = read_head_poses(head_pose_path)
        if not poses.times_ns.size:
            print(f"  warning: no head poses in {head_pose_path}; the quest rig stays unposed")
            return
        log_pose_track(
            recording,
            schema.rig_path(plan.quest_rig),
            times_ns=poses.times_ns,
            translations_xyz=poses.translations_xyz,
            quaternions_xyzw=poses.quaternions_xyzw,
            capture_time=True,
        )
