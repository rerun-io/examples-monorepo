"""LaMAria: ETH CVG's Aria Gen1 SLAM benchmark → one exoego:v2 base-layer rrd per sequence.

Upstream is a plain Apache-indexed archive at ``cvg-data.inf.ethz.ch/lamaria/``
(github.com/cvg/lamaria), four directories deep::

    raw_data/{training,test}/<seq>.vrs             897 MB … 10 GB per sequence
    aria_calibrations/{training,test}/<seq>.json   ~2.7 kB, the published body-frame calibration
    ground_truth/pseudo_dense/<seq>.txt           training only, 0.2–2.7 MB
    ground_truth/sparse/<seq>.json                only sequences with surveyed control points

The raw tree mirrors the **official** layout, so the upstream evaluation tools
run on it unchanged::

    <root>/<split>/<seq>/raw_data/<seq>.vrs
    <root>/<split>/<seq>/aria_calibrations/<seq>.json
    <root>/<split>/<seq>/ground_truth/pGT/<seq>.txt
    <root>/<split>/<seq>/ground_truth/control_points/<seq>.json

**Raw is scratch.** The default selection alone is 18.2 GB of VRS, so
``download`` fetches only the small files (a few MB) and writes a
``manifest.json`` recording what the archive held; ``convert`` then fetches
**one** VRS, encodes its three camera streams, writes the rrd, and deletes the
VRS and the temp mp4s again. ``--keep-raw`` keeps them. The archive is flaky (it
was down for hours during development), so every fetch resumes and a stalled one
is retried.

**Clocks.** Every VRS timestamp is Aria DEVICE time in nanoseconds, and the
published pGT and control points are stamped on that same clock. ``video_time``
is therefore that clock **unshifted**, so a ground-truth pose lines up with its
frame 1:1 with no retiming anywhere.

**Frames.** The rig frame is imu-right, which is what LaMAria's published
calibration uses as its body frame, so the rig node states
``reference = "imu_00"`` and every ``rig_T_sensor`` in ``dataforge.aria`` is
directly comparable with that file's ``T_b_s``. ``cam_00`` is camera-slam-left,
``cam_01`` camera-slam-right, ``cam_02`` camera-rgb; ``imu_00`` is imu-right
(identity ``rig_T_imu`` by construction) and ``imu_01`` imu-left, 129 mm away.
The base layer logs NO transform on the rig node and NO root
``ViewCoordinates``: the gt layer establishes the world frame and owns both.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import requests
import rerun as rr
import rerun.blueprint as rrb
import serde
import serde.json
from beartype.roar import BeartypeException
from jaxtyping import Float64, Int64
from numpy import ndarray
from projectaria_tools.core import data_provider
from simplecv.camera_parameters import Fisheye62Parameters
from simplecv.rerun_log_utils import log_pinhole

from dataforge import aria, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    FrameKind,
    FrameSource,
    ImuChannel,
    encode_frames_to_mp4,
    log_imu,
    log_rig_node,
    log_video_stream,
    require_av1_nvenc,
    resolve_ffmpeg,
)

LamariaSplit: TypeAlias = Literal["training", "test"]
"""Which half of the benchmark a sequence belongs to; only ``training`` ships ground truth."""

SPLITS: tuple[LamariaSplit, ...] = ("training", "test")
"""Both raw_data index pages ``download`` consults to resolve a sequence's split."""

LamariaSet: TypeAlias = Literal["controlled", "additional"]
"""Upstream's two collections: the ``R_*`` controlled experimental set, and the
``sequence_<n>_<m>`` additional set recorded around the city."""

DEFAULT_SEQUENCES: tuple[str, ...] = ("R_01_easy", "R_04_medium", "R_11_5cp", "sequence_1_19", "sequence_4_11")
"""The five sequences dataforge converts by default: 18.2 GB of VRS spanning both
sets, all three difficulty tiers, one low-light capture, and every ground-truth
shape (pGT alone, and pGT with 5, 14 and 15 surveyed control points)."""

MANIFEST_NAME: str = "manifest.json"
"""What ``download`` writes under ``root`` and every other verb reads."""

RAW_DATA_DIR: str = "raw_data"
"""Archive (and local) directory holding the VRS files, per split."""
CALIBRATION_DIR: str = "aria_calibrations"
"""Archive (and local) directory holding the published body-frame calibrations, per split."""
PSEUDO_GT_REMOTE_DIR: str = "ground_truth/pseudo_dense"
"""Archive directory of the dense pseudo ground truth; it lands locally in ``ground_truth/pGT``."""
CONTROL_POINTS_REMOTE_DIR: str = "ground_truth/sparse"
"""Archive directory of the surveyed control points; it lands locally in ``ground_truth/control_points``."""
PSEUDO_GT_LOCAL_DIR: str = "ground_truth/pGT"
"""Where the official layout keeps the pGT inside a sequence directory."""
CONTROL_POINTS_LOCAL_DIR: str = "ground_truth/control_points"
"""Where the official layout keeps the control points inside a sequence directory."""

INDEX_TIMEOUT_S: float = 30.0
"""Per-request timeout for an index page; they are a few kilobytes of HTML."""

RIG: int = 0
"""One Aria per sequence; the glasses are ``rig_00``."""
RIG_REFERENCE: str = "imu_00"
"""The rig frame *is* imu-right's, LaMAria's published body frame."""
GT_SOURCE: str = "gt"
"""``/world/runs/gt/`` — the run name the gt layer writes its trajectory under."""

FOLLOW_FORWARD: tuple[float, float, float] = (0.018, -0.967, -0.253)
"""Where the wearer looks, in the rig (imu-right) frame.

The Aria device frame *is* camera-slam-left's frame (``device_T_cam`` for
``1201-1`` is the identity), so the published ``cam0.T_b_s`` rotation is
``rig_R_cam0`` and its third column is that camera's optical axis (RDF) in the
rig frame. Typed in from R_01_easy's published calibration; a test re-derives it
from the reference fixture."""
FOLLOW_UP: tuple[float, float, float] = (-0.198, 0.245, -0.949)
"""The wearer's up, in the rig frame: the negated second column of the same
rotation (RDF's ``y`` is image-down, and glasses are worn upright)."""
FOLLOW_BACK_M: float = 0.9
"""How far behind the wearer the follow eye sits, along their own forward."""
FOLLOW_UP_M: float = 0.45
"""How far above the wearer the follow eye sits, along their own up."""
FOLLOW_AHEAD_M: float = 0.3
"""How far ahead of the wearer the eye aims, so the shot leads the motion."""
FOLLOW_TRAIL_SECONDS: float = -10.0
"""Cursor-relative start of the gt trail's visible window in the Follow view."""
IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres; the SLAM baseline is ~11 cm, so the three frusta stay legible."""

NOMINAL_FPS: dict[aria.AriaStreamId, int] = {
    aria.SLAM_LEFT_STREAM_ID: 20,
    aria.SLAM_RIGHT_STREAM_ID: 20,
    aria.RGB_STREAM_ID: 10,
}
"""Container frame rate per camera stream. Only the mp4 header cares:
``log_video_stream(times_ns=...)`` stamps every sample with its real capture time."""
FRAME_KINDS: dict[aria.AriaStreamId, FrameKind] = {
    aria.SLAM_LEFT_STREAM_ID: "gray8",
    aria.SLAM_RIGHT_STREAM_ID: "gray8",
    aria.RGB_STREAM_ID: "rgb24",
}
"""How each stream's decoded frames reach the encoder: single-plane gray for the
SLAM pair, and interleaved RGB (not BGR) for camera-rgb."""
CAMERA_KINDS: dict[aria.AriaStreamId, str] = {
    aria.SLAM_LEFT_STREAM_ID: "grayscale",
    aria.SLAM_RIGHT_STREAM_ID: "grayscale",
    aria.RGB_STREAM_ID: "rgb",
}
"""exoego:v2 content hint on each camera node."""
VRS_FETCH_ATTEMPTS: int = 4
"""How many times a stalled VRS transfer is resumed before ``convert`` gives up."""


@serde.serde
@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """One sequence as ``download`` resolved it from the archive's index pages."""

    sequence: str
    """Upstream sequence name, e.g. ``R_01_easy``; also the identity's only part."""
    split: LamariaSplit
    """Which ``raw_data/<split>/`` index listed the VRS."""
    vrs_url: str
    """Absolute URL ``convert`` fetches the VRS from."""
    vrs_display_bytes: int
    """Apache's **rounded** display size (``897M`` → 940 572 672). Good for a
    budget or a summary line and useless for verification, so it is never passed
    to ``http_fetch`` as an expected size."""
    has_pseudo_gt: bool
    """Whether ``ground_truth/pseudo_dense/`` lists this sequence (training only)."""
    has_control_points: bool
    """Whether ``ground_truth/sparse/`` lists this sequence (only surveyed ones)."""


@serde.serde
@dataclass(frozen=True, slots=True)
class LamariaManifest:
    """``<root>/manifest.json``: what the archive held when ``download`` last ran."""

    base_url: str
    """Archive root the records were resolved against."""
    sequences: list[SequenceRecord]
    """One record per selected sequence, sorted by name."""


@dataclass(frozen=True, slots=True)
class LamariaSource:
    """One sequence as discovery found it; ``convert`` needs nothing else from the tree."""

    sequence: str
    """Upstream sequence name."""
    split: LamariaSplit
    """``training`` or ``test``; part of the raw path and of the capture properties."""
    vrs_url: str
    """Where the VRS is fetched from on demand."""
    vrs_display_bytes: int
    """Apache's rounded size, for the fetch's progress line only."""
    vrs_path: Path
    """``<root>/<split>/<seq>/raw_data/<seq>.vrs``; deleted after convert unless ``--keep-raw``."""
    calibration_path: Path
    """``.../aria_calibrations/<seq>.json``, the published calibration; always kept."""
    pseudo_gt_path: Path | None
    """``.../ground_truth/pGT/<seq>.txt``, or ``None`` when the archive has none."""
    control_points_path: Path | None
    """``.../ground_truth/control_points/<seq>.json``, or ``None`` when the sequence was not surveyed."""


@dataclass(frozen=True, slots=True)
class CameraStream:
    """One camera stream of an open VRS, in the form ``convert`` consumes it.

    The frames are a lazy iterator on purpose: a sequence is thousands of
    1408x1408 RGB planes, and they go straight from the VRS decoder into
    ffmpeg's stdin without a decoded frame ever landing on disk.
    """

    stream_id: aria.AriaStreamId
    """Which Aria stream this is; it names the camera node and picks its ``kind``."""
    camera: Fisheye62Parameters
    """The camera's calibration, extrinsics holding ``rig_T_cam``."""
    frames: Iterator[bytes]
    """Raw planes in presentation order, one per frame, native orientation."""
    times_ns: Int64[ndarray, "n_frames"]
    """Capture times on Aria's device clock, one per frame, in the same order."""
    frame_source: FrameSource
    """How ``frames`` is laid out for the encoder (``gray8`` or ``rgb24``, plus the size)."""
    fps: int
    """Nominal container frame rate; the real per-sample times come from ``times_ns``."""


@dataclass(frozen=True, slots=True)
class ImuStream:
    """One IMU stream of an open VRS: both channels, and where the sensor sits."""

    stream_id: aria.AriaStreamId
    """Which Aria stream this is; it names the IMU node."""
    gyro: ImuChannel
    """Angular velocity in rad/s, raw and unrectified, at the sensor's native rate."""
    accel: ImuChannel
    """Linear acceleration in m/s^2, on the same timestamps."""
    rig_T_imu: Float64[ndarray, "4 4"]
    """The sensor's pose in the rig frame; the identity for imu-right, which *is* the rig."""


@dataclass(frozen=True, slots=True)
class SequenceStreams:
    """Everything a base-layer recording needs out of one VRS.

    This is the seam the VRS sits behind: ``convert`` never touches
    projectaria-tools directly, so the orchestration around it (encode, log,
    delete) is testable with synthetic streams.
    """

    cameras: tuple[CameraStream, ...]
    """Camera streams in ``cam_00``, ``cam_01``, ``cam_02`` order."""
    imus: tuple[ImuStream, ...]
    """IMU streams in ``imu_00`` (imu-right), ``imu_01`` (imu-left) order."""


@dataclass(frozen=True, slots=True)
class RecordingSummary:
    """What one converted sequence turned out to hold; ``convert`` prints it."""

    num_frames: int
    """Video samples written for camera-slam-left, the pGT's own camera."""
    duration_s: float
    """Span of every logged stream, in seconds."""
    control_point_count: int
    """Surveyed control points the sequence ships; 0 when it was never surveyed."""


@dataclass
class LamariaConfig(DataforgeDatasetConfig):
    """LaMAria: Aria Gen1 egocentric SLAM sequences, fetched from ETH CVG on demand."""

    command: ClassVar[str] = "lamaria"
    """Registry key, catalog dataset name, and identity ``dataset`` part."""

    _target: type = field(default_factory=lambda: LamariaDataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "lamaria")
    """Raw tree in the official layout. Point it at local NVMe: every VRS written
    here is deleted again once its rrd exists."""
    sequences: tuple[str, ...] = DEFAULT_SEQUENCES
    """Sequences to download and convert; a name the archive does not list is an error."""
    keep_raw: bool = False
    """Keep the fetched VRS and the encoded mp4s instead of deleting them."""
    base_url: str = "https://cvg-data.inf.ethz.ch/lamaria/"
    """Archive root; every index page and every file hangs off it."""


def sequence_set(sequence: str) -> LamariaSet:
    """Which upstream collection a sequence name belongs to.

    Args:
        sequence: Upstream sequence name.

    Returns:
        ``controlled`` for the ``R_*`` controlled experimental set, ``additional``
        for the ``sequence_<n>_<m>`` captures.
    """
    return "controlled" if sequence.startswith("R_") else "additional"


def sequence_challenge(sequence: str) -> str | None:
    """The difficulty tier the controlled set encodes in its own name.

    ``R_01_easy`` → ``easy``, ``R_11_5cp`` → ``5cp``. The additional set names
    carry no tier (``sequence_1_19``), so they get ``None`` rather than a guess.

    Args:
        sequence: Upstream sequence name.

    Returns:
        The trailing name token for the controlled set, else ``None``.
    """
    if sequence_set(sequence) != "controlled":
        return None
    return sequence.rsplit("_", 1)[-1]


def open_streams(vrs_path: Path) -> SequenceStreams:
    """Open one VRS and expose its three cameras and two IMUs in rig-frame form.

    The whole projectaria-tools surface of this dataset lives here, so the rest
    of ``convert`` is testable without a 900 MB file. The provider stays alive
    through the frame generators that reference it.

    Args:
        vrs_path: The sequence's ``.vrs``.

    Returns:
        The camera and IMU streams, in ``cam_MM`` / ``imu_MM`` order.
    """
    provider: data_provider.VrsDataProvider = aria.open_vrs(vrs_path)
    rig: aria.AriaRig = aria.AriaRig.from_provider(provider)
    cameras: list[CameraStream] = []
    for stream_id in aria.CAMERA_STREAM_IDS:
        camera: Fisheye62Parameters = rig.cameras[stream_id]
        cameras.append(
            CameraStream(
                stream_id=stream_id,
                camera=camera,
                # Native orientation, no rotation: the sideways Aria frames are what
                # the calibration describes, so rotating them would invalidate it.
                frames=(image.tobytes() for _, image in aria.iter_frames(provider, stream_id)),
                times_ns=aria.frame_timestamps_ns(provider, stream_id),
                frame_source=FrameSource(FRAME_KINDS[stream_id], width=camera.intrinsics.width, height=camera.intrinsics.height),
                fps=NOMINAL_FPS[stream_id],
            )
        )
    imus: list[ImuStream] = []
    for stream_id in aria.IMU_STREAM_IDS:
        channels: aria.ImuSamples = aria.read_imu(provider, stream_id)
        imus.append(ImuStream(stream_id=stream_id, gyro=channels[0], accel=channels[1], rig_T_imu=rig.rig_T_imu[stream_id]))
    return SequenceStreams(cameras=tuple(cameras), imus=tuple(imus))


def follow_eye_controls() -> rrb.EyeControls3D:
    """Chase camera for the Follow view, expressed in the rig (imu-right) frame.

    The view's origin is the rig node, so a fixed eye in this frame rides the
    glasses: it sits ``FOLLOW_BACK_M`` behind the wearer and ``FOLLOW_UP_M``
    above them and aims ``FOLLOW_AHEAD_M`` in front, which keeps both the wearer
    and the ground they walk over in shot.

    Returns:
        The eye controls the Follow view of both blueprints uses.
    """
    forward_xyz: Float64[ndarray, "3"] = np.array(FOLLOW_FORWARD, dtype=np.float64)
    up_xyz: Float64[ndarray, "3"] = np.array(FOLLOW_UP, dtype=np.float64)
    position_xyz: Float64[ndarray, "3"] = -FOLLOW_BACK_M * forward_xyz + FOLLOW_UP_M * up_xyz
    look_target_xyz: Float64[ndarray, "3"] = FOLLOW_AHEAD_M * forward_xyz
    # EyeControls3D is marked unstable by the SDK; re-validate this factory on Rerun bumps.
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.FirstPerson,
        position=tuple(position_xyz.tolist()),
        look_target=tuple(look_target_xyz.tolist()),
        eye_up=FOLLOW_UP,
        spin_speed=0.0,
    )


def camera_views() -> list[rrb.Spatial2DView]:
    """One 2D pane per camera stream, labelled the way the VRS names it."""
    return [
        rrb.Spatial2DView(
            name=aria.STREAM_LABELS[stream_id],
            origin=schema.pinhole_path(RIG, index),
            contents=f"{schema.pinhole_path(RIG, index)}/**",
        )
        for index, stream_id in enumerate(aria.CAMERA_STREAM_IDS)
    ]


def build_blueprint() -> rrb.Blueprint:
    """Default layout: the rig in 3D beside the camera grid, over the imu-right plots.

    Returns:
        The blueprint embedded in every LaMAria base-layer rrd and registered as
        the catalog dataset's default.
    """
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="World",
                        origin="/",
                        line_grid=True,
                        # The overview shows the whole gt path; its cursor trail stays hidden.
                        # Overrides on entities a base-only recording lacks are simply inert.
                        overrides={schema.trail_path(GT_SOURCE): rrb.EntityBehavior(visible=False)},
                    ),
                    # Follow-cam (rerun-io/eye_control_example pattern): the view's origin
                    # IS the rig frame, so a fixed first-person eye in that frame rides the
                    # glasses. Inert until the gt layer animates rig_00.
                    rrb.Spatial3DView(
                        name="Follow",
                        origin=schema.rig_path(RIG),
                        contents="/**",
                        line_grid=True,
                        overrides={
                            schema.trajectory_path(GT_SOURCE): rrb.EntityBehavior(visible=False),
                            schema.trail_path(GT_SOURCE): rrb.VisibleTimeRanges(
                                rrb.VisibleTimeRange(
                                    schema.TIMELINE,
                                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=FOLLOW_TRAIL_SECONDS),
                                    end=rrb.TimeRangeBoundary.cursor_relative(),
                                )
                            ),
                        },
                        eye_controls=follow_eye_controls(),
                    ),
                ),
                rrb.Grid(*camera_views(), grid_columns=2, name="Synchronized cameras"),
                column_shares=[3, 2],
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(
                    name="Gyroscope",
                    origin=schema.imu_path(RIG, 0),
                    contents=schema.gyro_path(RIG, 0),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
                rrb.TimeSeriesView(
                    name="Accelerometer",
                    origin=schema.imu_path(RIG, 0),
                    contents=schema.accel_path(RIG, 0),
                    plot_legend=rrb.PlotLegend(visible=True),
                ),
            ),
            row_shares=[3, 1],
        ),
        rrb.TimePanel(timeline=schema.TIMELINE),
        collapse_panels=True,
    )


def build_table_blueprint() -> rrb.Blueprint:
    """Segment-table preview card: the 3D rig with no video textures, plus slam-left.

    Every visible table row renders through this at once, so exactly one video
    stream is decoded and the other two are *excluded* rather than hidden.
    """
    video_exclusions: list[str] = [f"- {schema.video_path(RIG, index)}/**" for index in range(len(aria.CAMERA_STREAM_IDS))]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Follow",
                origin=schema.rig_path(RIG),
                contents=["/**", *video_exclusions, f"- {schema.trail_path(GT_SOURCE)}/**"],
                line_grid=True,
                eye_controls=follow_eye_controls(),
            ),
            camera_views()[0],
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline=schema.TIMELINE),
    )


class LamariaDataset(DataforgeDataset[LamariaConfig, LamariaSource]):
    """Converts LaMAria sequences into exoego:v2 base-layer recordings."""

    def archive_url(self, *parts: str) -> str:
        """Absolute URL of one archive path, e.g. ``raw_data/training/R_01_easy.vrs``."""
        return "/".join([self.config.base_url.rstrip("/"), *parts])

    def index(self, directory: str) -> list[transports.IndexEntry]:
        """List one Apache index page under ``base_url`` as ``(name, size_bytes)``.

        Args:
            directory: Archive-relative directory, e.g. ``raw_data/training``.

        Returns:
            One entry per listed file, in page order.

        Raises:
            requests.HTTPError: If the archive answered 4xx/5xx — the flaky-archive
                case, and one worth failing on rather than treating as "empty".
        """
        page: requests.Response = requests.get(f"{self.archive_url(directory)}/", timeout=INDEX_TIMEOUT_S)
        page.raise_for_status()
        return transports.parse_apache_index(page.text)

    def manifest(self) -> LamariaManifest:
        """Read ``<root>/manifest.json``, the record ``download`` left behind."""
        path: Path = self.config.root / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"no {MANIFEST_NAME} at {path}; run `dataforge-download lamaria` first")
        return serde.json.from_json(LamariaManifest, path.read_text())

    def source(self, record: SequenceRecord) -> LamariaSource:
        """Place one manifest record in the local official layout."""
        sequence_dir: Path = self.config.root / record.split / record.sequence
        return LamariaSource(
            sequence=record.sequence,
            split=record.split,
            vrs_url=record.vrs_url,
            vrs_display_bytes=record.vrs_display_bytes,
            vrs_path=sequence_dir / RAW_DATA_DIR / f"{record.sequence}.vrs",
            calibration_path=sequence_dir / CALIBRATION_DIR / f"{record.sequence}.json",
            pseudo_gt_path=sequence_dir / PSEUDO_GT_LOCAL_DIR / f"{record.sequence}.txt" if record.has_pseudo_gt else None,
            control_points_path=sequence_dir / CONTROL_POINTS_LOCAL_DIR / f"{record.sequence}.json" if record.has_control_points else None,
        )

    def discover(self) -> list[tuple[SequenceIdentity, LamariaSource]]:
        """Pair every selected sequence whose small files are on disk with its source.

        The manifest says what the archive holds; the small files on disk are the
        evidence that ``download`` finished for a sequence. A selected name that
        fails either check is announced and skipped rather than fatal: the
        selection defaults to all five sequences, so a narrower
        ``download --sequences`` must still leave the rest convertible, and a
        name the archive really does not have is already a hard error at
        ``download``.
        """
        records: dict[str, SequenceRecord] = {record.sequence: record for record in self.manifest().sequences}
        pairs: list[tuple[SequenceIdentity, LamariaSource]] = []
        for name in sorted(set(self.config.sequences)):
            record: SequenceRecord | None = records.get(name)
            if record is None:
                print(f"  warning: skipping {name}: {MANIFEST_NAME} does not list it; run `dataforge-download lamaria --sequences {name}`")
                continue
            source: LamariaSource = self.source(record)
            missing: list[Path] = [
                path
                for path in (source.calibration_path, source.pseudo_gt_path, source.control_points_path)
                if path is not None and not path.is_file()
            ]
            if missing:
                print(f"  warning: skipping {name}: {len(missing)} small file(s) not on disk, first {missing[0]}; re-run dataforge-download")
                continue
            pairs.append((SequenceIdentity(dataset=self.config.name, parts=(name,)), source))
        return pairs

    def download(self) -> None:
        """Resolve the archive's index pages, write the manifest, fetch the small files.

        Deliberately **not** a bulk fetch: the default selection alone is 18.2 GB
        of VRS and ``convert`` pulls one sequence at a time, so what is worth
        having up front is the few megabytes of calibration and ground truth
        plus a record of what the archive holds. Which ``raw_data/<split>/``
        index lists a sequence is what resolves its split; the two ground-truth
        indexes resolve what it ships.
        """
        selected: list[str] = sorted(set(self.config.sequences))
        splits: dict[str, LamariaSplit] = {}
        vrs_display_bytes: dict[str, int] = {}
        for split in SPLITS:
            for name, size in self.index(f"{RAW_DATA_DIR}/{split}"):
                sequence: str = name.removesuffix(".vrs")
                if name.endswith(".vrs") and sequence in selected:
                    splits[sequence] = split
                    vrs_display_bytes[sequence] = size
        unlisted: list[str] = [name for name in selected if name not in splits]
        if unlisted:
            raise ValueError(f"no raw_data index at {self.config.base_url} lists a VRS for {', '.join(unlisted)}")

        calibrated: set[str] = {
            name.removesuffix(".json")
            for split in sorted(set(splits.values()))
            for name, _ in self.index(f"{CALIBRATION_DIR}/{split}")
            if name.endswith(".json")
        }
        uncalibrated: list[str] = [name for name in selected if name not in calibrated]
        if uncalibrated:
            raise ValueError(f"{CALIBRATION_DIR} lists no published calibration for {', '.join(uncalibrated)}; the archive layout changed")
        with_pseudo_gt: set[str] = {name.removesuffix(".txt") for name, _ in self.index(PSEUDO_GT_REMOTE_DIR) if name.endswith(".txt")}
        with_control_points: set[str] = {name.removesuffix(".json") for name, _ in self.index(CONTROL_POINTS_REMOTE_DIR) if name.endswith(".json")}

        records: list[SequenceRecord] = [
            SequenceRecord(
                sequence=name,
                split=splits[name],
                vrs_url=self.archive_url(RAW_DATA_DIR, splits[name], f"{name}.vrs"),
                vrs_display_bytes=vrs_display_bytes[name],
                has_pseudo_gt=name in with_pseudo_gt,
                has_control_points=name in with_control_points,
            )
            for name in selected
        ]
        manifest_path: Path = self.config.root / MANIFEST_NAME
        with writing.atomic_write(manifest_path) as temp_path:
            temp_path.write_text(serde.json.to_json(LamariaManifest(base_url=self.config.base_url, sequences=records)))

        for record in records:
            source: LamariaSource = self.source(record)
            transports.http_fetch(self.archive_url(CALIBRATION_DIR, record.split, f"{record.sequence}.json"), dest=source.calibration_path)
            if source.pseudo_gt_path is not None:
                transports.http_fetch(self.archive_url(PSEUDO_GT_REMOTE_DIR, f"{record.sequence}.txt"), dest=source.pseudo_gt_path)
            if source.control_points_path is not None:
                transports.http_fetch(self.archive_url(CONTROL_POINTS_REMOTE_DIR, f"{record.sequence}.json"), dest=source.control_points_path)

        total_display_bytes: int = sum(record.vrs_display_bytes for record in records)
        print(
            f"lamaria: {len(records)} sequence(s) from {self.config.base_url}, "
            f"{total_display_bytes / 1e9:.1f} GB of VRS for convert to fetch one at a time"
        )
        for record in records:
            ground_truth: str = ", ".join(
                label for label, present in (("pGT", record.has_pseudo_gt), ("control points", record.has_control_points)) if present
            )
            print(f"  {record.sequence:<16} {record.split:<8} {record.vrs_display_bytes / 1e9:5.1f} GB  {ground_truth or 'no ground truth'}")
        print(f"  small files → {self.config.root}/<split>/<seq>/, manifest → {manifest_path}")

    def fetch_vrs(self, source: LamariaSource) -> Path:
        """Fetch one sequence's VRS, resuming and retrying a stalled transfer.

        The archive hangs up mid-transfer often enough that a single attempt at a
        multi-gigabyte file is not a plan: ``http_fetch`` leaves what it got on
        disk and every retry appends with a ``Range`` request. Nothing is ever
        deleted here, so even an exhausted retry budget leaves the next run less
        to do.

        The manifest's size is deliberately **not** passed as ``expected_size``:
        it is Apache's rounded display size, so it would reject every file.

        Args:
            source: The sequence to fetch.

        Returns:
            The local VRS path.

        Raises:
            RuntimeError: Every attempt stalled; the partial file is kept.
        """
        if source.vrs_path.is_file():
            print(f"  {source.vrs_path.stat().st_size / 1e9:.2f} GB already in {source.vrs_path.parent}; the fetch resumes or verifies it")
        else:
            print(f"  fetching {source.vrs_display_bytes / 1e9:.1f} GB from {source.vrs_url}")
        last_failure: str = ""
        for attempt in range(1, VRS_FETCH_ATTEMPTS + 1):
            try:
                return transports.http_fetch(source.vrs_url, dest=source.vrs_path)
            except BeartypeException:
                raise
            except (requests.RequestException, ValueError) as failure:
                last_failure = f"{type(failure).__name__}: {failure}"
                have: int = source.vrs_path.stat().st_size if source.vrs_path.is_file() else 0
                print(f"  warning: attempt {attempt}/{VRS_FETCH_ATTEMPTS} stalled at {have / 1e9:.2f} GB ({last_failure}); resuming")
        raise RuntimeError(f"{VRS_FETCH_ATTEMPTS} attempts at {source.vrs_url} all stalled, last: {last_failure}")

    def convert(self, identity: SequenceIdentity, source: LamariaSource, *, force: bool) -> Path:
        """Fetch one VRS, encode it, write the base-layer rrd, and delete the raw.

        The NVENC check comes before the fetch: a machine that cannot encode AV1
        should find out in a second rather than after a multi-gigabyte download.
        Failure keeps the VRS (a retry then resumes rather than refetching) but
        removes the temp mp4s, so the next attempt starts from a clean scratch
        directory. Nothing outside the sequence's own directory is touched.
        """
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        if writing.should_skip(target, force=force):
            print(f"skip {identity.sequence_key} → {target}")
            return target

        require_av1_nvenc(resolve_ffmpeg())
        self.fetch_vrs(source)
        work_dir: Path = source.vrs_path.parent / f"{source.sequence}-mp4"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary: RecordingSummary = self.write_recording(identity, source, work_dir=work_dir, target=target)
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            retained: int = source.vrs_path.stat().st_size if source.vrs_path.is_file() else 0
            print(f"  kept {retained / 1e9:.2f} GB of VRS in {source.vrs_path.parent} so a retry resumes instead of refetching")
            raise

        if self.config.keep_raw:
            print(f"  keeping raw: {source.vrs_path.name} and the mp4s in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            source.vrs_path.unlink(missing_ok=True)
        print(
            f"done {identity.sequence_key} → {target} ({len(aria.CAMERA_STREAM_IDS)} cameras, {summary.num_frames} frames, "
            f"{summary.duration_s:.1f} s, {summary.control_point_count} control points)"
        )
        return target

    def write_recording(self, identity: SequenceIdentity, source: LamariaSource, *, work_dir: Path, target: Path) -> RecordingSummary:
        """Encode every camera stream and write one sequence's base-layer rrd.

        Every camera is encoded before the recording opens: the VRS is read once,
        stream by stream, and ``log_video_stream`` remuxes the finished mp4s.
        The gt layer is a sibling rrd of the same recording id written by this
        same call (PR 3), which is why the encode and the base write are split
        out of ``convert`` rather than inlined into it.

        Args:
            identity: Sequence identity; names the recording and the rrd file.
            source: The discovered sequence, its VRS already on disk.
            work_dir: Scratch directory for the temp mp4s, beside the VRS.
            target: Final base-layer rrd path, written through ``atomic_recording``.

        Returns:
            What the sequence turned out to hold, for the caller's summary line.
        """
        streams: SequenceStreams = open_streams(source.vrs_path)
        vrs_bytes: int = source.vrs_path.stat().st_size
        clips: list[Path] = []
        for index, stream in enumerate(streams.cameras):
            clip: Path = work_dir / f"cam_{index:02d}.mp4"
            encode_frames_to_mp4(stream.frames, clip, source=stream.frame_source, fps=stream.fps)
            clips.append(clip)

        # The control points are the gt layer's material; the base layer reads them
        # only to say how many a consumer should expect, and PR 3 reuses the read.
        control_point_count: int = 0 if source.control_points_path is None else len(aria.read_control_points(source.control_points_path).points)
        clocks: list[Int64[ndarray, "n_samples"]] = [stream.times_ns for stream in streams.cameras if stream.times_ns.size]
        clocks += [imu.gyro.times_ns for imu in streams.imus if imu.gyro.times_ns.size]
        if not clocks:
            raise ValueError(f"{source.vrs_path} carries no timestamped camera or IMU stream")
        start_time_ns: int = min(int(times_ns[0]) for times_ns in clocks)
        end_time_ns: int = max(int(times_ns[-1]) for times_ns in clocks)

        num_frames: int = 0
        with writing.atomic_recording(
            target, application_id="dataforge", recording_id=identity.recording_id, default_blueprint=build_blueprint()
        ) as recording:
            # Deliberately NO ViewCoordinates at "/" and no transform on the rig node:
            # the gt layer establishes the world frame, so it owns both (a static
            # transform here would permanently shadow its temporal world_T_rig).
            log_rig_node(recording, RIG, reference=RIG_REFERENCE, num_cameras=len(streams.cameras), name="aria", kind="ego")
            for index, (stream, clip) in enumerate(zip(streams.cameras, clips, strict=True)):
                rr.log(
                    schema.cam_path(RIG, index),
                    rr.AnyValues(name=aria.STREAM_LABELS[stream.stream_id], kind=CAMERA_KINDS[stream.stream_id]),
                    static=True,
                    recording=recording,
                )
                log_pinhole(
                    stream.camera,
                    cam_log_path=Path(schema.cam_path(RIG, index)),
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                    static=True,
                    recording=recording,
                )
                # times_ns, not shift_ns: the mp4's own PTS is a nominal-rate fiction,
                # and these are Aria's device timestamps, logged unshifted.
                samples: int = log_video_stream(recording, clip, schema.video_path(RIG, index), times_ns=stream.times_ns)
                if stream.stream_id == aria.SLAM_LEFT_STREAM_ID:
                    num_frames = samples
            for index, imu in enumerate(streams.imus):
                # imu-right *is* the rig frame, so its pose is the identity by
                # construction; log_imu's default says so exactly, without the
                # 1e-17 residue of inverting and re-multiplying one transform.
                reference_imu: bool = imu.stream_id == aria.IMU_RIGHT_STREAM_ID
                log_imu(
                    recording,
                    RIG,
                    index,
                    gyro=imu.gyro,
                    accel=imu.accel,
                    name=aria.STREAM_LABELS[imu.stream_id],
                    rig_T_imu=None if reference_imu else rr.Transform3D(translation=imu.rig_T_imu[:3, 3], mat3x3=imu.rig_T_imu[:3, :3]),
                )
            writing.send_capture_properties(
                recording,
                identity,
                num_cameras=len(streams.cameras),
                num_frames=num_frames,
                split=source.split,
                set=sequence_set(source.sequence),
                challenge=sequence_challenge(source.sequence),
                control_point_count=control_point_count,
                has_pseudo_gt=source.pseudo_gt_path is not None,
                start_time_ns=start_time_ns,
                duration_s=(end_time_ns - start_time_ns) / 1e9,
                vrs_bytes=vrs_bytes,
            )
        return RecordingSummary(
            num_frames=num_frames, duration_s=(end_time_ns - start_time_ns) / 1e9, control_point_count=control_point_count
        )

    def default_blueprint(self) -> rrb.Blueprint:
        """Corpus-wide layout: every Aria Gen1 sequence carries the same three cameras."""
        return build_blueprint()

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card for the dataset's segment table."""
        return build_table_blueprint()
