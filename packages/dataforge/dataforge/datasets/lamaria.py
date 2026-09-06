"""LaMAria: ETH CVG's Aria Gen1 SLAM benchmark → a base and a gt exoego:v2 rrd per sequence.

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

**Clocks.** Every VRS timestamp is Aria DEVICE time in nanoseconds, and the
published pGT and control points are stamped on that same clock. ``video_time``
is therefore that clock **unshifted**, so a ground-truth pose lines up with its
frame 1:1 with no retiming anywhere.

**Frames.** The rig frame is imu-right, which is what LaMAria's published
calibration uses as its body frame, so the rig node states
``reference = "imu_00"``. The base layer logs NO transform on the rig node and
NO root ``ViewCoordinates``: the gt layer establishes the world frame and owns
both.

The verbs, the entity layout, the two world frames and the control-point checks
are documented in ``packages/dataforge/README.md``.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import serde
import serde.json
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from projectaria_tools.core import data_provider
from scipy.spatial.transform import Rotation
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

RIG: int = 0
"""One Aria per sequence; the glasses are ``rig_00``."""
RIG_REFERENCE: str = "imu_00"
"""The rig frame *is* imu-right's, LaMAria's published body frame."""
GT_SOURCE: str = "gt"
"""``/world/runs/gt/`` — the run name the gt layer writes its trajectory under."""

CameraKind: TypeAlias = Literal["grayscale", "rgb"]
"""exoego:v2 content hint on a camera node: what a consumer will decode."""

GtWorld: TypeAlias = Literal["mps", "lv95"]
"""Which world frame a sequence's ground truth is expressed in."""

MPS_WORLD_MAX_INDEX: int = 10
"""Highest ``R_NN`` index posed in the MPS frame; ``R_11`` onwards is surveyed in LV95/LN02."""

WorldUpAxis: TypeAlias = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
"""Signed axis of a world frame that gravity points *away* from."""

WORLD_UP: WorldUpAxis = "+z"
"""The up axis every LaMAria world is documented to have, and what the gt layer
states as ``rr.ViewCoordinates.RIGHT_HAND_Z_UP``.

Both frames are Z-up by construction — MPS's is gravity-aligned, and LV95/LN02 is a
projected national grid with levelled heights — so this is a *published* axis rather
than a guess. ``convert`` still measures it from gravity on every sequence
(``measured_world_up``) and warns rather than reorienting one rrd on its own."""

WORLD_UP_MIN_FRACTION_OF_G: float = 0.8
"""How much of |g| the measured axis must carry before ``convert`` accepts the measurement
as unambiguous; below it the averaging window was not near rest and the warning fires.

The five default sequences measure 0.92 to 1.01 g, so this leaves a visible margin
for a wearer who is already moving at the start rather than tracking the corpus."""

STANDARD_GRAVITY_MS2: float = 9.80665
"""Standard gravity; ``measured_world_up`` reports its result as a fraction of this."""
MEASURED_UP_WINDOW_NS: int = 2_000_000_000
"""How much of a sequence's start ``measured_world_up`` averages over: a wearer has
usually not started walking yet, so the mean there is nearly pure gravity."""

GT_TRAJECTORY_COLOR: tuple[int, int, int] = (110, 180, 255)
"""Fixed tint of the whole gt path; one trajectory is one quantity, not a per-row class."""
GT_TRAJECTORY_WIDTH_UI_POINTS: float = 1.5
"""Line width of the gt path, in **screen** points rather than metres.

A LaMAria walk is 170 m to 1.1 km long, so the World view frames hundreds of
metres at once and any honest metric width is a small fraction of a pixel there
(0.01 m over a 400 m shot is 0.06 px, i.e. invisible). Rerun reads a negative
radius as UI points, which keeps the overview path one visible hairline at every
zoom. The trail below stays metric on purpose: it rides the wearer in the Follow
view, where real centimetres are the point."""
GT_TRAIL_COLOR: tuple[int, int, int] = (255, 215, 90)
"""Fixed tint of the recent-motion trail; warm, so it reads against the cool full path."""
GT_TRAIL_RADIUS_M: float = 0.02
"""Point radius of the trail, in metres, at the pGT's 20 Hz."""

CONTROL_POINT_COLOR: tuple[int, int, int] = (120, 255, 160)
"""Tint of a fully surveyed (levelled) control point."""
CONTROL_POINT_UNLEVELLED_COLOR: tuple[int, int, int] = (255, 130, 90)
"""Tint of a control point the survey never levelled; its height is a placeholder, not a measurement."""
CONTROL_POINT_RADIUS_FLOOR_M: float = 0.1
"""Smallest radius a control point is drawn with. The survey uncertainties are
centimetres, which is invisible against a 1 km walk, so the sphere is a marker
whose radius only *grows* with a genuinely uncertain point."""
UNLEVELLED_LABEL_SUFFIX: str = " (no height)"
"""What an unlevelled point's label says, so a viewer never reads its ``z`` as data."""
CONTROL_POINT_MAX_DISTANCE_M: float = 50.0
"""How far a levelled control point may sit from the rig trajectory before ``convert``
refuses the sequence: its tag was photographed by these cameras, so a bigger gap means
the world frame or the origin translation is wrong, not that the walk was long.

This aborts where a low measured world-up fraction only warns, and the asymmetry is
deliberate: the up axis is declared once for the whole corpus and a weak measurement
says the averaging window was moving, while a distant point is per-sequence evidence
that these coordinates do not belong to this walk."""
CP_UV_RADIUS_PX: float = 4.0
"""Marker radius of a control-point detection, in pixels of the native 640x480
SLAM image; it is drawn in ``CONTROL_POINT_COLOR``, the same green as the 3D point."""

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
"""How far ahead of the wearer the eye aims, so the shot leads the motion.

The three distances were tuned together on R_01_easy, for a shot that holds the
three camera frusta and the last ten seconds of trail in view at once without the
ground filling it."""
FOLLOW_TRAIL_SECONDS: float = -10.0
"""Cursor-relative start of the gt trail's visible window in the Follow view."""
IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres; the SLAM baseline is ~11 cm, so the three frusta stay legible."""


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """What a converter has to know about one Aria camera stream beyond its calibration."""

    fps: int
    """Nominal container frame rate. Only the mp4 header cares:
    ``log_video_stream(times_ns=...)`` stamps every sample with its real capture time."""
    frame_kind: FrameKind
    """How its decoded frames reach the encoder: single-plane gray for the SLAM
    pair, interleaved RGB (not BGR) for camera-rgb."""
    kind: CameraKind
    """exoego:v2 content hint on the camera node."""


CAMERA_SPECS: dict[aria.AriaStreamId, CameraSpec] = {
    aria.SLAM_LEFT_STREAM_ID: CameraSpec(fps=20, frame_kind="gray8", kind="grayscale"),
    aria.SLAM_RIGHT_STREAM_ID: CameraSpec(fps=20, frame_kind="gray8", kind="grayscale"),
    aria.RGB_STREAM_ID: CameraSpec(fps=10, frame_kind="rgb24", kind="rgb"),
}
"""One entry per camera stream a LaMAria VRS carries."""


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
    """``IndexEntry.display_bytes`` of the listed VRS: a budget line, not a size."""
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
    """The manifest's display size, for the fetch's progress line only."""
    vrs_path: Path
    """``<root>/<split>/<seq>/raw_data/<seq>.vrs``; deleted after convert unless ``--keep-raw``."""
    calibration_path: Path
    """``.../aria_calibrations/<seq>.json``, the published calibration; always kept."""
    pseudo_gt_path: Path | None
    """``.../ground_truth/pGT/<seq>.txt``, or ``None`` when the archive has none."""
    control_points_path: Path | None
    """``.../ground_truth/control_points/<seq>.json``, or ``None`` when the sequence was not surveyed."""

    @property
    def has_ground_truth(self) -> bool:
        """Whether the archive publishes anything a gt layer could be built from."""
        return self.pseudo_gt_path is not None or self.control_points_path is not None


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
    frames: Iterator[bytes | memoryview]
    """Raw planes in presentation order, one per frame, native orientation. A
    decoded frame is handed over as its own buffer, not as a copy: a 2.4-minute
    sequence is 10 GB of memcpy otherwise."""
    times_ns: Int64[ndarray, "n_frames"]
    """Capture times on Aria's device clock, one per frame, in the same order."""


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
class GtTrajectory:
    """A sequence's pGT as the rig moves through it, in the columns ``rr.Transform3D`` takes."""

    times_ns: Int64[ndarray, "n_poses"]
    """Pose times on Aria's device clock — the pGT's own stamps, unshifted."""
    translations_xyz: Float64[ndarray, "n_poses 3"]
    """``world_t_rig``, metres."""
    quaternions_xyzw: Float64[ndarray, "n_poses 4"]
    """``world_R_rig``, scalar **last**, which is the order ``Transform3D`` wants."""
    length_m: float
    """Path length walked, summed over consecutive poses; 0.0 for fewer than two."""
    duration_s: float
    """Span from the first pose to the last; 0.0 for fewer than two."""


@dataclass(frozen=True, slots=True)
class WorldUp:
    """A measurement of which world axis is up, and how much of gravity landed on it.

    A dataclass and not a ``NamedTuple``: beartype resolves a NamedTuple's field
    annotations as forward references, which a ``Literal`` alias like
    ``WorldUpAxis`` is not.
    """

    axis: WorldUpAxis
    """The dominant signed world axis of the mean upward acceleration."""
    fraction_of_g: float
    """The mean's component along that axis as a fraction of standard gravity. A health
    check, not a calibration: near 1 means the window really was near rest and the axis
    is unambiguous, much less means the mean is not gravity."""


@dataclass(frozen=True, slots=True)
class RecordingSummary:
    """What one converted sequence turned out to hold; ``convert`` prints it."""

    num_frames: int
    """Video samples written for camera-slam-left, the pGT's own camera."""
    duration_s: float
    """Span of every logged stream, in seconds."""
    control_point_count: int = 0
    """Surveyed control points the sequence ships; 0 when it was never surveyed."""
    num_poses: int = 0
    """Ground-truth poses written into the gt layer; 0 when the sequence ships no pGT."""
    trajectory_len_m: float = 0.0
    """Path length of those poses, in metres."""
    num_detections: int = 0
    """Control-point detections written under the camera pinholes."""
    world_up: WorldUp | None = None
    """This sequence's own gravity measurement, or ``None`` when it has no poses to rotate with."""


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
    sequences: tuple[str, ...] | None = None
    """Sequences to select. Unset means ``DEFAULT_SEQUENCES`` at ``download`` and
    every downloaded sequence at ``convert``; a name the archive does not list is
    an error at ``download``."""
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


def gt_world(sequence: str) -> GtWorld:
    """Which world frame a sequence's published ground truth lives in.

    Upstream posed the first ten controlled sequences with MPS, whose world is
    Aria's own gravity-aligned frame, and surveyed everything from ``R_11``
    onwards — the control-point sequences and the whole additional set — in
    Switzerland's LV95/LN02 grid. Both are Z-up; they differ by hundreds of
    kilometres of easting, which ``aria.CUSTOM_ORIGIN_XYZ`` takes back out.

    Args:
        sequence: Upstream sequence name.

    Returns:
        ``mps`` for ``R_01``…``R_10``, ``lv95`` for everything else — including a
        controlled name whose second token is not a number, which the archive
        does not publish and which would only mislabel one property string.
    """
    index: str = sequence.split("_")[1] if sequence_set(sequence) == "controlled" else ""
    return "mps" if index.isdigit() and int(index) <= MPS_WORLD_MAX_INDEX else "lv95"


def rig_trajectory(pseudo_gt: aria.PseudoGt, *, rig_T_cam0: Float64[ndarray, "4 4"]) -> GtTrajectory:
    """Move the published camera trajectory onto the rig node, as Rerun columns.

    The pGT poses camera-slam-left (``world_T_cam0``) while the schema animates the
    *rig*, which is imu-right, so every pose is composed with ``cam0_T_rig``:
    ``world_T_rig = world_T_cam0 @ inv(rig_T_cam0)``. Rows are kept exactly as
    published, repeats and all — the corpus repeats a pose verbatim while the
    wearer stands still, and dropping those would change how a viewer draws a stop.

    Args:
        pseudo_gt: The sequence's published trajectory.
        rig_T_cam0: camera-slam-left's pose in the rig frame, i.e. what
            ``Fisheye62Parameters.extrinsics.world_T_cam`` holds for ``cam_00``
            (its "world" *is* the rig) and what the published calibration calls
            ``cam0.T_b_s``.

    Returns:
        The rig's pose per published stamp, plus the path length and span.
    """
    cam0_T_rig: Float64[ndarray, "4 4"] = np.linalg.inv(rig_T_cam0)
    world_T_rig: Float64[ndarray, "n_poses 4 4"] = pseudo_gt.world_T_cam0 @ cam0_T_rig
    translations_xyz: Float64[ndarray, "n_poses 3"] = np.ascontiguousarray(world_T_rig[:, :3, 3])
    quaternions_xyzw: Float64[ndarray, "n_poses 4"] = np.asarray(
        Rotation.from_matrix(world_T_rig[:, :3, :3]).as_quat(), dtype=np.float64
    ).reshape(-1, 4)
    steps_m: Float64[ndarray, "n_steps"] = np.linalg.norm(np.diff(translations_xyz, axis=0), axis=1)
    return GtTrajectory(
        times_ns=pseudo_gt.times_ns,
        translations_xyz=translations_xyz,
        quaternions_xyzw=quaternions_xyzw,
        length_m=float(steps_m.sum()),
        duration_s=float((pseudo_gt.times_ns[-1] - pseudo_gt.times_ns[0]) / 1e9) if pseudo_gt.times_ns.size else 0.0,
    )


def measured_world_up(gt: GtTrajectory, accel: ImuChannel, *, window_ns: int = MEASURED_UP_WINDOW_NS) -> WorldUp:
    """Measure which world axis is up, from gravity as imu-right reads it.

    LaMAria states its worlds are Z-up, and this is what checks the claim instead
    of trusting it. An accelerometer at rest measures the *reaction* to gravity,
    so its reading points **up**; imu-right *is* the rig frame, so rotating each
    sample into the world with the ground truth's own orientation
    (``world_R_rig @ a_rig``) and averaging yields a vector along the world's up
    axis. Only the first couple of seconds count: a wearer has usually not
    started walking yet, and the mean gets noisier the longer the window.

    Args:
        gt: The sequence's rig trajectory, whose rotations do the rotating.
        accel: imu-right's accelerometer channel in m/s^2, on the same device clock.
        window_ns: Length of the averaging window, from the first time both streams cover.

    Returns:
        The dominant signed world axis and the fraction of |g| it carried.

    Raises:
        ValueError: Either stream is empty, or they do not overlap inside the window.
    """
    if gt.times_ns.size == 0 or accel.times_ns.size == 0:
        raise ValueError("measuring the world up axis needs both a gt pose and an accelerometer sample")
    start_ns: int = max(int(gt.times_ns[0]), int(accel.times_ns[0]))
    inside: Bool[ndarray, "n_samples"] = (accel.times_ns >= start_ns) & (accel.times_ns < start_ns + window_ns)
    if not inside.any():
        raise ValueError(f"no accelerometer sample within {window_ns / 1e9:g} s of {start_ns}, where the ground truth starts")

    window_times_ns: Int64[ndarray, "n_window"] = accel.times_ns[inside]
    after: Int64[ndarray, "n_window"] = np.clip(np.searchsorted(gt.times_ns, window_times_ns), 0, gt.times_ns.size - 1)
    before: Int64[ndarray, "n_window"] = np.clip(after - 1, 0, gt.times_ns.size - 1)
    nearest: Int64[ndarray, "n_window"] = np.where(
        np.abs(gt.times_ns[before] - window_times_ns) <= np.abs(gt.times_ns[after] - window_times_ns), before, after
    )
    world_accel_xyz: Float64[ndarray, "n_window 3"] = Rotation.from_quat(gt.quaternions_xyzw[nearest]).apply(accel.values_xyz[inside])
    mean_xyz: Float64[ndarray, "3"] = world_accel_xyz.mean(axis=0)

    axis_index: int = int(np.argmax(np.abs(mean_xyz)))
    # Axis names by column index, signed by the mean's own direction.
    names: tuple[WorldUpAxis, ...] = ("+x", "+y", "+z") if mean_xyz[axis_index] >= 0.0 else ("-x", "-y", "-z")
    return WorldUp(axis=names[axis_index], fraction_of_g=float(abs(mean_xyz[axis_index]) / STANDARD_GRAVITY_MS2))


def validate_ground_truth(sequence: str, points: tuple[aria.ControlPoint, ...], trajectory: GtTrajectory) -> None:
    """Report every surveyed point's closest approach to the walk, and refuse a distant one.

    Every point's tag was photographed by these cameras, so a levelled point has
    to sit within a few metres of where the wearer walked: that distance is the
    one check that catches a wrong world frame or a missing origin translation,
    which no amount of self-consistent maths would. It runs as a preflight, so a
    wrong frame costs one VRS read rather than a published pair of layers. No
    pose is nothing to measure against, so a control-point-only sequence passes
    as it stands rather than being rejected.

    Args:
        sequence: Upstream sequence name, for the error message.
        points: The sequence's surveyed points, already translated by ``CUSTOM_ORIGIN_XYZ``.
        trajectory: The rig's pose per published pGT stamp; may be empty.

    Raises:
        ValueError: A levelled control point sits further than
            ``CONTROL_POINT_MAX_DISTANCE_M`` from the trajectory.
    """
    if not trajectory.times_ns.size:
        return
    too_far: list[str] = []
    for point in points:
        offsets_xyz: Float64[ndarray, "n_poses 3"] = trajectory.translations_xyz - point.position_xyz_m
        # An unlevelled point's z is the origin's height rather than a measurement,
        # so only its horizontal offset means anything.
        measured_xyz: Float64[ndarray, "n_poses n_axes"] = offsets_xyz if point.has_height else offsets_xyz[:, :2]
        distance_m: float = float(np.linalg.norm(measured_xyz, axis=1).min())
        measured_from: str = "in 3D" if point.has_height else f"horizontally{UNLEVELLED_LABEL_SUFFIX}"
        print(f"  control point {point.name:<10} came within {distance_m:8.2f} m of the trajectory, {measured_from}")
        if point.has_height and distance_m > CONTROL_POINT_MAX_DISTANCE_M:
            too_far.append(f"{point.name} at {distance_m:.1f} m")
    if too_far:
        raise ValueError(
            f"{sequence}: levelled control point(s) {', '.join(too_far)} sit further than {CONTROL_POINT_MAX_DISTANCE_M:g} m "
            f"from the rig trajectory, but their tags were photographed by these cameras — the world frame or the "
            f"origin translation is wrong"
        )


def log_control_points(recording: rr.RecordingStream, points: tuple[aria.ControlPoint, ...]) -> None:
    """Log the surveyed points as one static, labelled ``Points3D`` in the world.

    A survey is a property of the world and not of a moment, so this is
    static. An unlevelled point is drawn in its own colour and says so in its
    label, because its ``z`` is the origin's height rather than a measurement,
    and its ``NaN`` height uncertainty never reaches Rerun: the radius is the
    largest uncertainty the survey *did* publish, floored so a centimetre-level
    point stays visible against a kilometre-long walk.

    Args:
        recording: Destination recording stream.
        points: The sequence's surveyed points, in file order.
    """
    radii_m: list[float] = []
    for point in points:
        published_uncertainty_m: Float64[ndarray, "n_axes"] = point.uncertainty_xyz_m[np.isfinite(point.uncertainty_xyz_m)]
        radii_m.append(max(CONTROL_POINT_RADIUS_FLOOR_M, float(published_uncertainty_m.max()) if published_uncertainty_m.size else 0.0))
    rr.log(
        schema.control_points_path(),
        rr.Points3D(
            positions=np.stack([point.position_xyz_m for point in points]),
            colors=[CONTROL_POINT_COLOR if point.has_height else CONTROL_POINT_UNLEVELLED_COLOR for point in points],
            radii=radii_m,
            labels=[f"{point.name}{'' if point.has_height else UNLEVELLED_LABEL_SUFFIX}" for point in points],
            # Rerun hides labels past a handful of points on its own; a survey is
            # exactly the case where every name is worth reading.
            show_labels=True,
        ),
        static=True,
        recording=recording,
    )


def log_control_point_detections(
    recording: rr.RecordingStream, detections: tuple[aria.ControlPointDetection, ...], *, labels_by_name: dict[str, str]
) -> None:
    """Log each camera's control-point detections under its own pinhole, columnar.

    One ``send_columns`` per camera at the detections' own device timestamps, so
    a detection lands on the frame it was made in. Upstream runs its tag
    detector on the SLAM pair only, so camera-rgb usually gets nothing — and
    then nothing is written under it, rather than an empty column.

    Args:
        recording: Destination recording stream.
        detections: Every detection of the sequence, sorted by stream then time.
        labels_by_name: Survey name → the label to draw, unlevelled suffix included.

    Raises:
        ValueError: A detection names a control point the survey never published.
    """
    unknown: set[str] = {detection.control_point for detection in detections} - set(labels_by_name)
    if unknown:
        raise ValueError(f"control point detection(s) name {', '.join(sorted(unknown))}, which the survey does not publish")
    for index, stream_id in enumerate(aria.CAMERA_STREAM_IDS):
        seen: list[aria.ControlPointDetection] = [detection for detection in detections if detection.stream_id == stream_id]
        if not seen:
            continue
        times_ns: Int64[ndarray, "n_detections"] = np.array([detection.timestamp_ns for detection in seen], dtype=np.int64)
        rr.log(
            schema.cp_uv_path(RIG, index),
            rr.Points2D.from_fields(colors=CONTROL_POINT_COLOR, radii=CP_UV_RADIUS_PX, show_labels=True),
            static=True,
            recording=recording,
        )
        rr.send_columns(
            schema.cp_uv_path(RIG, index),
            indexes=[rr.TimeColumn(schema.TIMELINE, duration=times_ns.astype("timedelta64[ns]"))],
            columns=rr.Points2D.columns(
                positions=np.stack([detection.uv_px for detection in seen]),
                labels=[labels_by_name[detection.control_point] for detection in seen],
            ),
            recording=recording,
        )


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
                # ``image.data`` and not ``tobytes()``: ffmpeg's stdin takes the
                # decoder's own buffer, so nothing is copied on the way there.
                frames=(image.data for _, image in aria.iter_frames(provider, stream_id)),
                times_ns=aria.frame_timestamps_ns(provider, stream_id),
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
        """Pair every manifest sequence whose ground truth is on disk with its source.

        The manifest is what ``download`` resolved from the archive, so it is the
        selection unless ``--sequences`` narrows it. What proves ``download``
        finished for a sequence is its ground truth on disk, which is also all
        ``convert`` reads from the tree: the VRS is fetched on demand and the
        published calibration is only ever a cross-check. A sequence missing
        those files is announced and skipped rather than fatal, since a name the
        archive really does not have is already a hard error at ``download``.
        """
        selected: set[str] | None = None if self.config.sequences is None else set(self.config.sequences)
        pairs: list[tuple[SequenceIdentity, LamariaSource]] = []
        for record in sorted(self.manifest().sequences, key=lambda listed: listed.sequence):
            if selected is not None and record.sequence not in selected:
                continue
            source: LamariaSource = self.source(record)
            missing: list[Path] = [path for path in (source.pseudo_gt_path, source.control_points_path) if path is not None and not path.is_file()]
            if missing:
                print(f"  warning: skipping {record.sequence}: {missing[0]} is not on disk; re-run dataforge-download")
                continue
            pairs.append((SequenceIdentity(dataset=self.config.name, parts=(record.sequence,)), source))
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
        selected: list[str] = sorted(set(DEFAULT_SEQUENCES if self.config.sequences is None else self.config.sequences))
        splits: dict[str, LamariaSplit] = {}
        vrs_display_bytes: dict[str, int] = {}
        for split in SPLITS:
            for name, display_bytes in transports.http_index(f"{self.archive_url(RAW_DATA_DIR, split)}/"):
                sequence: str = name.removesuffix(".vrs")
                if name.endswith(".vrs") and sequence in selected:
                    splits[sequence] = split
                    vrs_display_bytes[sequence] = display_bytes
        unlisted: list[str] = [name for name in selected if name not in splits]
        if unlisted:
            raise ValueError(f"no raw_data index at {self.config.base_url} lists a VRS for {', '.join(unlisted)}")

        calibrated: set[str] = {
            name.removesuffix(".json")
            for split in sorted(set(splits.values()))
            for name, _ in transports.http_index(f"{self.archive_url(CALIBRATION_DIR, split)}/")
            if name.endswith(".json")
        }
        uncalibrated: list[str] = [name for name in selected if name not in calibrated]
        if uncalibrated:
            raise ValueError(f"{CALIBRATION_DIR} lists no published calibration for {', '.join(uncalibrated)}; the archive layout changed")
        with_pseudo_gt: set[str] = {
            name.removesuffix(".txt") for name, _ in transports.http_index(f"{self.archive_url(PSEUDO_GT_REMOTE_DIR)}/") if name.endswith(".txt")
        }
        with_control_points: set[str] = {
            name.removesuffix(".json")
            for name, _ in transports.http_index(f"{self.archive_url(CONTROL_POINTS_REMOTE_DIR)}/")
            if name.endswith(".json")
        }

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

    def convert(self, identity: SequenceIdentity, source: LamariaSource, *, force: bool) -> Path:
        """Fetch one VRS, encode it, write both rrd layers, and delete the raw.

        The base and the gt layer come out of the *same* VRS read, so they are
        skipped and rebuilt as a unit: half a sequence on disk means paying for
        the multi-gigabyte download again anyway, and rebuilding the base layer
        with it costs one encode against a corpus that would otherwise be
        silently incomplete. A sequence the archive publishes no ground truth for
        (the whole test split) has no gt layer to wait for, so it is done once
        its base rrd exists.

        The NVENC check comes before the fetch: a machine that cannot encode AV1
        should find out in a second rather than after a multi-gigabyte download.
        Failure keeps the VRS (a retry then resumes rather than refetching) but
        removes the temp mp4s, so the next attempt starts from a clean scratch
        directory. Nothing outside the sequence's own directory is touched.
        """
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        gt_target: Path = paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)
        if writing.should_skip(target, force=force) and (not source.has_ground_truth or writing.should_skip(gt_target, force=force)):
            print(f"skip {identity.sequence_key} → {target}{f' + {gt_target}' if source.has_ground_truth else ''}")
            return target

        require_av1_nvenc(resolve_ffmpeg())
        if source.vrs_path.is_file():
            print(f"  {source.vrs_path.stat().st_size / 1e9:.2f} GB already in {source.vrs_path.parent}; the fetch resumes or verifies it")
        else:
            print(f"  fetching {source.vrs_display_bytes / 1e9:.1f} GB from {source.vrs_url}")
        transports.http_fetch(source.vrs_url, dest=source.vrs_path)
        work_dir: Path = source.vrs_path.parent / f"{source.sequence}-mp4"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary: RecordingSummary = self.write_recording(identity, source, work_dir=work_dir, target=target, gt_target=gt_target)
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            retained: int = source.vrs_path.stat().st_size if source.vrs_path.is_file() else 0
            print(f"  kept {retained / 1e9:.2f} GB of VRS in {source.vrs_path.parent} so a retry resumes instead of refetching")
            raise

        if self.config.keep_raw:
            print(f"  keeping raw: {source.vrs_path.name} and the mp4s in {work_dir}")
        else:
            # A removal that fails on the success path is a real problem (a full
            # disk, a stale mount): only the failure path above swallows it.
            shutil.rmtree(work_dir)
            source.vrs_path.unlink(missing_ok=True)
        measured: str = "" if summary.world_up is None else f", world up {summary.world_up.axis} at {summary.world_up.fraction_of_g:.2f} g"
        print(
            f"done {identity.sequence_key} → {target}{f' + {gt_target}' if source.has_ground_truth else ''} "
            f"({len(aria.CAMERA_STREAM_IDS)} cameras, {summary.num_frames} frames, {summary.duration_s:.1f} s, "
            f"{summary.num_poses} gt poses over {summary.trajectory_len_m:.1f} m, "
            f"{summary.control_point_count} control points, {summary.num_detections} detections{measured})"
        )
        return target

    def write_recording(
        self, identity: SequenceIdentity, source: LamariaSource, *, work_dir: Path, target: Path, gt_target: Path
    ) -> RecordingSummary:
        """Encode every camera stream and write both rrd layers of one sequence.

        The ground truth is read and checked first, so a wrong world frame fails
        the sequence before a single frame is encoded. Every camera is then
        encoded before the first recording opens: the VRS is read once, stream by
        stream, and ``log_video_stream`` remuxes the finished mp4s. The base layer
        closes before the gt layer opens — one VRS read feeds both, but each is
        its own recording stream and its own atomic replace, so a failure in the
        second cannot half-write the first.

        Args:
            identity: Sequence identity; names both recordings and both rrd files.
            source: The discovered sequence, its VRS already on disk.
            work_dir: Scratch directory for the temp mp4s, beside the VRS.
            target: Final base-layer rrd path, written through ``atomic_recording``.
            gt_target: Final gt-layer rrd path, likewise; untouched when the
                archive publishes no ground truth for the sequence.

        Returns:
            What the sequence turned out to hold, for the caller's summary line.
        """
        streams: SequenceStreams = open_streams(source.vrs_path)
        vrs_bytes: int = source.vrs_path.stat().st_size

        # The control points are the gt layer's material, read once here: the base
        # layer only says how many a consumer should expect.
        control_points: aria.ControlPointSet | None = (
            None if source.control_points_path is None else aria.read_control_points(source.control_points_path)
        )
        control_point_count: int = 0 if control_points is None else len(control_points.points)
        trajectory: GtTrajectory | None = None
        if source.has_ground_truth:
            published: aria.PseudoGt = (
                aria.read_pseudo_gt(source.pseudo_gt_path)
                if source.pseudo_gt_path is not None
                else aria.PseudoGt(times_ns=np.zeros(0, dtype=np.int64), world_T_cam0=np.zeros((0, 4, 4), dtype=np.float64))
            )
            # The pGT poses camera-slam-left, so cam_00's own extrinsics are what
            # move the trajectory onto the rig.
            rig_T_cam0: Float64[ndarray, "4 4"] = np.asarray(streams.cameras[0].camera.extrinsics.world_T_cam, dtype=np.float64)
            trajectory = rig_trajectory(published, rig_T_cam0=rig_T_cam0)
            validate_ground_truth(source.sequence, () if control_points is None else control_points.points, trajectory)

        clips: list[Path] = []
        for index, stream in enumerate(streams.cameras):
            clip: Path = work_dir / f"cam_{index:02d}.mp4"
            spec: CameraSpec = CAMERA_SPECS[stream.stream_id]
            layout: FrameSource = FrameSource(spec.frame_kind, width=stream.camera.intrinsics.width, height=stream.camera.intrinsics.height)
            encode_frames_to_mp4(stream.frames, clip, source=layout, fps=spec.fps)
            clips.append(clip)

        clocks: list[Int64[ndarray, "n_samples"]] = [stream.times_ns for stream in streams.cameras if stream.times_ns.size]
        clocks += [imu.gyro.times_ns for imu in streams.imus if imu.gyro.times_ns.size]
        if not clocks:
            raise ValueError(f"{source.vrs_path} carries no timestamped camera or IMU stream")
        start_time_ns: int = min(int(times_ns[0]) for times_ns in clocks)
        end_time_ns: int = max(int(times_ns[-1]) for times_ns in clocks)
        duration_s: float = (end_time_ns - start_time_ns) / 1e9

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
                    rr.AnyValues(name=aria.STREAM_LABELS[stream.stream_id], kind=CAMERA_SPECS[stream.stream_id].kind),
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
                duration_s=duration_s,
                vrs_bytes=vrs_bytes,
            )

        if trajectory is None:
            print(f"  no ground truth published for {source.sequence}, so no gt layer: nothing establishes a world frame")
            return RecordingSummary(num_frames=num_frames, duration_s=duration_s)

        # imu-right's accelerometer is what measures the world up axis; imu_00 *is* the rig.
        world_up: WorldUp | None = self.write_gt_recording(
            identity, source, trajectory=trajectory, accel=streams.imus[0].accel, control_points=control_points, target=gt_target
        )
        return RecordingSummary(
            num_frames=num_frames,
            duration_s=duration_s,
            control_point_count=control_point_count,
            num_poses=int(trajectory.times_ns.size),
            trajectory_len_m=trajectory.length_m,
            num_detections=0 if control_points is None else len(control_points.detections),
            world_up=world_up,
        )

    def write_gt_recording(
        self,
        identity: SequenceIdentity,
        source: LamariaSource,
        *,
        trajectory: GtTrajectory,
        accel: ImuChannel,
        control_points: aria.ControlPointSet | None,
        target: Path,
    ) -> WorldUp | None:
        """Write one sequence's gt layer: the rig's pose, its path, and the surveyed points.

        This layer is what establishes a world frame at all, so it — not the base
        layer — owns the root ``ViewCoordinates`` and the transform on the rig
        node. The axis stated is the *published* one; this sequence's own gravity
        measurement is reported and warned about rather than used to reorient one
        rrd on its own.

        Args:
            identity: Sequence identity; names the recording and the rrd file.
            source: The discovered sequence, for its name and its world frame.
            trajectory: The rig's pose per published pGT stamp; may be empty when
                the sequence ships control points only.
            accel: imu-right's accelerometer channel, which measures the world up axis.
            control_points: The surveyed points and their detections, or ``None``.
            target: Final gt-layer rrd path, written through ``atomic_recording``.

        Returns:
            The measured world up axis, or ``None`` when there was no pose to rotate by.

        Raises:
            ValueError: A detection names a point the survey never published.
        """
        world_up: WorldUp | None = None
        if trajectory.times_ns.size:
            world_up = measured_world_up(trajectory, accel)
            if world_up.axis != WORLD_UP or world_up.fraction_of_g < WORLD_UP_MIN_FRACTION_OF_G:
                print(
                    f"  warning: lamaria declares world up {WORLD_UP} but {source.sequence} measured {world_up.axis} "
                    f"carrying {world_up.fraction_of_g:.2f} of |g|; the rrd still states the declared axis"
                )

        points: tuple[aria.ControlPoint, ...] = () if control_points is None else control_points.points
        labels_by_name: dict[str, str] = {
            point.name: f"{point.name}{'' if point.has_height else UNLEVELLED_LABEL_SUFFIX}" for point in points
        }
        gt_index: list[rr.TimeColumn] = [rr.TimeColumn(schema.TIMELINE, duration=trajectory.times_ns.astype("timedelta64[ns]"))]
        with writing.atomic_recording(target, application_id="dataforge", recording_id=identity.recording_id) as recording:
            # The right-handed axes WORLD_UP names; every LaMAria world is Z-up.
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
            if trajectory.times_ns.size:
                # from_parent stays unset: the stored value is world_T_rig, the
                # child-to-parent step, which is what every child frustum rides.
                rr.send_columns(
                    schema.rig_path(RIG),
                    indexes=gt_index,
                    columns=rr.Transform3D.columns(translation=trajectory.translations_xyz, quaternion=trajectory.quaternions_xyzw),
                    recording=recording,
                )
                # Two views of one trajectory: the static strip is the whole path for the
                # overview, and the per-pose points are what the blueprint's cursor-relative
                # time range turns into a recent-motion trail in the Follow view.
                rr.log(
                    schema.trajectory_path(GT_SOURCE),
                    rr.LineStrips3D(
                        [trajectory.translations_xyz],
                        colors=GT_TRAJECTORY_COLOR,
                        radii=rr.components.Radius.ui_points(GT_TRAJECTORY_WIDTH_UI_POINTS),
                    ),
                    static=True,
                    recording=recording,
                )
                rr.log(
                    schema.trail_path(GT_SOURCE),
                    rr.Points3D.from_fields(colors=GT_TRAIL_COLOR, radii=GT_TRAIL_RADIUS_M),
                    static=True,
                    recording=recording,
                )
                rr.send_columns(
                    schema.trail_path(GT_SOURCE),
                    indexes=gt_index,
                    columns=rr.Points3D.columns(positions=trajectory.translations_xyz),
                    recording=recording,
                )
            if points:
                log_control_points(recording, points)
            if control_points is not None:
                log_control_point_detections(recording, control_points.detections, labels_by_name=labels_by_name)
            recording.send_recording_name(identity.recording_id)
            # world_up_fraction_of_g belongs to the *measured* axis, which convert warns
            # about when it is not the declared one; a sequence with no pose has neither,
            # and AnyValues drops the untyped None rather than writing an empty value.
            recording.send_property(
                "gt",
                rr.AnyValues(
                    num_poses=int(trajectory.times_ns.size),
                    trajectory_len_m=trajectory.length_m,
                    duration_s=trajectory.duration_s,
                    world_up=WORLD_UP,
                    world_up_fraction_of_g=None if world_up is None else world_up.fraction_of_g,
                    control_point_count=len(points),
                    num_detections=0 if control_points is None else len(control_points.detections),
                    gt_world=gt_world(source.sequence),
                ),
            )
        return world_up

    def default_blueprint(self) -> rrb.Blueprint:
        """Corpus-wide layout: every Aria Gen1 sequence carries the same three cameras."""
        return build_blueprint()

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card for the dataset's segment table."""
        return build_table_blueprint()
