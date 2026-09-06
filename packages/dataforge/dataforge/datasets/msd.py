"""Monado SLAM Datasets (MSD): download-on-demand VR headset captures → a base and a gt rrd per sequence.

Upstream is the HuggingFace dataset ``collabora/monado-slam-datasets`` (CC-BY 4.0),
a tree of per-sequence zip archives::

    M_monado_datasets/<device_dir>/<collection>/<SEQ>.zip     (or .z01 .z02 … .zip)
    M_monado_datasets/<device_dir>/extras/calibration.json

Inside an archive: ``<SEQ>/mav0/cam<N>/data/<ts>.png`` grayscale frames plus a
``cam<N>/data.csv`` index, ``imu0/data.csv`` (~1 kHz), ``gt/data.csv`` (~1 kHz),
and on the Reverb G2 / Odyssey+ a 50 Hz ``mag0/data.csv``. Where a stream also
ships ``data.raw.csv`` / ``data.extra.csv`` siblings, the converter reads only
``data.csv``.

**Raw is never kept.** The corpus is ~350 GB of PNG sequences, so ``discover()``
enumerates the *remote* tree and ``convert()`` fetches exactly one sequence,
streams its PNGs through the AV1 encoder, and deletes the archive again. What
survives on disk is the rrd. ``raw_budget_gb`` caps the scratch directory so a
batch run cannot silently fill the NVMe with leftovers from failed sequences;
the Index and G2 long sessions (66 GB and 55 GB of split archives) are larger
than the cap and are processed anyway, with a warning, as accepted exceptions.

**One dataset per device.** A catalog dataset holds exactly one default
blueprint, hence one camera layout, and the three headsets have two, four and
two cameras. ``--device`` therefore picks both the corpus and the catalog
dataset name (``msd-index``, ``msd-g2``, ``msd-odyssey``).

**Two layers, one fetch.** ``convert`` writes a ``base`` rrd (video, IMU,
magnetometer) and a ``gt`` rrd (``world_T_rig`` at the full ~1 kHz ground-truth
rate, plus its path and trail) under the same recording id, so the catalog
stacks them onto one segment. Both come out of one archive read and are
therefore skipped and rebuilt together: half a sequence on disk means paying
for the download again regardless.

**Clocks.** Every csv timestamp is nanoseconds on one monotonic device clock
(values around 1e13, not a Unix epoch). ``video_time`` is that clock minus ``t0``,
the earliest sample of any stream *including* ``gt`` — the two layers must share
this origin, and the gt file is usually the earliest stream. Nothing is resampled.

**World axes.** MSD documents none: the Index's ground truth comes from SteamVR
Lighthouse and the G2's and Odyssey+'s from an undocumented MoCap rig. So the up
axis is *measured*, from gravity as the accelerometer reads it rotated into the
world by the ground truth (``measured_world_up``), and then fixed per device in
``MSD_DEVICES`` so every rrd of a device agrees. ``convert`` re-measures every
sequence and warns on a disagreement rather than silently reorienting one rrd.

**Frames.** ``rig_T_cam`` comes from basalt's ``T_imu_cam``, the camera pose in
the IMU frame; the rig frame *is* the IMU frame (``reference = "imu_00"``), so
``world`` in the ``Extrinsics`` below means the rig. This is the same convention
as simplecv's RoboCap loader, one inversion away (Kalibr states ``T_cam_imu``).
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import ClassVar, Literal, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import serde
import serde.json
from huggingface_hub import HfApi, RepoFile
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_log_utils import log_pinhole

from dataforge import paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    FrameSource,
    ImuChannel,
    encode_frames_to_mp4,
    log_imu,
    log_magnetometer,
    log_rig_node,
    log_video_stream,
    require_av1_nvenc,
    resolve_ffmpeg,
)

REPO_ID: str = "collabora/monado-slam-datasets"
"""HuggingFace dataset repo holding every MSD device."""
REPO_ROOT: str = "M_monado_datasets"
"""Top-level directory inside the repo; every device tree hangs off it."""
RIG: int = 0
"""MSD is one headset; the whole device is ``rig_00``."""
RIG_REFERENCE: str = "imu_00"
"""The rig frame *is* the IMU frame: every extrinsic in the calibration is a ``T_imu_cam``."""
IMU: int = 0
"""Every MSD sequence ships exactly one IMU, ``imu0`` → ``imu_00``."""
MAG: int = 0
"""The G2 and the Odyssey+ ship exactly one magnetometer, ``mag0`` → ``mag_00``."""
IMAGE_PLANE_DISTANCE: float = 0.1
"""Frustum length in metres; a headset's baseline is ~10 cm, so the frusta stay legible."""
PART_SUFFIX_RE: re.Pattern[str] = re.compile(r"^\.z\d+$")
"""Info-ZIP split-volume suffix (``.z01``, ``.z02``, …); the closing volume is the plain ``.zip``."""
CAMERA_TIMESTAMP_FIELD: str = "#timestamp [ns]"
"""Header cell every MSD csv starts with; the gt file pads the later cells with spaces."""
IMU_VALUE_COLUMNS: int = 6
"""``imu0/data.csv`` value columns: gyro xyz (rad/s) then accel xyz (m/s^2)."""
MAG_VALUE_COLUMNS: int = 3
"""``mag0/data.csv`` value columns: the field xyz, in the sensor's own units."""
GT_VALUE_COLUMNS: int = 7
"""``gt/data.csv`` value columns: position xyz (m) then the quaternion **wxyz**, scalar first."""
IDENTITY_QUATERNION_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
"""What a dropout row's rotation becomes, so its translation still applies."""
QUATERNION_NORM_TOLERANCE: float = 1e-3
"""How far a gt quaternion's norm may sit from 1 before the row counts as a dropout."""

MsdDeviceChoice: TypeAlias = Literal["index", "g2", "odyssey"]
"""``--device``: which headset's corpus to work on, and which catalog dataset."""
CameraModel: TypeAlias = Literal["kb4", "pinhole-radtan8"]
"""``camera_type`` values basalt writes in ``calibration.json``; MSD uses no others."""
WorldUpAxis: TypeAlias = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
"""Signed axis of a tracking world that gravity points *away* from."""
GtSource: TypeAlias = Literal["lighthouse", "mocap"]
"""What produced a device's ground truth: SteamVR Lighthouse, or a MoCap system."""

WORLD_UP_VIEW_COORDINATES: dict[WorldUpAxis, rr.components.ViewCoordinates] = {
    "+x": rr.ViewCoordinates.RIGHT_HAND_X_UP,
    "-x": rr.ViewCoordinates.RIGHT_HAND_X_DOWN,
    "+y": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
    "-y": rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
    "+z": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    "-z": rr.ViewCoordinates.RIGHT_HAND_Z_DOWN,
}
"""Root ``ViewCoordinates`` per world up axis, right-handed throughout.

Rerun's ``RIGHT_HAND_*`` aliases are exactly this table (``RIGHT_HAND_Y_UP`` is
``RUB``, ``RIGHT_HAND_Z_UP`` is ``RFU``), so naming the up axis is the whole
decision — the remaining two axes are then fixed by handedness. MSD's csvs say
nothing about world axes at all, hence the empirical ``measured_world_up``.
"""
POSITIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("+x", "+y", "+z")
"""Axis names by column index, for a positive mean; the negative row is below."""
NEGATIVE_WORLD_AXES: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = ("-x", "-y", "-z")
"""Axis names by column index, for a negative mean."""
STANDARD_GRAVITY_MS2: float = 9.80665
"""Standard gravity; ``measured_world_up`` reports its result as a fraction of this."""
MEASURED_UP_WINDOW_NS: int = 2_000_000_000
"""How much of a sequence's start ``measured_world_up`` averages over."""
GT_TRAJECTORY_COLOR: tuple[int, int, int] = (110, 180, 255)
"""Fixed tint of the whole gt path; one trajectory is one quantity, not a per-row class."""
GT_TRAJECTORY_RADIUS_M: float = 0.002
"""Line radius of the gt path, in metres — thin, because it overlays the rig itself."""
GT_TRAIL_COLOR: tuple[int, int, int] = (255, 215, 90)
"""Fixed tint of the recent-motion trail; warm, so it reads against the cool full path."""
GT_TRAIL_RADIUS_M: float = 0.004
"""Point radius of the trail, in metres; a 1 kHz trail is dense, so the dots stay small."""


@dataclass(frozen=True, slots=True)
class MsdDevice:
    """Everything that varies between the three headsets."""

    hf_dir: str
    """Device directory under ``M_monado_datasets`` (e.g. ``MI_valve_index``)."""
    collections: tuple[str, ...]
    """Sequence collections to enumerate, relative to ``hf_dir``; the
    ``*C_calibration`` collections are deliberately excluded (they hold
    calibration-target recordings, not trajectories)."""
    num_cameras: int
    """Cameras in the headset, logged as ``cam_00..cam_NN``."""
    has_magnetometer: bool
    """Whether the sequences ship a ``mag0/data.csv``."""
    label: str
    """Human device name for the rig node's ``name`` AnyValue."""
    world_up: WorldUpAxis
    """Up axis of this device's tracking world, **measured** with
    ``measured_world_up`` on a real sequence rather than assumed, and then fixed
    here so every rrd of the device carries the same root ``ViewCoordinates``.
    ``convert`` re-measures each sequence and warns when it disagrees."""
    gt_source: GtSource
    """What produced ``gt/data.csv``; goes into the gt layer's properties."""


MSD_DEVICES: dict[MsdDeviceChoice, MsdDevice] = {
    "index": MsdDevice(
        hf_dir="MI_valve_index",
        collections=(
            "MIO_others",
            "MIP_playing/MIPB_beat_saber",
            "MIP_playing/MIPP_pistol_whip",
            "MIP_playing/MIPT_thrill_of_the_fight",
        ),
        num_cameras=2,
        has_magnetometer=False,
        label="valve-index",
        # Measured on MIO09_short_1_updown: +y carries 1.00 of |g|. OpenVR worlds
        # are right-handed Y-up, so Lighthouse ground truth agreeing is expected.
        world_up="+y",
        gt_source="lighthouse",
    ),
    "g2": MsdDevice(
        hf_dir="MG_reverb_g2",
        collections=("MGO_others",),
        num_cameras=4,
        has_magnetometer=True,
        label="reverb-g2",
        # Measured on MGO09_short_1_updown; the MoCap rig documents no convention.
        world_up="+y",
        gt_source="mocap",
    ),
    "odyssey": MsdDevice(
        hf_dir="MO_odyssey_plus",
        collections=("MOO_others",),
        num_cameras=2,
        has_magnetometer=True,
        label="odyssey-plus",
        # Measured on MOO09_short_1_updown; same undocumented MoCap rig as the G2.
        world_up="+y",
        gt_source="mocap",
    ),
}
"""The three headsets MSD covers, keyed by the ``--device`` literal."""


@dataclass(frozen=True, slots=True)
class MsdSource:
    """One remote sequence as discovery found it; ``convert`` needs nothing else."""

    collection_path: str
    """Repo-relative directory holding the archive(s), e.g. ``M_monado_datasets/MI_valve_index/MIO_others``."""
    sequence: str
    """Archive stem, which is also the top-level directory inside it (e.g. ``MIO09_short_1_updown``)."""
    archive_paths: tuple[str, ...]
    """Repo-relative archive files, ``.z01``… first and the closing ``.zip`` last."""
    archive_bytes: int
    """Total download size of ``archive_paths``, from the repo listing."""


@dataclass
class MsdConfig(DataforgeDatasetConfig):
    """Monado SLAM Datasets: one VR headset's stereo/quad camera + IMU sequences."""

    command: ClassVar[str] = "msd"
    """Registry key and CLI subcommand; ``--device`` picks the catalog dataset."""

    _target: type = field(default_factory=lambda: MsdDataset)
    """Dataset class instantiated by ``setup()``."""
    device: MsdDeviceChoice = "index"
    """Headset to work on; also the catalog dataset (``msd-<device>``)."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "msd")
    """Scratch directory for archives, extractions and temp mp4s. Point it at
    local NVMe: every byte written here is deleted again after the sequence."""
    keep_raw: bool = False
    """Keep the fetched archive and the encoded mp4s instead of deleting them."""
    raw_budget_gb: float = 50.0
    """Cap on what ``root`` may hold. A sequence whose archives alone exceed it is
    processed anyway with a warning; leftovers that would breach it are an error."""
    revision: str | None = None
    """Repo branch, tag or commit; ``None`` takes the default branch."""

    @property
    def name(self) -> str:
        """Catalog dataset and identity ``dataset`` part: one per device layout."""
        return f"{self.command}-{self.device}"


def list_collection_files(repo_id: str, collection_path: str, revision: str | None = None) -> list[tuple[str, int]]:
    """List one remote collection directory as ``(repo-relative path, size in bytes)``.

    The whole HF listing surface of this dataset, isolated so tests can replace
    it with a literal tree.

    Args:
        repo_id: Hub dataset repo, normally ``REPO_ID``.
        collection_path: Repo-relative directory to list (not recursive).
        revision: Branch, tag or commit; ``None`` takes the default branch.

    Returns:
        One entry per *file* directly inside the directory; subdirectories are dropped.
    """
    entries = HfApi().list_repo_tree(repo_id, path_in_repo=collection_path, repo_type="dataset", revision=revision)
    return [(entry.path, entry.size) for entry in entries if isinstance(entry, RepoFile)]


def repo_revision(repo_id: str, revision: str | None = None) -> str | None:
    """Resolve a branch/tag to the commit sha stamped into every converted rrd.

    Isolated from ``list_collection_files`` because ``convert`` needs it without
    a listing, and a test needs it without a network.
    """
    return HfApi().repo_info(repo_id, repo_type="dataset", revision=revision).sha


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltPose:
    """One ``T_imu_cam`` entry: the camera's pose in the IMU frame, quaternion xyzw."""

    px: float
    """Translation x, metres."""
    py: float
    """Translation y, metres."""
    pz: float
    """Translation z, metres."""
    qx: float
    """Quaternion x."""
    qy: float
    """Quaternion y."""
    qz: float
    """Quaternion z."""
    qw: float
    """Quaternion w (basalt writes xyzw fields, scalar last)."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltIntrinsics:
    """Projection and distortion terms of one camera, in basalt's flat layout.

    Both models share the ``fx fy cx cy`` head. ``kb4`` fills ``k1..k4`` only;
    ``pinhole-radtan8`` fills all eight plus a ``rpmax`` validity radius that
    Rerun has no place for and that pyserde drops with every other unknown key.
    """

    fx: float
    """Focal length in x, pixels."""
    fy: float
    """Focal length in y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    k1: float = 0.0
    """First radial term (both models)."""
    k2: float = 0.0
    """Second radial term (both models)."""
    k3: float = 0.0
    """Third radial term: kb4's fourth-order coefficient, radtan8's third."""
    k4: float = 0.0
    """Fourth radial term."""
    k5: float = 0.0
    """Fifth radial term (radtan8 only)."""
    k6: float = 0.0
    """Sixth radial term (radtan8 only)."""
    p1: float = 0.0
    """First tangential term (radtan8 only)."""
    p2: float = 0.0
    """Second tangential term (radtan8 only)."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltCamera:
    """One camera's model tag and its coefficients."""

    camera_type: CameraModel
    """``kb4`` (Kannala-Brandt fisheye) or ``pinhole-radtan8`` (Brown-Conrady)."""
    intrinsics: BasaltIntrinsics
    """The coefficients themselves."""


@serde.serde
@dataclass(frozen=True, slots=True)
class BasaltCalibrationValue:
    """The one member of basalt's ``value0`` wrapper that dataforge reads."""

    T_imu_cam: list[BasaltPose]  # noqa: N815 — basalt's own key; renaming it would need a serde alias for no gain
    """Camera poses in the IMU frame, one per camera, in camera order."""
    intrinsics: list[BasaltCamera]
    """Camera models, one per camera, in the same order."""
    resolution: list[list[int]]
    """``[width, height]`` per camera, in the same order."""


@serde.serde
@dataclass(frozen=True, slots=True)
class MsdCalibration:
    """A device's ``extras/calibration.json``, as basalt writes it."""

    value0: BasaltCalibrationValue
    """cereal's single-root wrapper; everything lives under it."""


def camera_parameters(calibration: MsdCalibration, index: int, *, name: str) -> PinholeParameters | Fisheye62Parameters:
    """Build one camera's simplecv parameters from the device calibration.

    The extrinsics are the camera's pose **in the rig frame**, because MSD's rig
    frame is the IMU frame (``RIG_REFERENCE``) and ``T_imu_cam`` is exactly that
    pose. simplecv's ``Extrinsics`` calls the parent frame "world", so the rig
    goes in as ``world_R_cam`` / ``world_t_cam`` — the same convention simplecv's
    RoboCap loader uses, where Kalibr's inverse ``T_cam_imu`` goes in as
    ``cam_R_world`` / ``cam_t_world``.

    Args:
        calibration: Parsed ``calibration.json`` of the device.
        index: Camera index, matching the ``cam<index>`` directory in a sequence.
        name: Stream label carried into the parameters.

    Returns:
        A ``Fisheye62Parameters`` for a ``kb4`` camera, a ``PinholeParameters``
        for a ``pinhole-radtan8`` one.
    """
    value: BasaltCalibrationValue = calibration.value0
    pose: BasaltPose = value.T_imu_cam[index]
    camera: BasaltCamera = value.intrinsics[index]
    terms: BasaltIntrinsics = camera.intrinsics
    width: int = value.resolution[index][0]
    height: int = value.resolution[index][1]

    rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
    rig_t_cam: Float64[ndarray, "3"] = np.array([pose.px, pose.py, pose.pz], dtype=np.float64)
    extrinsics: Extrinsics = Extrinsics(world_R_cam=rig_R_cam, world_t_cam=rig_t_cam)
    intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF", fl_x=terms.fx, fl_y=terms.fy, cx=terms.cx, cy=terms.cy, height=height, width=width
    )
    if camera.camera_type == "kb4":
        return Fisheye62Parameters(
            name=name,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            distortion=KannalaBrandtDistortion(k1=terms.k1, k2=terms.k2, k3=terms.k3, k4=terms.k4),
        )
    return PinholeParameters(
        name=name,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        distortion=BrownConradyDistortion(
            k1=terms.k1, k2=terms.k2, p1=terms.p1, p2=terms.p2, k3=terms.k3, k4=terms.k4, k5=terms.k5, k6=terms.k6
        ),
    )


@serde.serde(type_check=serde.coerce)
@dataclass(frozen=True, slots=True)
class CameraRow:
    """One row of a ``cam<N>/data.csv``: when a frame was captured, and where it is."""

    timestamp_ns: int = serde.field(rename=CAMERA_TIMESTAMP_FIELD)
    """Capture time on the device clock, nanoseconds."""
    filename: str = serde.field(rename="filename")
    """PNG file name inside the camera's ``data/`` directory."""


@dataclass(frozen=True, slots=True)
class TimestampedSamples:
    """One purely numeric MSD csv, split into its clock and its value columns."""

    times_ns: Int64[ndarray, "n_samples"]
    """Sample times on the device clock, nanoseconds."""
    values: Float64[ndarray, "n_samples n_values"]
    """Every column after the timestamp, in file order."""


@dataclass(frozen=True, slots=True)
class GtTrajectory:
    """One ``gt/data.csv`` as Rerun wants it: ``world_T_rig`` per sample, quaternion xyzw."""

    times_ns: Int64[ndarray, "n_poses"]
    """Pose times on the device clock, nanoseconds — already the IMU's clock."""
    translations_xyz: Float64[ndarray, "n_poses 3"]
    """``world_t_rig``, metres."""
    quaternions_xyzw: Float64[ndarray, "n_poses 4"]
    """``world_R_rig``, scalar **last**; a dropout row holds identity."""
    num_sanitized: int
    """Rows whose quaternion was not unit-norm and was replaced with identity."""


def gt_trajectory(samples: TimestampedSamples) -> GtTrajectory:
    """Split a parsed ``gt/data.csv`` into the columns ``rr.Transform3D`` takes.

    Two file-format facts live here and nowhere else. The csv writes the
    quaternion **wxyz** (EuRoC's scalar-first layout) while ``Transform3D``
    wants xyzw; and a tracking dropout is written as a degenerate quaternion.
    A non-unit quaternion breaks the rotation chain from that row on, so every
    child frustum of the rig stops tracking — slam-evals hit exactly this on
    ROVER-T265's ``0 0 0 0`` rows (``slam_evals/ingest/columns.py``) and settled
    on the same per-row repair: identity rotation, translation untouched, and
    the count reported rather than hidden.

    Args:
        samples: ``read_numeric_csv(..., num_values=GT_VALUE_COLUMNS)`` output.

    Returns:
        The pose columns, plus how many rows had to be repaired.

    Raises:
        ValueError: The file holds a header and nothing else. gt's first stamp is
            also the sequence's clock origin, so an empty one has no silent default.
    """
    if samples.times_ns.size == 0:
        raise ValueError("gt/data.csv has no data rows, so the sequence has neither poses nor a clock origin")
    translations_xyz: Float64[ndarray, "n_poses 3"] = np.ascontiguousarray(samples.values[:, 0:3])
    quaternions_xyzw: Float64[ndarray, "n_poses 4"] = np.ascontiguousarray(samples.values[:, [4, 5, 6, 3]])
    norms: Float64[ndarray, "n_poses"] = np.linalg.norm(quaternions_xyzw, axis=1)
    dropped: Bool[ndarray, "n_poses"] = np.abs(norms - 1.0) > QUATERNION_NORM_TOLERANCE
    quaternions_xyzw[dropped] = IDENTITY_QUATERNION_XYZW
    return GtTrajectory(
        times_ns=samples.times_ns,
        translations_xyz=translations_xyz,
        quaternions_xyzw=quaternions_xyzw,
        num_sanitized=int(dropped.sum()),
    )


def measured_world_up(gt: GtTrajectory, accel: TimestampedSamples, *, window_ns: int = MEASURED_UP_WINDOW_NS) -> tuple[WorldUpAxis, float]:
    """Measure which world axis is up, from gravity as the accelerometer sees it.

    MSD ships no statement of its world axes: the Index's ground truth comes
    from SteamVR Lighthouse and the G2's and Odyssey+'s from a MoCap rig whose
    convention is undocumented. Gravity settles it. An accelerometer at rest
    measures the *reaction* to gravity, so its reading points **up**; rotating
    each sample into the world with the ground truth's own orientation
    (``world_R_rig @ a_rig``) and averaging therefore yields a vector along the
    world's up axis. Only the first couple of seconds are used: a headset is
    typically still on the floor or on a head that has not started moving, so
    the mean is nearly pure gravity there and gets noisier the longer the window.

    Args:
        gt: The sequence's ground truth, already in xyzw order and sanitized.
        accel: Accelerometer samples in m/s^2 — a 3-column ``TimestampedSamples``
            (``inertial.values[:, 3:6]``), on the same device clock as ``gt``.
        window_ns: Length of the averaging window, from the first sample both
            streams cover.

    Returns:
        The dominant signed world axis, and the mean's component along it as a
        fraction of standard gravity — a health check, not a calibration: a
        value near 1 means the window really was near rest and the axis is
        unambiguous, while a much smaller one means the mean is not gravity.

    Raises:
        ValueError: Either stream is empty, or they do not overlap inside the window.
    """
    if gt.times_ns.size == 0 or accel.times_ns.size == 0:
        raise ValueError("measuring the world up axis needs both a gt pose and an accelerometer sample")
    start_ns: int = max(int(gt.times_ns[0]), int(accel.times_ns[0]))
    inside: Bool[ndarray, "n_samples"] = (accel.times_ns >= start_ns) & (accel.times_ns < start_ns + window_ns)
    if not inside.any():
        raise ValueError(f"no accelerometer sample within {window_ns / 1e9:g} s of {start_ns}, where the gt starts")

    window_times_ns: Int64[ndarray, "n_window"] = accel.times_ns[inside]
    after: Int64[ndarray, "n_window"] = np.clip(np.searchsorted(gt.times_ns, window_times_ns), 0, gt.times_ns.size - 1)
    before: Int64[ndarray, "n_window"] = np.clip(after - 1, 0, gt.times_ns.size - 1)
    nearest: Int64[ndarray, "n_window"] = np.where(
        np.abs(gt.times_ns[before] - window_times_ns) <= np.abs(gt.times_ns[after] - window_times_ns), before, after
    )
    world_accel_xyz: Float64[ndarray, "n_window 3"] = Rotation.from_quat(gt.quaternions_xyzw[nearest]).apply(accel.values[inside])
    mean_xyz: Float64[ndarray, "3"] = world_accel_xyz.mean(axis=0)

    axis_index: int = int(np.argmax(np.abs(mean_xyz)))
    names: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = POSITIVE_WORLD_AXES if mean_xyz[axis_index] >= 0.0 else NEGATIVE_WORLD_AXES
    return names[axis_index], float(abs(mean_xyz[axis_index]) / STANDARD_GRAVITY_MS2)


def read_camera_index(data: bytes) -> list[CameraRow]:
    """Parse a ``cam<N>/data.csv`` into typed rows, in file order.

    Args:
        data: Whole csv, as read out of the archive.

    Returns:
        One ``CameraRow`` per data row; the file order *is* the presentation order.
    """
    reader: csv.DictReader[str] = csv.DictReader(io.StringIO(data.decode()))
    reader.fieldnames = [name.strip() for name in reader.fieldnames or []]
    return [serde.from_dict(CameraRow, {name: row[name] for name in (CAMERA_TIMESTAMP_FIELD, "filename")}) for row in reader]


def read_numeric_csv(data: bytes, *, num_values: int) -> TimestampedSamples:
    """Parse one all-numeric MSD csv (``imu0``, ``mag0``, ``gt``).

    ``numpy.loadtxt`` rather than a per-row typed dataclass: these run at 1 kHz,
    so an hour-long sequence is 3.6M rows, where a pyserde row costs 9.8 s and
    2.1 GB of live objects against loadtxt's 0.8 s and one array. The timestamps
    are read in their own ``int64`` pass instead of being cast down from the
    float table, so no stamp can be rounded no matter how large the clock grows.

    Args:
        data: Whole csv, as read out of the archive.
        num_values: Columns to keep after the timestamp (6 for an IMU, 3 for a
            magnetometer); trailing columns are ignored.

    Returns:
        The clock and the value table, row-aligned.
    """
    times_ns: Int64[ndarray, "n_samples"] = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1, usecols=0, dtype=np.int64, ndmin=1)
    values: Float64[ndarray, "n_samples n_values"] = np.loadtxt(
        io.BytesIO(data), delimiter=",", skiprows=1, usecols=range(1, 1 + num_values), dtype=np.float64, ndmin=2
    )
    return TimestampedSamples(times_ns=times_ns, values=values)


def resolve_seven_zip() -> Path:
    """Locate the 7-Zip CLI that reads MSD's multi-volume archives.

    conda-forge's ``7zip`` package installs the modern ``7zz`` and keeps ``7z``
    as a compatibility name, so both are accepted.
    """
    for binary in ("7zz", "7z"):
        found: str | None = shutil.which(binary)
        if found is not None:
            return Path(found)
    raise FileNotFoundError("no 7zz/7z on PATH; the conda-forge '7zip' package provides it (see [feature.dataforge.dependencies])")


class MemberReader(AbstractContextManager["MemberReader"], ABC):
    """Reads named members out of one sequence archive.

    Single and multi-volume archives need completely different machinery, but a
    converter only ever asks two things of either: give me this csv, and give me
    these PNGs in this order. Those two methods are the whole seam.
    """

    @abstractmethod
    def csv_bytes(self, member: str) -> bytes:
        """Whole contents of one small member, e.g. ``<SEQ>/mav0/imu0/data.csv``."""

    @abstractmethod
    def png_frames(self, members: Sequence[str]) -> Iterator[bytes]:
        """Encoded PNG bytes of ``members``, in the given order, one frame at a time.

        Members must all live in one directory (one camera's ``data/``): a
        multi-volume archive is extracted a directory at a time, so mixing
        cameras in one call would defeat the point of the extraction budget.
        """


class ZipMemberReader(MemberReader):
    """A plain single-file ``.zip``, read in-process with the stdlib."""

    def __init__(self, archive: Path) -> None:
        self.archive: zipfile.ZipFile = zipfile.ZipFile(archive)

    def csv_bytes(self, member: str) -> bytes:
        return self.archive.read(member)

    def png_frames(self, members: Sequence[str]) -> Iterator[bytes]:
        for member in members:
            yield self.archive.read(member)

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        self.archive.close()


class SevenZipMemberReader(MemberReader):
    """An Info-ZIP multi-volume set (``.z01``…``.zip``), read through the 7-Zip CLI.

    Python's ``zipfile`` opens the closing volume — its central directory is
    intact — and then fails on the first member whose data crosses a volume
    boundary, so it cannot be used at all here. 7-Zip cannot stream either, so
    ``png_frames`` extracts the members' directory into ``work_dir``, yields the
    files from there, and deletes them again: peak scratch is one camera's PNGs
    rather than the whole sequence.
    """

    def __init__(self, closing_volume: Path, work_dir: Path) -> None:
        self.archive: Path = closing_volume
        self.work_dir: Path = work_dir
        self.binary: Path = resolve_seven_zip()

    def _run(self, arguments: Sequence[str], *, capture: bool) -> bytes:
        completed: subprocess.CompletedProcess[bytes] = subprocess.run(
            [str(self.binary), *arguments], capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{self.binary.name} exited {completed.returncode} on {self.archive.name}:\n{completed.stderr.decode(errors='replace')}")
        return completed.stdout if capture else b""

    def csv_bytes(self, member: str) -> bytes:
        return self._run(["x", "-so", "-bso0", "-bsp0", str(self.archive), member], capture=True)

    def png_frames(self, members: Sequence[str]) -> Iterator[bytes]:
        if not members:
            return
        directory: str = str(PurePosixPath(members[0]).parent)
        extract_dir: Path = self.work_dir / "extract"
        shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        self._run(["x", f"-o{extract_dir}", "-y", "-bso0", "-bsp0", str(self.archive), f"{directory}/*"], capture=False)
        try:
            for member in members:
                yield (extract_dir / member).read_bytes()
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None) -> None:
        shutil.rmtree(self.work_dir / "extract", ignore_errors=True)


def open_member_reader(archives: Sequence[Path], work_dir: Path) -> MemberReader:
    """Pick the reader one sequence's archive files need.

    Args:
        archives: Local volume paths, parts first and the closing ``.zip`` last.
        work_dir: Scratch directory the multi-volume reader extracts into.

    Returns:
        A stdlib reader for a single file, the 7-Zip-backed one for a volume set.
    """
    if len(archives) == 1:
        return ZipMemberReader(archives[0])
    return SevenZipMemberReader(archives[-1], work_dir)


def group_archives(entries: Sequence[tuple[str, int]]) -> dict[str, list[tuple[str, int]]]:
    """Group a collection listing into ``stem → volumes``, dropping non-archives.

    An MSD sequence is either one ``<stem>.zip`` or an Info-ZIP multi-volume set
    (``<stem>.z01``, ``<stem>.z02``, …, ``<stem>.zip``). Volumes come back parts
    first and ascending, with the closing ``.zip`` last, which is the order 7-Zip
    wants them named in.

    Args:
        entries: ``(repo-relative path, size)`` pairs from one collection directory.

    Returns:
        One entry per sequence stem; README files and anything else are dropped.
    """
    grouped: dict[str, list[tuple[str, int]]] = {}
    for path, size in entries:
        suffix: str = Path(path).suffix
        if suffix != ".zip" and PART_SUFFIX_RE.match(suffix) is None:
            continue
        grouped.setdefault(Path(path).stem, []).append((path, size))
    # ".zip" sorts after ".z01".."z99" lexicographically, so one sort gives both
    # the ascending part order and the closing volume's place at the end.
    return {stem: sorted(volumes) for stem, volumes in sorted(grouped.items())}


def nominal_fps(times_ns: Int64[ndarray, "n_samples"]) -> int:
    """Container frame rate for one camera, from the median inter-sample gap.

    Only the mp4 header cares: ``log_video_stream(times_ns=...)`` stamps every
    sample with its real capture time afterwards, so this just has to be close
    enough that a player without the rrd shows the clip at the right speed. The
    median (not the mean) keeps one dropped frame from halving it.

    Args:
        times_ns: Capture times of the camera's frames, ascending.

    Returns:
        Frames per second, at least 1.
    """
    if times_ns.size < 2:
        return 1
    median_gap_ns: float = float(np.median(np.diff(times_ns)))
    if median_gap_ns <= 0.0:
        raise ValueError(f"camera timestamps do not advance (median gap {median_gap_ns} ns); the csv is unusable")
    return max(1, round(1e9 / median_gap_ns))


def follow_eye_controls() -> rrb.EyeControls3D:
    """First-person eye controls for the follow view, in the rig (IMU) frame.

    Carried over from RoboCap: MSD's world axes are only established by the gt
    layer, so the eye is placed relative to the headset itself and revisited
    when that layer lands.
    """
    # EyeControls3D is marked unstable by the SDK; re-validate this factory on Rerun bumps.
    return rrb.EyeControls3D(
        kind=rrb.Eye3DKind.FirstPerson,
        position=(0.8, -0.8, -0.6),
        look_target=(0.0, 0.0, 0.0),
        eye_up=(0.0, 0.0, -1.0),
        spin_speed=0.0,
    )


def camera_views(num_cameras: int) -> list[rrb.Spatial2DView]:
    """One 2D pane per camera, labelled the way the archives name them (``cam0``…)."""
    return [
        rrb.Spatial2DView(name=f"cam{index}", origin=schema.pinhole_path(RIG, index), contents=f"{schema.pinhole_path(RIG, index)}/**")
        for index in range(num_cameras)
    ]


def build_blueprint(num_cameras: int, *, has_magnetometer: bool) -> rrb.Blueprint:
    """Default layout: the rig in 3D beside a camera grid, over the sensor plots.

    Args:
        num_cameras: Cameras the device carries; the grid holds one pane each.
        has_magnetometer: Whether to add the third plot pane.

    Returns:
        The blueprint embedded in every base-layer rrd of this device and
        registered as the catalog dataset's default.
    """
    plots: list[rrb.TimeSeriesView] = [
        rrb.TimeSeriesView(
            name="Gyroscope", origin=schema.imu_path(RIG, IMU), contents=schema.gyro_path(RIG, IMU), plot_legend=rrb.PlotLegend(visible=True)
        ),
        rrb.TimeSeriesView(
            name="Accelerometer",
            origin=schema.imu_path(RIG, IMU),
            contents=schema.accel_path(RIG, IMU),
            plot_legend=rrb.PlotLegend(visible=True),
        ),
    ]
    if has_magnetometer:
        plots.append(
            rrb.TimeSeriesView(
                name="Magnetometer",
                origin=schema.mag_path(RIG, MAG),
                contents=schema.field_path(RIG, MAG),
                plot_legend=rrb.PlotLegend(visible=True),
            )
        )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="Rig",
                        origin="/",
                        line_grid=True,
                        # The overview shows the whole gt path; its cursor trail stays hidden.
                        # Overrides on entities a base-only recording lacks are simply inert.
                        overrides={schema.trail_path("gt"): rrb.EntityBehavior(visible=False)},
                    ),
                    # Follow-cam (rerun-io/eye_control_example pattern): the view's origin
                    # IS the rig frame, so a fixed first-person eye in that frame rides the
                    # headset. Inert until the gt layer animates rig_00.
                    rrb.Spatial3DView(
                        name="Follow",
                        origin=schema.rig_path(RIG),
                        contents="/**",
                        line_grid=True,
                        overrides={
                            schema.trajectory_path("gt"): rrb.EntityBehavior(visible=False),
                            schema.trail_path("gt"): rrb.VisibleTimeRanges(
                                rrb.VisibleTimeRange(
                                    schema.TIMELINE,
                                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=-10.0),
                                    end=rrb.TimeRangeBoundary.cursor_relative(),
                                )
                            ),
                        },
                        eye_controls=follow_eye_controls(),
                    ),
                ),
                rrb.Grid(*camera_views(num_cameras), grid_columns=2, name="Synchronized cameras"),
                column_shares=[3, 2],
            ),
            rrb.Horizontal(*plots),
            row_shares=[3, 1],
        ),
        rrb.TimePanel(timeline=schema.TIMELINE),
        collapse_panels=True,
    )


def build_table_blueprint(num_cameras: int) -> rrb.Blueprint:
    """Segment-table preview card: the 3D rig with no video textures, plus ``cam0``.

    Every visible table row renders through this at once, so exactly one video
    stream is decoded and the rest are *excluded* rather than hidden.

    Args:
        num_cameras: Cameras the device carries; all but ``cam0``'s video are excluded.
    """
    video_exclusions: list[str] = [f"- {schema.video_path(RIG, index)}/**" for index in range(num_cameras)]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="Follow",
                origin=schema.rig_path(RIG),
                contents=["/**", *video_exclusions, f"- {schema.trail_path('gt')}/**"],
                line_grid=True,
                eye_controls=follow_eye_controls(),
            ),
            rrb.Spatial2DView(name="cam0", origin=schema.pinhole_path(RIG, 0), contents=f"{schema.pinhole_path(RIG, 0)}/**"),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline=schema.TIMELINE),
    )


@dataclass(frozen=True, slots=True)
class RecordingSummary:
    """What one converted sequence turned out to hold; ``convert`` prints it."""

    num_frames: int
    """Longest per-camera video sample count."""
    num_poses: int
    """Ground-truth poses logged into the gt layer."""
    measured_up: WorldUpAxis
    """World up axis this sequence's own gravity measurement found."""
    measured_up_fraction: float
    """Fraction of standard gravity the measured axis carried; near 1 is a clean measurement."""


class MsdDataset(DataforgeDataset[MsdConfig, MsdSource]):
    """Converts one MSD headset's sequences into exoego:v2 base-layer recordings."""

    def __init__(self, config: MsdConfig) -> None:
        super().__init__(config)
        self.device: MsdDevice = MSD_DEVICES[config.device]

    def discover(self) -> list[tuple[SequenceIdentity, MsdSource]]:
        """Enumerate the **remote** tree: raw archives never stay on disk to walk.

        Collections are listed in the device's declared order and sequences
        sorted within each, so the same repo revision always yields the same list.
        """
        pairs: list[tuple[SequenceIdentity, MsdSource]] = []
        for collection in self.device.collections:
            collection_path: str = f"{REPO_ROOT}/{self.device.hf_dir}/{collection}"
            entries: list[tuple[str, int]] = list_collection_files(REPO_ID, collection_path, revision=self.config.revision)
            leaf: str = collection.rsplit("/", 1)[-1]
            for stem, volumes in group_archives(entries).items():
                pairs.append(
                    (
                        SequenceIdentity(dataset=self.config.name, parts=(leaf, stem)),
                        MsdSource(
                            collection_path=collection_path,
                            sequence=stem,
                            archive_paths=tuple(path for path, _ in volumes),
                            archive_bytes=sum(size for _, size in volumes),
                        ),
                    )
                )
        return pairs

    def default_blueprint(self) -> rrb.Blueprint:
        """Device-wide layout: every sequence of one headset has the same sensors."""
        return build_blueprint(self.device.num_cameras, has_magnetometer=self.device.has_magnetometer)

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card for the device's segment table."""
        return build_table_blueprint(self.device.num_cameras)

    @property
    def calibration_path(self) -> str:
        """Repo-relative path of the device's basalt calibration."""
        return f"{REPO_ROOT}/{self.device.hf_dir}/extras/calibration.json"

    def download(self) -> None:
        """Fetch the calibration, prove the machine can encode, and print the plan.

        Deliberately **not** a bulk fetch: the corpus is hundreds of gigabytes of
        PNG archives and ``convert`` pulls one sequence at a time, so the only
        thing worth having up front is the few-kilobyte calibration. The NVENC
        check is here because a machine that cannot encode AV1 should find out
        now rather than after the first multi-gigabyte download.
        """
        transports.hf_fetch(REPO_ID, allow_patterns=[self.calibration_path], local_dir=self.config.root, revision=self.config.revision)
        require_av1_nvenc(resolve_ffmpeg())
        discovered: list[tuple[SequenceIdentity, MsdSource]] = self.discover()
        per_collection: dict[str, list[MsdSource]] = {}
        for identity, source in discovered:
            per_collection.setdefault(identity.parts[0], []).append(source)
        total_bytes: int = sum(source.archive_bytes for _, source in discovered)
        print(f"{self.config.name}: {len(discovered)} sequence(s), {total_bytes / 1e9:.1f} GB of archives to stream through convert")
        for collection, sources in per_collection.items():
            print(f"  {collection}: {len(sources)} sequence(s), {sum(source.archive_bytes for source in sources) / 1e9:.1f} GB")
        print(f"  calibration → {self.config.root / self.calibration_path}; archives are fetched and deleted one sequence at a time")

    def calibration(self) -> MsdCalibration:
        """Load the device calibration, fetching it if ``download`` was skipped."""
        local_path: Path = self.config.root / self.calibration_path
        if not local_path.is_file():
            transports.hf_fetch(REPO_ID, allow_patterns=[self.calibration_path], local_dir=self.config.root, revision=self.config.revision)
        return serde.json.from_json(MsdCalibration, local_path.read_text())

    def enforce_raw_budget(self, source: MsdSource, archives: Sequence[Path]) -> None:
        """Refuse to fetch when ``root`` plus this sequence would breach the budget.

        A sequence whose archives *alone* exceed the budget (the Index and G2
        long sessions) is an accepted exception and only warns — there is no smaller
        unit to convert. Anything else is a leftover from an earlier failure, and
        silently blowing past the cap is how a batch run fills a disk.

        Args:
            source: Sequence about to be fetched.
            archives: Its local volume paths, excluded from the on-disk tally so
                a retry of an already-downloaded sequence is not double-counted.

        Raises:
            RuntimeError: Leftovers in ``root`` would push the fetch over the budget.
        """
        budget_bytes: int = int(self.config.raw_budget_gb * 1e9)
        if source.archive_bytes > budget_bytes:
            print(
                f"  warning: {source.sequence} needs {source.archive_bytes / 1e9:.1f} GB of archives, above the "
                f"{self.config.raw_budget_gb:g} GB raw budget; converting it anyway as an accepted exception"
            )
            return
        if not self.config.root.is_dir():
            return
        own: set[Path] = {archive.resolve() for archive in archives}
        leftovers: list[Path] = [path for path in self.config.root.rglob("*") if path.is_file() and path.resolve() not in own]
        occupied: int = sum(path.stat().st_size for path in leftovers)
        if occupied + source.archive_bytes <= budget_bytes:
            return
        biggest: list[Path] = sorted(leftovers, key=lambda path: path.stat().st_size, reverse=True)[:3]
        named: str = ", ".join(f"{path.relative_to(self.config.root)} ({path.stat().st_size / 1e9:.2f} GB)" for path in biggest)
        raise RuntimeError(
            f"{self.config.root} already holds {occupied / 1e9:.2f} GB and {source.sequence} needs "
            f"{source.archive_bytes / 1e9:.2f} GB more, over the {self.config.raw_budget_gb:g} GB raw budget. "
            f"Delete the leftovers of an earlier failed sequence first: {named}"
        )

    def convert(self, identity: SequenceIdentity, source: MsdSource, *, force: bool) -> Path:
        """Fetch one sequence, encode it, write its two rrd layers, and delete the raw.

        Both layers come out of the *same* archive fetch, so they are skipped and
        rebuilt as a unit: half a sequence on disk means the multi-gigabyte
        download has to happen again anyway, and rebuilding the base layer with
        it costs one encode against a corpus that would otherwise be silently
        incomplete. The base path is what the caller gets, because that is the
        layer every dataforge verb keys on.

        Failure keeps the archive (a retry then skips the multi-gigabyte
        download) but removes the mp4s and any extraction directory, so the next
        attempt starts from a clean scratch dir. Nothing outside ``root`` is ever
        deleted.
        """
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        gt_target: Path = paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)
        if writing.should_skip(target, force=force) and writing.should_skip(gt_target, force=force):
            print(f"skip {identity.sequence_key} → {target} + {gt_target}")
            return target

        archives: list[Path] = [self.config.root / archive_path for archive_path in source.archive_paths]
        self.enforce_raw_budget(source, archives)
        on_disk: int = sum(1 for archive in archives if archive.is_file())
        if on_disk:
            print(f"  {on_disk}/{len(archives)} archive volume(s) already in {self.config.root}; the fetch only verifies them")
        transports.hf_fetch(REPO_ID, allow_patterns=list(source.archive_paths), local_dir=self.config.root, revision=self.config.revision)

        work_dir: Path = self.config.root / "work" / source.sequence
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            summary: RecordingSummary = self.write_recording(
                identity, source, archives=archives, work_dir=work_dir, target=target, gt_target=gt_target
            )
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            retained: int = sum(archive.stat().st_size for archive in archives if archive.is_file())
            print(f"  kept {retained / 1e9:.2f} GB of archives in {self.config.root} so a retry skips the download")
            raise

        if self.config.keep_raw:
            print(f"  keeping raw: {len(archives)} archive volume(s) and the mp4s in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            for archive in archives:
                archive.unlink(missing_ok=True)
        print(
            f"done {identity.sequence_key} → {target} + {gt_target} "
            f"({self.device.num_cameras} cameras, {summary.num_frames} frames, {summary.num_poses} gt poses, "
            f"world up {summary.measured_up} at {summary.measured_up_fraction:.2f} g)"
        )
        return target

    def write_recording(
        self, identity: SequenceIdentity, source: MsdSource, *, archives: Sequence[Path], work_dir: Path, target: Path, gt_target: Path
    ) -> RecordingSummary:
        """Encode every camera, read every csv, and write both rrd layers of one sequence.

        The base layer closes before the gt layer opens: one archive read feeds
        both, but each is its own recording stream and its own atomic replace, so
        a failure in the second cannot half-write the first.

        Args:
            identity: Sequence identity; names both recordings and both rrd files.
            source: The discovered sequence, whose stem is the archive's top directory.
            archives: Local archive volume paths, the closing ``.zip`` last.
            work_dir: Scratch directory for the encoded mp4s and any extraction.
            target: Final base-layer rrd path, written through ``atomic_recording``.
            gt_target: Final gt-layer rrd path, likewise.

        Returns:
            What the sequence turned out to hold, for the caller's summary line.
        """
        calibration: MsdCalibration = self.calibration()
        cameras: list[PinholeParameters | Fisheye62Parameters] = [
            camera_parameters(calibration, index, name=f"cam{index}") for index in range(self.device.num_cameras)
        ]
        mav0: str = f"{source.sequence}/mav0"
        camera_times: list[Int64[ndarray, "n_samples"]] = []
        clips: list[Path] = []
        magnetometer: TimestampedSamples | None = None

        # Everything the archive holds is read here, before the recording opens: the
        # archive reader for a split sequence extracts one camera at a time, so the
        # mp4s must all exist before any of them is remuxed into the recording.
        with open_member_reader(archives, work_dir) as reader:
            for index in range(self.device.num_cameras):
                rows: list[CameraRow] = read_camera_index(reader.csv_bytes(f"{mav0}/cam{index}/data.csv"))
                if not rows:
                    raise ValueError(f"{source.sequence} cam{index} has an empty data.csv")
                times_ns: Int64[ndarray, "n_samples"] = np.array([row.timestamp_ns for row in rows], dtype=np.int64)
                clip: Path = work_dir / f"cam{index}.mp4"
                encode_frames_to_mp4(
                    reader.png_frames([f"{mav0}/cam{index}/data/{row.filename}" for row in rows]),
                    clip,
                    source=FrameSource("png"),
                    fps=nominal_fps(times_ns),
                )
                camera_times.append(times_ns)
                clips.append(clip)
            inertial: TimestampedSamples = read_numeric_csv(reader.csv_bytes(f"{mav0}/imu0/data.csv"), num_values=IMU_VALUE_COLUMNS)
            if self.device.has_magnetometer:
                magnetometer = read_numeric_csv(reader.csv_bytes(f"{mav0}/mag0/data.csv"), num_values=MAG_VALUE_COLUMNS)
            # The gt *layer* is a sibling rrd, but both layers share one zero-based
            # video_time, so the base layer has to know gt's clock origin too.
            gt: GtTrajectory = gt_trajectory(read_numeric_csv(reader.csv_bytes(f"{mav0}/gt/data.csv"), num_values=GT_VALUE_COLUMNS))

        streams: list[Int64[ndarray, "n_samples"]] = [*camera_times, inertial.times_ns]
        if magnetometer is not None and magnetometer.times_ns.size:
            streams.append(magnetometer.times_ns)
        start_time_ns: int = min(int(gt.times_ns[0]), *(int(times_ns[0]) for times_ns in streams))
        # Deliberately not bounded by gt: duration_ns describes the *sensor* layer.
        duration_ns: int = max(int(times_ns[-1]) for times_ns in streams) - start_time_ns
        num_frames: int = max(times_ns.size for times_ns in camera_times)

        with writing.atomic_recording(
            target,
            application_id="dataforge",
            recording_id=identity.recording_id,
            default_blueprint=build_blueprint(self.device.num_cameras, has_magnetometer=self.device.has_magnetometer),
        ) as recording:
            # Deliberately NO ViewCoordinates at "/": the gt layer owns the root
            # ViewCoordinates, because it is what establishes a world frame at all.
            log_rig_node(recording, RIG, reference=RIG_REFERENCE, num_cameras=self.device.num_cameras, name=self.device.label, kind="ego")
            for index, (camera, clip, times_ns) in enumerate(zip(cameras, clips, camera_times, strict=True)):
                rr.log(schema.cam_path(RIG, index), rr.AnyValues(name=f"cam{index}", kind="grayscale"), static=True, recording=recording)
                log_pinhole(
                    camera,
                    cam_log_path=Path(schema.cam_path(RIG, index)),
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                    static=True,
                    recording=recording,
                )
                log_video_stream(recording, clip, schema.video_path(RIG, index), times_ns=times_ns - start_time_ns)
            log_imu(
                recording,
                RIG,
                IMU,
                gyro=ImuChannel(times_ns=inertial.times_ns - start_time_ns, values_xyz=inertial.values[:, :3]),
                accel=ImuChannel(times_ns=inertial.times_ns - start_time_ns, values_xyz=inertial.values[:, 3:6]),
                name="imu0",
            )
            if magnetometer is not None:
                log_magnetometer(
                    recording,
                    RIG,
                    MAG,
                    field=ImuChannel(times_ns=magnetometer.times_ns - start_time_ns, values_xyz=magnetometer.values),
                    name="mag0",
                )
            writing.send_capture_properties(
                recording,
                identity,
                num_cameras=self.device.num_cameras,
                num_frames=num_frames,
                start_time_ns=start_time_ns,
                device=self.config.device,
                device_label=self.device.label,
                collection=identity.parts[0],
                hf_revision=repo_revision(REPO_ID, self.config.revision),
                duration_ns=duration_ns,
            )

        accel: TimestampedSamples = TimestampedSamples(times_ns=inertial.times_ns, values=inertial.values[:, 3:6])
        measured: tuple[WorldUpAxis, float] = measured_world_up(gt, accel)
        if measured[0] != self.device.world_up:
            print(
                f"  warning: {self.config.device} declares world_up {self.device.world_up} but {source.sequence} "
                f"measured {measured[0]} carrying {measured[1]:.2f} of |g|; the rrd still states the declared axis"
            )

        gt_times_ns: Int64[ndarray, "n_poses"] = gt.times_ns - start_time_ns
        gt_index: list[rr.TimeColumn] = [rr.TimeColumn(schema.TIMELINE, duration=gt_times_ns.astype("timedelta64[ns]"))]
        with writing.atomic_recording(gt_target, application_id="dataforge", recording_id=identity.recording_id) as recording:
            # The gt layer establishes a world frame at all, so it — not the base
            # layer — owns the root ViewCoordinates. The axis is the device's
            # declared one, not this sequence's measurement: every rrd of a device
            # must agree, and a disagreement is the warning above, not a silent
            # per-sequence reorientation.
            rr.log("/", WORLD_UP_VIEW_COORDINATES[self.device.world_up], static=True, recording=recording)
            rr.send_columns(
                schema.rig_path(RIG),
                indexes=gt_index,
                columns=rr.Transform3D.columns(translation=gt.translations_xyz, quaternion=gt.quaternions_xyzw),
                recording=recording,
            )
            # Two views of one trajectory: the static strip is the whole path for the
            # overview, and the per-pose points are what the blueprint's cursor-relative
            # time range turns into a recent-motion trail in the follow view.
            rr.log(
                schema.trajectory_path("gt"),
                rr.LineStrips3D([gt.translations_xyz], colors=GT_TRAJECTORY_COLOR, radii=GT_TRAJECTORY_RADIUS_M),
                static=True,
                recording=recording,
            )
            rr.log(
                schema.trail_path("gt"),
                rr.Points3D.from_fields(colors=GT_TRAIL_COLOR, radii=GT_TRAIL_RADIUS_M),
                static=True,
                recording=recording,
            )
            rr.send_columns(
                schema.trail_path("gt"),
                indexes=gt_index,
                columns=rr.Points3D.columns(positions=gt.translations_xyz),
                recording=recording,
            )
            recording.send_recording_name(identity.recording_id)
            recording.send_property(
                "gt",
                rr.AnyValues(
                    num_poses=gt.times_ns.size,
                    duration_ns=int(gt.times_ns[-1] - gt.times_ns[0]),
                    num_sanitized=gt.num_sanitized,
                    source=self.device.gt_source,
                    world_up=self.device.world_up,
                    measured_up=measured[0],
                    measured_up_fraction=measured[1],
                ),
            )
        return RecordingSummary(
            num_frames=num_frames, num_poses=int(gt.times_ns.size), measured_up=measured[0], measured_up_fraction=measured[1]
        )
