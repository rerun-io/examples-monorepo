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
streams its PNGs through the AV1 encoder, and deletes the archive again; what
survives is the rrd. ``enforce_raw_budget`` caps the scratch directory so a batch
run cannot silently fill the NVMe with leftovers from failed sequences.

**One dataset per device, two layers per sequence.** A catalog dataset holds one
default blueprint, hence one camera layout, and the three headsets have two, four
and two cameras — so ``--device`` picks the corpus *and* the catalog dataset
(``msd-index``, ``msd-g2``, ``msd-odyssey``). Each sequence becomes a ``base`` rrd
(video, IMU, magnetometer) and a ``gt`` rrd (``world_T_rig`` at the full ~1 kHz
rate, plus its path and trail) under one recording id, so the catalog stacks them
onto one segment. Both come out of one archive read, so they are skipped and
rebuilt together: half a sequence on disk means paying for the download again.

**Clocks.** Every csv timestamp is nanoseconds on one monotonic device clock
(values around 1e13, not a Unix epoch). ``video_time`` is that clock minus ``t0``,
the earliest sample of any stream *including* ``gt`` — the two layers must share
this origin, and the gt file is usually the earliest stream. Nothing is resampled.

**Per-device claims.** The world up axis and the follow frame are not in the
corpus; they are measured and derived from real sequences and then written down
per device. ``MSD_DEVICES`` states both, and why; ``convert`` re-checks both
against the sequence in front of it and warns on a disagreement rather than
silently reorienting one rrd.

**Frames.** ``rig_T_cam`` comes from basalt's ``T_imu_cam``, the camera pose in
the IMU frame; the rig frame *is* the IMU frame (``reference = "imu_00"``). The
calibration reader is ``dataforge.basalt``, and the archive readers — one
sequence can be a plain zip or an Info-ZIP volume set — are ``dataforge.archives``.
"""

from __future__ import annotations

import csv
import functools
import io
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
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

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.archives import MemberReader, group_archives, open_member_reader
from dataforge.basalt import BasaltCalibration, BasaltCamera, FollowFrame, camera_parameters, follow_frame
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    FrameSource,
    ImuChannel,
    encode_frames_to_mp4,
    log_camera_node,
    log_imu,
    log_magnetometer,
    log_pose_track,
    log_rig_node,
    log_video_stream,
    require_av1_nvenc,
    resolve_ffmpeg,
    time_column,
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
FOLLOW_BACK_M: float = 0.9
"""How far behind the headset the follow eye sits, along the device's own forward."""
FOLLOW_UP_M: float = 0.45
"""How far above the headset the follow eye sits, along the device's own up."""
FOLLOW_AHEAD_M: float = 0.3
"""How far ahead of the headset the follow eye aims, so the shot leads the motion."""
FOLLOW_FRAME_TOLERANCE_DEG: float = 5.0
"""How far a declared ``FollowFrame`` axis may sit from the calibration's before ``convert`` warns."""

MsdDeviceChoice: TypeAlias = Literal["index", "g2", "odyssey"]
"""``--device``: which headset's corpus to work on, and which catalog dataset."""
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
nothing about world axes at all; see ``MSD_DEVICES`` for how each device's is fixed.
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
    """Up axis of this device's tracking world; every rrd of the device carries
    the matching root ``ViewCoordinates``. Measured, not assumed — see ``MSD_DEVICES``."""
    follow: FollowFrame
    """Forward and up of this headset in its rig frame, which place its follow
    camera. Derived from the device's calibration — see ``MSD_DEVICES``."""
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
        # Measured on MIO09_short_1_updown: +y at 0.96 of |g|. OpenVR worlds are
        # right-handed Y-up, so Lighthouse ground truth agreeing is expected. The
        # measurement is not a restatement of the raw samples: over that window the
        # accelerometer's own mean points along the headset's -x, and only the gt
        # rotation (123..142 deg from identity there) turns it into world +y.
        world_up="+y",
        # follow_frame(calibration.json) on 2026-09-06: the front pair looks along rig +z
        # and its 13.4 cm baseline runs along rig y, so up is rig -x — the axis MIO09's
        # raw accelerometer mean already picked out. The Index alone carries no camera
        # roll: image-up lands within 1.2 deg of this up.
        follow=FollowFrame(forward=(0.009, 0.0, 1.0), up=(-1.0, 0.001, 0.009)),
        gt_source="lighthouse",
    ),
    "g2": MsdDevice(
        hf_dir="MG_reverb_g2",
        collections=("MGO_others",),
        num_cameras=4,
        has_magnetometer=True,
        label="reverb-g2",
        # Measured on MGO09_short_1_updown: +y at 0.98 of |g|. The MoCap rig
        # documents no convention, and the sign comes entirely from the gt
        # rotation — the raw accelerometer mean points along the headset's -y.
        world_up="+y",
        # follow_frame(calibration.json) on 2026-09-06 over cam0/cam1, the front pair
        # (cam2 and cam3 look along rig -x and +x, i.e. sideways). Their 10.8 cm baseline
        # runs along rig x, so up is rig -y — again where MGO09's raw accelerometer mean
        # points, and it puts the front pair 15.6 deg below the horizon, where tracking
        # cameras are aimed. All four G2 cameras are mounted rolled: image-up is rig +x,
        # a full 90 deg off this up, which is why the baseline and not image-up fixes it.
        follow=FollowFrame(forward=(-0.001, 0.268, 0.963), up=(0.006, -0.963, 0.268)),
        gt_source="mocap",
    ),
    "odyssey": MsdDevice(
        hf_dir="MO_odyssey_plus",
        collections=("MOO_others",),
        num_cameras=2,
        has_magnetometer=True,
        label="odyssey-plus",
        # Measured on MOO09_short_1_updown: +y at 0.93 of |g|; same undocumented
        # MoCap rig as the G2, and the same Y-up answer.
        world_up="+y",
        # follow_frame(calibration.json) on 2026-09-06: a 10.6 cm baseline along rig x
        # and an optical axis pitched 21.3 deg down from it, so up is rig -y. Like the
        # Index and unlike the G2 these cameras are upright — image-up is 0.3 deg away.
        follow=FollowFrame(forward=(0.002, 0.364, 0.932), up=(0.0, -0.932, 0.364)),
        gt_source="mocap",
    ),
}
"""The three headsets MSD covers, keyed by the ``--device`` literal.

Two of each device's fields are claims about data rather than facts read out of
the corpus, and this is where both are settled.

**The world up axis.** MSD documents none: the Index's ground truth comes from
SteamVR Lighthouse and the G2's and Odyssey+'s from an undocumented MoCap rig.
Gravity settles it — ``measured_world_up`` rotates the first seconds of
accelerometer samples into the world with the ground truth's own orientation and
averages, and an accelerometer at rest reads +g pointing *up*. All three headsets
measure ``+y`` on their ``*09_short_1_updown`` sequence, at 0.96, 0.98 and 0.93
of |g|, so all three state ``RIGHT_HAND_Y_UP`` at the root of their gt layer.

**The follow frame.** ``follow_frame`` reads it out of the device's own
``calibration.json``, but the constants are written down here instead of being
derived at use: ``register`` builds a device's default blueprint from the
registry alone, with no sequence and no calibration on disk, so the eye has to be
placeable without one.

Both are re-checked every ``convert`` (see ``warn_on_device_claims``), which
warns rather than reorienting or re-aiming a single rrd out of step with the rest.
"""


@dataclass(frozen=True, slots=True)
class MsdSource:
    """One remote sequence as discovery found it; ``convert`` needs nothing else."""

    collection: str
    """Leaf name of the collection the archive sits in, e.g. ``MIO_others``; also the identity's first part."""
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


@dataclass(frozen=True, slots=True)
class MeasuredUp:
    """What one sequence's own gravity measurement found; the gt layer records both."""

    axis: WorldUpAxis
    """The dominant signed world axis the mean acceleration points along."""
    fraction: float
    """That component as a fraction of standard gravity; near 1 is a clean measurement."""


def measured_world_up(gt: GtTrajectory, accel: ImuChannel, *, window_ns: int = MEASURED_UP_WINDOW_NS) -> MeasuredUp:
    """Measure which world axis is up, from gravity as the accelerometer sees it.

    An accelerometer at rest measures the *reaction* to gravity, so its reading
    points **up**; rotating each sample into the world with the ground truth's
    own orientation (``world_R_rig @ a_rig``) and averaging therefore yields a
    vector along the world's up axis. Only the first couple of seconds are used:
    a headset is typically still on the floor or on a head that has not started
    moving, so the mean is nearly pure gravity there and gets noisier the longer
    the window. Why this is measured at all, and what the three devices answer,
    is on ``MSD_DEVICES``.

    Args:
        gt: The sequence's ground truth, already in xyzw order and sanitized.
        accel: Accelerometer samples in m/s^2, on the same clock as ``gt``.
        window_ns: Length of the averaging window, from the first sample both
            streams cover.

    Returns:
        The axis and how much of gravity it carried — a health check, not a
        calibration: a much smaller fraction means the mean is not gravity.

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
    world_accel_xyz: Float64[ndarray, "n_window 3"] = Rotation.from_quat(gt.quaternions_xyzw[nearest]).apply(accel.values_xyz[inside])
    mean_xyz: Float64[ndarray, "3"] = world_accel_xyz.mean(axis=0)

    axis_index: int = int(np.argmax(np.abs(mean_xyz)))
    names: tuple[WorldUpAxis, WorldUpAxis, WorldUpAxis] = POSITIVE_WORLD_AXES if mean_xyz[axis_index] >= 0.0 else NEGATIVE_WORLD_AXES
    return MeasuredUp(axis=names[axis_index], fraction=float(abs(mean_xyz[axis_index]) / STANDARD_GRAVITY_MS2))


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


def follow_eye(follow: FollowFrame) -> rrb.EyeControls3D:
    """The device's chase camera: its own forward and up at this package's distances."""
    return blueprints.follow_eye_controls(follow.forward, follow.up, back_m=FOLLOW_BACK_M, up_m=FOLLOW_UP_M, ahead_m=FOLLOW_AHEAD_M)


def camera_views(num_cameras: int) -> list[rrb.Spatial2DView]:
    """One 2D pane per camera, labelled the way the archives name them (``cam0``…)."""
    return [blueprints.camera_view(f"cam{index}", RIG, index) for index in range(num_cameras)]


def build_blueprint(num_cameras: int, *, has_magnetometer: bool, follow: FollowFrame) -> rrb.Blueprint:
    """Default layout: the rig in 3D beside a camera grid, over the sensor plots.

    Args:
        num_cameras: Cameras the device carries; the grid holds one pane each.
        has_magnetometer: Whether to add the third plot pane.
        follow: The device's forward and up in the rig frame; it orients the Follow view.

    Returns:
        The blueprint embedded in every base-layer rrd of this device and
        registered as the catalog dataset's default.
    """
    plots: list[rrb.TimeSeriesView] = [
        blueprints.sensor_plot(name, origin, contents)
        for name, origin, contents in (
            ("Gyroscope", schema.imu_path(RIG, IMU), schema.gyro_path(RIG, IMU)),
            ("Accelerometer", schema.imu_path(RIG, IMU), schema.accel_path(RIG, IMU)),
        )
    ]
    if has_magnetometer:
        plots.append(blueprints.sensor_plot("Magnetometer", schema.mag_path(RIG, MAG), schema.field_path(RIG, MAG)))
    return blueprints.rig_blueprint(
        camera_views(num_cameras),
        rig=RIG,
        run_source=schema.GT_RUN_SOURCE,
        eye_controls=follow_eye(follow),
        plots=plots,
    )


def build_table_blueprint(num_cameras: int, *, follow: FollowFrame) -> rrb.Blueprint:
    """Segment-table preview card: the 3D rig with no video textures, plus ``cam0``.

    Args:
        num_cameras: Cameras the device carries; all but ``cam0``'s video are excluded.
        follow: The device's forward and up in the rig frame; it orients the Follow view.
    """
    return blueprints.table_blueprint(
        num_cameras,
        rig=RIG,
        run_source=schema.GT_RUN_SOURCE,
        eye_controls=follow_eye(follow),
        front_pane=blueprints.camera_view("cam0", RIG, 0),
    )


@dataclass(frozen=True, slots=True)
class SequencePaths:
    """Where one sequence's conversion reads and writes; ``convert`` works these out once."""

    archives: tuple[Path, ...]
    """Local archive volume paths, parts first and the closing ``.zip`` last."""
    work_dir: Path
    """Scratch directory for the encoded mp4s and any extraction; deleted unless ``--keep-raw``."""
    target: Path
    """Final base-layer rrd path."""
    gt_target: Path
    """Final gt-layer rrd path, a sibling of the same recording id."""


@dataclass(frozen=True, slots=True)
class SequenceStreams:
    """Everything one archive read yields; both layers are written from it and nothing else.

    Every clock here is already zero-based ``video_time`` — the archive's device
    clock minus ``start_time_ns`` — because the two layers must share one origin
    and the shift is a property of the sequence, not of either layer.
    """

    camera_times_ns: tuple[Int64[ndarray, "n_samples"], ...]
    """Frame times per camera, in ``cam0``…``camN`` order."""
    clips: tuple[Path, ...]
    """The encoded mp4 of each camera, in the same order."""
    gyro: ImuChannel
    """Angular velocity in rad/s."""
    accel: ImuChannel
    """Linear acceleration in m/s^2; also what the world-up measurement reads."""
    magnetometer: ImuChannel | None
    """Field samples in the sensor's own units, or ``None`` on a device without one."""
    gt: GtTrajectory
    """Ground truth, quaternions already xyzw and dropouts already repaired."""
    start_time_ns: int
    """The device-clock origin every stream above was shifted by; a recording property."""
    duration_ns: int
    """Span of the *sensor* streams; deliberately not bounded by gt."""

    @property
    def num_frames(self) -> int:
        """Longest per-camera sample count, which is what the capture properties report."""
        return max(times_ns.size for times_ns in self.camera_times_ns)


class MsdDataset(DataforgeDataset[MsdConfig, MsdSource]):
    """Converts one MSD headset's sequences into a base and a gt layer each."""

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
                            collection=leaf,
                            sequence=stem,
                            archive_paths=tuple(path for path, _ in volumes),
                            archive_bytes=sum(size for _, size in volumes),
                        ),
                    )
                )
        return pairs

    def default_blueprint(self) -> rrb.Blueprint:
        """Device-wide layout: every sequence of one headset has the same sensors."""
        return build_blueprint(self.device.num_cameras, has_magnetometer=self.device.has_magnetometer, follow=self.device.follow)

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card for the device's segment table."""
        return build_table_blueprint(self.device.num_cameras, follow=self.device.follow)

    @property
    def calibration_path(self) -> str:
        """Repo-relative path of the device's basalt calibration."""
        return f"{REPO_ROOT}/{self.device.hf_dir}/extras/calibration.json"

    @functools.cached_property
    def revision(self) -> str | None:
        """Repo commit sha stamped into every rrd, resolved once per dataset instance.

        Lazily, and deliberately not inside a recording: a batch run asks the hub
        once instead of once per sequence, and ``convert`` warms it before it
        encodes anything, so a transient network failure cannot land on a
        finished encode.
        """
        return repo_revision(REPO_ID, self.config.revision)

    def fetch_calibration(self) -> Path:
        """Fetch the device's ``calibration.json`` into ``root`` and return where it landed.

        The one place the calibration is pulled from the hub: ``download`` calls
        it to have the file up front, and ``calibration()`` calls it when a
        ``convert`` runs against a scratch dir that ``download`` never touched.
        """
        transports.hf_fetch(REPO_ID, allow_patterns=[self.calibration_path], local_dir=self.config.root, revision=self.config.revision)
        return self.config.root / self.calibration_path

    def calibration(self) -> BasaltCalibration:
        """Load the device calibration, fetching it if ``download`` was skipped."""
        local_path: Path = self.config.root / self.calibration_path
        if not local_path.is_file():
            local_path = self.fetch_calibration()
        return serde.json.from_json(BasaltCalibration, local_path.read_text())

    def download(self) -> None:
        """Fetch the calibration, prove the machine can encode, and print the plan.

        Deliberately **not** a bulk fetch: the corpus is hundreds of gigabytes of
        PNG archives and ``convert`` pulls one sequence at a time, so the only
        thing worth having up front is the few-kilobyte calibration. The NVENC
        check is here because a machine that cannot encode AV1 should find out
        now rather than after the first multi-gigabyte download.
        """
        local_calibration: Path = self.fetch_calibration()
        require_av1_nvenc(resolve_ffmpeg())
        discovered: list[tuple[SequenceIdentity, MsdSource]] = self.discover()
        per_collection: dict[str, list[MsdSource]] = {}
        for _, source in discovered:
            per_collection.setdefault(source.collection, []).append(source)
        total_bytes: int = sum(source.archive_bytes for _, source in discovered)
        print(f"{self.config.name}: {len(discovered)} sequence(s), {total_bytes / 1e9:.1f} GB of archives to stream through convert")
        for collection, sources in per_collection.items():
            print(f"  {collection}: {len(sources)} sequence(s), {sum(source.archive_bytes for source in sources) / 1e9:.1f} GB")
        print(f"  calibration → {local_calibration}; archives are fetched and deleted one sequence at a time")

    def sequence_paths(self, identity: SequenceIdentity, source: MsdSource) -> SequencePaths:
        """Everywhere this sequence's conversion touches disk, worked out in one place."""
        output_root: Path = paths.output_root()
        return SequencePaths(
            archives=tuple(self.config.root / archive_path for archive_path in source.archive_paths),
            work_dir=self.config.root / "work" / source.sequence,
            target=paths.rrd_path(output_root, layer=paths.BASE_LAYER, identity=identity),
            gt_target=paths.rrd_path(output_root, layer=paths.GT_LAYER, identity=identity),
        )

    def enforce_raw_budget(self, source: MsdSource, archives: Sequence[Path]) -> None:
        """Refuse to fetch when ``root`` plus this sequence would breach the budget.

        The cap covers this sequence's archives **plus** whatever else is already
        in ``root``. Two things ride on top of it and are not counted: one
        camera's extracted PNGs (split archives only) and the temp mp4s, both of
        which live in ``work/`` for the length of one sequence.

        A sequence whose archives *alone* exceed the budget (the Index and G2 long
        sessions) is an accepted exception and only warns — there is no smaller
        unit to convert — but its leftovers are still tallied, because leftovers
        that breach the cap on their own are an error whatever is being fetched.
        HuggingFace's own bookkeeping under ``root/.cache`` (including resumable
        ``.incomplete`` blobs) is excluded: it is not a leftover anyone should delete.

        Args:
            source: Sequence about to be fetched.
            archives: Its local volume paths, excluded from the on-disk tally so
                a retry of an already-downloaded sequence is not double-counted.

        Raises:
            RuntimeError: What is in ``root`` would push the fetch over the budget.
        """
        budget_bytes: int = int(self.config.raw_budget_gb * 1e9)
        oversized: bool = source.archive_bytes > budget_bytes
        if oversized:
            print(
                f"  warning: {source.sequence} needs {source.archive_bytes / 1e9:.1f} GB of archives, above the "
                f"{self.config.raw_budget_gb:g} GB raw budget; converting it anyway as an accepted exception"
            )
        if not self.config.root.is_dir():
            return

        # One walk, one stat per file: this runs before every fetch, and a scratch
        # dir mid-batch holds a whole sequence's PNGs.
        cache_dir: Path = self.config.root / ".cache"
        own: set[Path] = set(archives)
        leftovers: list[tuple[Path, int]] = []
        for directory, subdirectories, names in os.walk(self.config.root):
            here: Path = Path(directory)
            if here == cache_dir:
                subdirectories.clear()
                continue
            for name in names:
                path: Path = here / name
                if path not in own:
                    leftovers.append((path, os.stat(path).st_size))
        occupied: int = sum(size for _, size in leftovers)
        # An oversized sequence has already been waved through, so only its
        # leftovers are still weighed against the cap.
        needed: int = 0 if oversized else source.archive_bytes
        if occupied + needed <= budget_bytes:
            return
        biggest: list[tuple[Path, int]] = sorted(leftovers, key=lambda entry: entry[1], reverse=True)[:3]
        named: str = ", ".join(f"{path.relative_to(self.config.root)} ({size / 1e9:.2f} GB)" for path, size in biggest)
        raise RuntimeError(
            f"{self.config.root} already holds {occupied / 1e9:.2f} GB of leftovers and {source.sequence} needs "
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
        locations: SequencePaths = self.sequence_paths(identity, source)
        if writing.should_skip(locations.target, force=force) and writing.should_skip(locations.gt_target, force=force):
            print(f"skip {identity.sequence_key} → {locations.target} + {locations.gt_target}")
            return locations.target

        self.enforce_raw_budget(source, locations.archives)
        # Both hub lookups happen before the archives are pulled and long before a
        # frame is encoded: neither a bad calibration nor a transient revision
        # lookup may throw away a multi-gigabyte download and an hour of encoding.
        calibration: BasaltCalibration = self.calibration()
        _ = self.revision

        on_disk: int = sum(1 for archive in locations.archives if archive.is_file())
        if on_disk:
            print(f"  {on_disk}/{len(locations.archives)} archive volume(s) already in {self.config.root}; the fetch only verifies them")
        transports.hf_fetch(REPO_ID, allow_patterns=list(source.archive_paths), local_dir=self.config.root, revision=self.config.revision)

        locations.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open_member_reader(locations.archives, locations.work_dir) as reader:
                streams: SequenceStreams = self.read_sequence(reader, source, work_dir=locations.work_dir)
            measured: MeasuredUp = measured_world_up(streams.gt, streams.accel)
            self.warn_on_device_claims(source, calibration=calibration, measured=measured)
            self.write_base_layer(identity, source, streams, locations.target, calibration=calibration)
            self.write_gt_layer(identity, streams, locations.gt_target, measured=measured)
        except BaseException:
            paths.remove_tree(locations.work_dir)
            retained: int = sum(archive.stat().st_size for archive in locations.archives if archive.is_file())
            print(f"  kept {retained / 1e9:.2f} GB of archives in {self.config.root} so a retry skips the download")
            raise

        if self.config.keep_raw:
            print(f"  keeping raw: {len(locations.archives)} archive volume(s) and the mp4s in {locations.work_dir}")
        else:
            paths.remove_tree(locations.work_dir)
            for archive in locations.archives:
                archive.unlink(missing_ok=True)
        print(
            f"done {identity.sequence_key} → {locations.target} + {locations.gt_target} "
            f"({self.device.num_cameras} cameras, {streams.num_frames} frames, {streams.gt.times_ns.size} gt poses, "
            f"world up {measured.axis} at {measured.fraction:.2f} g)"
        )
        return locations.target

    def read_sequence(self, reader: MemberReader, source: MsdSource, *, work_dir: Path) -> SequenceStreams:
        """Read every csv and encode every camera: the whole archive, in one pass.

        The csvs come first so a sequence that is missing a stream fails before
        the expensive half, and the mp4s all exist before any of them is remuxed
        — a split archive's reader extracts one camera at a time, so a converter
        cannot hold a reader open across the two.

        Args:
            reader: Open reader over the sequence's archive volume(s).
            source: The discovered sequence, whose stem is the archive's top directory.
            work_dir: Scratch directory the encoded mp4s are written into.

        Raises:
            ValueError: A camera index, ``imu0/data.csv`` or ``gt/data.csv`` holds
                no data rows, so the sequence has no clock to place the rest on.
        """
        mav0: str = f"{source.sequence}/mav0"
        camera_rows: list[list[CameraRow]] = []
        for index in range(self.device.num_cameras):
            rows: list[CameraRow] = read_camera_index(reader.csv_bytes(f"{mav0}/cam{index}/data.csv"))
            if not rows:
                raise ValueError(f"{source.sequence} cam{index} has an empty data.csv")
            camera_rows.append(rows)
        camera_times_ns: list[Int64[ndarray, "n_samples"]] = [
            np.array([row.timestamp_ns for row in rows], dtype=np.int64) for rows in camera_rows
        ]
        inertial: TimestampedSamples = read_numeric_csv(reader.csv_bytes(f"{mav0}/imu0/data.csv"), num_values=IMU_VALUE_COLUMNS)
        if inertial.times_ns.size == 0:
            raise ValueError(f"{source.sequence} imu0/data.csv has no data rows, so the sequence has no inertial stream")
        magnetometer: TimestampedSamples | None = (
            read_numeric_csv(reader.csv_bytes(f"{mav0}/mag0/data.csv"), num_values=MAG_VALUE_COLUMNS) if self.device.has_magnetometer else None
        )
        # The gt *layer* is a sibling rrd, but both layers share one zero-based
        # video_time, so the base layer has to know gt's clock origin too.
        gt: GtTrajectory = gt_trajectory(read_numeric_csv(reader.csv_bytes(f"{mav0}/gt/data.csv"), num_values=GT_VALUE_COLUMNS))

        sensor_times_ns: list[Int64[ndarray, "n_samples"]] = [*camera_times_ns, inertial.times_ns]
        if magnetometer is not None and magnetometer.times_ns.size:
            sensor_times_ns.append(magnetometer.times_ns)
        start_time_ns: int = min(int(gt.times_ns[0]), *(int(times_ns[0]) for times_ns in sensor_times_ns))
        # Deliberately not bounded by gt: duration_ns describes the *sensor* layer.
        duration_ns: int = max(int(times_ns[-1]) for times_ns in sensor_times_ns) - start_time_ns

        clips: list[Path] = []
        for index, (rows, times_ns) in enumerate(zip(camera_rows, camera_times_ns, strict=True)):
            clip: Path = work_dir / f"cam{index}.mp4"
            encode_frames_to_mp4(
                reader.png_frames([f"{mav0}/cam{index}/data/{row.filename}" for row in rows]),
                clip,
                source=FrameSource("png"),
                fps=nominal_fps(times_ns),
            )
            clips.append(clip)

        inertial_times_ns: Int64[ndarray, "n_samples"] = inertial.times_ns - start_time_ns
        return SequenceStreams(
            camera_times_ns=tuple(times_ns - start_time_ns for times_ns in camera_times_ns),
            clips=tuple(clips),
            gyro=ImuChannel(times_ns=inertial_times_ns, values_xyz=inertial.values[:, :3]),
            accel=ImuChannel(times_ns=inertial_times_ns, values_xyz=inertial.values[:, 3:6]),
            magnetometer=(
                None if magnetometer is None else ImuChannel(times_ns=magnetometer.times_ns - start_time_ns, values_xyz=magnetometer.values)
            ),
            gt=replace(gt, times_ns=gt.times_ns - start_time_ns),
            start_time_ns=start_time_ns,
            duration_ns=duration_ns,
        )

    def warn_on_device_claims(self, source: MsdSource, *, calibration: BasaltCalibration, measured: MeasuredUp) -> None:
        """Re-check both per-device claims against this sequence and say so on a disagreement.

        Neither claim is re-applied: every rrd of a device must carry the same
        world axes and the same follow eye, so one sequence disagreeing is news,
        not a reason to reorient that rrd alone. See ``MSD_DEVICES``.
        """
        # The declared axes are rounded to three decimals and so are a hair short of
        # unit length; dividing by the norms keeps that rounding out of the angles.
        derived: FollowFrame = follow_frame(calibration)
        declared_axes: Float64[ndarray, "2 3"] = np.array([self.device.follow.forward, self.device.follow.up], dtype=np.float64)
        derived_axes: Float64[ndarray, "2 3"] = np.array([derived.forward, derived.up], dtype=np.float64)
        cosines: Float64[ndarray, "2"] = np.einsum("ij,ij->i", declared_axes, derived_axes) / np.linalg.norm(declared_axes, axis=1)
        deviations_deg: Float64[ndarray, "2"] = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
        if float(deviations_deg.max()) > FOLLOW_FRAME_TOLERANCE_DEG:
            print(
                f"  warning: {self.config.device} declares a follow frame its calibration disagrees with — "
                f"forward off by {deviations_deg[0]:.1f} deg, up off by {deviations_deg[1]:.1f} deg, over the "
                f"{FOLLOW_FRAME_TOLERANCE_DEG:g} deg tolerance; the blueprint still uses the declared frame"
            )
        if measured.axis != self.device.world_up:
            print(
                f"  warning: {self.config.device} declares world_up {self.device.world_up} but {source.sequence} "
                f"measured {measured.axis} carrying {measured.fraction:.2f} of |g|; the rrd still states the declared axis"
            )

    def write_base_layer(
        self, identity: SequenceIdentity, source: MsdSource, streams: SequenceStreams, target: Path, *, calibration: BasaltCalibration
    ) -> int:
        """Write the sensor layer: every camera's video, the IMU, the magnetometer.

        Returns:
            Frames the longest camera holds, which is the recording's ``num_frames``.
        """
        with writing.atomic_recording(target, recording_id=identity.recording_id, default_blueprint=self.default_blueprint()) as recording:
            # Deliberately NO ViewCoordinates at "/": the gt layer owns the root
            # ViewCoordinates, because it is what establishes a world frame at all.
            log_rig_node(recording, RIG, reference=RIG_REFERENCE, num_cameras=self.device.num_cameras, name=self.device.label, kind="ego")
            for index, (clip, times_ns) in enumerate(zip(streams.clips, streams.camera_times_ns, strict=True)):
                # The model tag saves a consumer from inferring the projection from the
                # distortion component. ``rpmax`` is radtan8's own key, so it is already
                # None on a kb4 camera and AnyValues leaves the key off entirely.
                basalt_camera: BasaltCamera = calibration.value0.intrinsics[index]
                log_camera_node(
                    recording,
                    RIG,
                    index,
                    camera_parameters(calibration, index, name=f"cam{index}"),
                    name=f"cam{index}",
                    kind="grayscale",
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                    camera_model=basalt_camera.camera_type,
                    distortion_valid_radius=basalt_camera.intrinsics.rpmax,
                )
                log_video_stream(recording, clip, schema.video_path(RIG, index), times_ns=times_ns)
            log_imu(recording, RIG, IMU, gyro=streams.gyro, accel=streams.accel, name="imu0")
            if streams.magnetometer is not None:
                log_magnetometer(recording, RIG, MAG, field=streams.magnetometer, name="mag0")
            writing.send_capture_properties(
                recording,
                identity,
                num_cameras=self.device.num_cameras,
                num_frames=streams.num_frames,
                start_time_ns=streams.start_time_ns,
                device=self.config.device,
                device_label=self.device.label,
                collection=source.collection,
                hf_revision=self.revision,
                duration_ns=streams.duration_ns,
            )
        return streams.num_frames

    def write_gt_layer(self, identity: SequenceIdentity, streams: SequenceStreams, gt_target: Path, *, measured: MeasuredUp) -> None:
        """Write the ground-truth layer: ``world_T_rig`` at the full rate, its path and its trail.

        Its own recording stream and its own atomic replace, opened only after
        the base layer has closed, so a failure here cannot half-write that one.
        The recording id is the base layer's, which is what makes the catalog
        stack the two onto one segment.
        """
        gt: GtTrajectory = streams.gt
        with writing.atomic_recording(gt_target, recording_id=identity.recording_id) as recording:
            # The gt layer establishes a world frame at all, so it — not the base
            # layer — owns the root ViewCoordinates. The axis is the device's
            # declared one, not this sequence's measurement: every rrd of a device
            # must agree, and a disagreement is a warning, not a silent
            # per-sequence reorientation.
            rr.log("/", WORLD_UP_VIEW_COORDINATES[self.device.world_up], static=True, recording=recording)
            log_pose_track(
                recording,
                schema.rig_path(RIG),
                times_ns=gt.times_ns,
                translations_xyz=gt.translations_xyz,
                quaternions_xyzw=gt.quaternions_xyzw,
            )
            # Two views of one trajectory: the static strip is the whole path for the
            # overview, and the per-pose points are what the blueprint's cursor-relative
            # time range turns into a recent-motion trail in the follow view.
            rr.log(
                schema.trajectory_path(schema.GT_RUN_SOURCE),
                rr.LineStrips3D([gt.translations_xyz], colors=GT_TRAJECTORY_COLOR, radii=GT_TRAJECTORY_RADIUS_M),
                static=True,
                recording=recording,
            )
            rr.log(
                schema.trail_path(schema.GT_RUN_SOURCE),
                rr.Points3D.from_fields(colors=GT_TRAIL_COLOR, radii=GT_TRAIL_RADIUS_M),
                static=True,
                recording=recording,
            )
            rr.send_columns(
                schema.trail_path(schema.GT_RUN_SOURCE),
                indexes=[time_column(gt.times_ns)],
                columns=rr.Points3D.columns(positions=gt.translations_xyz),
                recording=recording,
            )
            recording.send_recording_name(identity.recording_id)
            recording.send_property(
                "gt",
                rr.AnyValues(
                    num_poses=int(gt.times_ns.size),
                    duration_ns=int(gt.times_ns[-1] - gt.times_ns[0]),
                    num_sanitized=gt.num_sanitized,
                    source=self.device.gt_source,
                    world_up=self.device.world_up,
                    measured_up=measured.axis,
                    measured_up_fraction=measured.fraction,
                ),
            )
