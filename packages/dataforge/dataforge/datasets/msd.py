"""Monado SLAM Datasets (MSD): reading one sequence archive — calibration, csvs, frames.

Upstream is the HuggingFace dataset ``collabora/monado-slam-datasets`` (CC-BY 4.0),
a tree of per-sequence zip archives::

    M_monado_datasets/<device_dir>/<collection>/<SEQ>.zip     (or .z01 .z02 … .zip)
    M_monado_datasets/<device_dir>/extras/calibration.json

Inside an archive: ``<SEQ>/mav0/cam<N>/data/<ts>.png`` grayscale frames plus a
``cam<N>/data.csv`` index, ``imu0/data.csv`` (~1 kHz), ``gt/data.csv`` (~1 kHz),
and on the Reverb G2 / Odyssey+ a 50 Hz ``mag0/data.csv``. Where a stream also
ships ``data.raw.csv`` / ``data.extra.csv`` siblings, the converter reads only
``data.csv``.

**Clocks.** Every csv timestamp is nanoseconds on one monotonic device clock
(values around 1e13, not a Unix epoch). ``video_time`` is that clock minus ``t0``,
the earliest sample of any stream *including* ``gt`` — the ground-truth layer
(a sibling rrd) has to share this origin, so its first row is read here even
though no gt is logged. Nothing is resampled.

**Frames.** ``rig_T_cam`` comes from basalt's ``T_imu_cam``, the camera pose in
the IMU frame; the rig frame *is* the IMU frame (``reference = "imu_00"``), so
``world`` in the ``Extrinsics`` below means the rig. This is the same convention
as simplecv's RoboCap loader, one inversion away (Kalibr states ``T_cam_imu``).
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Literal, TypeAlias

import numpy as np
import serde
import serde.json
from jaxtyping import Float64, Int64
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

CAMERA_TIMESTAMP_FIELD: str = "#timestamp [ns]"
"""Header cell every MSD csv starts with; the gt file pads the later cells with spaces."""

CameraModel: TypeAlias = Literal["kb4", "pinhole-radtan8"]
"""``camera_type`` values basalt writes in ``calibration.json``; MSD uses no others."""


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


def first_timestamp_ns(data: bytes) -> int:
    """First data row's timestamp, without parsing the rest of the file.

    The ``gt`` csv is read only for this: the ground-truth *layer* is a sibling
    rrd, but both layers must be on the same zero-based ``video_time``, so its
    clock origin has to be known here.

    Args:
        data: Whole csv, as read out of the archive.

    Raises:
        ValueError: The file holds a header and nothing else.
    """
    reader: Iterator[list[str]] = csv.reader(io.StringIO(data.decode()))
    next(reader, None)
    first: list[str] | None = next(reader, None)
    if first is None:
        raise ValueError("csv has no data rows, so it cannot contribute a clock origin")
    return int(first[0])


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
