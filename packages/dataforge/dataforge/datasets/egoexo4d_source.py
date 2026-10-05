"""Ego-Exo4D raw tree readers: take metadata, GoPro calibration, the take clock and the Aria trajectory.

Layout as the official ``egoexo`` CLI writes it under the raw root: ``takes.json``,
``takes/<take>/frame_aligned_videos/*.mp4``, ``takes/<take>/trajectory/{gopro_calibs,closed_loop_trajectory}.csv``,
``takes/<take>/<aria>_noimagestreams.vrs`` and ``captures/<capture>/timesync.csv``. Field meanings follow the
public docs (docs.ego-exo4d-data.org) and projectaria-tools' MPS readers.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import SerdeError, coerce, from_dict, serde
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion

from dataforge import aria
from dataforge.records import read_csv_columns, read_json

FPS: int = 30
"""Rate of every frame-aligned video and of the HM fits."""
EXO_SIZE: tuple[int, int] = (1920, 1080)
"""Stored size of a landscape GoPro (native 3840x2160): the size the HM fits were made at."""
KB4: str = "kb4"
"""``camera_model`` of the GoPros: OpenCV fisheye, gopro_calibs' ``KANNALABRANDTK3`` (four radial terms)."""


@serde
@dataclass(frozen=True, slots=True)
class AlignedVideo:
    """One frame-aligned stream of a camera (unknown keys allowed: the release's schema, not ours)."""

    relative_path: str
    """Path under the take's ``root_dir``."""


@serde
@dataclass(frozen=True, slots=True)
class CaptureCamera:
    """One device of the capture."""

    cam_id: str
    """``cam01`` … for GoPros, ``aria01`` … for the headset."""
    is_ego: bool
    """True for the wearer's Aria."""


@serde
@dataclass(frozen=True, slots=True)
class TakeCapture:
    """The capture a take was cut from."""

    capture_name: str
    """Directory under ``captures/`` holding ``timesync.csv``."""
    cameras: list[CaptureCamera]
    """Every device of the capture."""


@serde
@dataclass(frozen=True, slots=True)
class Take:
    """One ``takes.json`` entry."""

    take_name: str
    """Human id, also the HM release's folder name."""
    take_uid: str
    """The id the release manifests key take files by."""
    root_dir: str
    """``takes/<take_name>`` under the raw root."""
    capture_uid: str
    """The id the ``captures`` manifest keys timesync by."""
    timesync_start_idx: int
    """Row of ``timesync.csv`` holding the take's first frame."""
    timesync_end_idx: int
    """Row one past the take's last frame (HM's convention; see ``take_rows``)."""
    capture: TakeCapture
    """The capture and its devices."""
    frame_aligned_videos: dict[str, dict[str, AlignedVideo]]
    """Streams by camera id, then readable stream id (``0`` for a GoPro; ``rgb``, ``slam-left``, ``slam-right``, ``et``)."""
    task_name: str | None = None
    """Task, e.g. ``Remove a Wheel``."""
    parent_task_name: str | None = None
    """Activity, e.g. ``Bike Repair``."""
    university_name: str | None = None
    """Recording lab."""

    @property
    def aria(self) -> str:
        """The wearer's Aria camera id (``aria01``)."""
        ids: list[str] = [camera.cam_id for camera in self.capture.cameras if camera.is_ego and camera.cam_id.startswith("aria")]
        if len(ids) != 1:
            raise ValueError(f"{self.take_name}: expected one ego Aria, found {ids}")
        return ids[0]

    def video(self, cam_id: str, stream: str) -> str:
        """Raw-root-relative path of one frame-aligned stream."""
        try:
            return f"{self.root_dir}/{self.frame_aligned_videos[cam_id][stream].relative_path}"
        except KeyError:
            raise ValueError(f"{self.take_name}: takes.json lists no frame-aligned {cam_id}/{stream} video") from None


def read_takes(path: Path) -> dict[str, Take]:
    """Every take in ``takes.json``, keyed by take name."""
    return {take.take_name: take for take in read_json(path, list[Take])}


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class GoproCalib:
    """One ``gopro_calibs.csv`` row: a static GoPro's world pose and KB4 lens at native resolution."""

    cam_uid: str
    tx_world_cam: float
    ty_world_cam: float
    tz_world_cam: float
    qx_world_cam: float
    qy_world_cam: float
    qz_world_cam: float
    qw_world_cam: float
    image_width: int
    image_height: int
    intrinsics_type: str
    intrinsics_0: float
    """fx"""
    intrinsics_1: float
    """fy"""
    intrinsics_2: float
    """cx"""
    intrinsics_3: float
    """cy"""
    intrinsics_4: float
    """k1"""
    intrinsics_5: float
    """k2"""
    intrinsics_6: float
    """k3"""
    intrinsics_7: float
    """k4"""
    quality: float
    """1 localized, 0 bad, -1 unknown."""

    def __post_init__(self) -> None:
        if self.intrinsics_type != "KANNALABRANDTK3":
            raise ValueError(f"{self.cam_uid}: intrinsics_type {self.intrinsics_type!r}, expected KANNALABRANDTK3")

    @property
    def world_T_cam(self) -> Float64[ndarray, "4 4"]:
        """Camera-to-world pose."""
        transform: Float64[ndarray, "4 4"] = np.eye(4)
        transform[:3, :3] = Rotation.from_quat([self.qx_world_cam, self.qy_world_cam, self.qz_world_cam, self.qw_world_cam]).as_matrix()
        transform[:3, 3] = [self.tx_world_cam, self.ty_world_cam, self.tz_world_cam]
        return transform

    def camera(self, width: int, height: int) -> Fisheye62Parameters:
        """The lens rescaled to a stored ``width`` x ``height`` image; the KB4 terms act on angles and do not scale."""
        scale: float = width / self.image_width
        if abs(height / self.image_height - scale) > 1e-9:
            raise ValueError(f"{self.cam_uid}: {width}x{height} is not a uniform rescale of {self.image_width}x{self.image_height}")
        pose: Float64[ndarray, "4 4"] = self.world_T_cam
        return Fisheye62Parameters(
            name=self.cam_uid,
            extrinsics=Extrinsics(world_R_cam=pose[:3, :3], world_t_cam=pose[:3, 3]),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RDF",
                fl_x=self.intrinsics_0 * scale,
                fl_y=self.intrinsics_1 * scale,
                cx=self.intrinsics_2 * scale,
                cy=self.intrinsics_3 * scale,
                width=width,
                height=height,
            ),
            distortion=KannalaBrandtDistortion(k1=self.intrinsics_4, k2=self.intrinsics_5, k3=self.intrinsics_6, k4=self.intrinsics_7),
        )


def read_gopro_calibs(path: Path) -> list[GoproCalib]:
    """Every row of a take's ``gopro_calibs.csv``, in file order (the order HM numbers its views in)."""
    with path.open(newline="") as stream:
        rows: list[dict[str, str]] = [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(stream)]
    try:
        return [from_dict(GoproCalib, row) for row in rows]
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def localized(calibs: list[GoproCalib]) -> list[GoproCalib]:
    """The GoPros Ego-Exo4D localized (quality 1): the exo cameras HM fit with, in file order."""
    return [calib for calib in calibs if calib.quality == 1.0]


def stored_size(calib: GoproCalib) -> tuple[int, int]:
    """``EXO_SIZE``, portrait for a portrait GoPro."""
    return EXO_SIZE if calib.image_width >= calib.image_height else (EXO_SIZE[1], EXO_SIZE[0])


def take_rows(take: Take) -> range:
    """timesync.csv data rows of the take's frames, frame ``i`` at row ``start + i``.

    The public sources disagree on the end bound (inclusive in the metadata docs, exclusive in HM and the
    official pose code); this follows HM, whose fits index the frames. ``convert`` checks the count
    against every video.
    """
    return range(take.timesync_start_idx, take.timesync_end_idx)


class TakeClock(NamedTuple):
    """The take's frame times and how many of them repeat an earlier stamp."""

    times_ns: Int64[ndarray, "n"]
    """Aria RGB capture time of each frame (device clock), a missing stamp replaced by the last one."""
    filled: int
    """Frames whose stamp was missing."""


def read_take_clock(path: Path, take: Take) -> TakeClock:
    """The take's frame times: the Aria RGB capture timestamps (device clock, ns), gaps forward-filled.

    Raises:
        ValueError: The column is missing, the first frame has no timestamp, or the rows run past the file.
    """
    column: str = f"{take.aria}_{aria.RGB_STREAM_ID}_capture_timestamp_ns"
    table: pa.Table = read_csv_columns(path, {column: pa.float64()})  # float: a missing stamp is an empty cell
    rows: range = take_rows(take)
    if rows.stop > table.num_rows:
        raise ValueError(f"{path}: take {take.take_name} needs rows {rows.start}..{rows.stop - 1}, the file has {table.num_rows}")
    stamps: Float64[ndarray, "n"] = table.column(column).to_numpy(zero_copy_only=False)[rows.start : rows.stop]
    present: Bool[ndarray, "n"] = np.isfinite(stamps)
    if not present[0]:
        raise ValueError(f"{path}: take {take.take_name} has no {column} on its first frame")
    filled: Int64[ndarray, "n"] = np.maximum.accumulate(np.where(present, np.arange(len(stamps)), 0))
    return TakeClock(stamps[filled].astype(np.int64), int((~present).sum()))
