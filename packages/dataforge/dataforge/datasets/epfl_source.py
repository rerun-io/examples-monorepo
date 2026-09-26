"""Sequential EPFL sidecar readers; no writes to the source tree."""

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched, islice, zip_longest
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import SerdeError, coerce, from_dict, serde

from dataforge import hands, schema

SOURCE_REVISION: str = "b91f2df8027e4d7e529939e78f856463f64a58cd"
CLOCK_SOURCE: str = "timestamps.txt (Azure Kinect device clock, µs), one row per RGB frame; row i = video frame i = pose row i"
BATCH_SIZE: int = 64
CAMERA_NAMES: tuple[str, ...] = ("output0", *(f"{bank}output{i}" for bank in ("A", "B") for i in range(4)), "hololens")
"""Numeric camera order used by upstream; HoloLens follows nine exo cameras."""
EXO_CAMERAS: tuple[str, ...] = CAMERA_NAMES[:-1]
EXO_RIGS: tuple[int, ...] = tuple(range(len(EXO_CAMERAS)))
EGO_RIG: int = len(EXO_CAMERAS)
EXO_SIZE: tuple[int, int] = (1280, 720)
"""Exo RGB width and height."""
HOLO_SIZE: tuple[int, int] = (896, 504)
"""HoloLens RGB width and height."""
HAND_L2_THRESHOLD: float = 0.06
"""Upstream rejects MANO fits whose residual reaches this value."""
BODY_L2_THRESHOLD: float = 0.09
"""Upstream rejects SMPL fits whose residual reaches this value."""


def read_timestamps(path: Path) -> Int64[ndarray, "n"]:
    """Preserve the unshifted device clock, including skipped device ticks."""
    with path.open() as handle:
        times = np.fromiter((int(line) * 1000 for line in handle), dtype=np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: timestamps must be nonempty and increasing")
    return times


@serde
@dataclass(frozen=True, slots=True)
class ExoCamera:
    """Static RGB calibration; unknown depth fields are outside this port."""

    K: Float64[ndarray, "3 3"]
    """RGB intrinsics."""
    dist: Float64[ndarray, "8"]
    """Full rational distortion coefficients."""
    word2cam: Float64[ndarray, "4 4"]
    """World to static camera, with the release's spelling."""
    size: ClassVar[tuple[int, int]] = EXO_SIZE
    """RGB width and height."""
    camera_model: ClassVar[str] = "OpenCV rational Brown-Conrady (8 coefficients)"
    """Projection model for the shipped distortion."""


@serde
@dataclass(frozen=True, slots=True)
class EgoCamera:
    """HoloLens intrinsics; its moving pose comes from a separate CSV."""

    K: Float64[ndarray, "3 3"]
    """RGB intrinsics."""
    dist: Float64[ndarray, "8"]
    """Shipped zero distortion coefficients."""
    size: ClassVar[tuple[int, int]] = HOLO_SIZE
    """RGB width and height."""
    camera_model: ClassVar[str] = "pinhole"
    """Undistorted HoloLens projection."""


def read_cameras(path: Path) -> tuple[dict[str, ExoCamera], EgoCamera]:
    """Decode the nine static cameras and separate HoloLens calibration."""
    try:
        raw = json.loads(path.read_text())
        return {name: from_dict(ExoCamera, raw[name]) for name in EXO_CAMERAS}, from_dict(EgoCamera, raw["hololens"])
    except (SerdeError, json.JSONDecodeError, KeyError) as error:
        raise ValueError(f"{path}: invalid camera calibration: {error}") from error


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Source layout, rejection rule and destinations of one released fit."""

    name: Literal["left", "right", "body"]
    """Fit name on a pose row."""
    file: Literal["pose3d_mano.csv", "pose3d_smpl.csv"]
    """Source CSV basename."""
    prefix: str
    """Prefix of parameter columns."""
    residual: str
    """Residual column name."""
    pose_size: int
    """Number of axis-angle parameters."""
    threshold: float
    """Residual at which the fit is rejected."""
    coco: slice
    """Destination slots in COCO-133."""
    source: slice
    """Joint rows in the source CSV."""
    parameters_path: str
    """Raw parameter entity."""
    mesh_path: str
    """Derived mesh entity."""
    albedo: tuple[int, int, int, int]
    """Mesh RGBA color."""


FITS: tuple[FitSpec, ...] = (
    FitSpec(
        "left",
        "pose3d_mano.csv",
        "left_",
        "l2_dist_left",
        48,
        HAND_L2_THRESHOLD,
        slice(91, 112),
        slice(0, 21),
        schema.hand_mano_path("left"),
        schema.hand_mesh_path("left"),
        hands.HAND_ALBEDO["left"],
    ),
    FitSpec(
        "right",
        "pose3d_mano.csv",
        "right_",
        "l2_dist_right",
        48,
        HAND_L2_THRESHOLD,
        slice(112, 133),
        slice(21, 42),
        schema.hand_mano_path("right"),
        schema.hand_mesh_path("right"),
        hands.HAND_ALBEDO["right"],
    ),
    FitSpec(
        "body",
        "pose3d_smpl.csv",
        "",
        "l2_dist",
        72,
        BODY_L2_THRESHOLD,
        slice(0, 17),
        slice(0, 17),
        "/world/gt/body/smpl",
        "/world/gt/body/mesh",
        (160, 190, 200, 110),
    ),
)
"""The three independent fits in each matched pair of source rows."""


@serde
@dataclass(frozen=True, slots=True)
class FitParameters:
    """One shipped model fit, preserving its raw parameters and residual."""

    poses: Float32[ndarray, "p"]
    """Full axis-angle pose with root placeholder."""
    Rh: Float32[ndarray, "3"]
    """Root rotation."""
    Th: Float32[ndarray, "3"]
    """Origin-pivot translation in metres."""
    shapes: Float32[ndarray, "10"]
    """Per-row shape coefficients."""
    l2_dist: float
    """Shipped residual, NaN when empty."""


@dataclass(frozen=True, slots=True)
class Fit:
    """Decoded parameters with the source fit specification."""

    parameters: FitParameters
    """Raw source values, including rejected parameters."""
    spec: FitSpec
    """Source layout and rejection rule."""

    @property
    def accepted(self) -> bool:
        """Upstream rejects empty, nonfinite, and threshold-or-larger residuals."""
        return bool(np.isfinite(self.parameters.l2_dist) and self.parameters.l2_dist < self.spec.threshold)


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class Measurements:
    """Typed measurement columns at the CSV boundary."""

    rgb_frameid: int
    """Original capture frame number."""
    kp3ds: Float32[ndarray, "j 3"]
    """Shipped world joints in metres."""
    kp3ds_conf: Float32[ndarray, "j"]
    """Shipped per-joint confidence."""

    def __post_init__(self) -> None:
        if len(self.kp3ds) != len(self.kp3ds_conf):
            raise ValueError("joint and confidence counts differ")


@dataclass(frozen=True, slots=True)
class PoseRow:
    """One matched pair of pose CSV rows."""

    rgb_frameid: int
    """Frame number in the untrimmed capture."""
    positions: Float32[ndarray, "133 3"]
    """Shipped COCO world positions, rejected fits masked."""
    confidence: Float32[ndarray, "133"]
    """Shipped confidence, never residual-derived."""
    left: Fit
    """Left MANO fit."""
    right: Fit
    """Right MANO fit."""
    body: Fit
    """SMPL fit."""


def array_cell(value: str) -> list:
    """CSV arrays are JSON with occasional lowercase nan tokens."""
    return json.loads(value.replace("nan", "NaN"))


def parse_pose_row(mano: dict[str, str], smpl: dict[str, str]) -> PoseRow:
    """Join the two fit streams by rgb_frameid and apply upstream per-fit gates."""
    measurements = [
        from_dict(Measurements, {"rgb_frameid": raw["rgb_frameid"], "kp3ds": array_cell(raw["kp3ds"]), "kp3ds_conf": array_cell(raw["kp3ds_conf"])})
        for raw in (mano, smpl)
    ]
    hand, body = measurements
    frame = hand.rgb_frameid
    if frame != body.rgb_frameid:
        raise ValueError(f"rgb_frameid mismatch: {frame} != {smpl['rgb_frameid']}")
    fits: list[Fit] = []
    xyz: Float32[ndarray, "133 3"] = np.full((133, 3), np.nan, dtype=np.float32)
    confidence: Float32[ndarray, "133"] = np.zeros(133, dtype=np.float32)
    sources = {"pose3d_mano.csv": (mano, hand), "pose3d_smpl.csv": (smpl, body)}
    for file, (_, measurement) in sources.items():
        expected: int = max(spec.source.stop for spec in FITS if spec.file == file)
        if len(measurement.kp3ds) != expected:
            raise ValueError(f"{file}: expected {expected} source joints")
    for spec in FITS:
        raw, measurement = sources[spec.file]
        fields = {name: array_cell(raw[spec.prefix + name]) for name in ("poses", "Rh", "Th", "shapes")}
        residual: str = raw[spec.residual]
        parameters: FitParameters = from_dict(FitParameters, {**fields, "l2_dist": float(residual) if residual.strip() else float("nan")})
        fit: Fit = Fit(parameters, spec)
        if fit.parameters.poses.size != spec.pose_size:
            raise ValueError("incorrect pose parameter count")
        fits.append(fit)
        if fit.accepted:
            xyz[spec.coco] = measurement.kp3ds[spec.source]
            confidence[spec.coco] = np.nan_to_num(measurement.kp3ds_conf[spec.source], nan=0.0)
    confidence[~np.isfinite(xyz).all(axis=1)] = 0.0
    return PoseRow(frame, xyz, confidence, *fits)


def pose_batches(root: Path, count: int, *, total: int | None = None) -> Iterator[list[PoseRow]]:
    """Read at most count rows in batches of 64; total=None means the whole file."""

    def rows() -> Iterator[PoseRow]:
        with (root / "pose3d_mano.csv").open(newline="") as left, (root / "pose3d_smpl.csv").open(newline="") as right:
            pairs = zip_longest(csv.DictReader(left), csv.DictReader(right))
            previous = None
            seen = 0
            for mano, smpl in islice(pairs, count):
                if mano is None or smpl is None:
                    raise ValueError(f"{root}: pose CSV row counts differ")
                try:
                    row = parse_pose_row(mano, smpl)
                except (SerdeError, ValueError, KeyError) as error:
                    raise ValueError(f"{root}: invalid pose row {seen}: {error}") from error
                if previous is not None and row.rgb_frameid != previous + 1:
                    raise ValueError(f"{root}: rgb_frameid is not consecutive at row {seen}")
                previous = row.rgb_frameid
                seen += 1
                yield row
            if seen != count or ((total is None or count == total) and next(pairs, None) is not None):
                raise ValueError(f"{root}: pose rows do not match expected count {count}")

    return (list(batch) for batch in batched(rows(), BATCH_SIZE))


def holo_batches(path: Path, count: int, *, total: int | None = None) -> Iterator[Float64[ndarray, "n 4 4"]]:
    """Invert cam_T_world; empty poses are NaN. total=None means the whole file."""

    def transforms() -> Iterator[Float64[ndarray, "4 4"]]:
        with path.open(newline="") as handle:
            rows = csv.DictReader(handle)
            seen = 0
            for row in islice(rows, count):
                raw = array_cell(row["world2holo"])
                yield np.linalg.inv(np.asarray(raw, dtype=np.float64)) if raw else np.full((4, 4), np.nan)
                seen += 1
            if seen != count or ((total is None or count == total) and next(rows, None) is not None):
                raise ValueError(f"{path}: HoloLens rows do not match expected count {count}")

    return (np.asarray(batch) for batch in batched(transforms(), BATCH_SIZE))
