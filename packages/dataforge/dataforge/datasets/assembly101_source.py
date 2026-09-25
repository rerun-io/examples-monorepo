"""Assembly101 JSON boundaries and columnar hand streams."""

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

import numpy as np
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import SerdeError, coerce, field, from_dict, serde

from dataforge.hands import coco133_from_hands_batch
from dataforge.records import read_json

NAS_ROOT: Path = Path("/mnt/nas")
FRAME_RATE: int = 60
ANNOTATION_RATE: int = 30
EGO_RIG: int = 8
EXO_SERIALS: tuple[str, ...] = ("C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404")
SOURCE_REVISION: str = "mirror 001839131530cee9b2deb9ca66c025998d10cba4; official bfc15ea5"
CLOCK_SOURCE: str = "frame_index/60: documented 60 fps; AssemblyPoses timestamp = t0 + k/60 (1 ms rounding); mirror PTS CFR 1/60"
POSE_MEMBERS: tuple[str, ...] = ("camera_extrinsics_fixed", "camera_extrinsics_ego", "landmarks3D", "landmarks2D", "hand_confidences", "timestamp")
T = TypeVar("T")


def read_record(path: Path, cls: type[T]) -> T:  # noqa: UP047
    """Decode a numpy-valued map through a dataclass root, naming decode failures."""
    try:
        return from_dict(cls, {"data": json.loads(path.read_text())})
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


def pose_path(root: Path, member: str, sequence: str) -> Path:
    """Member location in the release's extracted archive layout."""
    return root / "assembly101_camera_and_hand_poses" / member / f"{sequence}.json"


def frame_times(frames: Int64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    """Source frame keys on the documented 60 Hz duration clock."""
    return np.rint(frames / FRAME_RATE * 1e9).astype(np.int64)


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class HandPair:
    """One shipped frame; missing hands remain missing."""

    left: Float32[ndarray, "21 d"] | None = field(rename="0", default=None)
    """Left Assembly-21 joints."""
    right: Float32[ndarray, "21 d"] | None = field(rename="1", default=None)
    """Right Assembly-21 joints."""

    def array(self, dimensions: int) -> Float32[ndarray, "2 21 d"]:
        result: Float32[ndarray, "2 21 d"] = np.full((2, 21, dimensions), np.nan, dtype=np.float32)
        for index, value in enumerate((self.left, self.right)):
            if value is not None:
                if value.shape != (21, dimensions):
                    raise ValueError(f"expected 21 x {dimensions} hand coordinates, got {value.shape}")
                result[index] = value
        return result


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class HandConfidence:
    """Per-hand confidence, applied to every present joint."""

    left: float = field(rename="0")
    """Left hand confidence."""
    right: float = field(rename="1")
    """Right hand confidence."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.left <= 1.0 or not 0.0 <= self.right <= 1.0:
            raise ValueError("hand confidence must lie in [0, 1]")


HandRows: TypeAlias = tuple[Int64[ndarray, "n"], Float32[ndarray, "n 133 d"], Float32[ndarray, "n 133"]]


def read_confidence(path: Path) -> dict[int, Float32[ndarray, "2"]]:
    """Decode a confidence stream once, retaining real frame keys."""
    rows: dict[str, HandConfidence] = read_json(path, dict[str, HandConfidence])
    return {int(key): np.array([row.left, row.right], dtype=np.float32) for key, row in rows.items()}


def frame_batches(rows: Mapping[str, object], frame_limit: int | None) -> Iterator[list[str]]:
    """Numeric frame keys below the limit, in 256-frame batches."""
    keys: list[str] = sorted(rows, key=int)
    if frame_limit is not None:
        keys = [key for key in keys if int(key) < frame_limit]
    for start in range(0, len(keys), 256):
        yield keys[start : start + 256]


def coco_rows(
    keys: list[str], pairs: list[dict[str, list[list[float | int]]]], confidence: dict[int, Float32[ndarray, "2"]], dimensions: int, scale: float
) -> HandRows:
    """Decode a batch into raw COCO rows; shared writers apply the confidence rule."""
    joints: Float32[ndarray, "n 2 21 d"] = np.stack([from_dict(HandPair, raw).array(dimensions) for raw in pairs]) * np.float32(scale)
    confidence_batch: Float32[ndarray, "n 2"] = np.stack([confidence[int(key)] for key in keys])
    positions, scores = coco133_from_hands_batch(joints, confidence_batch)
    return np.array(keys, dtype=np.int64), positions, scores


def read_hand_rows(
    path: Path,
    confidence: dict[int, Float32[ndarray, "2"]],
    *,
    dimensions: int,
    scale: float,
    frame_limit: int | None,
) -> Iterator[HandRows]:
    """Read 3D once; sort numeric keys and release consumed source frames in batches."""
    with path.open() as handle:
        rows: dict[str, dict[str, list[list[float | int]]]] = json.load(handle)
    for keys in frame_batches(rows, frame_limit):
        yield coco_rows(keys, [rows.pop(key) for key in keys], confidence, dimensions, scale)


def read_pixels(
    path: Path, confidence: dict[int, Float32[ndarray, "2"]], camera_scales: dict[str, float], frame_limit: int | None
) -> Iterator[tuple[str, HandRows]]:
    """Read the large 2D member once; keep only one 256-frame camera batch in addition to it."""
    with path.open() as handle:
        rows: dict[str, dict[str, dict[str, list[list[float | int]]]]] = json.load(handle)
    for keys in frame_batches(rows, frame_limit):
        for camera, scale in camera_scales.items():
            yield camera, coco_rows(keys, [rows[key].pop(camera) for key in keys], confidence, 2, scale)
        for key in keys:
            del rows[key]


def read_transforms(path: Path) -> dict[str, Float64[ndarray, "4 4"]]:
    """Read fixed world-from-camera transforms; translations remain in source mm."""
    return read_record(path, FixedTransforms).data


@serde
@dataclass(frozen=True, slots=True)
class FixedTransforms:
    """Typed envelope for the source's bare camera map."""

    data: dict[str, Float64[ndarray, "4 4"]]
    """World-from-camera transforms, source mm."""


@serde
@dataclass(frozen=True, slots=True)
class EgoTransforms:
    """Typed envelope for the source's frame-keyed headset transforms."""

    data: dict[str, dict[str, Float64[ndarray, "4 4"]]]
    """World-from-camera transforms, source mm."""


@serde
@dataclass(frozen=True, slots=True)
class ManifestRow:
    """Mirror coverage facts distinguish absent poses from incomplete downloads."""

    sequence_name: str
    """Upstream sequence directory."""
    video_only: Literal["True", "False"]
    """CSV boolean spelling."""


def read_manifest(root: Path) -> dict[str, bool]:
    """Return pose expectations when the mirror inventory is present."""
    path: Path = root / "manifests/sequences.csv"
    if not path.is_file():
        return {}
    result: dict[str, bool] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: ManifestRow = from_dict(ManifestRow, raw)
            result[row.sequence_name] = row.video_only == "False"
    return result
