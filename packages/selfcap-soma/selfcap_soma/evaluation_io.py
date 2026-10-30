"""Bounded camera and detector reads for the single-segment experiment."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from datafusion import col, lit
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import RrdReader

from selfcap_soma.camera_geometry import Camera, EgoCamera
from selfcap_soma.catalog import seconds


@dataclass(frozen=True, slots=True)
class PhoneView:
    camera: Camera
    name: str
    pixels: Float64[ndarray, "n 133 2"]
    scores: Float64[ndarray, "n 133"]


def first(table: pa.Table, name: str) -> Float64[ndarray, "k"]:
    return np.asarray(table[name].drop_null()[0].as_py(), dtype=np.float64).reshape(-1)


def nearest_indices(source: Float64[ndarray, "m"], target: Float64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    right = np.clip(np.searchsorted(source, target), 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(source[left] - target) <= abs(source[right] - target), left, right)


def read_views(dataset: DatasetEntry, segment: str, times: Float64[ndarray, "n"]) -> tuple[list[PhoneView], EgoCamera]:
    view = dataset.filter_segments(segment)
    cameras = [f"/world/rig_{i:02d}/cam_00" for i in range(4)]
    oak = "/world/rig_04"
    oak_camera = f"{oak}/cam_00"
    static = view.filter_contents([*cameras, *(f"{c}/pinhole" for c in cameras), f"{oak_camera}/pinhole"]).reader(index=None).to_arrow_table()
    paths = [f"{c}/pinhole/pred/rtmw/coco133_uv" for c in cameras]
    samples = (
        view.filter_contents(paths)
        .reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= lit(int((times[0] - 0.05) * 1e9)))
        .filter(col("video_time").cast(pa.int64()) <= lit(int((times[-1] + 0.05) * 1e9)))
        .to_arrow_table()
    )
    phones: list[PhoneView] = []
    for camera, path in zip(cameras, paths, strict=True):
        matrix = first(static, f"{camera}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T
        pose = np.eye(4)
        pose[:3, :3] = first(static, f"{camera}:Transform3D:mat3x3").reshape(3, 3).T
        pose[:3, 3] = first(static, f"{camera}:Transform3D:translation")
        position = f"{path}:Points2D:positions"
        rows = samples.filter(samples[position].is_valid())
        stamps = seconds(rows["video_time"])
        indices = nearest_indices(stamps, times)
        pixels = np.asarray(rows[position].to_pylist(), dtype=np.float64)[indices]
        scores = np.asarray(rows[f"{path}:simplecv.KeypointConfidence2D:confidences"].to_pylist(), dtype=np.float64)[indices]
        valid = abs(stamps[indices] - times) < 0.021
        pixels[~valid] = np.nan
        scores[~valid] = 0.0
        phones.append(PhoneView(Camera(matrix, pose, (0.0, 0.0), camera), camera.split("/")[2], pixels, scores))
    poses = (
        view.filter_contents([oak])
        .reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= lit(int((times[0] - 0.1) * 1e9)))
        .filter(col("video_time").cast(pa.int64()) <= lit(int((times[-1] + 0.1) * 1e9)))
        .select("video_time", f"{oak}:Transform3D:mat3x3", f"{oak}:Transform3D:translation")
        .to_arrow_table()
    )
    poses = poses.filter(poses[f"{oak}:Transform3D:mat3x3"].is_valid())
    stamps = seconds(poses["video_time"])
    transforms = np.tile(np.eye(4), (len(stamps), 1, 1))
    transforms[:, :3, :3] = np.asarray(poses[f"{oak}:Transform3D:mat3x3"].to_pylist(), dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
    transforms[:, :3, 3] = np.asarray(poses[f"{oak}:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    size = first(static, f"{oak_camera}/pinhole:Pinhole:resolution")
    ego = EgoCamera(
        4,
        0,
        first(static, f"{oak_camera}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T,
        first(static, f"{oak_camera}/pinhole:simplecv.components.DistortionCoefficients"),
        (int(size[0]), int(size[1])),
        np.eye(4),
        stamps,
        transforms,
    )
    if len(ego.distortion) != 14:
        raise ValueError("expected the full 14-coefficient OAK lens")
    return phones, ego


@dataclass(frozen=True, slots=True)
class ImageTrack:
    name: str
    times: Float64[ndarray, "n"]
    points: Float64[ndarray, "n j 2"]


def read_image_tracks(path: Path) -> list[ImageTrack]:
    """Read only 2D inference tracks from the corrected EgoHand recording."""
    if not path.exists():
        return []
    grouped: dict[str, list[pa.RecordBatch]] = {}
    for chunk in RrdReader(path).stream().to_chunks():
        name = str(chunk.entity_path)
        if name.startswith(("/image/egohand/", "/image/sapiens/")):
            batch = chunk.to_record_batch()
            if "Points2D:positions" in batch.schema.names and "video_time" in batch.schema.names:
                grouped.setdefault(name, []).append(batch)
    tracks: list[ImageTrack] = []
    for name, batches in grouped.items():
        table = pa.Table.from_batches(batches)
        stamps = seconds(table["video_time"])
        points = np.asarray(table["Points2D:positions"].to_pylist(), dtype=np.float64)
        order = np.argsort(stamps)
        tracks.append(ImageTrack(name, stamps[order], points[order]))
    return tracks
