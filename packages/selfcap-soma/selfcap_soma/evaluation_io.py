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
from simplecv.data.skeleton.soma_hand import SOMA_HAND25_TO_COCO21

from selfcap_soma.camera_geometry import Camera, EgoCamera
from selfcap_soma.catalog import ExoSource, read_rigs, seconds


@dataclass(frozen=True, slots=True)
class PhoneView:
    camera: Camera
    name: str
    pixels: Float64[ndarray, "n 133 2"]
    scores: Float64[ndarray, "n 133"]


def first(table: pa.Table, name: str) -> Float64[ndarray, "k"]:
    if name not in table.column_names or not len(table[name].drop_null()):
        raise ValueError(f"missing calibration value: {name}")
    return np.asarray(table[name].drop_null()[0].as_py(), dtype=np.float64).reshape(-1)


def missing_camera_calibration(table: pa.Table, camera: str, extrinsics: bool = True, distorted: bool = False) -> list[str]:
    """List absent, empty or nonfinite calibration fields before reading a camera."""
    required = {f"{camera}/pinhole:Pinhole:image_from_camera": 9}
    if extrinsics:
        required.update({f"{camera}:Transform3D:translation": 3, f"{camera}:Transform3D:mat3x3": 9})
    if distorted:
        required.update({f"{camera}/pinhole:Pinhole:resolution": 2, f"{camera}/pinhole:simplecv.components.DistortionCoefficients": 14})
    missing = []
    for name, size in required.items():
        if name not in table.column_names or not len(table[name].drop_null()):
            missing.append(name)
            continue
        values = first(table, name)
        if values.size != size or not np.isfinite(values).all():
            missing.append(name)
    return missing


def nearest_indices(source: Float64[ndarray, "m"], target: Float64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    right = np.clip(np.searchsorted(source, target), 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(source[left] - target) <= abs(source[right] - target), left, right)


def read_views(
    dataset: DatasetEntry, segment: str, times: Float64[ndarray, "n"], source: ExoSource = "rtmw"
) -> tuple[list[PhoneView], EgoCamera | None]:
    view = dataset.filter_segments(segment)
    rigs = read_rigs(dataset, segment)
    cameras = [f"{rig}/cam_00" for rig in rigs.exo]
    oak = rigs.oak
    oak_camera = f"{oak}/cam_00"
    static = view.filter_contents([*cameras, *(f"{c}/pinhole" for c in cameras), f"{oak_camera}/pinhole"]).reader(index=None).to_arrow_table()
    columns = view.schema().column_names()
    paths = []
    calibrated = []
    for camera in cameras:
        missing = missing_camera_calibration(static, camera)
        if missing:
            print(f"SKIP_CAMERA {segment} {camera}: missing calibration {missing}", flush=True)
            continue
        candidates = [
            name.split(":", 1)[0]
            for name in columns
            if name.startswith(camera + "/pinhole/")
            and name.endswith("/coco133_uv:Points2D:positions")
            and f"/{source}/" in name
            and name.split(":", 1)[0] + ":simplecv.KeypointConfidence2D:confidences" in columns
        ]
        if not candidates:
            print(f"SKIP_CAMERA {segment} {camera}: no {source} 2D track", flush=True)
            continue
        if len(candidates) != 1:
            raise ValueError(f"ambiguous {source} 2D tracks for {camera}: {candidates}")
        calibrated.append(camera)
        paths.append(candidates[0])
    samples = (
        None
        if not paths
        else (
            view.filter_contents(paths)
            .reader(index="video_time")
            .filter(col("video_time").cast(pa.int64()) >= lit(int((times[0] - 0.05) * 1e9)))
            .filter(col("video_time").cast(pa.int64()) <= lit(int((times[-1] + 0.05) * 1e9)))
            .to_arrow_table()
        )
    )
    phones: list[PhoneView] = []
    for camera, path in zip(calibrated, paths, strict=True):
        assert samples is not None
        matrix = first(static, f"{camera}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T
        pose = np.eye(4)
        pose[:3, :3] = first(static, f"{camera}:Transform3D:mat3x3").reshape(3, 3).T
        pose[:3, 3] = first(static, f"{camera}:Transform3D:translation")
        position = f"{path}:Points2D:positions"
        rows = samples.filter(samples[position].is_valid())
        stamps = seconds(rows["video_time"])
        if not len(stamps):
            print(f"SKIP_CAMERA {segment} {camera}: no detections in requested window", flush=True)
            continue
        indices = nearest_indices(stamps, times)
        pixels = np.asarray(rows[position].to_pylist(), dtype=np.float64)[indices]
        scores = np.asarray(rows[f"{path}:simplecv.KeypointConfidence2D:confidences"].to_pylist(), dtype=np.float64)[indices]
        valid = abs(stamps[indices] - times) < 0.021
        pixels[~valid] = np.nan
        scores[~valid] = 0.0
        phones.append(PhoneView(Camera(matrix, pose, (0.0, 0.0), camera), camera.split("/")[2], pixels, scores))
    missing = missing_camera_calibration(static, oak_camera, extrinsics=False, distorted=True)
    if missing:
        print(f"SKIP_CAMERA {segment} {oak_camera}: missing calibration {missing}", flush=True)
        return phones, None
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
    if not len(stamps):
        print(f"SKIP_CAMERA {segment} {oak}: no calibrated pose samples", flush=True)
        return phones, None
    transforms = np.tile(np.eye(4), (len(stamps), 1, 1))
    transforms[:, :3, :3] = np.asarray(poses[f"{oak}:Transform3D:mat3x3"].to_pylist(), dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
    transforms[:, :3, 3] = np.asarray(poses[f"{oak}:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    size = first(static, f"{oak_camera}/pinhole:Pinhole:resolution")
    ego = EgoCamera(
        int(oak.rsplit("_", 1)[1]),
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


def read_egohand_rgb(dataset: DatasetEntry, segment: str, times: Float64[ndarray, "n"]) -> list[ImageTrack]:
    """Read registered RGB hand image evidence once, preserving absent detections."""
    view = dataset.filter_segments(segment)
    oak = read_rigs(dataset, segment).oak
    paths = [
        column.split(":", 1)[0]
        for column in view.schema().column_names()
        if column.startswith(f"{oak}/cam_00/pinhole/") and "/egohand/" in column and column.endswith(":Points2D:positions")
    ]
    if not paths:
        return []
    table = (
        view.filter_contents(paths)
        .reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= lit(int((times[0] - 0.05) * 1e9)))
        .filter(col("video_time").cast(pa.int64()) <= lit(int((times[-1] + 0.05) * 1e9)))
        .to_arrow_table()
    )
    result = []
    for path in paths:
        column = path + ":Points2D:positions"
        rows = table.filter(table[column].is_valid())
        stamps = seconds(rows["video_time"])
        points = np.full((len(stamps), 21, 2), np.nan)
        for index, row in enumerate(rows[column].to_pylist()):
            if len(row) == 25:
                points[index] = np.asarray(row, dtype=np.float64)[list(SOMA_HAND25_TO_COCO21)]
            elif len(row) == 21:
                points[index] = np.asarray(row, dtype=np.float64)
        order = np.argsort(stamps)
        result.append(ImageTrack(path, stamps[order], points[order]))
    return result
