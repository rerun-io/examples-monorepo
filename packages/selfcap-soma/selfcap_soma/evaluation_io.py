"""Bounded camera and detector reads for SOMA layers."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from dataforge.clocks import nearest_framesets as nearest_indices
from dataforge.oak_mount import read_oak_track
from dataforge.selfcap_schema import OAK_HAND_BASES, oak_egohand_path
from dataforge.triangulation import HEADSET_COVERED, Camera, EgoCamera
from datafusion import col, lit
from jaxtyping import Float64
from numpy import ndarray

from selfcap_soma.catalog import ExoSource, SegmentCatalog, exo_view_track, floats, seconds
from selfcap_soma.overlay_style import MIN_OBSERVED_SCORE


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


def read_views(
    catalog: SegmentCatalog, segment: str, times: Float64[ndarray, "n"], source: ExoSource = "rtmw"
) -> tuple[list[PhoneView], EgoCamera | None]:
    view, rigs = catalog.view, catalog.rigs
    cameras = [f"{rig}/cam_00" for rig in rigs.exo]
    oak = rigs.oak
    oak_camera = f"{oak}/cam_00"
    static = view.filter_contents([*cameras, *(f"{c}/pinhole" for c in cameras), f"{oak_camera}/pinhole"]).reader(index=None).to_arrow_table()
    columns = catalog.columns
    paths = []
    calibrated = []
    for camera in cameras:
        missing = missing_camera_calibration(static, camera)
        if missing:
            print(f"SKIP_CAMERA {segment} {camera}: missing calibration {missing}", flush=True)
            continue
        track = exo_view_track(columns, camera, source)
        if track is None:
            print(f"SKIP_CAMERA {segment} {camera}: no {source} 2D track", flush=True)
            continue
        calibrated.append(camera)
        paths.append(track)
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
        pixels = floats(rows[position], -1, 133, 2)[indices]
        scores = floats(rows[f"{path}:simplecv.KeypointConfidence2D:confidences"], -1, 133)[indices]
        # The stored entity keeps every detection; a view holds only joints fitting and scoring may use.
        valid = (abs(stamps[indices] - times) < 0.021)[:, None] & (scores >= MIN_OBSERVED_SCORE)
        valid[:, HEADSET_COVERED] = False
        pixels[~valid] = np.nan
        scores[~valid] = 0.0
        phones.append(PhoneView(Camera(matrix, pose, (0.0, 0.0), camera), camera.split("/")[2], pixels, scores))
    missing = missing_camera_calibration(static, oak_camera, extrinsics=False, distorted=True)
    if missing:
        print(f"SKIP_CAMERA {segment} {oak_camera}: missing calibration {missing}", flush=True)
        return phones, None
    track = read_oak_track(view, oak, rigs.quest, start_ns=int((times[0] - 0.1) * 1e9), end_ns=int((times[-1] + 0.1) * 1e9))
    if track is None:
        print(f"SKIP_CAMERA {segment} {oak}: no fitted OAK mount", flush=True)
        return phones, None
    stamps, transforms = track
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
    return phones, ego


@dataclass(frozen=True, slots=True)
class ImageTrack:
    name: str
    times: Float64[ndarray, "n"]
    points: Float64[ndarray, "n j 2"]


def read_egohand_rgb(catalog: SegmentCatalog, times: Float64[ndarray, "n"]) -> list[ImageTrack]:
    """Read registered RGB hand image evidence once, preserving absent detections."""
    view, oak = catalog.view, catalog.rigs.oak
    path = oak_egohand_path(int(oak.rsplit("_", 1)[1]), 0)
    column = path + ":Points2D:positions"
    if column not in catalog.columns:
        return []
    table = (
        view.filter_contents([path])
        .reader(index="video_time")
        .filter(col("video_time").cast(pa.int64()) >= lit(int((times[0] - 0.05) * 1e9)))
        .filter(col("video_time").cast(pa.int64()) <= lit(int((times[-1] + 0.05) * 1e9)))
        .to_arrow_table()
    )
    rows = table.filter(table[column].is_valid())
    stamps = seconds(rows["video_time"])
    points = floats(rows[column], -1, 133, 2)
    scores = floats(rows[path + ":confidence"], -1, 133)
    points[~(scores >= MIN_OBSERVED_SCORE)] = np.nan
    order = np.argsort(stamps)
    return [ImageTrack(f"{path}/{side}", stamps[order], points[order, base:base + 21]) for side, base in OAK_HAND_BASES]
