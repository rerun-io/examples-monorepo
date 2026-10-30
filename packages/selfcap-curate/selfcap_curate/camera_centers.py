"""Optical-centre projections for checking the SelfCap camera calibration."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa
from dataforge.logging_toolkit import time_column
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun import Points2D
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk
from scipy.spatial.transform import Rotation
from simplecv.catalog_rig_layout import catalog_components
from simplecv.rrd_query_utils import first_valid_value

from selfcap_curate.keypoints import Camera, EgoCamera, camera_transforms, project


@dataclass(frozen=True, slots=True)
class CenterTrack:
    label: str
    """Short camera name shown beside its centre."""
    kind: Literal["oak", "quest"]
    """Device kind determining the point colour."""
    times_s: Float64[ndarray, "p"]
    """Aligned video_time of the source rig poses."""
    world_T_rig: Float64[ndarray, "p 4 4"]
    """Calibrated rig poses in Quest world."""
    rig_T_cam: Float64[ndarray, "4 4"]
    """Camera extrinsic, including the resolved camera-50 lever arm when borrowed."""


def read_quest_centers(calibration: DatasetEntry, base: DatasetEntry, segment: str, rig: int) -> list[CenterTrack]:
    """Use the base's left-eye head track and the calib layer's resolved Quest extrinsics."""
    path: str = f"/world/rig_{rig:02d}"
    view = base.filter_segments(segment)
    videos: set[str] = {item.entity_path for item in catalog_components(view.schema()) if item.archetype == "VideoStream"}
    poses: pa.Table = (
        view.filter_contents(path)
        .reader(index="video_time")
        .select("video_time", f"{path}:Transform3D:translation", f"{path}:Transform3D:quaternion")
        .to_arrow_table()
    )
    poses = poses.filter(poses[f"{path}:Transform3D:quaternion"].is_valid())
    if len(poses) < 2:
        return []
    stamps: Float64[ndarray, "p"] = poses["video_time"].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64) * 1e-9
    transforms: Float64[ndarray, "p 4 4"] = np.tile(np.eye(4), (len(poses), 1, 1))
    transforms[:, :3, 3] = np.asarray(poses[f"{path}:Transform3D:translation"].to_pylist(), dtype=np.float64).reshape(-1, 3)
    transforms[:, :3, :3] = Rotation.from_quat(
        np.asarray(poses[f"{path}:Transform3D:quaternion"].to_pylist(), dtype=np.float64).reshape(-1, 4)
    ).as_matrix()
    extrinsics: pa.Table = (
        calibration.filter_segments(segment).filter_contents([f"{path}/cam_00", f"{path}/cam_01"]).reader(index=None).to_arrow_table()
    )
    result: list[CenterTrack] = []
    for cam in (0, 1):
        camera: str = f"{path}/cam_{cam:02d}"
        if f"{camera}/pinhole/video" not in videos:
            continue
        matrix: Float64[ndarray, "4 4"] = np.eye(4)
        if cam == 1 or f"{camera}:Transform3D:translation" in extrinsics.column_names:
            translation: str = f"{camera}:Transform3D:translation"
            rotation: str = f"{camera}:Transform3D:mat3x3"
            if translation not in extrinsics.column_names or rotation not in extrinsics.column_names:
                continue  # Unresolved faulty-left mode must not invent a camera-50 baseline.
            matrix[:3, 3] = np.asarray(first_valid_value(extrinsics[translation]), dtype=np.float64).reshape(3)
            matrix[:3, :3] = np.asarray(first_valid_value(extrinsics[rotation]), dtype=np.float64).reshape(3, 3).T
        label: str = "Quest L" if cam == 0 else "Quest R" if f"{path}/cam_00/pinhole/video" in videos else "Quest 50"
        result.append(CenterTrack(label, "quest", stamps, transforms, matrix))
    return result


def project_centers(camera: Camera, centers: Float64[ndarray, "n k 3"]) -> Float32[ndarray, "n k 2"]:
    """Project world centres through the exo lens; clip behind-camera and out-of-image pixels."""
    pixels: Float64[ndarray, "m 3"] = project(camera, centers.reshape(-1, 3))
    width, height = camera.size
    valid: Bool[ndarray, "m"] = np.isfinite(pixels).all(axis=1) & (pixels[:, 2] > 1e-5)
    valid &= (pixels[:, 0] >= 0.0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0.0) & (pixels[:, 1] < height)
    return np.where(valid[:, None], pixels[:, :2], np.nan).reshape(*centers.shape[:2], 2).astype(np.float32)


def camera_center_chunks(
    cameras: dict[str, Camera], oak: tuple[EgoCamera, ...], quest: list[CenterTrack], frame_times: dict[str, Int64[ndarray, "n"]]
) -> Iterator[Chunk]:
    """One labelled Points2D entity per exo video, on its own native video frame clock."""
    moving: list[CenterTrack] = [
        CenterTrack(("OAK rgb", "OAK L", "OAK R")[camera.cam], "oak", camera.times_s, camera.world_T_rig, camera.rig_T_cam) for camera in oak
    ] + quest
    for phone, camera in cameras.items():
        stamps: Int64[ndarray, "n"] = frame_times[phone]
        times: Float64[ndarray, "n"] = stamps.astype(np.float64) * 1e-9
        others: list[Camera] = [other for other in cameras.values() if other.entity != camera.entity]
        labels: list[str] = [other.entity.split("/")[2] for other in others] + [track.label for track in moving]
        if not labels:
            continue
        centers: Float64[ndarray, "n k 3"] = np.stack(
            [
                *(np.broadcast_to(other.world_T_cam[:3, 3], (len(times), 3)) for other in others),
                *(camera_transforms(times, track.times_s, track.world_T_rig, track.rig_T_cam)[:, :3, 3] for track in moving),
            ],
            axis=1,
        )
        colors: UInt8[ndarray, "k 3"] = np.asarray(
            [(70, 150, 255)] * len(others) + [(255, 170, 0) if track.kind == "oak" else (60, 220, 110) for track in moving],
            dtype=np.uint8,
        )
        entity: str = f"{camera.entity}/pinhole/camera_centers"
        yield Chunk.from_columns(
            entity,
            indexes=[],
            columns=[*Points2D.columns(labels=labels, colors=colors).partition([len(labels)]), *Points2D.columns(radii=4.0, show_labels=[True])],
        )
        pixels: Float32[ndarray, "n k 2"] = project_centers(camera, centers)
        yield Chunk.from_columns(
            entity, indexes=[time_column(stamps)], columns=Points2D.columns(positions=pixels.reshape(-1, 2)).partition([len(labels)] * len(times))
        )
