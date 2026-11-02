"""Optical-centre projections for checking the SelfCap camera calibration."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from dataforge.logging_toolkit import time_column
from dataforge.oak_mount import StaticCamera, read_rig_track, read_static_cameras
from dataforge.schema import rig_path
from dataforge.triangulation import Camera, EgoCamera, camera_transforms, project
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun import Points2D
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk
from simplecv.catalog_rig_layout import catalog_components


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
    """The base's left-eye head track and each Quest camera's extrinsic, calib over base: the factory right eye of a healthy
    headset is in the base, a borrowed camera 50 in the calib layer."""
    path: str = rig_path(rig)
    view = base.filter_segments(segment)
    videos: set[str] = {item.entity_path for item in catalog_components(view.schema()) if item.archetype == "VideoStream"}
    cameras: dict[int, str] = {cam: f"{path}/cam_{cam:02d}" for cam in (0, 1) if f"{path}/cam_{cam:02d}/pinhole/video" in videos}
    track: tuple[Float64[ndarray, "p"], Float64[ndarray, "p 4 4"]] | None = read_rig_track(view, path) if cameras else None
    if track is None or len(track[0]) < 2:
        return []
    static: dict[str, StaticCamera] = read_static_cameras(calibration.filter_segments(segment), view, list(cameras.values()))
    result: list[CenterTrack] = []
    for cam, camera in cameras.items():
        matrix: Float64[ndarray, "4 4"] | None = static[camera].rig_T_cam
        if matrix is None and cam == 1:
            continue  # Unresolved faulty-left mode must not invent a camera-50 baseline.
        label: str = "Quest L" if cam == 0 else "Quest R" if 0 in cameras else "Quest 50"
        result.append(CenterTrack(label, "quest", track[0], track[1], np.eye(4) if matrix is None else matrix))
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
