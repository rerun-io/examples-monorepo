"""A finished colsfm run as a Rerun catalog layer under `/world/runs/<source>`.

The layer carries what the run produced, in the world frame of the trajectory it started from, so it overlays the
segment it was read from:

- `/world/runs/<source>/rig`: the optimised `world_T_rig` per registered rig frame, on `video_time` (the reference
  keyframe's timestamp), sent as columns;
- `/world/runs/<source>/rig/<camera>`: the refined `rig_T_cam` and the camera's pinhole, so the frusta ride the rig;
- `/world/runs/<source>/points`: the sparse model, coloured from the images, static, so every layout shows the map;
- `/world/runs/<source>/points_growing`: the same points, each on `video_time` at the first rig frame that observes it
  (one row per rig frame); a layout that opts in (a cursor window from the start) grows the cloud along the walk;
- `/world/runs/<source>/trajectory`: the whole optimised path as one line strip;
- the `colsfm` recording property: the run's headline numbers (registered images, points, reprojection, wall time).

The recording id is the segment id, which is what lets a catalog register the file as a layer of that segment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from colsfm.benchmark import ReconstructionMetrics, RunArtifacts, read_run, reconstruction_metrics
from colsfm.frames_meta import CameraParams, FramesMeta, KeyframeMeta
from colsfm.rerun_log import camera_k_matrix

APPLICATION_ID: Final[str] = "colsfm"
"""Application id of a layer written here."""

DEFAULT_SOURCE: Final[str] = "colsfm"
"""Run source under `/world/runs/`."""

TRAJECTORY_RGB: Final[tuple[int, int, int]] = (255, 140, 0)
"""Optimised trajectory and frusta: orange, apart from the dataforge gt green."""

IMAGE_PLANE_DISTANCE: Final[float] = 0.1
"""Frustum length in metres."""


@dataclass(frozen=True)
class LayerSummary:
    """What a written layer holds."""

    path: Path
    """The `.rrd` written."""
    num_rig_poses: int
    """Rig frames with an optimised pose."""
    num_points: int
    """Sparse points."""
    metrics: ReconstructionMetrics
    """Registered images, points, reprojection error of the model."""


def rig_poses(frames_meta: FramesMeta, registered: set[str] | None = None) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 4 4"]]:
    """One optimised `world_T_rig` per rig frame whose reference keyframe was registered.

    Args:
        frames_meta: The run's optimised `frames_meta`.
        registered: Image names the reconstruction registered; `None` keeps every rig frame.

    Returns:
        Timestamps in nanoseconds (from the keyframe's microseconds) and the poses, in time order.
    """
    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    times: list[int] = []
    poses: list[Float64[ndarray, "4 4"]] = []
    for rig_frame in frames_meta.rig_frames():
        keyframe: KeyframeMeta = by_id[rig_frame.reference_keyframe_id]
        if registered is not None and keyframe.image_name not in registered:
            continue
        pose: Float64[ndarray, "4 4"] = np.eye(4)
        pose[:3, :] = frames_meta.world_T_vehicle(keyframe).matrix()
        times.append(rig_frame.timestamp_microseconds * 1000)
        poses.append(pose)
    order: Int64[ndarray, " n"] = np.argsort(np.asarray(times, dtype=np.int64), kind="stable")
    return np.asarray(times, dtype=np.int64)[order], np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)[order]


def _log_camera(recording: rr.RecordingStream, entity: str, camera: CameraParams) -> None:
    rig_T_cam: Float64[ndarray, "3 4"] = camera.vehicle_T_cam.matrix()
    recording.log(entity, rr.Transform3D(translation=rig_T_cam[:, 3], mat3x3=rig_T_cam[:, :3]), static=True)
    if camera.camera_matrix is None and camera.projection_matrix is None:
        return
    # The export rule (rectified projection matrix for the pinhole family, raw camera matrix otherwise), so the layer's
    # frusta agree with the walkthrough's and with the exported COLMAP cameras.
    k: Float64[ndarray, "3 3"] = camera_k_matrix(camera)
    recording.log(
        f"{entity}/pinhole",
        rr.Pinhole(
            image_from_camera=k,
            resolution=[camera.image_width, camera.image_height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=IMAGE_PLANE_DISTANCE,
            color=TRAJECTORY_RGB,
        ),
        static=True,
    )


def read_rrd_trajectory(rrd_path: Path, entity: str, timeline: str) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 4 4"]]:
    """Read a temporal `Transform3D` (quaternion + translation) of one entity out of an `.rrd`, as `world_T_<entity>`.

    Args:
        rrd_path: The recording.
        entity: Entity path carrying the transform, e.g. `/world/runs/slam_rs/rig`.
        timeline: Duration timeline to read it on.

    Returns:
        Timestamps (ns) and poses, sorted, one per distinct timestamp.

    Raises:
        ValueError: When the entity carries no temporal pose on `timeline`.
    """
    from rerun.experimental import RrdReader
    from scipy.spatial.transform import Rotation

    times: list[int] = []
    quaternions: list[Float64[ndarray, " 4"]] = []
    translations: list[Float64[ndarray, " 3"]] = []
    for chunk in RrdReader(str(rrd_path)).store().stream():
        if str(chunk.entity_path) != entity:
            continue
        batch = chunk.to_record_batch()
        names: set[str] = set(batch.schema.names)
        if not {timeline, "Transform3D:quaternion", "Transform3D:translation"} <= names:
            continue
        for stamp, quaternion, translation in zip(
            batch.column(timeline).to_pylist(),
            batch.column("Transform3D:quaternion").to_pylist(),
            batch.column("Transform3D:translation").to_pylist(),
            strict=True,
        ):
            if stamp is None or quaternion is None or translation is None:
                continue
            times.append(int(stamp.value) if hasattr(stamp, "value") else int(stamp))
            quaternions.append(np.asarray(quaternion, dtype=np.float64).reshape(4))
            translations.append(np.asarray(translation, dtype=np.float64).reshape(3))
    if not times:
        raise ValueError(f"{rrd_path}: {entity} carries no temporal Transform3D on {timeline}")
    order: Int64[ndarray, " n"] = np.argsort(np.asarray(times, dtype=np.int64), kind="stable")
    t_ns: Int64[ndarray, " n"] = np.asarray(times, dtype=np.int64)[order]
    keep: np.ndarray = np.concatenate([[True], np.diff(t_ns) > 0])
    poses: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (int(keep.sum()), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(np.asarray(quaternions)[order][keep]).as_matrix()
    poses[:, :3, 3] = np.asarray(translations)[order][keep]
    return t_ns[keep], poses


def log_points_by_first_view(
    recording: rr.RecordingStream,
    entity: str,
    xyz: Float64[ndarray, "p 3"],
    rgb: UInt8[ndarray, "p 3"],
    seen_ns: Int64[ndarray, " p"],
    timeline: str,
) -> None:
    """Log the cloud as one row per first-seen time (a rig frame), each holding the points that rig frame first observes.

    Rows, not one chunk per point: 1.6 M points of a 1 km clip land in one row per rig frame. The radius is one static value.

    Args:
        recording: The layer's recording.
        entity: The points entity.
        xyz: Positions, metres.
        rgb: Colours.
        seen_ns: Per point, when it is first observed.
        timeline: The segment's video timeline.
    """
    order: Int64[ndarray, " p"] = np.argsort(seen_ns, kind="stable")
    times, counts = np.unique(seen_ns[order], return_counts=True)
    recording.send_columns(
        entity,
        indexes=[rr.TimeColumn(timeline, duration=times.astype("timedelta64[ns]"))],
        columns=rr.Points3D.columns(positions=xyz[order], colors=rgb[order]).partition(counts.tolist()),
    )
    recording.log(entity, rr.Points3D.from_fields(radii=rr.Radius.ui_points(1.5)), static=True)


def open_layer(rrd_path: Path, segment: str) -> rr.RecordingStream:
    """A recording that saves to `rrd_path` and carries `segment` as its recording id (a layer of that segment).

    Args:
        rrd_path: Where the layer goes; parent directories are created.
        segment: The catalog segment id.

    Returns:
        The recording stream; flush it when done.
    """
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording: rr.RecordingStream = rr.RecordingStream(APPLICATION_ID, recording_id=segment, send_properties=False)
    recording.save(rrd_path)
    return recording


def log_rig_trajectory(
    recording: rr.RecordingStream,
    root: str,
    times_ns: Int64[ndarray, " n"],
    world_T_rig: Float64[ndarray, "n 4 4"],
    timeline: str,
    color: tuple[int, int, int] = TRAJECTORY_RGB,
) -> None:
    """Log `world_T_rig` over time at `<root>/rig` (as columns) and the whole path at `<root>/trajectory`.

    Args:
        recording: The layer's recording.
        root: `/world/runs/<source>`.
        times_ns: One timestamp per pose, nanoseconds on `timeline`.
        world_T_rig: The poses.
        timeline: The segment's video timeline.
        color: Line colour.
    """
    recording.send_columns(
        f"{root}/rig",
        indexes=[rr.TimeColumn(timeline, duration=times_ns.astype("timedelta64[ns]"))],
        columns=rr.Transform3D.columns(translation=world_T_rig[:, :3, 3], mat3x3=world_T_rig[:, :3, :3]),
    )
    recording.log(f"{root}/trajectory", rr.LineStrips3D([world_T_rig[:, :3, 3]], colors=[color], radii=rr.Radius.ui_points(2.0)), static=True)


def write_run_layer(run_dir: Path, rrd_path: Path, *, segment: str, source: str = DEFAULT_SOURCE, timeline: str = "video_time") -> LayerSummary:
    """Write one finished run as a layer of `segment`.

    Args:
        run_dir: The run's output directory (`sparse/`, `kpmap/`, `runtime.csv`, `summary.json`).
        rrd_path: Where the layer goes; parent directories are created.
        segment: The catalog segment (recording id) the run was read from.
        source: Run source under `/world/runs/`.
        timeline: Timeline of the rig poses; the segment's video timeline.

    Returns:
        What was written.
    """
    artifacts: RunArtifacts = read_run(run_dir, source)
    reconstruction: pycolmap.Reconstruction = artifacts.reconstruction
    registered: set[str] = {image.name for image in reconstruction.images.values() if image.has_pose}
    # One pose per input frame when the run wrote them; a run without them carries its registered keyframes.
    times_ns, world_T_rig = rig_poses(artifacts.every_frame) if artifacts.every_frame is not None else rig_poses(artifacts.frames_meta, registered)
    metrics: ReconstructionMetrics = reconstruction_metrics(reconstruction)

    root: str = f"/world/runs/{source}"
    recording: rr.RecordingStream = open_layer(rrd_path, segment)
    log_rig_trajectory(recording, root, times_ns, world_T_rig, timeline)
    for camera in artifacts.frames_meta.cameras.values():
        _log_camera(recording, f"{root}/rig/{camera.sensor_name}", camera)
    points: Float64[ndarray, "p 3"] = artifacts.points_xyz()
    colors: UInt8[ndarray, "p 3"] = artifacts.points_rgb().astype(np.uint8)
    recording.log(f"{root}/points", rr.Points3D(points, colors=colors, radii=rr.Radius.ui_points(1.5)), static=True)
    # Each image on its rig frame's clock, the clock of `<root>/rig`: one row per rig frame, not per camera image.
    image_t_ns: dict[int, int] = {
        keyframe_id: rig_frame.timestamp_microseconds * 1000 for rig_frame in artifacts.frames_meta.rig_frames() for keyframe_id in rig_frame.keyframe_ids
    }
    seen_ns: Int64[ndarray, " p"] = np.fromiter(
        (min(image_t_ns[element.image_id] for element in point.track.elements) for point in reconstruction.points3D.values()),
        dtype=np.int64,
        count=len(reconstruction.points3D),
    )
    log_points_by_first_view(recording, f"{root}/points_growing", points, colors, seen_ns, timeline)
    recording.send_property(
        "colsfm",
        rr.AnyValues(
            registered_images=metrics.registered_images,
                        num_points=metrics.num_points3D,
            mean_reprojection_error_px=metrics.mean_reprojection_error_px,
            total_runtime_s=float(sum(artifacts.runtime_seconds_by_stage.values())),
            run_dir=str(run_dir),
        ),
    )
    recording.flush()
    return LayerSummary(path=rrd_path, num_rig_poses=int(len(times_ns)), num_points=int(points.shape[0]), metrics=metrics)
