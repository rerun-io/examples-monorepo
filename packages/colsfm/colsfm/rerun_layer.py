"""A finished colsfm run as a Rerun catalog layer under `/world/runs/<source>`.

The layer carries what the run produced, in the world frame of the trajectory it started from, so it overlays the
segment it was read from:

- `/world/runs/<source>/rig`: the optimised `world_T_rig` per registered rig frame, on `video_time` (the reference
  keyframe's timestamp), sent as columns;
- `/world/runs/<source>/rig/<camera>`: the refined `rig_T_cam` and the camera's pinhole, so the frusta ride the rig;
- `/world/runs/<source>/rig/<camera>/pinhole/video`: for a catalog input, a `VideoFrameReference` at each rig frame to
  that camera's sample in the segment's own `VideoStream`, so each frustum shows the picture the run read (the same
  camera, pinhole, video layout as the dataset's rig; nothing is copied or encoded again);
- `/world/runs/<source>/points`: the sparse model, coloured from the images, static, so every layout shows the map;
- `/world/runs/<source>/points_growing`: the same points, each on `video_time` at the first rig frame that observes it
  (one row per rig frame); a layout that opts in (a cursor window from the start) grows the cloud along the walk;
- the optimised path in dataforge's names (`log_walk_path`): `/world/runs/<source>/trail`, one segment per rig frame on
  `video_time` (a layout's visible time range makes the trail), green at the first rig frame through yellow to red at
  the last (`walk_colors`); `trajectory`, the whole path in the same colours, one row at the last rig frame;
  `endpoints/start` (with a ring, static) and `endpoints/end` (from the last rig frame on);
- `/world/runs/<source>/rig_view`, for a catalog input whose segment has viewing faces (dataforge's Insta360 `cube_view`
  layer, `pinhole/video_view`): a pose at every viewing frame, the prior re-anchored on the registered rig frames
  (`colsfm.frame_poses.view_rig_poses`, the every-frame export's rule), with each camera's frustum referencing its
  viewing face of that frame. `rig` keeps the registered rig frames only;
- `/world/runs/<source>/position_difference`: how far each optimised rig frame lies from the prior pose the run started
  from (the run's own `keyframes/frames_meta.json`), in metres, on `video_time`; left out for a run without that file;
- the `colsfm` recording property: the run's headline numbers (registered images, points, reprojection, wall time).

The recording id is the segment id, which is what lets a catalog register the file as a layer of that segment.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
import rerun as rr
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from colsfm.benchmark import ReconstructionMetrics, RunArtifacts, read_run, reconstruction_metrics
from colsfm.frame_manifest import FrameSourceManifest
from colsfm.frame_poses import ViewRigSource, view_rig_poses
from colsfm.frames_meta import FRAMES_META_NAME, CameraParams, FramesMeta, KeyframeMeta, read_frames_meta
from colsfm.rerun_log import camera_k_matrix
from colsfm.run_lifecycle import KEYFRAME_DIR_NAME

APPLICATION_ID: Final[str] = "colsfm"
"""Application id of a layer written here."""

DEFAULT_SOURCE: Final[str] = "colsfm"
"""Run source under `/world/runs/`."""

TRAJECTORY_RGB: Final[tuple[int, int, int]] = (255, 140, 0)
"""Optimised trajectory and frusta: orange, apart from the dataforge gt green."""

IMAGE_PLANE_DISTANCE: Final[float] = 0.1
"""Frustum length in metres."""

WALK_GO_HUE: Final[float] = 120.0
"""Hue of the path at the first rig frame: green."""
WALK_STOP_HUE: Final[float] = 0.0
"""Hue of the path at the last rig frame: red, through yellow and orange (HSV, so no muddy brown between)."""
WALK_SATURATION: Final[float] = 0.85
"""Saturation of the walk gradient."""
PATH_RADIUS_UI: Final[float] = 2.0
"""Path stroke, ui points."""
START_RING_RADIUS_UI: Final[float] = 2.5
"""Stroke of the ring around the start dot, ui points (as slam-rs's ``walk_path``)."""
RING_POINTS: Final[int] = 49
"""Points on the start ring (the first repeated last)."""
ENDPOINT_RADIUS_UI: Final[float] = 9.0
"""Radius of the start and end dots, ui points."""
START_RING_M: Final[float] = 0.25
"""Radius of the ring around the start dot, metres: a walk that passes its start again keeps the start readable."""


def walk_colors(t_ns: Int64[ndarray, " n"], first_ns: int, last_ns: int) -> UInt8[ndarray, "n 3"]:
    """The walk gradient at each stamp: green at `first_ns`, through yellow, to red at `last_ns` (HSV hue `WALK_GO_HUE` to
    `WALK_STOP_HUE`). slam-rs draws its raw run with the same gradient (`slam_rs.vio_log.walk_colors`; a test in each
    package pins the same colours)."""
    fraction: Float64[ndarray, " n"] = np.clip((t_ns - first_ns) / max(last_ns - first_ns, 1), 0.0, 1.0)
    rgb: list[tuple[float, float, float]] = [colorsys.hsv_to_rgb((WALK_GO_HUE + (WALK_STOP_HUE - WALK_GO_HUE) * f) / 360.0, WALK_SATURATION, 1.0) for f in fraction]
    return np.round(np.asarray(rgb, dtype=np.float64).reshape(-1, 3) * 255.0).astype(np.uint8)


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


def log_position_difference(
    recording: rr.RecordingStream,
    entity: str,
    prior_t_ns: Int64[ndarray, " n"],
    prior_xyz: Float64[ndarray, "n 3"],
    t_ns: Int64[ndarray, " m"],
    xyz: Float64[ndarray, "m 3"],
    timeline: str,
) -> None:
    """Log the distance between the optimised and the prior rig position at every rig frame both hold.

    The frames are matched by timestamp: the run keeps the prior's stamps, so nothing is interpolated.

    Args:
        recording: The layer's recording.
        entity: The scalar entity.
        prior_t_ns: Timestamps of the prior rig frames, nanoseconds, increasing.
        prior_xyz: Their positions, metres.
        t_ns: Timestamps of the optimised rig frames, nanoseconds, increasing.
        xyz: Their positions, metres.
        timeline: The segment's video timeline.
    """
    shared, in_prior, in_run = np.intersect1d(prior_t_ns, t_ns, return_indices=True)
    if shared.size == 0:
        return
    difference_m: Float64[ndarray, " k"] = np.linalg.norm(xyz[in_run] - prior_xyz[in_prior], axis=1)
    recording.send_columns(
        entity,
        indexes=[rr.TimeColumn(timeline, duration=shared.astype("timedelta64[ns]"))],
        columns=rr.Scalars.columns(scalars=difference_m),
    )
    recording.log(entity, rr.SeriesLines(names="distance to the prior [m]"), static=True)


def log_frame_references(recording: rr.RecordingStream, root: str, frames_meta: FramesMeta, manifest: FrameSourceManifest) -> None:
    """Log, under each rig camera's pinhole, a reference to its sample of every rig frame in the segment's video.

    A row at the rig frame's time (the time `<root>/rig` takes that frame's pose) holds the sample's own stamp on the
    stream, so pose and picture change together. The stream is the one the run read the keyframes from.

    Args:
        recording: The layer's recording.
        root: `/world/runs/<source>`.
        frames_meta: The rig frames the layer poses: the keyframes name their samples (`image_name`).
        manifest: The run's frame source: each camera's stream and the stamp of each named sample.
    """
    by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    for stream in manifest.streams:
        sample_ns: dict[str, int] = dict(zip(stream.names, stream.t_ns, strict=True))
        shown: list[tuple[int, int, int]] = [
            (rig_frame.timestamp_microseconds * 1000, sample_ns[by_id[keyframe_id].image_name], by_id[keyframe_id].camera_params_id)
            for rig_frame in frames_meta.rig_frames()
            for keyframe_id in rig_frame.keyframe_ids
            if by_id[keyframe_id].image_name in sample_ns
        ]
        if not shown:
            continue
        entity: str = f"{root}/rig/{frames_meta.cameras[shown[0][2]].sensor_name}/pinhole/video"
        recording.send_columns(
            entity,
            indexes=[rr.TimeColumn(manifest.timeline, duration=np.asarray([row for row, _, _ in shown], dtype="timedelta64[ns]"))],
            columns=rr.VideoFrameReference.columns_nanos(np.asarray([sample for _, sample, _ in shown], dtype=np.int64)),
        )
        recording.log(entity, rr.VideoFrameReference.from_fields(video_reference=f"/{stream.entity.lstrip('/')}"), static=True)


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
    color: tuple[int, int, int] | None = None,
) -> None:
    """Log `world_T_rig` over time at `<root>/rig` (as columns) and the path (`log_walk_path`).

    Args:
        recording: The layer's recording.
        root: `/world/runs/<source>`.
        times_ns: One timestamp per pose, nanoseconds on `timeline`, increasing.
        world_T_rig: The poses.
        timeline: The segment's video timeline.
        color: One colour for the path in place of the walk gradient (a comparison source beside a colsfm run).
    """
    recording.send_columns(
        f"{root}/rig",
        indexes=[rr.TimeColumn(timeline, duration=times_ns.astype("timedelta64[ns]"))],
        columns=rr.Transform3D.columns(translation=world_T_rig[:, :3, 3], mat3x3=world_T_rig[:, :3, :3]),
    )
    log_walk_path(recording, root, times_ns, world_T_rig[:, :3, 3], timeline, color)


def log_walk_path(
    recording: rr.RecordingStream,
    root: str,
    times_ns: Int64[ndarray, " n"],
    positions: Float64[ndarray, "n 3"],
    timeline: str,
    color: tuple[int, int, int] | None = None,
) -> None:
    """Log a path in dataforge's names: a segment per pose at `<root>/trail`, the whole path at `<root>/trajectory` from
    the last pose on, the start dot and ring (static) and the end dot (from the last pose on) at `<root>/endpoints`; in the
    walk gradient, or in one `color`. Nothing with fewer than two poses.
    """
    if len(times_ns) < 2:
        return
    colors: UInt8[ndarray, "n 3"] = (
        walk_colors(times_ns, int(times_ns[0]), int(times_ns[-1])) if color is None else np.tile(np.asarray(color, dtype=np.uint8), (len(times_ns), 1))
    )
    strips = rr.LineStrips3D.columns(strips=list(np.stack([positions[:-1], positions[1:]], axis=1)), colors=colors[1:])
    radius = rr.Radius.ui_points(PATH_RADIUS_UI)
    recording.send_columns(f"{root}/trail", indexes=[rr.TimeColumn(timeline, duration=times_ns[1:].astype("timedelta64[ns]"))], columns=strips.partition([1] * (len(times_ns) - 1)))
    recording.log(f"{root}/trail", rr.LineStrips3D.from_fields(radii=radius), static=True)
    last = [rr.TimeColumn(timeline, duration=times_ns[-1:].astype("timedelta64[ns]"))]
    recording.send_columns(f"{root}/trajectory", indexes=last, columns=strips.partition([len(times_ns) - 1]))
    recording.log(f"{root}/trajectory", rr.LineStrips3D.from_fields(radii=radius), static=True)
    endpoint = rr.Radius.ui_points(ENDPOINT_RADIUS_UI)
    recording.log(f"{root}/endpoints/start", rr.Points3D(positions[:1], colors=colors[:1], radii=endpoint, labels=["start"], show_labels=True), static=True)
    angles: Float64[ndarray, " a"] = np.linspace(0.0, 2.0 * np.pi, RING_POINTS)
    ring: Float64[ndarray, "a 3"] = positions[0] + START_RING_M * np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    recording.log(f"{root}/endpoints/start_ring", rr.LineStrips3D([ring], colors=colors[:1], radii=rr.Radius.ui_points(START_RING_RADIUS_UI)), static=True)
    recording.send_columns(f"{root}/endpoints/end", indexes=last, columns=rr.Points3D.columns(positions=positions[-1:], colors=colors[-1:]))
    recording.log(f"{root}/endpoints/end", rr.Points3D.from_fields(radii=endpoint, labels=["end"], show_labels=True), static=True)


def log_view_rig(
    recording: rr.RecordingStream,
    root: str,
    frames_meta: FramesMeta,
    manifest: FrameSourceManifest,
    rig_t_ns: Int64[ndarray, " k"],
    world_T_rig: Float64[ndarray, "k 4 4"],
    source: ViewRigSource,
) -> None:
    """Log `<root>/rig_view`: the rig at every viewing frame (`view_rig_poses`), its cameras as under `<root>/rig`, and
    under each camera's pinhole a reference to that camera's viewing face (`video_view`, beside the stream the run read)
    at each frame.

    Args:
        recording: The layer's recording.
        root: `/world/runs/<source>`.
        frames_meta: The run's cameras and keyframes (which camera reads which stream).
        manifest: The run's frame source: each camera's stream.
        rig_t_ns: Registered rig frame stamps, increasing.
        world_T_rig: The run's poses at them.
        source: The prior and the viewing frames.
    """
    view_t_ns, world_T_view = view_rig_poses(rig_t_ns, world_T_rig, source)
    if len(view_t_ns) == 0:
        return
    index = [rr.TimeColumn(manifest.timeline, duration=view_t_ns.astype("timedelta64[ns]"))]
    recording.send_columns(f"{root}/rig_view", indexes=index, columns=rr.Transform3D.columns(translation=world_T_view[:, :3, 3], mat3x3=world_T_view[:, :3, :3]))
    by_name: dict[str, KeyframeMeta] = {keyframe.image_name: keyframe for keyframe in frames_meta.keyframes}
    for stream in manifest.streams:
        named: KeyframeMeta | None = next((by_name[name] for name in stream.names if name in by_name), None)
        if named is None:
            continue
        camera: CameraParams = frames_meta.cameras[named.camera_params_id]
        entity: str = f"{root}/rig_view/{camera.sensor_name}"
        _log_camera(recording, entity, camera)
        video_view: str | None = source.view_entities.get("/" + stream.entity.lstrip("/"))
        if video_view is None:
            continue
        recording.log(f"{entity}/pinhole/video", rr.VideoFrameReference.from_fields(video_reference=video_view), static=True)
        recording.send_columns(f"{entity}/pinhole/video", indexes=index, columns=rr.VideoFrameReference.columns_nanos(view_t_ns))


def write_run_layer(
    run_dir: Path,
    rrd_path: Path,
    *,
    segment: str,
    source: str = DEFAULT_SOURCE,
    timeline: str = "video_time",
    frames: FrameSourceManifest | None = None,
    view: ViewRigSource | None = None,
) -> LayerSummary:
    """Write one finished run as a layer of `segment`.

    Args:
        run_dir: The run's output directory (`sparse/`, `kpmap/`, `runtime.csv`, `summary.json`).
        rrd_path: Where the layer goes; parent directories are created.
        segment: The catalog segment (recording id) the run was read from.
        source: Run source under `/world/runs/`.
        timeline: Timeline of the rig poses; the segment's video timeline.
        frames: The run's frame source, for a catalog input: the frusta then reference their pictures in the segment's
            video streams (`log_frame_references`).
        view: With `frames`, the prior and the segment's viewing frames: the layer then carries `rig_view` (`log_view_rig`).

    Returns:
        What was written.

    Raises:
        ValueError: If ``view`` comes without ``frames`` (the viewing rig references the streams ``frames`` names).
    """
    if view is not None and frames is None:
        raise ValueError("a viewing rig needs the run's frame source (frames)")
    artifacts: RunArtifacts = read_run(run_dir, source)
    reconstruction: pycolmap.Reconstruction = artifacts.reconstruction
    registered: set[str] = {image.name for image in reconstruction.images.values() if image.has_pose}
    # One pose per input frame when the run wrote them; a run without them carries its registered keyframes.
    posed: FramesMeta = artifacts.frames_meta if artifacts.every_frame is None else artifacts.every_frame
    times_ns, world_T_rig = rig_poses(posed) if artifacts.every_frame is not None else rig_poses(posed, registered)
    metrics: ReconstructionMetrics = reconstruction_metrics(reconstruction)

    root: str = f"/world/runs/{source}"
    recording: rr.RecordingStream = open_layer(rrd_path, segment)
    log_rig_trajectory(recording, root, times_ns, world_T_rig, timeline)
    for camera in artifacts.frames_meta.cameras.values():
        _log_camera(recording, f"{root}/rig/{camera.sensor_name}", camera)
    if frames is not None:
        log_frame_references(recording, root, posed, frames)
        if view is not None:
            log_view_rig(recording, root, artifacts.frames_meta, frames, times_ns, world_T_rig, view)
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
    # The keyframes the run selected, with the poses it was given: what the optimisation started from.
    prior_path: Path = run_dir / KEYFRAME_DIR_NAME / FRAMES_META_NAME
    if prior_path.is_file():
        prior_ns, world_T_prior = rig_poses(read_frames_meta(prior_path))
        log_position_difference(recording, f"{root}/position_difference", prior_ns, world_T_prior[:, :3, 3], times_ns, world_T_rig[:, :3, 3], timeline)
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
