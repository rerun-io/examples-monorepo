"""Log a finished MetricTrail run (its ViPE artifacts) to a Rerun recording; layout in :mod:`metrictrail.entities`."""

import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from serde import serde

from metrictrail import entities
from metrictrail.artifacts import ColoredCloud, iter_depth_zip, read_intrinsics, subsample_cloud
from metrictrail.blueprint import SceneView, scene_view
from metrictrail.eval.protocols import VslamlabDataset
from metrictrail.eval.trajectory import apply_alignment, running_rmse
from metrictrail.eval.vipe_output import MetricTrailTrajectory, RunRecord, read_run_record, read_trajectory
from metrictrail.frames import DisplayRaster, FrameReader, FrameSource, fit_for_display
from metrictrail.ground_truth import AlignedRun, GtContext, align_to_gt
from metrictrail.palette import GT_RGB, RUNNING_RMSE_RGB, RunLabel, run_rgb
from metrictrail.properties import PropertyValue, any_values
from metrictrail.rerun_log import FrameVideo, SourceRaster, log_depth, log_pinholes, set_frame_time
from metrictrail.surface import ColoredMesh

GT_VIEW: dict[VslamlabDataset, rr.components.ViewCoordinates] = {
    "rgbdtum": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    "7scenes": rr.ViewCoordinates.RDF,
    "kitti": rr.ViewCoordinates.RDF,
}
"""World axes of each benchmark's GT frame (TUM mocap: z up; 7-Scenes and KITTI: a camera frame, y down)."""
Z_UP: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, 0.0, 1.0), (-0.7071, -0.7071, 0.0))
"""(up, back) of a z-up world."""
RDF_UP: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
"""(up, back) of a camera-like RDF world (ViPE's world: the first camera's frame)."""


@dataclass(frozen=True, slots=True)
class LoggedRun:
    """What :func:`log_run_outputs` logged: the numbers and the eyes the blueprint frames them with."""

    metrics: "RunMetrics"
    """The ``metrics`` property."""
    world_eye: SceneView | None
    """Eye of the ``world`` view (None without a finite pose)."""
    metric_eye: SceneView | None
    """Eye of the ``metric`` view (None without GT)."""


@dataclass(frozen=True, slots=True)
class RunOutputs:
    """The artifacts of one sequence in a MetricTrail output directory."""

    run_dir: Path
    """``pipeline.output.path``."""
    name: str
    """Artifact name (the input directory name or video stem)."""
    trajectory: MetricTrailTrajectory
    """``pose/<name>.npz``."""
    intrinsics: Float64[ndarray, "n 4"]
    """``intrinsics/<name>.npz`` per tracked frame."""
    record: RunRecord | None
    """``vipe/<name>_metrictrail.json``, if the run wrote it."""


def read_run_outputs(run_dir: Path, name: str) -> RunOutputs:
    """Read pose, intrinsics and run record of ``name`` in a MetricTrail output directory."""
    trajectory: MetricTrailTrajectory = read_trajectory(run_dir, name)
    record_path: Path = run_dir / "vipe" / f"{name}_metrictrail.json"
    record: RunRecord | None = read_run_record(record_path) if record_path.is_file() else None
    return RunOutputs(
        run_dir=run_dir,
        name=name,
        trajectory=trajectory,
        intrinsics=read_intrinsics(run_dir, name, trajectory.world_T_cam.shape[0]),
        record=record,
    )


def artifact_name(run_dir: Path) -> str:
    """The only ``pose/<name>.npz`` in an output directory.

    Raises:
        ValueError: If there is none or more than one (pass the name explicitly then).
    """
    names: list[str] = sorted(path.stem for path in (run_dir / "pose").glob("*.npz"))
    if len(names) != 1:
        raise ValueError(f"{run_dir / 'pose'} holds {names}; pass the artifact name")
    return names[0]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a recording's data came from (the ``provenance`` recording property)."""

    dataset_path: str
    """The sequence the run read (VSLAM-LAB sequence directory when there is GT, else the input)."""
    frames_read: str
    """Frame directory or video ViPE read."""
    run_output_dir: str
    """MetricTrail output directory."""
    producer: str
    """Which tree ran (upstream ViPE or our port) and which tool logged it."""
    env: str
    """Python environment that logged it."""
    utc: str
    """When it was logged."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Trajectory numbers of one run (the ``metrics`` recording property); GT fields are null without GT."""

    num_tracked: int
    """Tracked frames (rows of the pose artifact)."""
    num_finite: int
    """Tracked frames with a finite pose."""
    num_gt_pairs: int | None
    """Tracked frames paired with GT by the paper's rule."""
    ate_sim3_m: float | None
    """ATE RMSE after Sim(3) Umeyama (the paper's metric), metres; for a figure, not the evaluation of record."""
    sim3_scale: float | None
    """Scale of that alignment (1.0 = exact metric scale)."""
    ate_se3_m: float | None
    """ATE RMSE after SE(3) alignment with scale 1 (metric-scale diagnostic), metres."""
    cloud_points: int | None
    """Points of the fused cloud before display subsampling."""
    mesh_triangles: int | None = None
    """Triangles of the TSDF mesh (:mod:`metrictrail.surface`)."""


@dataclass(frozen=True, slots=True)
class LogOptions:
    """How :func:`log_run_outputs` logs."""

    label: RunLabel
    """Which tree produced the run (entity names and colour)."""
    log_frames: bool
    """Log ``cam``/``cam/depth`` from the depth artifact and, for a frame-directory input, ``cam/rgb`` (a run without
    live capture; a video input is the caller's ``input/video``, :func:`metrictrail.rerun_log.log_input_video`)."""
    raster_policy: DisplayRaster
    """How the input frames reached the tracking raster (display fit of ``cam/rgb``)."""
    depth_range_m: tuple[float, float]
    """Turbo range of depth images, metres."""
    depth_every: int = 1
    """Log every k-th tracked frame's depth image and RGB frame."""
    max_cloud_points: int = 400_000
    """Display subsample of the fused cloud."""
    image_plane_distance_m: float = 0.08
    """Frustum size of the trajectory camera in the 3D views, metres."""


def rotation_of(gt_T_est: Float64[ndarray, "4 4"]) -> Float64[ndarray, "3 3"]:
    """The rotation of a similarity matrix (its 3x3 block divided by the scale)."""
    block: Float64[ndarray, "3 3"] = gt_T_est[:3, :3]
    return block / np.cbrt(np.linalg.det(block))


def normal_colors(normals_xyz: Float32[ndarray, "n 3"], gt_T_est: Float64[ndarray, "4 4"]) -> UInt8[ndarray, "n 3"]:
    """Unit normals rotated by an alignment's rotation, as 8-bit RGB ``(n + 1) / 2`` of the target frame's axes."""
    rotated: Float64[ndarray, "n 3"] = normals_xyz.astype(np.float64) @ rotation_of(gt_T_est).T
    return np.clip(np.round((rotated + 1.0) * 127.5), 0, 255).astype(np.uint8)


def log_mesh(rec: rr.RecordingStream, mesh: ColoredMesh, gt_T_est: Float64[ndarray, "4 4"], colour_path: str, normals_path: str) -> None:
    """The mesh moved by an alignment, twice: fused colours (lit) at ``colour_path``, normals as RGB at ``normals_path``."""
    vertices: Float32[ndarray, "v 3"] = apply_alignment(gt_T_est, mesh.vertices_xyz.astype(np.float64)).astype(np.float32)
    normals: Float32[ndarray, "v 3"] = (mesh.normals_xyz.astype(np.float64) @ rotation_of(gt_T_est).T).astype(np.float32)
    lit = rr.Mesh3D(vertex_positions=vertices, triangle_indices=mesh.triangles, vertex_normals=normals, vertex_colors=mesh.colors_rgb)
    rec.log(colour_path, lit, static=True)
    # The normals copy carries its normals as colours only: no lighting needed, 12 bytes per vertex fewer.
    colours = normal_colors(mesh.normals_xyz, gt_T_est)
    rec.log(normals_path, rr.Mesh3D(vertex_positions=vertices, triangle_indices=mesh.triangles, vertex_colors=colours), static=True)


def log_run_outputs(
    rec: rr.RecordingStream,
    outputs: RunOutputs,
    options: LogOptions,
    *,
    frames: FrameSource | None,
    context: GtContext | None,
    cloud_world: ColoredCloud | None,
    provenance: Provenance,
    mesh_world: ColoredMesh | None = None,
    video_time_ns: Int64[ndarray, "s"] | None = None,
) -> LoggedRun:
    """Log trajectory, GT, cloud, errors, frames and depth of a finished run, and its properties.

    Args:
        rec: Target recording.
        outputs: The run's artifacts.
        options: What to log and how.
        frames: The input the run read (``cam/rgb`` when ``options.log_frames``); None skips the frames.
        context: GT of the sequence; None logs the trajectory in ViPE's world frame with no errors.
        cloud_world: The fused metric cloud in ViPE's world frame (the frame of ``pose/<name>.npz``); None skips it.
        provenance: The ``provenance`` property.
        mesh_world: The TSDF mesh in ViPE's world frame; None skips it.
        video_time_ns: ``video_time`` of each source frame (the input video's PTS); None: the GT stamps with GT, else no
            ``video_time``.

    Returns:
        The numbers also sent as the ``metrics`` property, and eyes that frame the trajectory.
    """
    trajectory: MetricTrailTrajectory = outputs.trajectory
    world_T_cam: Float64[ndarray, "n 4 4"] = trajectory.world_T_cam
    num_tracked: int = world_T_cam.shape[0]
    finite_rows: Int64[ndarray, "k"] = np.flatnonzero(np.isfinite(world_T_cam.reshape(num_tracked, 16)).all(axis=1)).astype(np.int64)
    color: tuple[int, int, int] = run_rgb(options.label)
    tracked: Int64[ndarray, "n"] = np.arange(num_tracked, dtype=np.int64)
    if video_time_ns is None and context is not None:
        video_time_ns = context.frames.timestamps_ns
    rec.log(entities.WORLD_ROOT, GT_VIEW[context.protocol.dataset] if context is not None else rr.ViewCoordinates.RDF, static=True)

    aligned: AlignedRun | None = align_to_gt(world_T_cam, trajectory.source_frame_ids, context) if context is not None else None
    sim3: Float64[ndarray, "4 4"] = aligned.sim3.gt_T_est if aligned is not None else np.eye(4)
    positions: Float64[ndarray, "k 3"] = apply_alignment(sim3, world_T_cam[finite_rows, :3, 3])
    rotations: Float64[ndarray, "k 3 3"] = np.einsum("ij,njk->nik", rotation_of(sim3), world_T_cam[finite_rows, :3, :3])
    rec.log(entities.world_trajectory(options.label), rr.LineStrips3D([positions], colors=[color], radii=rr.Radius.ui_points(2.0)), static=True)
    intrinsics0: Float64[ndarray, "4"] = outputs.intrinsics[finite_rows[0]] if finite_rows.size else np.full(4, np.nan)
    raster_hw: tuple[int, int] | None = None
    if outputs.record is not None and outputs.record.input_raster.prepared_hw is not None:
        raster_hw = (outputs.record.input_raster.prepared_hw[0], outputs.record.input_raster.prepared_hw[1])
    if raster_hw is None:
        raster_hw = (round(2 * intrinsics0[3]), round(2 * intrinsics0[2]))
    rec.log(
        entities.world_cam(options.label),
        rr.Pinhole(
            focal_length=intrinsics0[:2],
            principal_point=intrinsics0[2:],
            resolution=[raster_hw[1], raster_hw[0]],
            image_plane_distance=options.image_plane_distance_m,
            color=color,
            line_width=-2.0,  # negative = UI points (rr.Radius.ui_points); a positive value is metres
        ),
        static=True,
    )
    indexes: list[rr.TimeColumn] = time_columns(finite_rows, trajectory.source_frame_ids, video_time_ns)
    rec.send_columns(entities.world_cam(options.label), indexes=indexes, columns=rr.Transform3D.columns(translation=positions, mat3x3=rotations))

    if aligned is not None:
        gt_rows: Int64[ndarray, "m"] = aligned.rows
        rec.log(entities.WORLD_GT_TRAJECTORY, rr.LineStrips3D([aligned.gt_xyz], colors=[GT_RGB], radii=rr.Radius.ui_points(2.0)), static=True)
        rec.log(entities.METRIC_GT_TRAJECTORY, rr.LineStrips3D([aligned.gt_xyz], colors=[GT_RGB], radii=rr.Radius.ui_points(2.0)), static=True)
        rec.log(entities.WORLD_GT_POSITION, rr.Points3D.from_fields(colors=[GT_RGB], radii=rr.Radius.ui_points(6.0)), static=True)
        gt_indexes: list[rr.TimeColumn] = time_columns(gt_rows, trajectory.source_frame_ids, video_time_ns)
        rec.send_columns(entities.WORLD_GT_POSITION, indexes=gt_indexes, columns=rr.Points3D.columns(positions=aligned.gt_xyz))
        metric_xyz: Float64[ndarray, "m 3"] = apply_alignment(aligned.se3.gt_T_est, world_T_cam[gt_rows, :3, 3])
        rec.log(entities.metric_trajectory(options.label), rr.LineStrips3D([metric_xyz], colors=[color], radii=rr.Radius.ui_points(2.0)), static=True)
        for path, values, line_color, name, width in (
            (entities.ATE_SIM3_ERR, aligned.sim3_err_m, color, "per-frame error, Sim(3) (m)", 1.5),
            (entities.ATE_SIM3_RUNNING, running_rmse(aligned.sim3_err_m), RUNNING_RMSE_RGB, "running ATE RMSE, Sim(3) (m)", 2.5),
            (entities.ATE_SE3_ERR, aligned.se3_err_m, color, "per-frame error, SE(3) scale 1 (m)", 1.5),
            (entities.ATE_SE3_RUNNING, running_rmse(aligned.se3_err_m), RUNNING_RMSE_RGB, "running ATE RMSE, SE(3) (m)", 2.5),
        ):
            rec.log(path, rr.SeriesLines(colors=[line_color], names=[name], widths=[width]), static=True)
            rec.send_columns(path, indexes=gt_indexes, columns=rr.Scalars.columns(scalars=values))

    cloud_points: int | None = None
    if cloud_world is not None:
        cloud_points = int(cloud_world.points_xyz.shape[0])
        shown: ColoredCloud = subsample_cloud(cloud_world, options.max_cloud_points)
        world_xyz: Float32[ndarray, "c 3"] = apply_alignment(sim3, shown.points_xyz.astype(np.float64)).astype(np.float32)
        rec.log(entities.WORLD_CLOUD, rr.Points3D(world_xyz, colors=shown.colors_rgb, radii=rr.Radius.ui_points(1.5)), static=True)
        if shown.normals_xyz is not None:
            rec.log(
                entities.WORLD_CLOUD_NORMALS,
                rr.Points3D(world_xyz, colors=normal_colors(shown.normals_xyz, sim3), radii=rr.Radius.ui_points(1.5)),
                static=True,
            )
        if aligned is not None:
            metric_cloud: Float32[ndarray, "c 3"] = apply_alignment(aligned.se3.gt_T_est, shown.points_xyz.astype(np.float64)).astype(np.float32)
            rec.log(entities.METRIC_CLOUD, rr.Points3D(metric_cloud, colors=shown.colors_rgb, radii=rr.Radius.ui_points(1.5)), static=True)
            if shown.normals_xyz is not None:
                rec.log(
                    entities.METRIC_CLOUD_NORMALS,
                    rr.Points3D(metric_cloud, colors=normal_colors(shown.normals_xyz, aligned.se3.gt_T_est), radii=rr.Radius.ui_points(1.5)),
                    static=True,
                )

    if mesh_world is not None:
        log_mesh(rec, mesh_world, sim3, entities.WORLD_MESH, entities.WORLD_MESH_NORMALS)
        if aligned is not None:
            log_mesh(rec, mesh_world, aligned.se3.gt_T_est, entities.METRIC_MESH, entities.METRIC_MESH_NORMALS)

    if options.log_frames:
        log_frames_and_depth(rec, outputs, options, frames, tracked, raster_hw, video_time_ns)

    metrics = RunMetrics(
        num_tracked=num_tracked,
        num_finite=int(finite_rows.shape[0]),
        num_gt_pairs=int(aligned.rows.shape[0]) if aligned is not None else None,
        ate_sim3_m=aligned.sim3.rmse_m if aligned is not None else None,
        sim3_scale=aligned.sim3.scale if aligned is not None else None,
        ate_se3_m=aligned.se3.rmse_m if aligned is not None else None,
        cloud_points=cloud_points,
        mesh_triangles=int(mesh_world.triangles.shape[0]) if mesh_world is not None else None,
    )
    rec.log(entities.SUMMARY, rr.TextDocument(summary_markdown(outputs, options.label, metrics), media_type=rr.MediaType.MARKDOWN), static=True)
    rec.send_property(entities.PROVENANCE_PROPERTY, any_values({item.name: getattr(provenance, item.name) for item in fields(provenance)}))
    rec.send_property(entities.METRICS_PROPERTY, any_values({item.name: getattr(metrics, item.name) for item in fields(metrics)}))
    if outputs.record is not None:
        rec.send_property(entities.RUN_PROPERTY, any_values(run_property(outputs.record)))
    up, back = Z_UP if context is not None and context.protocol.dataset == "rgbdtum" else RDF_UP
    world_eye: SceneView | None = scene_view(positions, np.array(up), np.array(back)) if positions.shape[0] else None
    metric_eye: SceneView | None = scene_view(aligned.gt_xyz, np.array(up), np.array(back)) if aligned is not None else None
    return LoggedRun(metrics=metrics, world_eye=world_eye, metric_eye=metric_eye)


def time_columns(rows: Int64[ndarray, "k"], source_frame_ids: Int64[ndarray, "n"], video_time_ns: Int64[ndarray, "s"] | None) -> list[rr.TimeColumn]:
    """``frame``, ``source_frame`` and, when known, ``video_time`` index columns of the tracked rows ``rows``
    (``video_time_ns`` indexed by source frame)."""
    sources: Int64[ndarray, "k"] = source_frame_ids[rows]
    indexes: list[rr.TimeColumn] = [
        rr.TimeColumn(entities.FRAME_TIMELINE, sequence=rows),
        rr.TimeColumn(entities.SOURCE_FRAME_TIMELINE, sequence=sources),
    ]
    if video_time_ns is not None:
        indexes.append(rr.TimeColumn(entities.VIDEO_TIMELINE, duration=video_time_ns[sources].astype("timedelta64[ns]")))
    return indexes


def log_frames_and_depth(
    rec: rr.RecordingStream,
    outputs: RunOutputs,
    options: LogOptions,
    frames: FrameSource | None,
    tracked: Int64[ndarray, "n"],
    raster_hw: tuple[int, int],
    video_time_ns: Int64[ndarray, "s"] | None,
) -> None:
    """``cam`` pinhole (and ``input`` for a video input), ``cam/rgb`` (a frame directory's frames on the raster, as a
    video) and ``cam/depth`` (depth artifact) of every k-th frame."""
    chosen: list[int] = [int(row) for row in tracked[:: options.depth_every]]
    source_frame_ids: Int64[ndarray, "n"] = outputs.trajectory.source_frame_ids
    raster: SourceRaster | None = None
    if frames is not None and frames.kind == "video" and outputs.record is not None:
        raster = SourceRaster.from_record(outputs.record.input_raster)
    read: FrameReader | None = frames.reader() if frames is not None and frames.kind == "frame_dir" else None
    rgb_video: FrameVideo | None = FrameVideo(rec, entities.CAM_RGB, fps=30.0) if read is not None else None
    depth_path: Path = outputs.run_dir / "depth" / f"{outputs.name}.zip"
    depth_by_row: dict[int, Float32[ndarray, "h w"]] = {}
    depth_iter = iter_depth_zip(depth_path, chosen) if depth_path.is_file() else iter(())
    for row in chosen:
        source: int = int(source_frame_ids[row])
        set_frame_time(rec, row, source, video_time_ns)
        intrinsics_px: Float64[ndarray, "4"] = outputs.intrinsics[row]
        if np.isfinite(intrinsics_px).all():
            log_pinholes(rec, intrinsics_px, raster_hw, raster)
        if rgb_video is not None and read is not None:
            rgb_video.log(fit_for_display(read(source), raster_hw, options.raster_policy))
        while row not in depth_by_row:
            nxt: tuple[int, Float32[ndarray, "h w"]] | None = next(depth_iter, None)
            if nxt is None:
                break
            depth_by_row[nxt[0]] = nxt[1]
        depth_m: Float32[ndarray, "h w"] | None = depth_by_row.pop(row, None)
        if depth_m is not None:
            log_depth(rec, depth_m, options.depth_range_m)
    if rgb_video is not None:
        rgb_video.close()


def run_property(record: RunRecord) -> dict[str, PropertyValue | None]:
    """The run-record fields sent as the ``run`` property (flat; lists become text)."""
    return {
        "name": record.name,
        "status": record.status,
        "tracking_status": record.tracking_status,
        "raster_policy": record.input_raster.policy,
        "raster_hw": "x".join(str(v) for v in record.input_raster.prepared_hw) if record.input_raster.prepared_hw else None,
        "frame_stride": record.input_raster.frame_stride,
        "intrinsics_source": record.intrinsics_source,
        "calibration_status": record.calibration_status,
        "bound_K_fx_fy_cx_cy": ", ".join(f"{v:.3f}" for v in record.bound_K_fx_fy_cx_cy) if record.bound_K_fx_fy_cx_cy else None,
        "runtime_seconds": record.runtime_seconds,
        "tracking_fps": record.runtime.tracking_fps if record.runtime is not None else None,
        "tracking_wall_seconds": record.runtime.tracking_wall_seconds if record.runtime is not None else None,
        "max_memory_allocated_gib": record.runtime.max_memory_allocated_gib if record.runtime is not None else None,
    }


def summary_markdown(outputs: RunOutputs, label: RunLabel, metrics: RunMetrics) -> str:
    """The ``summary`` card."""
    who: str = "our port" if label == "ours" else "upstream ViPE (unmodified)"
    lines: list[str] = [f"### MetricTrail, {who}: `{outputs.name}`", "", f"- tracked frames: **{metrics.num_tracked}** ({metrics.num_finite} finite)"]
    if metrics.mesh_triangles is not None:
        lines.append(f"- TSDF mesh: **{metrics.mesh_triangles:,}** triangles")
    if metrics.cloud_points is not None:
        lines.append(f"- fused metric cloud: **{metrics.cloud_points:,}** points")
    if metrics.ate_sim3_m is not None and metrics.sim3_scale is not None and metrics.ate_se3_m is not None:
        lines += [
            f"- ATE RMSE, Sim(3) (the paper's metric): **{100.0 * metrics.ate_sim3_m:.2f} cm** over {metrics.num_gt_pairs} frames",
            f"- Sim(3) scale: **{metrics.sim3_scale:.3f}** (1.000 = exact metric scale)",
            f"- ATE RMSE, SE(3) with scale 1: **{100.0 * metrics.ate_se3_m:.2f} cm**",
            "- figure numbers; the evaluation of record is `metrictrail.apis.vslamlab_eval`",
        ]
    else:
        lines.append("- no ground truth for this input")
    if outputs.record is not None and outputs.record.runtime is not None and outputs.record.runtime.tracking_fps is not None:
        lines.append(f"- tracking: {outputs.record.runtime.tracking_fps:.1f} frames/s (run record)")
    return "\n".join(lines)


def make_provenance(dataset_path: Path, frames_read: Path, run_dir: Path, producer: str, env: Path) -> Provenance:
    """A :class:`Provenance` stamped now."""
    return Provenance(
        dataset_path=str(dataset_path),
        frames_read=str(frames_read),
        run_output_dir=str(run_dir.resolve()),
        producer=producer,
        env=str(env),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
