"""The overview of one catalog run: input trajectory, colsfm's refinement and ground truth, with the clip's video.

`log_overview` writes, under `/overview` on the segment's `video_time` timeline:

- `world/gt/trajectory` (green), `world/input/trajectory` (blue) and `world/colsfm/trajectory` (orange), the last two
  aligned onto ground truth by one SE(3) Umeyama fit each (scale held at 1, as SE(3) ATE is scored);
- `world/colsfm/rig`: the refined rig pose over time, and under it every camera with its refined extrinsic, its pinhole
  and its video — the segment's own `VideoStream` packets, copied from the catalog, not re-encoded;
- `world/colsfm/points`: the sparse model, aligned the same way;
- `notes`: the numbers table (ATE of input and output under SE(3) and Sim(3), registered images, points,
  reprojection, wall time per stage).

`overview_tab` lays those out as one blueprint tab, to sit in front of the stage-by-stage walkthrough tabs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_video_codec import video_codec
from simplecv.catalog_video_reader import CatalogVideoReader, StreamIndex, iter_chunks, read_video_index

from colsfm.benchmark import ReconstructionMetrics, RunArtifacts, reconstruction_metrics
from colsfm.catalog import CatalogCameraModel, CatalogTrajectory
from colsfm.rerun_layer import rig_poses
from colsfm.rigid_fit import RigidAlignment, align_rigid
from colsfm.trajectory import MIN_ALIGNMENT_SAMPLES

ROOT: Final[str] = "/overview"
"""Entity root of the overview, apart from the walkthrough's stage roots."""

POINT_CROP_FACTOR: Final[float] = 3.0
"""The overview keeps the points within this many path extents of the path's centre."""

GT_RGB: Final[tuple[int, int, int]] = (60, 200, 90)
INPUT_RGB: Final[tuple[int, int, int]] = (60, 140, 255)
OUTPUT_RGB: Final[tuple[int, int, int]] = (255, 140, 0)


@dataclass(frozen=True)
class TrajectoryError:
    """ATE of one trajectory against ground truth, in metres; unscored (None) when too few poses overlap it."""

    se3_rmse_m: float | None
    """RMSE after a rigid fit (scale held at 1)."""
    sim3_rmse_m: float | None
    """RMSE after a similarity fit."""
    sim3_scale: float | None
    """The similarity fit's scale."""
    num_matched: int
    """Poses joined to ground truth."""

    def table_cells(self) -> str:
        """This trajectory's cells in the notes table.

        Returns:
            `SE(3) | Sim(3) | scale | poses`, with `unscored` in place of the numbers when there are none.
        """
        if self.se3_rmse_m is None or self.sim3_rmse_m is None or self.sim3_scale is None:
            return f"unscored | unscored | – | {self.num_matched}"
        return f"{100 * self.se3_rmse_m:.2f} cm | {100 * self.sim3_rmse_m:.2f} cm | {self.sim3_scale:.4f} | {self.num_matched}"


@dataclass(frozen=True)
class OverviewNumbers:
    """What the overview's table says."""

    input_error: TrajectoryError
    """The starting trajectory."""
    output_error: TrajectoryError
    """colsfm's refined trajectory."""
    metrics: ReconstructionMetrics
    """Registered images, points, reprojection."""
    total_images: int
    """Images handed to the pipeline."""
    stage_seconds: dict[str, float]
    """Wall time per stage, from `runtime.csv`."""


def _join(
    times_ns: Int64[ndarray, " n"], poses: Float64[ndarray, "n 4 4"], ground_truth: CatalogTrajectory
) -> tuple[Float64[ndarray, "m 3"], Float64[ndarray, "m 3"]]:
    inside: np.ndarray = (times_ns >= ground_truth.t_ns[0]) & (times_ns <= ground_truth.t_ns[-1])
    return poses[inside, :3, 3], ground_truth.at(times_ns[inside])[:, :3, 3]


def trajectory_error(
    times_ns: Int64[ndarray, " n"], poses: Float64[ndarray, "n 4 4"], ground_truth: CatalogTrajectory
) -> tuple[TrajectoryError, RigidAlignment | None]:
    """ATE of `poses` against ground truth interpolated at the same instants, and the SE(3) fit that scores it.

    Fewer than `MIN_ALIGNMENT_SAMPLES` poses inside the ground truth's time span determine no fit: the error comes
    back unscored and the fit None, and the overview draws that trajectory where it is instead of aborting.

    Args:
        times_ns: Pose timestamps on the segment timeline.
        poses: `world_T_rig` per timestamp.
        ground_truth: The ground-truth trajectory.

    Returns:
        SE(3) and Sim(3) RMSE, and the SE(3) fit (which moves the trajectory into the ground-truth frame for display).
    """
    source, target = _join(times_ns, poses, ground_truth)
    if len(source) < MIN_ALIGNMENT_SAMPLES:
        return TrajectoryError(se3_rmse_m=None, sim3_rmse_m=None, sim3_scale=None, num_matched=len(source)), None
    rigid: RigidAlignment = align_rigid(source, target)
    similarity: RigidAlignment = align_rigid(source, target, estimate_scale=True)
    error: TrajectoryError = TrajectoryError(
        se3_rmse_m=rigid.rmse_meters, sim3_rmse_m=similarity.rmse_meters, sim3_scale=similarity.scale, num_matched=len(source)
    )
    return error, rigid


def log_overview(
    *,
    dataset: DatasetEntry,
    segment: str,
    cameras: Sequence[CatalogCameraModel],
    artifacts: RunArtifacts,
    input_times_ns: Int64[ndarray, " n"],
    input_poses: Float64[ndarray, "n 4 4"],
    ground_truth: CatalogTrajectory,
    timeline: str = "video_time",
    recording: rr.RecordingStream | None = None,
) -> OverviewNumbers:
    """Log the overview of one finished catalog run into `recording` (the global one by default).

    Args:
        dataset: The catalog dataset the run read.
        segment: The segment id.
        cameras: The cameras the run used, in entity order (`colsfm.catalog.read_catalog_cameras`).
        artifacts: The finished run (`colsfm.benchmark.read_run`).
        input_times_ns: The starting trajectory's timestamps.
        input_poses: The starting trajectory.
        ground_truth: Ground truth on the same timeline.
        timeline: The segment's video timeline.
        recording: Where to log.

    Returns:
        The numbers the notes table shows.
    """
    output_times_ns, output_poses = (
        rig_poses(artifacts.every_frame)
        if artifacts.every_frame is not None
        else rig_poses(artifacts.frames_meta, {image.name for image in artifacts.reconstruction.images.values() if image.has_pose})
    )
    input_error, input_fit = trajectory_error(input_times_ns, input_poses, ground_truth)
    output_error, output_fit = trajectory_error(output_times_ns, output_poses, ground_truth)
    input_xyz: Float64[ndarray, "n 3"] = input_poses[:, :3, 3] if input_fit is None else input_fit.apply(input_poses[:, :3, 3])
    path_xyz: Float64[ndarray, "n 3"] = output_poses[:, :3, 3] if output_fit is None else output_fit.apply(output_poses[:, :3, 3])

    world: str = f"{ROOT}/world"
    gt_xyz: Float64[ndarray, "g 3"] = ground_truth.at(ground_truth.t_ns[:: max(1, len(ground_truth.t_ns) // 4000)])[:, :3, 3]
    rr.log(f"{world}/gt/trajectory", rr.LineStrips3D([gt_xyz], colors=[GT_RGB], radii=rr.Radius.ui_points(2.0)), static=True, recording=recording)
    rr.log(
        f"{world}/input/trajectory",
        rr.LineStrips3D([input_xyz], colors=[INPUT_RGB], radii=rr.Radius.ui_points(1.5)),
        static=True,
        recording=recording,
    )
    rr.log(
        f"{world}/colsfm/trajectory",
        rr.LineStrips3D([path_xyz], colors=[OUTPUT_RGB], radii=rr.Radius.ui_points(2.0)),
        static=True,
        recording=recording,
    )
    # Points farther from the path than POINT_CROP_FACTOR x its own extent are dropped from the overview (they are
    # kept in the walkthrough's stage 7): a handful of far triangulations would otherwise set the 3D view's framing.
    points_xyz: Float64[ndarray, "p 3"] = artifacts.points_xyz() if output_fit is None else output_fit.apply(artifacts.points_xyz())
    centre: Float64[ndarray, " 3"] = path_xyz.mean(axis=0)
    extent: float = float(np.linalg.norm(path_xyz - centre, axis=1).max()) + 1.0
    near: np.ndarray = np.linalg.norm(points_xyz - centre, axis=1) <= POINT_CROP_FACTOR * extent
    rr.log(
        f"{world}/colsfm/points",
        rr.Points3D(points_xyz[near], colors=artifacts.points_rgb().astype(np.uint8)[near], radii=rr.Radius.ui_points(1.5)),
        static=True,
        recording=recording,
    )

    # The refined rig, moved into the ground-truth frame, carrying the cameras and their video.
    rig_rotations: Float64[ndarray, "n 3 3"] = output_poses[:, :3, :3] if output_fit is None else output_fit.apply_rotations(output_poses[:, :3, :3])
    rig_entity: str = f"{world}/colsfm/rig"
    rr.send_columns(
        rig_entity,
        indexes=[rr.TimeColumn(timeline, duration=output_times_ns.astype("timedelta64[ns]"))],
        columns=rr.Transform3D.columns(translation=path_xyz, mat3x3=rig_rotations),
        recording=recording,
    )
    refined_by_name = {camera.sensor_name: camera for camera in artifacts.frames_meta.cameras.values()}
    entities: list[str] = [f"{rig_entity}/{camera.name}" for camera in cameras]
    for camera, entity in zip(cameras, entities, strict=True):
        refined = refined_by_name.get(camera.name)
        rig_T_cam: Float64[ndarray, "3 4"] = refined.vehicle_T_cam.matrix() if refined is not None else camera.rig_T_cam[:3, :]
        rr.log(entity, rr.Transform3D(translation=rig_T_cam[:, 3], mat3x3=rig_T_cam[:, :3]), static=True, recording=recording)
        rr.log(
            f"{entity}/pinhole",
            rr.Pinhole(
                image_from_camera=camera.k_matrix,
                resolution=[camera.width, camera.height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.12,
                color=OUTPUT_RGB,
            ),
            static=True,
            recording=recording,
        )
    relay_source_video(
        dataset,
        segment,
        [f"{camera.entity}/pinhole/video" for camera in cameras],
        [f"{entity}/pinhole/video" for entity in entities],
        timeline,
        recording,
    )

    numbers: OverviewNumbers = OverviewNumbers(
        input_error=input_error,
        output_error=output_error,
        metrics=reconstruction_metrics(artifacts.reconstruction),
        total_images=len(artifacts.frames_meta.keyframes),
        stage_seconds={str(stage): float(seconds) for stage, seconds in artifacts.runtime_seconds_by_stage.items()},
    )
    rr.log(f"{ROOT}/notes", rr.TextDocument(notes_markdown(segment, numbers), media_type=rr.MediaType.MARKDOWN), static=True, recording=recording)
    return numbers


def relay_source_video(
    dataset: DatasetEntry, segment: str, sources: Sequence[str], targets: Sequence[str], timeline: str, recording: rr.RecordingStream | None
) -> None:
    """Re-log each source ``VideoStream``'s stored samples at its target, never decoded, a chunk of every camera at a time.

    Memory is a few chunks, not the session.

    Args:
        dataset: Dataset holding the segment.
        segment: Segment to read.
        sources: The source ``VideoStream`` entities.
        targets: One per source, in the same order.
        timeline: Read from, and logged on.
        recording: None logs to the global recording.
    """
    index: tuple[StreamIndex, ...] = read_video_index(dataset, segment, sources, timeline)
    for target, stream in zip(targets, index, strict=True):
        rr.log(target, rr.VideoStream(codec=video_codec(stream.codec)), static=True, recording=recording)
    reader: CatalogVideoReader = CatalogVideoReader(dataset, segment, index, timeline)
    for chunk in iter_chunks(reader, range(reader.num_chunks)):
        for target, stream, packets in zip(targets, index, chunk.streams, strict=True):
            if packets.samples:
                stop: int = packets.first + len(packets.samples)
                rr.send_columns(
                    target,
                    indexes=[rr.TimeColumn(timeline, duration=stream.t_ns[packets.first : stop].astype("timedelta64[ns]"))],
                    columns=rr.VideoStream.columns(sample=packets.samples, is_keyframe=packets.keyframes),
                    recording=recording,
                )


def overview_eye(ground_truth: CatalogTrajectory) -> tuple[list[float], list[float]]:
    """A 3D eye that looks at the ground-truth path from above and to the side, far enough to see all of it.

    Args:
        ground_truth: The ground-truth trajectory (the overview's frame).

    Returns:
        `(position, look_target)`.
    """
    xyz: Float64[ndarray, "n 3"] = ground_truth.translations
    centre: Float64[ndarray, " 3"] = xyz.mean(axis=0)
    radius: float = float(np.linalg.norm(xyz - centre, axis=1).max()) + 0.5
    offset: Float64[ndarray, " 3"] = np.array([1.2, -1.2, 1.4]) * radius
    return (centre + offset).tolist(), centre.tolist()


def notes_markdown(segment: str, numbers: OverviewNumbers) -> str:
    """The overview's numbers as Markdown.

    Args:
        segment: The segment id.
        numbers: The run's numbers.

    Returns:
        A Markdown document.
    """
    rows: list[str] = [
        f"# colsfm on `{segment}`",
        "",
        "| trajectory | SE(3) ATE | Sim(3) ATE | Sim(3) scale | poses |",
        "|---|---:|---:|---:|---:|",
        f"| input | {numbers.input_error.table_cells()} |",
        f"| colsfm | {numbers.output_error.table_cells()} |",
        "",
        f"Registered {numbers.metrics.registered_images} / {numbers.total_images} images, {numbers.metrics.num_points3D} points, "
        f"mean reprojection {numbers.metrics.mean_reprojection_error_px:.3f} px.",
        "",
        "| stage | seconds |",
        "|---|---:|",
        *[f"| {stage} | {seconds:.1f} |" for stage, seconds in numbers.stage_seconds.items()],
        f"| **total** | **{sum(numbers.stage_seconds.values()):.1f}** |",
    ]
    return "\n".join(rows)


def overview_tab(camera_names: Sequence[str], eye: tuple[Sequence[float], Sequence[float]] | None = None) -> rrb.Container:
    """The Overview tab: the 3D comparison, every camera's video, and the numbers.

    Args:
        camera_names: The cameras `log_overview` logged, in entity order.
        eye: `(position, look_target)` of the 3D view, e.g. from `overview_eye`; the viewer frames everything without it.

    Returns:
        A blueprint container.
    """
    rig: str = f"{ROOT}/world/colsfm/rig"
    return rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial3DView(
                origin=f"{ROOT}/world",
                name="GT (green) · input (blue) · colsfm (orange)",
                eye_controls=None if eye is None else rrb.EyeControls3D(position=eye[0], look_target=eye[1]),
            ),
            rrb.Grid(
                *[rrb.Spatial2DView(origin=f"{rig}/{name}/pinhole", name=name, contents=["+ $origin/**"]) for name in camera_names],
                grid_columns=max(1, len(camera_names)),
            ),
            row_shares=[3.0, 1.0],
        ),
        rrb.TextDocumentView(origin=f"{ROOT}/notes", name="Numbers"),
        column_shares=[3.0, 1.2],
        name="Overview",
    )
