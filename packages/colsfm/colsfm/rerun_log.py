"""Log a two-run cuSFM comparison into one Rerun recording.

The layout follows `demo_rerun.py`: two exoego rigs in the same world, logged
with simplecv's `log_rig_static` / `log_rig_pose_stream`, plus the trajectory
polylines the RoboCap catalog's slam layer uses.

```
/world
  /rig_00                       run A (the NVIDIA blob)   — world_T_rig on `frame_time`
    /cam_NN/pinhole             PinholeWithDistortion + frustum, static
  /rig_01                       run B (the colsfm pipeline)
  /points/rig_00                run A's sparse cloud, static, in the world frame
  /points/rig_01                run B's sparse cloud
  /runs/rig_00                  AnyValues{num_poses, source}
    /trajectory                 static LineStrips3D, per-segment colour ramp
    /endpoints                  static Points3D labelled start/end
  /runs/rig_01
  /runs/input                   the trajectory in the dataset's frames_meta.json
  /runs/ground_truth            ground_truth.txt, when the dataset ships one
/report                         the markdown report as a TextDocument
```

Two deliberate departures from the brief:

1. **The clouds sit at `/world/points/rig_NN`, not `/world/rig_NN/points`.**
   A rig node carries a *temporal* `world_T_rig`, and Rerun composes transforms
   down the tree — a cloud parented to the rig would be dragged along the
   trajectory and smear across the scene. The per-run identity is kept in the
   entity name instead, and the shared `/world/points` prefix is what the
   blueprint scopes.
2. **No imagery.** Video and JPEGs are out of scope here; the demo owns them.

Colour convention, inherited from the demo so the two recordings read alike:
run A (blob) orange, run B (colsfm) blue, input pale blue, ground truth green.
A run whose points are all black — the blob without `--output_rgb`, NOTES.md
gotcha 11 — falls back to its flat run colour rather than rendering an
invisible cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from simplecv.camera_parameters import (
    Extrinsics,
    Fisheye62Parameters,
    Intrinsics,
    KannalaBrandtDistortion,
    PinholeParameters,
)
from simplecv.rerun_rig_logger import log_rig_pose_stream, log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration, RigPoseStream, entity_id

from colsfm.bench_report import render_markdown_report
from colsfm.benchmark import Comparison, RigTrack, RunArtifacts
from colsfm.cameras import PINHOLE_FAMILY_MODELS
from colsfm.frames_meta import CameraParams, FramesMeta
from colsfm.geometry import Matrix3

Rgb: TypeAlias = tuple[int, int, int]
"""An opaque colour, 0-255 per channel."""

Rgba: TypeAlias = tuple[int, int, int, int]
"""A colour with an explicit alpha, 0-255 per channel."""

TIMELINE: Final[str] = "frame_time"
"""Duration timeline the rig poses are logged on, keyed by keyframe timestamp."""

WORLD_PATH: Final[str] = "world"
"""Root of the spatial tree; `log_rig_static` appends `rig_<NN>` to it."""

REPORT_PATH: Final[str] = "report"
"""Entity holding the markdown report as a `TextDocument`."""

RUN_A_INDEX: Final[int] = 0
"""Rig index of the reference run, so it lands at `/world/rig_00`."""

RUN_B_INDEX: Final[int] = 1
"""Rig index of the candidate run, so it lands at `/world/rig_01`."""

RUN_A_COLOR: Final[Rgb] = (255, 150, 40)
"""Orange — the demo's cuSFM hue, reused for the blob reference run."""

RUN_B_COLOR: Final[Rgb] = (90, 170, 255)
"""Blue — the demo's input hue, reused for the colsfm run."""

INPUT_COLOR: Final[Rgb] = (150, 150, 160)
"""Grey for the input trajectory, so neither run's hue is claimed by it."""

GROUND_TRUTH_COLOR: Final[Rgb] = (80, 230, 120)
"""Green for ground truth, the one independent reference."""

TRAJECTORY_START_COLOR: Final[Rgba] = (60, 220, 90, 255)
"""Endpoint marker for the first pose, as the demo colours it."""

TRAJECTORY_END_COLOR: Final[Rgba] = (235, 60, 70, 255)
"""Endpoint marker for the last pose, as the demo colours it."""

TRAJECTORY_RADIUS: Final[float] = 0.004
"""Polyline radius in scene units, matching the demo and the catalog slam layer."""

ENDPOINT_RADIUS: Final[float] = 0.02
"""Radius of the start/end markers."""

POINT_RADIUS: Final[float] = 0.01
"""Sparse-point radius in scene units."""

IMAGE_PLANE_DISTANCE: Final[float] = 0.1
"""Frustum length in metres; Galileo's cameras sit centimetres apart."""

COLOUR_RAMP_FLOOR: Final[float] = 0.35
"""Darkest fraction of the hue at the start of a trajectory (direction of travel)."""

NANOSECONDS_PER_MICROSECOND: Final[int] = 1000
"""`log_rig_pose_stream` takes nanoseconds; every timestamp here is microseconds.

The logger then converts those nanoseconds to float64 *seconds* before building
the `TimeColumn`, and float64 holds ~380 ns at Galileo's 1.7e9-second epoch, so
the timeline is accurate to well under a microsecond but not bit-exact. Every
metric joins on the integer microseconds in the metadata, never on this column."""


@dataclass(frozen=True, slots=True)
class TrajectorySources:
    """The reference trajectories logged alongside the two runs."""

    input_track: RigTrack
    """Rig trajectory implied by the dataset's input `frames_meta.json`."""
    ground_truth: RigTrack | None
    """Rig trajectory from `ground_truth.txt`, or None when the dataset ships none."""


@dataclass(frozen=True, slots=True)
class BenchmarkScene:
    """Everything one benchmark recording is logged from.

    The four travel together and never separately, and `run_a` and `run_b` are
    the same type, so passing them positionally invited a silent swap that would
    have drawn run B at `/world/rig_00` with run A's metrics beside it.
    """

    comparison: Comparison
    """The result of `colsfm.benchmark.compare_runs`, rendered into the report pane."""
    run_a: RunArtifacts
    """Reference run's artifacts (the blob); lands at `/world/rig_00`."""
    run_b: RunArtifacts
    """Candidate run's artifacts (the colsfm pipeline); lands at `/world/rig_01`."""
    sources: TrajectorySources
    """Input and ground-truth trajectories drawn alongside the two runs."""


def _camera_k_matrix(camera: CameraParams) -> Matrix3:
    """Pick the intrinsic matrix cuSFM's export rules pick for this model.

    `colsfm.cameras.PINHOLE_FAMILY_MODELS` are exported from the rectified 3x4
    `projection_matrix`, the distorted models from the raw 3x3 `camera_matrix`
    (export.md §3.3). Either choice falls back to the other matrix, because a
    dataset may ship only one of the two and the frustum is worth drawing from
    whichever it has — which is why this does not just call
    `colmap_camera().calibration_matrix()`.

    Args:
        camera: One camera's calibration.

    Returns:
        Float64 intrinsics with shape `[3, 3]`.

    Raises:
        ValueError: When the calibration carries neither matrix.
    """
    from_projection: Matrix3 | None = (
        None if camera.projection_matrix is None else np.asarray(camera.projection_matrix[:3, :3], dtype=np.float64)
    )
    from_camera_matrix: Matrix3 | None = (
        None if camera.camera_matrix is None else np.asarray(camera.camera_matrix, dtype=np.float64)
    )
    preferred_first: bool = camera.projection_model in PINHOLE_FAMILY_MODELS
    ordered: tuple[Matrix3 | None, Matrix3 | None] = (
        (from_projection, from_camera_matrix) if preferred_first else (from_camera_matrix, from_projection)
    )
    for k_matrix in ordered:
        if k_matrix is not None:
            return k_matrix
    raise ValueError(f"camera {camera.camera_params_id} ({camera.sensor_name}) has neither a projection nor a camera matrix")


def build_rig(frames_meta: FramesMeta, index: int, track: RigTrack) -> Rig:
    """Assemble an exoego `Rig` from a run's calibration plus its rig trajectory.

    Cameras are ordered by `camera_params_id`, so `cam_NN` is stable across the
    two rigs. The rig frame is cuSFM's vehicle (FLU) frame, so `rig_T_cam` is the
    metadata's `sensor_to_vehicle_transform` verbatim.

    Args:
        frames_meta: The run's `kpmap/keyframes/frames_meta.json`.
        index: Rig entity index; 0 lands at `/world/rig_00`.
        track: The rig trajectory to drive the rig with.

    Returns:
        The rig, ready for `log_rig_static` and `log_rig_pose_stream`.
    """
    sensors: list[CameraSensor] = []
    for camera_index, camera_params_id in enumerate(sorted(frames_meta.cameras)):
        camera: CameraParams = frames_meta.cameras[camera_params_id]
        # `Rigid3d.matrix()` is 3x4; `Extrinsics`' "world" slot is the *parent*
        # frame, which here is the rig, so this pair is `rig_T_cam`.
        vehicle_T_cam: Float64[ndarray, "3 4"] = np.asarray(camera.vehicle_T_cam.matrix(), dtype=np.float64)
        extrinsics: Extrinsics = Extrinsics(world_R_cam=vehicle_T_cam[:3, :3], world_t_cam=vehicle_T_cam[:3, 3])
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=_camera_k_matrix(camera),
            height=camera.image_height,
            width=camera.image_width,
        )
        parameters: PinholeParameters | Fisheye62Parameters
        if camera.projection_model == "OPENCV_FISHEYE":
            coefficients: list[float] = ([*np.asarray(camera.distortion_coefficients, dtype=np.float64).tolist()] + [0.0] * 4)[:4]
            parameters = Fisheye62Parameters(
                name=camera.sensor_name,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                distortion=KannalaBrandtDistortion(
                    k1=coefficients[0], k2=coefficients[1], k3=coefficients[2], k4=coefficients[3]
                ),
            )
        else:
            parameters = PinholeParameters(name=camera.sensor_name, extrinsics=extrinsics, intrinsics=intrinsics)
        sensors.append(CameraSensor(index=camera_index, name=camera.sensor_name, kind="rgb", pinhole=parameters))

    return Rig(
        index=index,
        calibration=RigCalibration(cameras=sensors, reference_index=0),
        pose_stream=RigPoseStream(world_t_rig=track.world_t_rig.copy(), world_R_rig=track.world_R_rig.copy()),
        image_plane_distance=IMAGE_PLANE_DISTANCE,
    )


def log_run_rig(artifacts: RunArtifacts, index: int, *, label: str, frustum_color: Rgb) -> None:
    """Log one run as a rig: static skeleton, tinted frusta, and its pose stream.

    Args:
        artifacts: The run, as read by `colsfm.benchmark.read_run`.
        index: Rig entity index.
        label: Provenance string logged on the rig node.
        frustum_color: Frustum tint identifying the run.
    """
    rig: Rig = build_rig(artifacts.frames_meta, index, artifacts.track)
    log_rig_static(rig, world_path=WORLD_PATH)
    log_rig_pose_stream(
        rig,
        timestamps_ns=artifacts.track.timestamps_microseconds * NANOSECONDS_PER_MICROSECOND,
        world_path=WORLD_PATH,
        timeline=TIMELINE,
    )
    rig_path: str = f"{WORLD_PATH}/{entity_id('rig', index)}"
    rr.log(rig_path, rr.AnyValues(source=label, run=artifacts.name, run_dir=str(artifacts.run_dir)), static=True)
    for sensor in rig.calibration.cameras:
        rr.log(
            f"{rig_path}/{entity_id('cam', sensor.index)}/pinhole",
            rr.Pinhole.from_fields(image_plane_distance=IMAGE_PLANE_DISTANCE, color=(*frustum_color, 255)),
            static=True,
        )


def log_points(artifacts: RunArtifacts, index: int, fallback_color: Rgb) -> int:
    """Log one run's sparse cloud in the world frame.

    A run written without `--output_rgb` has every point at `(0, 0, 0)` (NOTES.md
    gotcha 11), which renders as an invisible black cloud. Those fall back to the
    run's flat colour so the two clouds stay distinguishable.

    Args:
        artifacts: The run.
        index: Rig entity index; the cloud lands at `/world/points/rig_<NN>`.
        fallback_color: Flat colour used when every point is black.

    Returns:
        The number of points logged.
    """
    positions: Float64[ndarray, "n 3"] = artifacts.points_xyz()
    if len(positions) == 0:
        return 0
    points_rgb: Int64[ndarray, "n 3"] = artifacts.points_rgb()
    all_black: bool = not points_rgb.any()
    colors: UInt8[ndarray, "n 3"] = (
        np.tile(np.asarray(fallback_color, dtype=np.uint8), (len(positions), 1)) if all_black else points_rgb.astype(np.uint8)
    )
    rr.log(
        f"{WORLD_PATH}/points/{entity_id('rig', index)}",
        rr.Points3D(positions=positions, colors=colors, radii=POINT_RADIUS),
        static=True,
    )
    return len(positions)


def log_trajectory(name: str, positions: Float64[ndarray, "n 3"], *, source: str, hue: Rgb) -> bool:
    """Log a trajectory polyline the way the RoboCap catalog's slam layer does.

    `runs/<name>` carries provenance, `runs/<name>/trajectory` is a static
    `LineStrips3D` of `n-1` two-point segments whose colour ramps from dark to
    full hue so the direction of travel is readable, and `runs/<name>/endpoints`
    marks start and end.

    Args:
        name: Leaf under `/world/runs`.
        positions: Float64 world positions with shape `[n, 3]`.
        source: Provenance string logged on the run node.
        hue: Base colour of the polyline.

    Returns:
        True when a polyline was logged; False when there were fewer than two poses.
    """
    if len(positions) < 2:
        return False
    segments: Float64[ndarray, "num_segments 2 3"] = np.stack([positions[:-1], positions[1:]], axis=1)
    ramp: Float64[ndarray, "num_segments"] = np.linspace(COLOUR_RAMP_FLOOR, 1.0, len(segments))
    colors: UInt8[ndarray, "num_segments 4"] = np.stack(
        [
            (ramp * hue[0]).astype(np.uint8),
            (ramp * hue[1]).astype(np.uint8),
            (ramp * hue[2]).astype(np.uint8),
            np.full(len(segments), 255, dtype=np.uint8),
        ],
        axis=-1,
    )
    run_path: str = f"{WORLD_PATH}/runs/{name}"
    rr.log(run_path, rr.AnyValues(num_poses=len(positions), source=source), static=True)
    rr.log(
        f"{run_path}/trajectory",
        rr.LineStrips3D(segments, colors=colors, radii=TRAJECTORY_RADIUS),
        static=True,
    )
    rr.log(
        f"{run_path}/endpoints",
        rr.Points3D(
            positions=np.stack([positions[0], positions[-1]]),
            colors=[TRAJECTORY_START_COLOR, TRAJECTORY_END_COLOR],
            labels=["start", "end"],
            radii=ENDPOINT_RADIUS,
        ),
        static=True,
    )
    return True


def build_blueprint(comparison: Comparison) -> rrb.Blueprint:
    """Three panes: both reconstructions, the trajectories alone, and the report.

    The scene view shows everything under `/world`, which is the only view that
    can resolve the rig transform trees. The trajectory view shares the same
    origin — polylines are logged in the world frame — but is scoped to
    `/world/runs/**`, so the two clouds cannot bury the four thin lines. The
    report rides alongside as a `TextDocumentView` so the numbers and the picture
    are read together.

    Args:
        comparison: Used only for the view names, so the panes say which run is which.

    Returns:
        The blueprint, with the side panels collapsed for clean screenshots.
    """
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin=f"/{WORLD_PATH}",
                name=f"Scene — rig_00 {comparison.run_a.name}, rig_01 {comparison.run_b.name}",
            ),
            rrb.Spatial3DView(
                origin=f"/{WORLD_PATH}",
                name="Trajectories only",
                contents=[f"+ /{WORLD_PATH}/runs/**"],
            ),
            rrb.TextDocumentView(origin=f"/{REPORT_PATH}", name="Benchmark report"),
            column_shares=[3.0, 2.0, 2.0],
        ),
        collapse_panels=True,
    )


def log_comparison(scene: BenchmarkScene) -> None:
    """Log the whole benchmark into the active recording.

    Args:
        scene: The comparison, both runs' artifacts, and the reference trajectories.
    """
    comparison: Comparison = scene.comparison
    run_a: RunArtifacts = scene.run_a
    run_b: RunArtifacts = scene.run_b
    sources: TrajectorySources = scene.sources

    rr.log("/", rr.ViewCoordinates.RFU, static=True)
    rr.send_blueprint(build_blueprint(comparison))

    log_run_rig(run_a, RUN_A_INDEX, label=f"A: {run_a.name} ({run_a.run_dir})", frustum_color=RUN_A_COLOR)
    log_run_rig(run_b, RUN_B_INDEX, label=f"B: {run_b.name} ({run_b.run_dir})", frustum_color=RUN_B_COLOR)

    points_a: int = log_points(run_a, RUN_A_INDEX, RUN_A_COLOR)
    points_b: int = log_points(run_b, RUN_B_INDEX, RUN_B_COLOR)
    print(f"[rerun] clouds: rig_00 {points_a} points, rig_01 {points_b} points")

    log_trajectory(
        entity_id("rig", RUN_A_INDEX),
        run_a.track.world_t_rig,
        source=f"A: {run_a.name} ({comparison.run_a.reconstruction.num_points3D} points)",
        hue=RUN_A_COLOR,
    )
    log_trajectory(
        entity_id("rig", RUN_B_INDEX),
        run_b.track.world_t_rig,
        source=f"B: {run_b.name} ({comparison.run_b.reconstruction.num_points3D} points)",
        hue=RUN_B_COLOR,
    )
    log_trajectory(
        "input",
        sources.input_track.world_t_rig,
        source=f"input frames_meta.json ({len(sources.input_track)} rig frames)",
        hue=INPUT_COLOR,
    )
    if sources.ground_truth is not None:
        log_trajectory(
            "ground_truth",
            sources.ground_truth.world_t_rig,
            source=f"ground_truth.txt ({len(sources.ground_truth)} poses)",
            hue=GROUND_TRUTH_COLOR,
        )

    rr.log(REPORT_PATH, rr.TextDocument(render_markdown_report(comparison), media_type="text/markdown"), static=True)


def save_comparison(path: Path, scene: BenchmarkScene, *, application_id: str = "colsfm_bench") -> Path:
    """Write the comparison to a standalone `.rrd`.

    Args:
        path: Destination `.rrd`; parent directories are created.
        scene: The comparison, both runs' artifacts, and the reference trajectories.
        application_id: Rerun application id stamped into the recording.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # A scoped `RecordingStream` rather than `rr.init` + `rr.save`: leaving the
    # `with` block flushes *and* closes the file sink, so the `.rrd` gets its
    # footer. A global recording left alive writes a footerless file the reader
    # can only rescue with a whole-file scan.
    recording: rr.RecordingStream = rr.RecordingStream(application_id, recording_id=f"{scene.comparison.dataset}_compare")
    recording.save(str(path))
    with recording:
        log_comparison(scene)
    recording.flush(timeout_sec=30.0)
    return path
