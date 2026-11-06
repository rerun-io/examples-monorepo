"""The MetricTrail catalog layer: one run on one VSLAM-LAB segment, stacked on dataforge's ``base`` and ``gt`` layers.

Implements ``/tmp/fleet-artifacts/metrictrail/p1/catalog/SPEC.md`` (sections 3, 4 and 2.6), proven there on fr1_desk.

Conventions (dataforge's, as its ``vslam_lab`` converter and ``slam_rs.catalog_layer`` use them; dataforge itself is
not a dependency of this package, so the few names are restated here):

* ``application_id = "dataforge"``, ``recording_id`` = the segment id (``vslam-lab-<benchmark>__<sequence>``),
  no automatic ``RecordingInfo`` (a derived layer is the same recording as its base);
* timeline ``video_time`` (duration): the benchmark's own nanosecond stamps, unshifted, so a tracked frame lands on the
  base video frame it was computed from;
* everything under ``/world/runs/<source>``.

Layout (``<root>`` = ``/world/runs/<source>``):

* ``<root>``: static ``Transform3D``, the SE(3) Umeyama fit from the estimate's world to the GT world (rigid, so the
  pinhole below stays drawable and the scale error stays visible); the paper's Sim(3) is in the properties;
* ``<root>/trajectory`` (static ``LineStrips3D``) and ``<root>/trail`` (one segment per frame on ``video_time``);
* ``<root>/cam`` (``Transform3D`` per tracked frame) and ``<root>/cam/pinhole`` (static ``Pinhole``, RDF);
* ``<root>/cam/pinhole/depth``: the run's metric depth (``depth/<name>.zip``, metres) at DPVO's keyframes;
* ``<root>/keyframes`` (static ``Points3D``) and ``<root>/points`` (static ``Points3D``, the metric cloud);
* recording property group ``<source>``: :class:`LayerProperties`.

Catalog ``.rrd`` files cannot be overwritten in place: :func:`write_layer` refuses an existing target, so a rebuild
writes a new path and registers it again.
"""

import os
from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rerun as rr
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde

from metrictrail.artifacts import ColoredCloud
from metrictrail.palette import OURS_RGB, UPSTREAM_RGB
from metrictrail.properties import any_values

LayerSource: TypeAlias = Literal["metrictrail", "metrictrail_upstream"]
"""Layer name and run source: our port, or unmodified upstream ViPE run on this machine."""

APPLICATION_ID: str = "dataforge"
"""dataforge's application id; every layer of a segment shares it."""
TIMELINE: str = "video_time"
"""dataforge's duration timeline."""
LAYER_COLORS: dict[LayerSource, tuple[int, int, int]] = {"metrictrail": OURS_RGB, "metrictrail_upstream": UPSTREAM_RGB}
"""Trajectory, trail and keyframe colour per source: the explainer legend (port magenta #E040FB, upstream amber #F5A623;
the gt layer is dataforge's and gets the legend's green from the blueprint)."""
POINTS_VOXEL_M: dict[str, float] = {"rgbdtum": 0.01, "7scenes": 0.02, "kitti": 0.10}
"""Voxel of the keyframe cloud per VSLAM-LAB dataset (SPEC section 3)."""
MAX_POINTS: int = 300_000
"""Cap of the keyframe cloud (SPEC section 3)."""
TIME_SOURCE: str = "VSLAM-LAB rgb.csv ts_rgb_0 (= base video_time); frames not in rgb.csv: int(float(stem)*1e9)"
"""How ``video_time`` was made (SPEC 3.1)."""
DEPTH_SOURCE: str = "upstream depth/<name>.zip: MoGe-2 metric z-depth (float16 m), K-conditioned on the bound K; keyframes only"
"""What ``cam/pinhole/depth`` holds."""


def run_root(source: LayerSource) -> str:
    """``/world/runs/<source>``."""
    return f"/world/runs/{source}"


def layer_entities(source: LayerSource) -> dict[str, str]:
    """Every entity path the layer writes, by role (the layer schema the tests pin)."""
    root: str = run_root(source)
    return {
        "root": root,
        "trajectory": f"{root}/trajectory",
        "trail": f"{root}/trail",
        "cam": f"{root}/cam",
        "pinhole": f"{root}/cam/pinhole",
        "depth": f"{root}/cam/pinhole/depth",
        "keyframes": f"{root}/keyframes",
        "points": f"{root}/points",
    }


def segment_id(benchmark_folder: str, sequence: str) -> str:
    """dataforge's segment id of a VSLAM-LAB sequence: ``vslam-lab-<benchmark>__<sequence>`` (benchmark lower-cased)."""
    return f"vslam-lab-{benchmark_folder.lower()}__{sequence}"


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerProperties:
    """The layer's recording property group ``property:<source>:*`` (SPEC section 4; lengths in metres)."""

    producer: str
    """Which tree ran: ``upstream ViPE @ ead2e00, unmodified (...)`` or ``examples-monorepo metrictrail @ <sha>``."""
    vipe_commit: str
    """ViPE commit of the tree that ran."""
    repo_commit: str | None
    """Port only: the monorepo commit of the code that ran (the run's own ``logs/commit.txt``, not the layer writer's)."""
    repo_changes: list[str] | None
    """Port only: uncommitted changes of the worktree the run used (``git status --short``); None when clean."""
    env: str
    """Pixi environment of the run (and lock hash for the port)."""
    host: str
    """Host that built the layer."""
    utc: str
    """When the layer was built."""
    dataset_path: str
    """MetricTrail frame directory the run read."""
    run_output_dir: str
    """MetricTrail output directory."""
    run_record: str
    """``vipe/<name>_metrictrail.json``."""
    upstream_case: str
    """Artifact / upstream eval case name."""
    seed: int
    """``pipeline.slam.seed`` of this run."""
    seeds: list[int] | None
    """Seeds of every run whose ATE is in ``paper_ate_m_per_seed``."""
    slam_preset: str | None
    """DPVO backend preset (run record ``tracking.preset``)."""
    raster_policy: str
    """Input raster policy."""
    tracking_stride: int
    """DPVO tracks every k-th input frame."""
    raster_hw: list[int]
    """Tracking raster ``[height, width]``."""
    intrinsics_source: str | None
    """``sharedk`` or the known-K source."""
    calibration_status: str | None
    """SharedK outcome."""
    k_fx_fy_cx_cy: list[float]
    """K the tracker used (run record ``bound_K_fx_fy_cx_cy``)."""
    num_frames: int
    """Frames in the input directory."""
    num_tracked: int
    """Tracked frames."""
    num_keyframes: int
    """Keyframes with a depth image in the layer."""
    num_frames_not_in_base: int
    """Tracked frames rgb.csv does not list (their stamp comes from the file name)."""
    time_source: str
    """How ``video_time`` was made."""
    display_alignment: str
    """What the run-root transform is."""
    se3_translation: list[float]
    """Run-root SE(3) translation."""
    se3_quaternion_xyzw: list[float]
    """Run-root SE(3) rotation."""
    alignment_gt_match: str
    """The paper's GT pairing rule."""
    sim3_scale: float
    """The paper's Sim(3): scale (p_gt = s R p_est + t)."""
    sim3_translation: list[float]
    """The paper's Sim(3): translation."""
    sim3_quaternion_xyzw: list[float]
    """The paper's Sim(3): rotation."""
    paper_ate_m: float
    """The paper's metric (evaluation report)."""
    paper_num_pairs: int
    """Its pair count."""
    se3_ate_m: float
    """SE(3), scale 1 diagnostic (evaluation report)."""
    vslamlab_ate_m: float | None
    """VSLAM-LAB's own ``evaluate_sequence`` number."""
    vslamlab_num_evaluated: int | None
    """Its pair count."""
    paper_ate_m_per_seed: list[float] | None
    """``paper_ate_m`` of every seed in ``seeds``."""
    paper_ate_m_mean: float | None
    """Mean over seeds."""
    paper_ate_m_std: float | None
    """Sample standard deviation over seeds (the pre-registered sigma)."""
    paper_table: str
    """Table and page of the paper's dataset number."""
    paper_dataset_value: float
    """The paper's dataset mean."""
    paper_units: str
    """Its units."""
    depth_source: str
    """What the depth images are."""
    points_voxel_m: float
    """Voxel of the keyframe cloud."""
    runtime_s: float | None
    """Whole-run wall clock."""
    tracking_fps: float | None
    """End-to-end tracking frames per second (run record ``tracking.end_to_end_fps``)."""
    max_memory_allocated_gib: float | None
    """CUDA peak allocation after model load."""


@dataclass(frozen=True, slots=True)
class LayerInputs:
    """Everything a layer holds, already read and aligned (no file access in :func:`write_layer` but the target)."""

    world_T_cam: Float64[ndarray, "n 4 4"]
    """Raw estimate, camera-to-world in the estimate's world, finite rows only."""
    times_ns: Int64[ndarray, "n"]
    """``video_time`` of each row (the benchmark's stamps), strictly increasing."""
    intrinsics_px: Float64[ndarray, "4"]
    """``fx, fy, cx, cy`` on the tracking raster."""
    raster_hw: tuple[int, int]
    """Tracking raster ``(height, width)``."""
    keyframe_rows: Int64[ndarray, "k"]
    """Rows (into ``world_T_cam``) with a keyframe depth."""
    keyframe_depth_m: list[Float32[ndarray, "h w"]]
    """Metric depth of each keyframe row, metres (0 = no depth)."""
    points: ColoredCloud
    """Keyframe cloud in the estimate's world."""
    gt_T_est_se3: Float64[ndarray, "4 4"]
    """Rigid alignment estimate world -> GT world (the run-root transform)."""


def matrix_to_quat_xyzw(rotation: Float64[ndarray, "3 3"]) -> list[float]:
    """Hamilton quaternion ``x, y, z, w`` of a rotation matrix."""
    return [float(value) for value in Rotation.from_matrix(rotation).as_quat()]


def backproject_keyframes(
    world_T_cam: Float64[ndarray, "n 4 4"],
    keyframe_rows: Int64[ndarray, "k"],
    depths_m: Iterable[Float32[ndarray, "h w"]],
    colors: Iterable[UInt8[ndarray, "h w 3"]],
    intrinsics_px: Float64[ndarray, "4"],
    *,
    stride_px: int,
    depth_range_m: tuple[float, float],
    voxel_m: float,
) -> ColoredCloud:
    """A simple keyframe cloud: every ``stride_px``-th pixel with depth in range, to world, first point per voxel.

    Not upstream's ``fuse_point_cloud`` (no multi-view consistency filter); used when no fused cloud is given.
    """
    fx, fy, cx, cy = (float(value) for value in intrinsics_px)
    points: list[Float64[ndarray, "p 3"]] = []
    point_colors: list[UInt8[ndarray, "p 3"]] = []
    for row, depth_m, rgb in zip(keyframe_rows.tolist(), depths_m, colors, strict=True):
        rows_px: Int64[ndarray, "a b"]
        cols_px: Int64[ndarray, "a b"]
        rows_px, cols_px = np.meshgrid(np.arange(0, depth_m.shape[0], stride_px), np.arange(0, depth_m.shape[1], stride_px), indexing="ij")
        z: Float64[ndarray, "a b"] = depth_m[rows_px, cols_px].astype(np.float64)
        ok: ndarray = np.isfinite(z) & (z >= depth_range_m[0]) & (z <= depth_range_m[1])
        cam_xyz: Float64[ndarray, "p 3"] = np.stack([(cols_px[ok] - cx) / fx * z[ok], (rows_px[ok] - cy) / fy * z[ok], z[ok]], axis=1)
        pose: Float64[ndarray, "4 4"] = world_T_cam[row]
        points.append(cam_xyz @ pose[:3, :3].T + pose[:3, 3])
        point_colors.append(rgb[rows_px[ok], cols_px[ok]])
    if not points:
        return ColoredCloud(points_xyz=np.zeros((0, 3), dtype=np.float32), colors_rgb=np.zeros((0, 3), dtype=np.uint8))
    all_points: Float64[ndarray, "p 3"] = np.concatenate(points)
    keep: Int64[ndarray, "q"] = np.sort(np.unique(np.floor(all_points / voxel_m).astype(np.int64), axis=0, return_index=True)[1]).astype(np.int64)
    return ColoredCloud(points_xyz=all_points[keep].astype(np.float32), colors_rgb=np.concatenate(point_colors)[keep])


def write_layer(target: Path, segment: str, source: LayerSource, inputs: LayerInputs, properties: LayerProperties) -> Path:
    """Write the layer ``.rrd`` to a temporary file and move it to ``target``.

    Raises:
        FileExistsError: If ``target`` exists (catalog files are never overwritten; write a new path).
        ValueError: If the inputs are non-finite, unsorted in time, or the keyframes do not match their depths.
    """
    if target.exists():
        raise FileExistsError(f"{target} exists; catalog layers are never overwritten in place, write a new path")
    if not np.isfinite(inputs.world_T_cam).all() or not np.all(np.diff(inputs.times_ns) > 0):
        raise ValueError("refusing non-finite poses or non-increasing timestamps")
    if inputs.keyframe_rows.shape[0] != len(inputs.keyframe_depth_m):
        raise ValueError(f"{inputs.keyframe_rows.shape[0]} keyframe rows but {len(inputs.keyframe_depth_m)} depth images")
    paths: dict[str, str] = layer_entities(source)
    color: tuple[int, int, int] = LAYER_COLORS[source]
    positions: Float64[ndarray, "n 3"] = inputs.world_T_cam[:, :3, 3]
    quats_xyzw: Float64[ndarray, "n 4"] = Rotation.from_matrix(inputs.world_T_cam[:, :3, :3]).as_quat()
    duration: rr.TimeColumn = rr.TimeColumn(TIMELINE, duration=inputs.times_ns.astype("timedelta64[ns]"))
    target.parent.mkdir(parents=True, exist_ok=True)
    temp: Path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    rec: rr.RecordingStream = rr.RecordingStream(APPLICATION_ID, recording_id=segment, send_properties=False)
    rec.save(temp)
    try:
        se3: Float64[ndarray, "4 4"] = inputs.gt_T_est_se3
        rec.log(paths["root"], rr.Transform3D(translation=se3[:3, 3], quaternion=rr.Quaternion(xyzw=matrix_to_quat_xyzw(se3[:3, :3]))), static=True)
        rec.log(paths["trajectory"], rr.LineStrips3D([positions], colors=[color], radii=rr.Radius.ui_points(1.5)), static=True)
        rec.log(paths["trail"], rr.LineStrips3D.from_fields(colors=[color], radii=rr.Radius.ui_points(3.0)), static=True)
        previous: Int64[ndarray, "n"] = np.maximum(np.arange(positions.shape[0]) - 1, 0)
        rec.send_columns(
            paths["trail"], indexes=[duration], columns=rr.LineStrips3D.columns(strips=list(np.stack([positions[previous], positions], axis=1)))
        )
        rec.send_columns(paths["cam"], indexes=[duration], columns=rr.Transform3D.columns(translation=positions, quaternion=quats_xyzw))
        rec.log(
            paths["pinhole"],
            rr.Pinhole(
                focal_length=inputs.intrinsics_px[:2],
                principal_point=inputs.intrinsics_px[2:],
                resolution=[inputs.raster_hw[1], inputs.raster_hw[0]],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.1,
            ),
            static=True,
        )
        rec.log(paths["keyframes"], rr.Points3D(positions[inputs.keyframe_rows], colors=[color], radii=rr.Radius.ui_points(3.0)), static=True)
        for row, depth_m in zip(inputs.keyframe_rows.tolist(), inputs.keyframe_depth_m, strict=True):
            # One column send per keyframe keeps the layer on video_time only (rec.log would add log_time).
            image: rr.DepthImage = rr.DepthImage(depth_m.astype(np.float16), meter=1.0)
            if image.buffer is None or image.format is None:
                raise ValueError(f"keyframe row {row}: depth image without buffer or format")
            rec.send_columns(
                paths["depth"],
                indexes=[rr.TimeColumn(TIMELINE, duration=inputs.times_ns[[row]].astype("timedelta64[ns]"))],
                columns=rr.DepthImage.columns(buffer=image.buffer.as_arrow_array(), format=image.format.as_arrow_array(), meter=[1.0]),
            )
        if inputs.points is not None:
            rec.log(
                paths["points"], rr.Points3D(inputs.points.points_xyz, colors=inputs.points.colors_rgb, radii=rr.Radius.ui_points(1.5)), static=True
            )
        rec.send_property(source, any_values({item.name: getattr(properties, item.name) for item in fields(properties)}))
        rec.flush()
    finally:
        rec.disconnect()
    os.replace(temp, target)
    target.chmod(0o664)  # the ssd-8tb NAS share gives new files mode 000
    return target
