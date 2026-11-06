"""Entity paths, timelines and property groups of a MetricTrail run recording (``metrictrail run`` / ``log-run``).

One recording per run. Timeline ``frame`` (sequence) is the tracked-frame index ``i`` (row ``i`` of ``pose/<name>.npz``,
DPVO's frame counter); ``source_frame`` (sequence) is the input frame it came from (``source_frame_ids[i]``) and
``video_time`` (duration) its VSLAM-LAB timestamp when the run has one.

Live capture (only while the pipeline runs; :mod:`metrictrail.capture`):

* ``cam`` Pinhole of the tracking raster; ``cam/rgb`` the frame DPVO tracks; ``cam/depth`` the MoGe-2 metric depth DPVO
  is given; ``cam/normals`` MoGe-2's camera-space surface normals of each new keyframe, from the same forward as the
  depth (RGB = (n + 1) / 2 of x right, y down, z forward); ``cam/dino_pca`` DINOv3 patch features on a fixed
  3-component PCA; ``cam/patches`` the newest keyframe's DPVO patches, coloured by their depth;
* ``dpvo/keyframes``, ``dpvo/keyframe_path``, ``dpvo/window`` (BA-window frustums), ``dpvo/current`` (newest keyframe
  pose + pinhole), ``dpvo/window_points`` (patch points of the window), in DPVO's world frame;
* ``metrics/dpvo/...`` scalars.

From the finished artifacts (:mod:`metrictrail.outputs`):

* ``cam``, ``cam/rgb``, ``cam/depth`` when there was no live capture (frames read from the input, depth from
  ``depth/<name>.zip``);
* ``world/<label>/trajectory`` (static), ``world/<label>/cam`` (pose over ``frame`` + pinhole), ``world/cloud`` (fused
  metric cloud), ``world/cloud_normals`` (the same points coloured by their fused MoGe-2 normal in that frame,
  RGB = (n + 1) / 2; Points3D carries no normal component; only after a live capture, which holds the normals);
  in the GT frame after the paper's Sim(3) alignment when GT exists, else in ViPE's world frame;
* ``world/gt/trajectory`` (static) and ``world/gt/position`` (over ``frame``);
* ``metric/<label>/trajectory``, ``metric/gt/trajectory``, ``metric/cloud``, ``metric/cloud_normals``: the same after
  an SE(3)-only alignment, scale fixed at 1, so metric scale shows against GT in metres;
* ``metrics/ate_sim3/{err_m,running_rmse_m}`` and ``metrics/ate_se3/{err_m,running_rmse_m}`` over ``frame``;
* ``summary`` (markdown) and ``notes/capture`` (what a recording without a live capture lacks).

Recording properties: ``provenance`` (dataset_path, run_output_dir, producer, env, utc), ``run`` (the run record),
``metrics`` (ATE numbers, scale, counts).
"""

from metrictrail.palette import RunLabel

FRAME_TIMELINE: str = "frame"
"""Tracked-frame index (sequence timeline)."""
SOURCE_FRAME_TIMELINE: str = "source_frame"
"""Input frame index of the tracked frame (sequence timeline)."""
VIDEO_TIMELINE: str = "video_time"
"""VSLAM-LAB timestamp (duration timeline, the benchmark's own nanoseconds, as dataforge logs it)."""

CAM: str = "cam"
CAM_RGB: str = "cam/rgb"
CAM_DEPTH: str = "cam/depth"
CAM_NORMALS: str = "cam/normals"
CAM_DINO_PCA: str = "cam/dino_pca"
CAM_PATCHES: str = "cam/patches"

DPVO_ROOT: str = "dpvo"
DPVO_KEYFRAMES: str = "dpvo/keyframes"
DPVO_KEYFRAME_PATH: str = "dpvo/keyframe_path"
DPVO_WINDOW: str = "dpvo/window"
DPVO_CURRENT: str = "dpvo/current"
DPVO_WINDOW_POINTS: str = "dpvo/window_points"
DPVO_METRICS: str = "metrics/dpvo"

WORLD_ROOT: str = "world"
WORLD_GT_TRAJECTORY: str = "world/gt/trajectory"
WORLD_GT_POSITION: str = "world/gt/position"
WORLD_CLOUD: str = "world/cloud"
WORLD_CLOUD_NORMALS: str = "world/cloud_normals"
METRIC_ROOT: str = "metric"
METRIC_GT_TRAJECTORY: str = "metric/gt/trajectory"
METRIC_CLOUD: str = "metric/cloud"
METRIC_CLOUD_NORMALS: str = "metric/cloud_normals"
ATE_SIM3_ERR: str = "metrics/ate_sim3/err_m"
ATE_SIM3_RUNNING: str = "metrics/ate_sim3/running_rmse_m"
ATE_SE3_ERR: str = "metrics/ate_se3/err_m"
ATE_SE3_RUNNING: str = "metrics/ate_se3/running_rmse_m"
SUMMARY: str = "summary"
NOTES_CAPTURE: str = "notes/capture"

PROVENANCE_PROPERTY: str = "provenance"
RUN_PROPERTY: str = "run"
METRICS_PROPERTY: str = "metrics"


def world_trajectory(label: RunLabel) -> str:
    """``world/<label>/trajectory``: the run's whole camera path (static)."""
    return f"world/{label}/trajectory"


def world_cam(label: RunLabel) -> str:
    """``world/<label>/cam``: the run's camera pose over ``frame`` and its pinhole."""
    return f"world/{label}/cam"


def metric_trajectory(label: RunLabel) -> str:
    """``metric/<label>/trajectory``: the SE(3)-aligned path (scale 1)."""
    return f"metric/{label}/trajectory"
