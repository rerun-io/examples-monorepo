"""Entity paths, timelines and property groups of a MetricTrail run recording (``metrictrail run`` / ``log-run``).

One recording per run. Timeline ``frame`` (sequence) is the tracked-frame index ``i`` (row ``i`` of ``pose/<name>.npz``,
DPVO's frame counter); ``source_frame`` (sequence) is the input frame it came from (``source_frame_ids[i]``) and
``video_time`` (duration) its VSLAM-LAB timestamp when the run has one.

Live capture (only while the pipeline runs; :mod:`metrictrail.capture`):

* ``dpvo/current/pinhole/video`` (the input video's own samples) and ``.../patches``, in the source camera's pinhole;
* ``cam`` Pinhole of the tracking raster; ``cam/rgb`` the frame DPVO tracks (frame directories; an H.264 video); ``cam/depth`` the MoGe-2 metric depth DPVO
  is given; ``cam/normals`` MoGe-2's camera-space surface normals of each new keyframe, from the same forward as the
  depth (RGB = (n + 1) / 2 of x right, y down, z forward); ``cam/dino_pca`` DINOv3 patch features on a fixed
  3-component PCA; ``cam/patches`` the newest keyframe's DPVO patches, coloured by their depth;
* ``dpvo/keyframes``, ``dpvo/keyframe_path``, ``dpvo/window`` (BA-window frustums), ``dpvo/current`` (newest keyframe
  pose; ``dpvo/current/pinhole`` its frustum), ``dpvo/window_points`` (patch points of the window), in DPVO's world
  frame; ``dpvo`` itself carries the gravity alignment (``Transform3D``, z up, re-fit every 10 keyframes);
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

CALIBRATION_TIMELINE: str = "calibration_step"
"""SharedK calibration's sweep: the index of the frame it read (sequence timeline); its data is also on ``video_time``
and ``source_frame``."""
CALIBRATION: str = "calibration"
"""SharedK calibration's own DROID-style keyframe SLAM (camera axes RDF, the first keyframe at the origin): the map it
optimises with the focal and then throws away; tracking starts again in ``dpvo``."""
CALIBRATION_KEYFRAMES: str = "calibration/keyframes"
"""One child per kept keyframe, ``f<sweep frame>``: its pose, ``camera`` (its frustum) and ``points``
(the bundle-adjusted disparities, in the keyframe's camera, after ViPE's multi-view depth filter)."""
CALIBRATION_PATH: str = "calibration/keyframe_path"
CALIBRATION_EDGES: str = "calibration/factor_graph"
"""The frontend's factor-graph edges between keyframe centres."""
CALIBRATION_NEWEST: str = "calibration/newest"
"""The newest keyframe in 2D, outside the 3D view."""
CALIBRATION_NEWEST_RGB: str = "calibration/newest/rgb"
CALIBRATION_NEWEST_PRIOR: str = "calibration/newest/moge2_depth"
"""MoGe-2 metric depth at the current focal: the prior the bundle adjustment pulls the disparities towards."""
CALIBRATION_NEWEST_DEPTH: str = "calibration/newest/droid_depth"
"""Depth of the bundle-adjusted disparities."""
CALIBRATION_FOCAL: str = "metrics/sharedk/focal_px"
"""SharedK's focal (tracking-raster px) after each backend solve."""
CALIBRATION_HEAD_FOCAL: str = "metrics/sharedk/moge2_focal_px"
"""MoGe-2's own focal on each of the first 32 frames; their median starts the fit."""
CALIBRATION_KEPT: str = "metrics/sharedk/keyframes"
"""Keyframes the sweep keeps, per frame read."""


def calibration_keyframe(step: int) -> str:
    """The entity of the calibration keyframe made at sweep step ``step``."""
    return f"{CALIBRATION_KEYFRAMES}/f{step:05d}"


DPVO_ROOT: str = "dpvo"
DPVO_KEYFRAMES: str = "dpvo/keyframes"
DPVO_KEYFRAME_PATH: str = "dpvo/keyframe_path"
DPVO_WINDOW: str = "dpvo/window"
DPVO_CURRENT: str = "dpvo/current"
"""Newest keyframe's pose; the Follow view rides it."""
DPVO_CURRENT_PINHOLE: str = "dpvo/current/pinhole"
"""Its frustum, a child, so a 3D view rooted at the camera stays 3D. For a video input it is the source camera (the
tracking raster's K mapped back through ViPE's resize and crop, :class:`metrictrail.rerun_log.SourceRaster`), and the
input video plays inside it."""
INPUT: str = DPVO_CURRENT_PINHOLE
"""The pinhole the input video sits in: the moving camera's frustum, so the video shows in the 3D views too."""
INPUT_VIDEO: str = f"{DPVO_CURRENT_PINHOLE}/video"
"""The input video's own samples (``rr.VideoStream``) on ``video_time``."""
INPUT_PATCHES: str = f"{DPVO_CURRENT_PINHOLE}/patches"
"""DPVO's newest-keyframe patches in source pixels, over the input video."""
DPVO_WINDOW_POINTS: str = "dpvo/window_points"
DPVO_MESH: str = "dpvo/mesh"
"""Live TSDF mesh of the app, one subtree per DPVO pass (``pass_<n>``, earlier passes cleared statically): changed
super-blocks under ``blocks/`` on the timeline while tracking, one mesh under ``full`` from the pass's last integrated
frame (:class:`simplecv.rerun_live_mesh.LiveMeshLog`)."""
DPVO_METRICS: str = "metrics/dpvo"

WORLD_ROOT: str = "world"
WORLD_GT_TRAJECTORY: str = "world/gt/trajectory"
WORLD_GT_POSITION: str = "world/gt/position"
WORLD_CLOUD: str = "world/cloud"
WORLD_CLOUD_NORMALS: str = "world/cloud_normals"
WORLD_MESH: str = "world/mesh"
WORLD_MESH_NORMALS: str = "world/mesh_normals"
METRIC_ROOT: str = "metric"
METRIC_GT_TRAJECTORY: str = "metric/gt/trajectory"
METRIC_CLOUD: str = "metric/cloud"
METRIC_CLOUD_NORMALS: str = "metric/cloud_normals"
METRIC_MESH: str = "metric/mesh"
METRIC_MESH_NORMALS: str = "metric/mesh_normals"
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
