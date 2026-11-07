"""High-level inference pipeline for DPVO visual odometry.

This module provides the main entry-point for running DPVO inference on
image directories or video files.  Videos decode on NVDEC and stay on the GPU;
image directories load in a background process.  Without a calibration file,
MoGe-2 estimates the intrinsics.  Rerun shows the camera trajectory, the point
cloud and the source video (logged once as a VideoStream) or images.

Both the CLI and Gradio UI consume the same :func:`run_dpvo_pipeline` generator.

Typical CLI usage::

    from dpvo.api.inference import DPVOInferenceConfig, run_dpvo_pipeline
    config = tyro.cli(DPVOInferenceConfig)
    for msg in run_dpvo_pipeline(dpvo_config=config.dpvo_config, ...):
        pass
"""

import importlib.util
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from itertools import chain, islice
from pathlib import Path
from timeit import default_timer as timer
from typing import Annotated, Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import tyro
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.rerun_log_utils import RerunTyroConfig, log_video
from torch import Tensor
from tqdm import tqdm

from dpvo.config import DPVOConfig
from dpvo.dpvo import DPVO
from dpvo.stream import VIDEO_SCALE, Frame, image_dir_frames, load_calib, num_frames, video_frames, video_hw
from dpvo.utils import DEFAULT_DEVICE, Timer

# ── Tyro subcommand aliases for DPVOConfig presets ──────────────────────
AccurateDPVOConfig = Annotated[
    DPVOConfig,
    tyro.conf.subcommand(name="accurate", default=DPVOConfig.accurate()),
]
"""Subcommand alias for the accurate preset."""

FastDPVOConfig = Annotated[
    DPVOConfig,
    tyro.conf.subcommand(name="fast", default=DPVOConfig.fast()),
]
"""Subcommand alias for the fast preset."""

SlamDPVOConfig = Annotated[
    DPVOConfig,
    tyro.conf.subcommand(name="slam-proximity", default=DPVOConfig.slam()),
]
"""Subcommand alias for the SLAM preset (proximity loop closure)."""

SlamClassicDPVOConfig = Annotated[
    DPVOConfig,
    tyro.conf.subcommand(name="slam-classic", default=DPVOConfig.slam_classic()),
]
"""Subcommand alias for the SLAM preset with classical (DBoW2) loop closure."""


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class DPVOPrediction:
    """Container for the final outputs of a DPVO inference run.

    Holds the optimized keyframe poses (translation + quaternion), their
    timestamps, and the associated 3-D point cloud with per-point colors.
    """

    final_poses: Float32[ndarray, "num_keyframes 7"]
    """Keyframe poses as ``[tx, ty, tz, qx, qy, qz, qw]`` in
    world-from-camera (``world_se3_cam``) convention."""
    tstamps: Float64[ndarray, "num_keyframes"]
    """Timestamp per keyframe. Integer frame indices for video/image streams,
    real-valued (nanoseconds/seconds) for evaluation datasets (EuRoC, TUM-RGBD)."""
    final_points: Float32[Tensor, "num_points 3"]
    """Reconstructed 3-D points in world coordinates.
    ``num_points = buffer_size * patches_per_frame``."""
    final_colors: UInt8[Tensor, "buffer_size num_patches 3"]
    """RGB colors for each reconstructed point."""


@dataclass
class DPVOPipelineHandle:
    """Mutable handle populated by :func:`run_dpvo_pipeline`.

    Callers inspect ``prediction`` after the generator is exhausted.
    """

    prediction: DPVOPrediction | None = field(default=None, repr=False)
    """Final prediction, populated after pipeline completes."""
    elapsed_time: float = 0.0
    """Total wall-clock seconds for the run."""


@dataclass
class DPVOInferenceConfig:
    """Configuration for a DPVO inference run (CLI entry point)."""

    rr_config: RerunTyroConfig
    """Rerun recording configuration (save path, application id, etc.)."""
    dpvo_config: AccurateDPVOConfig | FastDPVOConfig | SlamDPVOConfig | SlamClassicDPVOConfig = field(
        default_factory=DPVOConfig.fast
    )
    """DPVO solver configuration preset."""
    imagedir: str = "data/movies/IMG_0493.mp4"
    """Path to image directory or video file."""
    network_path: str = "checkpoints/dpvo.pth"
    """Path to DPVO network weights."""
    calib: str | None = None
    """Path to calibration file (``fx fy cx cy`` of the native video or images). If None, MoGe-2 estimates it."""
    stride: int = 2
    """Frame stride for sampling."""
    skip: int = 0
    """Number of leading frames to skip."""
    max_frames: int = 0
    """Maximum number of frames to process (0 = no limit)."""


# ── Camera orientation ──────────────────────────────────────────────────

# RDF (OpenCV) → RUB (OpenGL) flip: negate Y and Z axes.
# auto_orient_and_center_poses(method="up") assumes column 1 of the
# rotation is the camera's up vector, which is true for RUB but not RDF.
_RDF_TO_RUB: Float64[ndarray, "3 3"] = np.diag([1.0, -1.0, -1.0])

# Recompute the orient transform every N keyframes to amortise the SVD cost.
_ORIENT_RECOMPUTE_INTERVAL: int = 10


def compute_orient_transform(
    cam_se3_world_buffer: Float32[Tensor, "buffer_size 7"],
) -> tuple[Float64[ndarray, "3 3"], Float64[ndarray, "3"]] | None:
    """Compute a gravity-alignment transform from the current pose buffer.

    Converts DPVO's internal cam-from-world (RDF) poses to world-from-camera
    (RUB), then runs ``auto_orient_and_center_poses(method="up")`` to find
    a rotation+translation that aligns the reconstruction's up-vector with
    the viewer's +Y axis.

    Args:
        cam_se3_world_buffer: The raw pose buffer ``[tx, ty, tz, qx, qy, qz, qw]``
            in cam-from-world convention (only nonzero slots are used).

    Returns:
        ``(orient_R, orient_t)`` — a 3x3 rotation and 3-vector translation
        to log on the scene root via ``rr.Transform3D``.  Returns ``None``
        if fewer than 3 nonzero poses are available.
    """
    poses_mask: Tensor = ~(cam_se3_world_buffer[:, :6] == 0).all(dim=1)
    nonzero_poses: Float32[ndarray, "n 7"] = cam_se3_world_buffer[poses_mask].numpy(force=True)

    if nonzero_poses.shape[0] < 3:
        return None

    # Build N × 4×4 world_T_cam matrices in RUB convention
    n: int = nonzero_poses.shape[0]
    world_T_cam_rub: Float64[ndarray, "n 4 4"] = np.zeros((n, 4, 4))
    for i in range(n):
        cam_R_world: Float64[ndarray, "3 3"] = Rotation.from_quat(nonzero_poses[i, 3:]).as_matrix()
        cam_T_world: Float64[ndarray, "4 4"] = np.eye(4)
        cam_T_world[:3, :3] = cam_R_world
        cam_T_world[:3, 3] = nonzero_poses[i, :3]
        world_T_cam: Float64[ndarray, "4 4"] = np.linalg.inv(cam_T_world)
        # RDF → RUB: flip the camera axes so column 1 is "up"
        world_T_cam[:3, :3] = world_T_cam[:3, :3] @ _RDF_TO_RUB
        world_T_cam_rub[i] = world_T_cam

    orient_34 = auto_orient_and_center_poses(
        world_T_cam_rub.astype(np.float32),
        method="up",
        center_method="poses",
    ).transform
    orient_R: Float64[ndarray, "3 3"] = orient_34[:, :3]
    orient_t: Float64[ndarray, "3"] = orient_34[:, 3]
    return orient_R, orient_t


# ── Rerun logging helpers ───────────────────────────────────────────────


def log_trajectory(
    parent_log_path: Path,
    cam_se3_world_buffer: Float32[Tensor, "buffer_size 7"],
    points: Float32[Tensor, "num_points 3"],
    colors: UInt8[Tensor, "buffer_size num_patches 3"],
    path_list: list[list[float]],
    last_orient_n: int = 0,
    log_points: bool = True,
) -> tuple[list[list[float]], int]:
    """Log the current SLAM state to Rerun for live visualization.

    Every call logs the most recent camera-to-world transform and the
    accumulated camera path as a 3-D line strip.  With ``log_points`` it also
    logs the point cloud (after radius-based outlier removal on the GPU);
    the caller sets it only when a new keyframe joined the buffer.

    Args:
        parent_log_path: Rerun entity path prefix (e.g. ``Path("world")``).
        cam_se3_world_buffer: All buffered keyframe poses
            ``[tx, ty, tz, qx, qy, qz, qw]`` in camera-from-world
            (``cam_se3_world``) convention.
        points: All buffered 3-D points (flattened across patches).
        colors: Per-point RGB colors matching ``points``.
        path_list: Running list of world-space camera positions; updated
            in-place and returned.
        last_orient_n: Number of poses used in the last orient recomputation.
            Pass 0 on the first call.
        log_points: Whether to log the point cloud.

    Returns:
        Tuple of ``(path_list, last_orient_n)`` — the updated path list and
        the pose count at which the orient transform was last recomputed.
    """
    cam_log_path: str = f"{parent_log_path}/camera"

    # Filter out zero-initialized (unused) buffer slots
    poses_mask: Tensor = ~(cam_se3_world_buffer[:, :6] == 0).all(dim=1)
    nonzero_poses: Float32[Tensor, "num_nonzero 7"] = cam_se3_world_buffer[poses_mask]

    if nonzero_poses.shape[0] == 0:
        return path_list, last_orient_n

    # Extract the most recent keyframe pose.
    # poses_ stores cam_se3_world (camera-from-world) internally, so the
    # raw [tx, ty, tz, qx, qy, qz, qw] encodes cam_T_world.
    cam_se3_world: Float32[ndarray, "7"] = nonzero_poses[-1].numpy(force=True)
    cam_T_world: Float64[ndarray, "4 4"] = np.eye(4)
    cam_T_world[:3, :3] = Rotation.from_quat(cam_se3_world[3:]).as_matrix()
    cam_T_world[:3, 3] = cam_se3_world[:3]

    world_T_cam: Float64[ndarray, "4 4"] = np.linalg.inv(cam_T_world)
    world_t_cam: Float64[ndarray, "3"] = world_T_cam[:3, 3]
    path_list.append(world_t_cam.tolist())

    rr.log(cam_log_path, rr.Transform3D(translation=world_t_cam, mat3x3=world_T_cam[:3, :3], from_parent=False))
    rr.log(f"{parent_log_path}/path", rr.LineStrips3D(strips=[path_list], colors=[255, 0, 0]))

    if log_points:
        points_mask: Tensor = ~(points == 0).all(dim=1)
        kept_points: Float32[Tensor, "n 3"] = points[points_mask]
        kept_colors: UInt8[Tensor, "n 3"] = colors.view(-1, 3)[points_mask]
        # Outlier removal: discard points whose distance from the camera
        # trajectory median exceeds 5x the maximum camera-center radius.
        # Use world-space camera positions (path_list) instead of
        # cam_se3_world translations, which live in a different frame.
        if len(path_list) >= 2:
            cam_positions: Float32[Tensor, "n_cams 3"] = torch.tensor(path_list, dtype=kept_points.dtype, device=kept_points.device)
            center: Float32[Tensor, "3"] = cam_positions.median(dim=0).values
            max_radius: Tensor = (cam_positions - center).norm(dim=1).max()
            inliers: Tensor = (kept_points - center).norm(dim=1) < max_radius * 5
            kept_points, kept_colors = kept_points[inliers], kept_colors[inliers]
        rr.log(
            f"{parent_log_path}/pointcloud",
            rr.Points3D(positions=kept_points.numpy(force=True), colors=kept_colors.numpy(force=True)),
        )

    # Recompute gravity-alignment transform periodically.
    # Fire immediately once we have >= 3 poses, then every N keyframes.
    n_poses: int = nonzero_poses.shape[0]
    if (last_orient_n == 0 and n_poses >= 3) or (n_poses - last_orient_n >= _ORIENT_RECOMPUTE_INTERVAL):
        orient = compute_orient_transform(cam_se3_world_buffer)
        if orient is not None:
            orient_R, orient_t = orient
            rr.log(f"{parent_log_path}", rr.Transform3D(mat3x3=orient_R, translation=orient_t))
            last_orient_n = n_poses

    return path_list, last_orient_n


# ── Blueprint ───────────────────────────────────────────────────────────

# Chase eye of the Follow view, in the camera frame (RDF: x right, y down, z forward). DPVO is monocular, so the
# units are its own scene scale (initial patch depth ~1): behind and above the camera, aimed ahead of it.
_FOLLOW_EYE_POSITION: tuple[float, float, float] = (0.0, -0.6, -1.5)
_FOLLOW_EYE_TARGET: tuple[float, float, float] = (0.0, 0.0, 1.5)
_FOLLOW_EYE_UP: tuple[float, float, float] = (0.0, -1.0, 0.0)


def dpvo_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    """Overview and Follow 3D views beside the camera, like dataforge's rig layout.

    The overview is rooted at ``/`` so the gravity-alignment transform on
    ``parent_log_path`` shows (per rerun#9244 a Transform3D on a view's own
    origin is invisible), and it draws the ground grid. The Follow view is
    rooted at the camera entity, so its eye, fixed in the camera frame, rides
    the camera; it has no grid because the camera frame has no ground plane.
    """
    camera_path: str = f"{parent_log_path}/camera"
    overview = rrb.Spatial3DView(name="Overview", origin="/", eye_controls=rrb.EyeControls3D(spin_speed=0.25))
    follow = rrb.Spatial3DView(
        name="Follow",
        origin=camera_path,
        contents="/**",
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=_FOLLOW_EYE_POSITION,
            look_target=_FOLLOW_EYE_TARGET,
            eye_up=_FOLLOW_EYE_UP,
            spin_speed=0.0,
        ),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(overview, follow),
            rrb.Spatial2DView(origin=f"{camera_path}/pinhole/image", name="Camera"),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


# ── Intrinsics ──────────────────────────────────────────────────────────

INTRINSICS_FRAMES: int = 8
"""Leading frames whose median MoGe-2 intrinsics calibrate a run without a calibration file."""


def load_intrinsics_model() -> Any:
    """MoGe-2 ViT-L geometry predictor: monoprior's TensorRT engines when TensorRT imports, else Torch."""
    from monopriors.models.moge_v2 import MoGeV2TorchPredictor, MoGeV2TrtPredictor

    if importlib.util.find_spec("tensorrt") is not None and importlib.util.find_spec("trtkit") is not None:
        return MoGeV2TrtPredictor(encoder="vitl", heads="geometry")
    return MoGeV2TorchPredictor(encoder="vitl", heads="geometry")


def estimate_intrinsics(bgr_b3hw: UInt8[Tensor, "b 3 h w"], model: Any) -> Float32[Tensor, "4"]:
    """Median ``[fx, fy, cx, cy]`` in pixels of the batch's MoGe-2 intrinsics."""
    rgb_bhw3: UInt8[Tensor, "b h w 3"] = bgr_b3hw.flip(1).permute(0, 2, 3, 1).contiguous()
    with torch.inference_mode():
        K_norm_b33: Float32[Tensor, "b 3 3"] = model.predict_geometry(rgb_bhw3).intrinsics_b33
    height, width = bgr_b3hw.shape[-2:]
    normalized: Float32[Tensor, "b 4"] = torch.stack(
        [K_norm_b33[:, 0, 0], K_norm_b33[:, 1, 1], K_norm_b33[:, 0, 2], K_norm_b33[:, 1, 2]], dim=1
    )
    scale: Float32[Tensor, "4"] = torch.tensor([width, height, width, height], dtype=torch.float32, device=normalized.device)
    return (normalized.median(dim=0).values * scale).float()


# ── Main pipeline generator ────────────────────────────────────────────


@torch.no_grad()
def run_dpvo_pipeline(
    *,
    dpvo_config: DPVOConfig,
    network_path: str,
    imagedir: str,
    calib: str | None = None,
    stride: int = 1,
    skip: int = 0,
    intrinsics_model: Any | None = None,
    parent_log_path: Path = Path("world"),
    handle: DPVOPipelineHandle | None = None,
    recording: rr.RecordingStream | None = None,
    timeit: bool = False,
    max_frames: int = 0,
    device: torch.device = DEFAULT_DEVICE,
) -> Generator[str, None, None]:
    """Run the full DPVO visual-odometry pipeline on an image dir or video.

    This is a **generator** that yields status strings after each processed
    frame.  Both the CLI and Gradio UI consume it identically — the CLI
    exhausts it silently while Gradio yields intermediate Rerun bytes.

    1. Logs a video once as an H.264 VideoStream on the ``video_time`` timeline.
    2. Decodes frames on the GPU (video) or in a background process (images).
    3. Without a calibration file, estimates the intrinsics with MoGe-2 from
       the median of the first :data:`INTRINSICS_FRAMES` frames.
    4. Feeds each frame into the DPVO SLAM system and logs the camera pose
       per frame and the point cloud when a keyframe joins the buffer.
    5. Populates ``handle.prediction`` and ``handle.elapsed_time``.

    Args:
        dpvo_config: DPVO solver configuration.
        network_path: File path to the pre-trained DPVO checkpoint.
        imagedir: Path to an image directory or video file.
        calib: Path to a calibration text file (``fx fy cx cy`` of the native
            video or images), or ``None`` to estimate intrinsics with MoGe-2.
        stride: Keep every *stride*-th frame.
        skip: Number of leading frames to discard.
        intrinsics_model: Pre-loaded MoGe-2 predictor (Gradio singleton).
            ``None`` loads one when needed (CLI path).
        parent_log_path: Rerun entity path prefix.
        handle: Mutable handle to store results in.
        recording: Rerun recording stream for Gradio (thread-local).
            ``None`` uses the active recording (CLI).
        timeit: If ``True``, print per-iteration timing via ``Timer``.
        max_frames: Maximum number of frames to process (0 = no limit).
        device: Device that decodes the frames and runs the tracker.

    Yields:
        Status message strings (e.g. ``"Frame 42/200"``).
    """
    is_video: bool = Path(imagedir).is_file()
    total_frames: int = num_frames(imagedir, stride, skip)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    cam_image_path: str = f"{parent_log_path}/camera/pinhole/image"
    rr.send_blueprint(dpvo_blueprint(parent_log_path))

    start: float = timer()

    video_times_ns: ndarray | None = None
    if is_video:
        yield "Logging the source video..."
        video_times_ns = log_video(
            Path(imagedir),
            Path(cam_image_path),
            timeline="video_time",
            recording=recording if recording is not None else rr.get_data_recording(),
            output_codec=rr.VideoCodec.H264,
        )

    frames: Iterator[Frame] = (
        video_frames(Path(imagedir), stride, skip, device=device) if is_video else image_dir_frames(imagedir, stride, skip, device=device)
    )
    if max_frames > 0:
        frames = islice(frames, max_frames)
    head: list[Frame] = list(islice(frames, INTRINSICS_FRAMES))
    if not head:
        yield "No frames processed"
        return

    # SLAM-resolution intrinsics; calibration files describe the native video or images.
    slam_scale: float = VIDEO_SCALE if is_video else 1.0
    if calib is not None:
        K_33, _ = load_calib(calib)
        native: Float32[ndarray, "4"] = np.array([K_33[0, 0], K_33[1, 1], K_33[0, 2], K_33[1, 2]], dtype=np.float32)
        intrinsics: Float32[Tensor, "4"] = torch.from_numpy(native * slam_scale).to(device)
    else:
        yield "Estimating camera intrinsics with MoGe-2..."
        intrinsics = estimate_intrinsics(
            torch.stack([frame.bgr_3hw for frame in head]),
            intrinsics_model if intrinsics_model is not None else load_intrinsics_model(),
        ).to(device)

    # The pinhole matches what the 2D view shows: the native video, or the cropped images.
    # Cropping keeps the top-left corner, so the principal point needs no shift.
    slam_h, slam_w = head[0].bgr_3hw.shape[1:]
    pinhole_hw: tuple[int, int] = video_hw(Path(imagedir)) if is_video else (slam_h, slam_w)
    pinhole_K: Float32[ndarray, "4"] = intrinsics.numpy(force=True) / slam_scale
    rr.log(
        f"{parent_log_path}/camera/pinhole",
        rr.Pinhole(
            height=pinhole_hw[0],
            width=pinhole_hw[1],
            focal_length=[float(pinhole_K[0]), float(pinhole_K[1])],
            principal_point=[float(pinhole_K[2]), float(pinhole_K[3])],
            image_plane_distance=1,
        ),
        static=True,
    )

    slam: DPVO = DPVO(dpvo_config, network_path, ht=slam_h, wd=slam_w, device=device)
    path_list: list[list[float]] = []
    last_orient_n: int = 0
    last_points_n: int = 0

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        for frame in chain(head, frames):
            if video_times_ns is not None:
                rr.set_time("video_time", duration=np.timedelta64(int(video_times_ns[frame.source_index]), "ns"))
            else:
                rr.set_time("timestep", sequence=frame.t)
                assert frame.bgr_hw3 is not None
                rr.log(cam_image_path, rr.Image(frame.bgr_hw3, color_model=rr.ColorModel.BGR).compress(jpeg_quality=90))

            with Timer("SLAM", enabled=timeit, device=device):
                slam(frame.t, frame.bgr_3hw, intrinsics)

            if slam.is_initialized:
                path_list, last_orient_n = log_trajectory(
                    parent_log_path=parent_log_path,
                    cam_se3_world_buffer=slam.poses_,
                    points=slam.points_,
                    colors=slam.colors_,
                    path_list=path_list,
                    last_orient_n=last_orient_n,
                    log_points=slam.n > last_points_n,
                )
                last_points_n = slam.n

            pbar.update(1)
            yield f"Frame {frame.t + 1}/{total_frames}"

    # Terminate: runs loop closure finalization + 12 BA iterations
    yield "Running final bundle adjustment..."

    total_time: float = timer() - start
    print(f"Total time: {total_time:.2f}s")

    final_poses, tstamps = slam.terminate()
    dpvo_pred: DPVOPrediction = DPVOPrediction(
        final_poses=final_poses,
        tstamps=tstamps,
        final_points=slam.points_,
        final_colors=slam.colors_,
    )

    if handle is not None:
        handle.prediction = dpvo_pred
        handle.elapsed_time = total_time


def dpvo_inference(config: DPVOInferenceConfig) -> None:
    """Run the DPVO inference pipeline from the command line.

    This is the main CLI entry point. It exhausts the
    :func:`run_dpvo_pipeline` generator silently and prints summary stats.

    Args:
        config: Inference configuration (typically from ``tyro.cli``).
    """
    handle: DPVOPipelineHandle = DPVOPipelineHandle()

    for _msg in run_dpvo_pipeline(
        dpvo_config=config.dpvo_config,
        network_path=config.network_path,
        imagedir=config.imagedir,
        calib=config.calib,
        stride=config.stride,
        skip=config.skip,
        handle=handle,
        max_frames=config.max_frames,
    ):
        pass  # CLI exhausts the generator silently

    if handle.prediction is not None:
        print(f"Processed in {handle.elapsed_time:.2f}s")
        print(f"Keyframes: {handle.prediction.final_poses.shape[0]}")
