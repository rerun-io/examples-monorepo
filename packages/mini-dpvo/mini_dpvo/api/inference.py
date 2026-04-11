"""High-level inference pipeline for DPVO visual odometry.

This module provides the main entry-point for running DPVO inference on
image directories or video files.  It handles frame reading (via multiprocessing),
optional auto-calibration using DUSt3R, the DPVO SLAM loop, and Rerun-based
visualization of camera trajectories, point clouds, and per-frame images.

Both the CLI and Gradio UI consume the same :func:`run_dpvo_pipeline` generator.

Typical CLI usage::

    from mini_dpvo.api.inference import DPVOInferenceConfig, run_dpvo_pipeline
    config = tyro.cli(DPVOInferenceConfig)
    for msg in run_dpvo_pipeline(dpvo_config=config.dpvo_config, ...):
        pass
"""

import os
from collections.abc import Generator
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from pathlib import Path
from timeit import default_timer as timer
from typing import Annotated

import cv2
import mmcv
import numpy as np
import rerun as rr
import torch
import tyro
from einops import rearrange
from jaxtyping import Float32, Float64, UInt8
from mini_dust3r.api import OptimizedResult, inferece_dust3r
from mini_dust3r.model import AsymmetricCroCo3DStereo
from scipy.spatial.transform import Rotation
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm

from mini_dpvo.config import DPVOConfig
from mini_dpvo.dpvo import DPVO
from mini_dpvo.stream import image_stream, video_stream
from mini_dpvo.utils import Timer

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


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class DPVOPrediction:
    """Container for the final outputs of a DPVO inference run.

    Holds the optimized keyframe poses (translation + quaternion), their
    timestamps, and the associated 3-D point cloud with per-point colors.
    """

    final_poses: Float32[np.ndarray, "num_keyframes 7"]
    """Keyframe poses as ``[tx, ty, tz, qx, qy, qz, qw]``."""
    tstamps: Float64[np.ndarray, "num_keyframes"]  # noqa: F821
    """Timestamp (frame index) of each keyframe."""
    final_points: Float32[torch.Tensor, "num_points 3"]
    """Reconstructed 3-D points in world coordinates."""
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"]
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
    dpvo_config: AccurateDPVOConfig | FastDPVOConfig = field(default_factory=DPVOConfig.fast)
    """DPVO solver configuration preset."""
    imagedir: str = "data/movies/IMG_0493.MOV"
    """Path to image directory or video file."""
    network_path: str = "checkpoints/dpvo.pth"
    """Path to DPVO network weights."""
    calib: str | None = None
    """Path to calibration file. If None, uses DUSt3R for estimation."""
    stride: int = 2
    """Frame stride for sampling."""
    skip: int = 0
    """Number of leading frames to skip."""


# ── Rerun logging helpers ───────────────────────────────────────────────


def log_trajectory(
    parent_log_path: Path,
    poses: Float32[torch.Tensor, "buffer_size 7"],
    points: Float32[torch.Tensor, "num_points 3"],
    colors: UInt8[torch.Tensor, "buffer_size num_patches 3"],
    intri_np: Float32[np.ndarray, "4"],
    bgr_hw3: UInt8[np.ndarray, "h w 3"],
    path_list: list[list[float]],
    jpg_quality: int = 90,
) -> list[list[float]]:
    """Log the current SLAM state to Rerun for live visualization.

    For the most recent keyframe this logs:
    * the camera image (JPEG-compressed),
    * the pinhole intrinsics,
    * the camera-to-world transform,
    * the accumulated camera path as a 3-D line strip, and
    * the point cloud (after radius-based outlier removal).

    Args:
        parent_log_path: Rerun entity path prefix (e.g. ``Path("world")``).
        poses: All buffered keyframe poses ``[tx, ty, tz, qx, qy, qz, qw]``.
        points: All buffered 3-D points (flattened across patches).
        colors: Per-point RGB colors matching ``points``.
        intri_np: Camera intrinsics as ``[fx, fy, cx, cy]``.
        bgr_hw3: Current frame in BGR channel order.
        path_list: Running list of world-space camera positions; updated
            in-place and returned.
        jpg_quality: JPEG compression quality for the image log (0--100).

    Returns:
        The updated ``path_list`` with the latest camera position appended.
    """
    cam_log_path: str = f"{parent_log_path}/camera"
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = mmcv.bgr2rgb(bgr_hw3)
    rr.log(
        f"{cam_log_path}/pinhole/image",
        rr.Image(rgb_hw3).compress(jpeg_quality=jpg_quality),
    )
    rr.log(
        f"{cam_log_path}/pinhole",
        rr.Pinhole(
            height=bgr_hw3.shape[0],
            width=bgr_hw3.shape[1],
            focal_length=[intri_np[0], intri_np[1]],
            principal_point=[intri_np[2], intri_np[3]],
            image_plane_distance=1,
        ),
    )

    # Filter out zero-initialized (unused) buffer slots
    poses_mask: torch.Tensor = ~(poses[:, :6] == 0).all(dim=1)
    points_mask: torch.Tensor = ~(points == 0).all(dim=1)

    nonzero_poses: Float32[torch.Tensor, "num_nonzero 7"] = poses[poses_mask]
    nonzero_points: Float32[torch.Tensor, "num_nonzero 3"] = points[points_mask]

    if nonzero_poses.shape[0] == 0:
        return path_list

    # Extract the most recent keyframe pose for the camera transform
    last_index: int = nonzero_poses.shape[0] - 1
    quat_pose: Float32[np.ndarray, "7"] = nonzero_poses[last_index].numpy(force=True)
    trans_quat: Float32[np.ndarray, "3"] = quat_pose[:3]
    rotation_quat: Rotation = Rotation.from_quat(quat_pose[3:])

    # Build the camera-to-world 4x4 transform (cam_T_world) and invert
    # to get world_T_cam for Rerun logging (child-from-parent convention).
    cam_R_world: Float64[np.ndarray, "3 3"] = rotation_quat.as_matrix()

    cam_T_world: Float64[np.ndarray, "4 4"] = np.eye(4)
    cam_T_world[:3, :3] = cam_R_world
    cam_T_world[0:3, 3] = trans_quat

    world_T_cam: Float64[np.ndarray, "4 4"] = np.linalg.inv(cam_T_world)

    path_list.append(world_T_cam[:3, 3].copy().tolist())

    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=world_T_cam[:3, 3],
            mat3x3=world_T_cam[:3, :3],
            from_parent=False,
        ),
    )

    # log path using linestrip
    rr.log(
        f"{parent_log_path}/path",
        rr.LineStrips3D(
            strips=[
                path_list,
            ],
            colors=[255, 0, 0],
        ),
    )

    # Outlier removal: discard points whose distance from the trajectory
    # median exceeds 5x the maximum camera-center radius.
    trajectory_center: Float32[np.ndarray, "3"] = np.median(nonzero_poses[:, :3].numpy(force=True), axis=0)

    def radii(a: Float32[np.ndarray, "n 3"]) -> Float32[np.ndarray, "n"]:
        """Compute Euclidean distance of each row to ``trajectory_center``."""
        return np.linalg.norm(a - trajectory_center, axis=1)

    points_np: Float32[np.ndarray, "num_points 3"] = nonzero_points.view(-1, 3).numpy(force=True)
    colors_np: UInt8[np.ndarray, "num_points 3"] = colors.view(-1, 3)[points_mask].numpy(force=True)
    inlier_mask: np.ndarray = (
        radii(points_np) < radii(nonzero_poses[:, :3].numpy(force=True)).max() * 5
    )
    points_filtered: Float32[np.ndarray, "num_inliers 3"] = points_np[inlier_mask]
    colors_filtered: UInt8[np.ndarray, "num_inliers 3"] = colors_np[inlier_mask]

    # log all points and colors at the same time
    rr.log(
        f"{parent_log_path}/pointcloud",
        rr.Points3D(
            positions=points_filtered,
            colors=colors_filtered,
        ),
    )
    return path_list


def log_final(
    parent_log_path: Path,
    final_poses: Float32[torch.Tensor, "num_keyframes 7"],
    tstamps: Float64[torch.Tensor, "num_keyframes"],  # noqa: F821
    final_points: Float32[torch.Tensor, "num_points 3"],
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"],
) -> None:
    """Log the final optimized per-keyframe camera transforms to Rerun.

    After DPVO terminates its SLAM loop and performs final bundle adjustment,
    this function logs one ``rr.Transform3D`` per keyframe so the user can
    inspect the full set of optimized camera poses.

    Args:
        parent_log_path: Rerun entity path prefix (e.g. ``Path("world")``).
        final_poses: Optimized keyframe poses ``[tx, ty, tz, qx, qy, qz, qw]``.
        tstamps: Timestamp (frame index) per keyframe.
        final_points: Final 3-D point cloud (unused here, reserved for
            future extensions).
        final_colors: Per-point colors (unused here).
    """
    for idx, (pose_quat, _tstamp) in enumerate(zip(final_poses, tstamps, strict=False)):
        cam_log_path: str = f"{parent_log_path}/camera_{idx}"
        trans_quat: Float32[np.ndarray, "3"] = pose_quat[:3]
        R_33: Float64[np.ndarray, "3 3"] = Rotation.from_quat(pose_quat[3:]).as_matrix()
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(translation=trans_quat, mat3x3=R_33, from_parent=False),
        )


# ── Frame I/O helpers ───────────────────────────────────────────────────


def create_reader(
    imagedir: str, calib: str | None, stride: int, skip: int, queue: Queue
) -> Process:
    """Create a multiprocessing ``Process`` that reads frames into a queue.

    If ``imagedir`` is a directory the reader uses :func:`image_stream`;
    otherwise it treats the path as a video file and uses
    :func:`video_stream`.

    Args:
        imagedir: Path to an image directory **or** a video file.
        calib: Path to a calibration file (``fx fy cx cy`` per line), or
            ``None`` to skip file-based calibration.
        stride: Sample every *stride*-th frame.
        skip: Number of leading frames to skip before sampling begins.
        queue: Shared ``Queue`` into which ``(timestep, bgr_hw3, intrinsics)``
            tuples are placed. A sentinel ``(-1, ...)`` signals end-of-stream.

    Returns:
        An unstarted ``Process`` ready to be ``.start()``-ed.
    """
    reader: Process
    if os.path.isdir(imagedir):
        reader = Process(
            target=image_stream, args=(queue, imagedir, calib, stride, skip)
        )
    else:
        reader = Process(
            target=video_stream, args=(queue, imagedir, calib, stride, skip)
        )

    return reader


def calculate_num_frames(video_or_image_dir: str, stride: int, skip: int) -> int:
    """Calculate the effective number of frames after applying skip and stride.

    For an image directory the total is the number of regular files; for a
    video file it is the frame count reported by OpenCV.  The returned value
    accounts for both ``skip`` (dropped leading frames) and ``stride``
    (sub-sampling interval).

    Args:
        video_or_image_dir: Path to an image directory or video file.
        stride: Sampling interval (keep every *stride*-th frame).
        skip: Number of leading frames to discard.

    Returns:
        The number of frames that will actually be processed.
    """
    total_frames: int = 0
    if os.path.isdir(video_or_image_dir):
        total_frames = len(
            [
                name
                for name in os.listdir(video_or_image_dir)
                if os.path.isfile(os.path.join(video_or_image_dir, name))
            ]
        )
    else:
        cap: cv2.VideoCapture = cv2.VideoCapture(video_or_image_dir)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    total_frames = (total_frames - skip) // stride
    return total_frames


def calib_from_dust3r(
    bgr_hw3: UInt8[np.ndarray, "height width 3"],
    model: AsymmetricCroCo3DStereo,
    device: str,
) -> Float32[np.ndarray, "3 3"]:
    """Estimate a 3x3 camera intrinsic matrix from a single image using DUSt3R.

    The image is temporarily saved to disk, fed through DUSt3R monocular
    calibration, and the resulting intrinsics are up-scaled from the DUSt3R
    processing resolution back to the original image dimensions.

    Args:
        bgr_hw3: Input image in BGR channel order with shape
            ``(height, width, 3)``.
        model: Pre-loaded DUSt3R ``AsymmetricCroCo3DStereo`` model.
        device: Torch device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        The 3x3 intrinsic matrix ``K`` scaled to the original image size.
    """
    tmp_path: Path = Path("/tmp/dpvo/tmp.png")
    # Save image to a temporary file for DUSt3R (expects a directory of images)
    mmcv.imwrite(bgr_hw3, str(tmp_path))
    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=tmp_path.parent,
        model=model,
        device=device,
        batch_size=1,
    )
    # Clean up the temporary file
    tmp_path.unlink()

    # Scale the predicted intrinsics from DUSt3R's internal resolution
    # back up to the original image dimensions.
    downscaled_h: int
    downscaled_w: int
    downscaled_h, downscaled_w, _ = optimized_results.rgb_hw3_list[0].shape
    orig_h: int
    orig_w: int
    orig_h, orig_w, _ = bgr_hw3.shape

    # Scaling factors
    scaling_factor_x: float = orig_w / downscaled_w
    scaling_factor_y: float = orig_h / downscaled_h

    # Apply per-axis scaling to fx, fy, cx, cy
    K_33_original: Float32[np.ndarray, "3 3"] = optimized_results.K_b33[0].copy()
    K_33_original[0, 0] *= scaling_factor_x  # fx
    K_33_original[1, 1] *= scaling_factor_y  # fy
    K_33_original[0, 2] *= scaling_factor_x  # cx
    K_33_original[1, 2] *= scaling_factor_y  # cy

    return K_33_original


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
    dust3r_model: AsymmetricCroCo3DStereo | None = None,
    parent_log_path: Path = Path("world"),
    handle: DPVOPipelineHandle | None = None,
    recording: rr.RecordingStream | None = None,
    timeit: bool = False,
) -> Generator[str, None, None]:
    """Run the full DPVO visual-odometry pipeline on an image dir or video.

    This is a **generator** that yields status strings after each processed
    frame.  Both the CLI and Gradio UI consume it identically — the CLI
    exhausts it silently while Gradio yields intermediate Rerun bytes.

    1. Spawns a background ``Process`` to read frames into a shared queue.
    2. Optionally estimates camera intrinsics with DUSt3R when no calibration
       file is provided.
    3. Feeds each frame into the DPVO SLAM system.
    4. Logs live trajectory, point cloud, and images to Rerun after the
       system is initialized.
    5. Runs 12 additional update iterations after all frames are consumed
       for final refinement.
    6. Populates ``handle.prediction`` and ``handle.elapsed_time``.

    Args:
        dpvo_config: DPVO solver configuration.
        network_path: File path to the pre-trained DPVO checkpoint.
        imagedir: Path to an image directory or video file.
        calib: Path to a calibration text file (``fx fy cx cy`` per line),
            or ``None`` to auto-estimate intrinsics via DUSt3R.
        stride: Keep every *stride*-th frame.
        skip: Number of leading frames to discard.
        dust3r_model: Pre-loaded DUSt3R model (Gradio singleton).
            ``None`` loads one on-the-fly (CLI path).
        parent_log_path: Rerun entity path prefix.
        handle: Mutable handle to store results in.
        recording: Rerun recording stream for Gradio (thread-local).
            ``None`` uses the global recording (CLI).
        timeit: If ``True``, print per-iteration timing via ``Timer``.

    Yields:
        Status message strings (e.g. ``"Frame 42/200"``).
    """
    slam: DPVO | None = None
    queue = Queue(maxsize=8)

    reader: Process = create_reader(imagedir, calib, stride, skip, queue)
    reader.start()

    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    start: float = timer()
    total_frames: int = calculate_num_frames(imagedir, stride, skip)

    # If no calibration file was provided, use DUSt3R to predict intrinsics
    # from the very first frame pulled off the reader queue.
    intri_np_dust3r: Float32[np.ndarray, "4"] | None = None
    if calib is None:
        yield "Estimating camera intrinsics with DUSt3R..."
        if dust3r_model is None:
            dust3r_device: str = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            ).to(dust3r_device)
        else:
            dust3r_device = next(dust3r_model.parameters()).device.type

        _, bgr_hw3, _ = queue.get()
        K_33_pred: Float32[np.ndarray, "3 3"] = calib_from_dust3r(bgr_hw3, dust3r_model, dust3r_device)
        intri_np_dust3r = np.array(
            [K_33_pred[0, 0], K_33_pred[1, 1], K_33_pred[0, 2], K_33_pred[1, 2]]
        )

    # path list for visualizing the trajectory
    path_list: list[list[float]] = []
    frame_idx: int = 0

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            t: int
            bgr_hw3: UInt8[np.ndarray, "h w 3"]
            intri_np: Float32[np.ndarray, "4"]
            (t, bgr_hw3, intri_np_calib) = queue.get()
            intri_np = intri_np_calib if calib is not None else intri_np_dust3r
            # queue will have a (-1, image, intrinsics) tuple when the reader is done
            if t < 0:
                break

            rr.set_time("timestep", sequence=t)

            bgr_3hw: UInt8[torch.Tensor, "c ht wd"] = rearrange(torch.from_numpy(bgr_hw3), "h w c -> c h w").cuda()
            intri_torch: Float32[torch.Tensor, "4"] = torch.from_numpy(intri_np).cuda()

            if slam is None:
                slam = DPVO(dpvo_config, network_path, ht=bgr_3hw.shape[1], wd=bgr_3hw.shape[2])

            with Timer("SLAM", enabled=timeit):
                slam(t, bgr_3hw, intri_torch)

            if slam.is_initialized:
                poses: Float32[torch.Tensor, "buffer_size 7"] = slam.poses_
                points: Float32[torch.Tensor, "num_points 3"] = (
                    slam.points_
                )
                colors: UInt8[torch.Tensor, "buffer_size num_patches 3"] = slam.colors_
                path_list = log_trajectory(
                    parent_log_path=parent_log_path,
                    poses=poses,
                    points=points,
                    colors=colors,
                    intri_np=intri_np,
                    bgr_hw3=bgr_hw3,
                    path_list=path_list,
                )

            frame_idx += 1
            pbar.update(1)
            yield f"Frame {frame_idx}/{total_frames}"

    # Run additional update iterations for final bundle-adjustment refinement
    yield "Running final bundle adjustment..."
    for _ in range(12):
        slam.update()

    total_time: float = timer() - start
    print(f"Total time: {total_time:.2f}s")

    reader.join()

    final_poses: Float32[torch.Tensor, "num_keyframes 7"]
    tstamps: Float64[torch.Tensor, "num_keyframes"]  # noqa: F821

    final_poses, tstamps = slam.terminate()
    final_points: Float32[torch.Tensor, "num_points 3"] = slam.points_
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"] = slam.colors_
    dpvo_pred: DPVOPrediction = DPVOPrediction(
        final_poses=final_poses,
        tstamps=tstamps,
        final_points=final_points,
        final_colors=final_colors,
    )

    if handle is not None:
        handle.prediction = dpvo_pred
        handle.elapsed_time = total_time
