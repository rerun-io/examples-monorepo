import numpy as np
import os
import torch
from pathlib import Path
from multiprocessing import Process, Queue
from yacs.config import CfgNode

from mini_dpvo.utils import Timer
from mini_dpvo.dpvo import DPVO
from mini_dpvo.stream import image_stream, video_stream

import rerun as rr
from jaxtyping import UInt8, Float64, Float32
from scipy.spatial.transform import Rotation
from dataclasses import dataclass

from timeit import default_timer as timer

import cv2
import mmcv
from tqdm import tqdm
from mini_dust3r.api import OptimizedResult, inferece_dust3r
from mini_dust3r.model import AsymmetricCroCo3DStereo


@dataclass
class DPVOPrediction:
    final_poses: Float32[torch.Tensor, "num_keyframes 7"]
    tstamps: Float64[torch.Tensor, "num_keyframes"]  # noqa: F821
    final_points: Float32[torch.Tensor, "buffer_size*num_patches 3"]
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"]


def log_trajectory(
    parent_log_path: Path,
    poses: Float32[torch.Tensor, "buffer_size 7"],
    points: Float32[torch.Tensor, "buffer_size*num_patches 3"],
    colors: UInt8[torch.Tensor, "buffer_size num_patches 3"],
    intri_np: Float64[np.ndarray, "4"],
    bgr_hw3: UInt8[np.ndarray, "h w 3"],
    path_list: list,
    jpg_quality: int = 90,
):
    cam_log_path = f"{parent_log_path}/camera"
    rgb_hw3 = mmcv.bgr2rgb(bgr_hw3)
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

    poses_mask = ~(poses[:, :6] == 0).all(dim=1)
    points_mask = ~(points == 0).all(dim=1)

    nonzero_poses = poses[poses_mask]
    nonzero_points = points[points_mask]

    last_index = nonzero_poses.shape[0] - 1
    # get last non-zero pose, and the index of the last non-zero pose
    quat_pose = nonzero_poses[last_index].numpy(force=True)
    trans_quat: Float32[np.ndarray, "3"] = quat_pose[:3]
    rotation_quat = Rotation.from_quat(quat_pose[3:])

    cam_R_world: Float64[np.ndarray, "3 3"] = rotation_quat.as_matrix()

    cam_T_world = np.eye(4)
    cam_T_world[:3, :3] = cam_R_world
    cam_T_world[0:3, 3] = trans_quat

    world_T_cam = np.linalg.inv(cam_T_world)

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

    # outlier removal
    trajectory_center = np.median(nonzero_poses[:, :3].numpy(force=True), axis=0)

    def radii(a):
        return np.linalg.norm(a - trajectory_center, axis=1)

    points_np = nonzero_points.view(-1, 3).numpy(force=True)
    colors_np = colors.view(-1, 3)[points_mask].numpy(force=True)
    inlier_mask = (
        radii(points_np) < radii(nonzero_poses[:, :3].numpy(force=True)).max() * 5
    )
    points_filtered = points_np[inlier_mask]
    colors_filtered = colors_np[inlier_mask]

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
    final_points: Float32[torch.Tensor, "buffer_size*num_patches 3"],
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"],
):
    for idx, (pose_quat, tstamp) in enumerate(zip(final_poses, tstamps)):
        cam_log_path = f"{parent_log_path}/camera_{idx}"
        trans_quat = pose_quat[:3]
        R_33 = Rotation.from_quat(pose_quat[3:]).as_matrix()
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(translation=trans_quat, mat3x3=R_33, from_parent=False),
        )


def create_reader(
    imagedir: str, calib: str | None, stride: int, skip: int, queue: Queue
) -> Process:
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
    # Determine the total number of frames
    total_frames = 0
    if os.path.isdir(video_or_image_dir):
        total_frames = len(
            [
                name
                for name in os.listdir(video_or_image_dir)
                if os.path.isfile(os.path.join(video_or_image_dir, name))
            ]
        )
    else:
        cap = cv2.VideoCapture(video_or_image_dir)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    total_frames = (total_frames - skip) // stride
    return total_frames


def calib_from_dust3r(
    bgr_hw3: UInt8[np.ndarray, "height width 3"],
    model: AsymmetricCroCo3DStereo,
    device: str,
) -> Float64[np.ndarray, "3 3"]:
    """
    Calculates the calibration matrix from mini-dust3r.

    Args:
        bgr_hw3: The input image in BGR format with shape (height, width, 3).
        model: The Dust3D-R model used for inference.
        device: The device to run the inference on.

    Returns:
        The calibration matrix with shape (3, 3).

    Raises:
        None.
    """
    tmp_path = Path("/tmp/dpvo/tmp.png")
    # save image
    mmcv.imwrite(bgr_hw3, str(tmp_path))
    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=tmp_path.parent,
        model=model,
        device=device,
        batch_size=1,
    )
    # DELETE tmp file
    tmp_path.unlink()

    # get predicted intrinsics in original image size
    downscaled_h, downscaled_w, _ = optimized_results.rgb_hw3_list[0].shape
    orig_h, orig_w, _ = bgr_hw3.shape

    # Scaling factors
    scaling_factor_x = orig_w / downscaled_w
    scaling_factor_y = orig_h / downscaled_h

    # Scale the intrinsic matrix to the original image size
    K_33_original = optimized_results.K_b33[0].copy()
    K_33_original[0, 0] *= scaling_factor_x  # fx
    K_33_original[1, 1] *= scaling_factor_y  # fy
    K_33_original[0, 2] *= scaling_factor_x  # cx
    K_33_original[1, 2] *= scaling_factor_y  # cy

    return K_33_original


@torch.no_grad()
def inference_dpvo(
    cfg: CfgNode,
    network_path: str,
    imagedir: str,
    calib: str,
    stride: int = 1,
    skip: int = 0,
    timeit: bool = False,
) -> tuple[DPVOPrediction, float]:
    slam = None
    queue = Queue(maxsize=8)

    reader: Process = create_reader(imagedir, calib, stride, skip, queue)
    reader.start()

    parent_log_path = Path("world")
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)

    start = timer()
    total_frames = calculate_num_frames(imagedir, stride, skip)

    # estimate camera intrinsics if not provided
    if calib is None:
        dust3r_device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(dust3r_device)
        _, bgr_hw3, _ = queue.get()
        K_33_pred = calib_from_dust3r(bgr_hw3, dust3r_model, dust3r_device)
        intri_np_dust3r = np.array(
            [K_33_pred[0, 0], K_33_pred[1, 1], K_33_pred[0, 2], K_33_pred[1, 2]]
        )

    # path list for visualizing the trajectory
    path_list = []

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            t: int
            bgr_hw3: UInt8[np.ndarray, "h w 3"]
            intri_np: Float64[np.ndarray, "4"]
            (t, bgr_hw3, intri_np_calib) = queue.get()
            intri_np = intri_np_calib if calib is not None else intri_np_dust3r
            # queue will have a (-1, image, intrinsics) tuple when the reader is done
            if t < 0:
                break

            rr.set_time_sequence(timeline="timestep", sequence=t)

            bgr_3hw: UInt8[torch.Tensor, "h w 3"] = (
                torch.from_numpy(bgr_hw3).permute(2, 0, 1).cuda()
            )
            intri_torch: Float64[torch.Tensor, "4"] = torch.from_numpy(intri_np).cuda()

            if slam is None:
                slam = DPVO(cfg, network_path, ht=bgr_3hw.shape[1], wd=bgr_3hw.shape[2])

            with Timer("SLAM", enabled=timeit):
                slam(t, bgr_3hw, intri_torch)

            if slam.is_initialized:
                poses: Float32[torch.Tensor, "buffer_size 7"] = slam.poses_
                points: Float32[torch.Tensor, "buffer_size*num_patches 3"] = (
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
            pbar.update(1)

    for _ in range(12):
        slam.update()

    total_time: float = timer() - start
    print(f"Total time: {total_time:.2f}s")

    reader.join()

    final_poses: Float32[torch.Tensor, "num_keyframes 7"]
    tstamps: Float64[torch.Tensor, "num_keyframes"]  # noqa: F821

    final_poses, tstamps = slam.terminate()
    final_points: Float32[torch.Tensor, "buffer_size*num_patches 3"] = slam.points_
    final_colors: UInt8[torch.Tensor, "buffer_size num_patches 3"] = slam.colors_
    dpvo_pred = DPVOPrediction(
        final_poses=final_poses,
        tstamps=tstamps,
        final_points=final_points,
        final_colors=final_colors,
    )
    return dpvo_pred, total_time
