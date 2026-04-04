import cv2
import numpy as np
import open3d as o3d
import torch
import tqdm
from pathlib import Path
from jaxtyping import Float32, UInt8
from typing import Literal
from serde import serde
from serde.json import to_json
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
import lietorch
from simplecv.ops import conventions
from mast3r_slam.mast3r_utils import frame_to_intir


@serde
class NSFrame:
    file_path: str
    transform_matrix: Float32[
        np.ndarray, "4 4"
    ]  # 4x4 camera transformation matrix in OpenGL format
    colmap_im_id: int


@serde
class NerfstudioData:
    w: int
    h: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    camera_model: Literal["OPENCV"]
    frames: list[NSFrame]
    applied_transform: Float32[np.ndarray, "3 4"]
    ply_file_path: Literal["sparse_pc.ply"]


def save_kf_to_nerfstudio(
    ns_save_path: Path,
    keyframes: SharedKeyframes,
    confidence_thresh: int = 100,
):
    """
    Save keyframes to NerfStudio format
    :param ns_save_path: Path to save the NerfStudio data
    :param keyframes: SharedKeyframes object
    :param confidence_thresh: Confidence threshold to apply to the keyframes

    :return: Open3D point cloud object
    """
    ns_save_path.mkdir(parents=True, exist_ok=True)
    # Create images subdirectory
    images_dir = ns_save_path / "images"
    images_dir.mkdir(exist_ok=True)

    ns_frames_list = []
    pcd_positions = []
    pcd_colors = []
    for i in tqdm.tqdm(range(len(keyframes)), desc="Processing keyframes"):
        keyframe = keyframes[i]
        rgb_img: Float32[torch.Tensor, "H W 3"] = keyframe.uimg
        rgb_img: UInt8[np.ndarray, "H W 3"] = (rgb_img * 255).numpy().astype(np.uint8)
        bgr_img: UInt8[np.ndarray, "H W 3"] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr_img.shape

        # Save the image with zero-padded numbering
        image_filename = f"frame_{i + 1:05d}.png"  # Format: frame_00001.png
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), bgr_img)
        relative_image_path = f"images/{image_filename}"

        se3_pose: lietorch.SE3 = as_SE3(keyframe.T_WC.cpu())
        matb4x4: Float32[np.ndarray, "1 4 4"] = (
            se3_pose.matrix().numpy().astype(dtype=np.float32)
        )
        # in RDF (OpenCV) Format
        mat4x4_cv: Float32[np.ndarray, "4 4"] = matb4x4[0]

        # in RUB (OpenGL) Format
        mat4x4_gl = conventions.convert_pose(
            mat4x4_cv,
            src_convention=conventions.CC.CV,
            dst_convention=conventions.CC.GL,
        )

        mask = keyframe.C.cpu().numpy() > confidence_thresh

        # Convert the mask from shape (h*w, 1) to shape (h*w,)
        mask = mask.squeeze()  # Remove the trailing dimension to get a 1D boolean array

        # Now apply the mask to both positions and colors
        positions: Float32[np.ndarray, "num_points 3"] = keyframe.X_canon.cpu().numpy()
        colors: UInt8[np.ndarray, "num_points 3"] = rgb_img.reshape(-1, 3)

        masked_positions = positions[mask]  # Now selects entire rows where mask is True
        masked_colors = colors[mask]

        # Convert to homogeneous coordinates (add 1 as 4th coordinate)
        homogeneous_positions = np.ones(
            (masked_positions.shape[0], 4), dtype=np.float32
        )
        homogeneous_positions[:, :3] = masked_positions

        # Apply transformation (points are column vectors: p_world = T_world_cam * p_cam)
        world_positions = (mat4x4_cv @ homogeneous_positions.T).T[:, :3]

        pcd_positions.append(world_positions)
        pcd_colors.append(masked_colors)

        ns_frames_list.append(
            NSFrame(
                file_path=relative_image_path,
                transform_matrix=mat4x4_gl,
                colmap_im_id=i,
            )
        )

    # stack all the point clouds
    pcd_positions: Float32[np.ndarray, "num_points 3"] = np.vstack(pcd_positions)
    pcd_colors: UInt8[np.ndarray, "num_points 3"] = np.vstack(pcd_colors)
    # normalize point colors to be between 0 and 1 and a float32
    pcd_colors: Float32[np.ndarray, "num_points 3"] = (
        pcd_colors.astype(np.float32) / 255.0
    )
    # Create an empty point cloud
    pcd = o3d.geometry.PointCloud()

    # Ensure your positions and colors are of the appropriate type (typically float64 for points)
    pcd.points = o3d.utility.Vector3dVector(pcd_positions.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors.astype(np.float64))

    # downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    # save point cloud to file
    o3d.io.write_point_cloud(str(ns_save_path / "sparse_pc.ply"), pcd)

    # use the last keyframe to get the focal and principal point
    focal, principal_point = frame_to_intir(keyframe)
    # save to nerfstudio format, assumes no distortion
    ns_data = NerfstudioData(
        w=w,
        h=h,
        fl_x=focal[0],
        fl_y=focal[1],
        cx=principal_point[0],
        cy=principal_point[1],
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
        camera_model="OPENCV",
        frames=ns_frames_list,
        applied_transform=np.eye(3, 4, dtype=np.float32),
        ply_file_path="sparse_pc.ply",
    )
    json_str: str = to_json(ns_data)
    with open(ns_save_path / "transforms.json", "w") as f:
        f.write(json_str)

    return pcd
