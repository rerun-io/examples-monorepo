from pathlib import Path
from typing import Literal

import cv2
import lietorch
import numpy as np
import open3d as o3d
import tqdm
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from serde import serde
from serde.json import to_json
from simplecv.ops import conventions
from torch import Tensor

from mast3r_slam.frame import Frame, SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.mast3r_utils import frame_to_intir


@serde
class NSFrame:
    """A single frame entry in NerfStudio's ``transforms.json`` format."""

    file_path: str
    """Relative path to the frame's image file."""
    transform_matrix: Float32[
        np.ndarray, "4 4"
    ]
    """4x4 camera transformation matrix in OpenGL (RUB) convention."""
    colmap_im_id: int
    """COLMAP-style image ID (sequential index)."""


@serde
class NerfstudioData:
    """Top-level NerfStudio dataset descriptor written to ``transforms.json``."""

    w: int
    """Image width in pixels."""
    h: int
    """Image height in pixels."""
    fl_x: float
    """Focal length along the x-axis."""
    fl_y: float
    """Focal length along the y-axis."""
    cx: float
    """Principal point x-coordinate."""
    cy: float
    """Principal point y-coordinate."""
    k1: float
    """First radial distortion coefficient."""
    k2: float
    """Second radial distortion coefficient."""
    p1: float
    """First tangential distortion coefficient."""
    p2: float
    """Second tangential distortion coefficient."""
    camera_model: Literal["OPENCV"]
    """Camera model identifier (always OPENCV for this exporter)."""
    frames: list[NSFrame]
    """List of per-frame pose entries."""
    applied_transform: Float32[ndarray, "3 4"]
    """3x4 affine transform applied to all poses (typically identity)."""
    ply_file_path: Literal["sparse_pc.ply"]
    """Relative path to the sparse point cloud PLY file."""


def save_kf_to_nerfstudio(
    ns_save_path: Path,
    keyframes: SharedKeyframes,
    confidence_thresh: int = 100,
) -> o3d.geometry.PointCloud:
    """Save keyframes to NerfStudio format and return the sparse point cloud.

    Exports images, camera poses (in OpenGL convention), and a merged sparse
    point cloud to the specified directory in NerfStudio's ``transforms.json``
    format.

    Args:
        ns_save_path: Directory to write NerfStudio data into.
        keyframes: Shared keyframe buffer to read from.
        confidence_thresh: Minimum confidence for a point to be included in
            the sparse point cloud.

    Returns:
        An Open3D ``PointCloud`` of the merged, downsampled sparse reconstruction.
    """
    ns_save_path.mkdir(parents=True, exist_ok=True)
    # Create images subdirectory
    images_dir: Path = ns_save_path / "images"
    images_dir.mkdir(exist_ok=True)

    ns_frames_list: list[NSFrame] = []
    pcd_positions: list[Float32[ndarray, "n_valid 3"]] = []
    pcd_colors: list[UInt8[ndarray, "n_valid 3"]] = []
    h: int = 0
    w: int = 0
    for i in tqdm.tqdm(range(len(keyframes)), desc="Processing keyframes"):
        keyframe: Frame = keyframes[i]
        rgb_img_float: Float32[Tensor, "H W 3"] = keyframe.uimg
        rgb_img: UInt8[ndarray, "H W 3"] = (rgb_img_float * 255).numpy().astype(np.uint8)
        bgr_img: UInt8[ndarray, "H W 3"] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        h, w, _ = bgr_img.shape

        # Save the image with zero-padded numbering
        image_filename: str = f"frame_{i + 1:05d}.png"  # Format: frame_00001.png
        image_path: Path = images_dir / image_filename
        cv2.imwrite(str(image_path), bgr_img)
        relative_image_path: str = f"images/{image_filename}"

        se3_pose: lietorch.SE3 = as_SE3(keyframe.world_T_cam.cpu())
        matb4x4: Float32[ndarray, "1 4 4"] = (
            se3_pose.matrix().numpy().astype(dtype=np.float32)
        )
        # in RDF (OpenCV) Format
        mat4x4_cv: Float32[ndarray, "4 4"] = matb4x4[0]

        # in RUB (OpenGL) Format
        mat4x4_gl: Float32[ndarray, "4 4"] = conventions.convert_pose(
            mat4x4_cv,
            src_convention=conventions.CC.CV,
            dst_convention=conventions.CC.GL,
        )

        assert keyframe.C is not None
        mask_raw: Bool[ndarray, "hw 1"] = keyframe.C.cpu().numpy() > confidence_thresh

        # Convert the mask from shape (h*w, 1) to shape (h*w,)
        mask: Bool[ndarray, "hw"] = mask_raw.squeeze()  # Remove the trailing dimension to get a 1D boolean array

        # Now apply the mask to both positions and colors
        assert keyframe.X_canon is not None
        positions: Float32[ndarray, "num_points 3"] = keyframe.X_canon.cpu().numpy()
        colors: UInt8[ndarray, "num_points 3"] = rgb_img.reshape(-1, 3)

        masked_positions: Float32[ndarray, "n_valid 3"] = positions[mask]  # Now selects entire rows where mask is True
        masked_colors: UInt8[ndarray, "n_valid 3"] = colors[mask]

        # Convert to homogeneous coordinates (add 1 as 4th coordinate)
        homogeneous_positions: Float32[ndarray, "n_valid 4"] = np.ones(
            (masked_positions.shape[0], 4), dtype=np.float32
        )
        homogeneous_positions[:, :3] = masked_positions

        # Apply transformation (points are column vectors: p_world = T_world_cam * p_cam)
        world_positions: Float32[ndarray, "n_valid 3"] = (mat4x4_cv @ homogeneous_positions.T).T[:, :3]

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
    pcd_positions_all: Float32[ndarray, "num_points 3"] = np.vstack(pcd_positions)
    pcd_colors_all: UInt8[ndarray, "num_points 3"] = np.vstack(pcd_colors)
    # normalize point colors to be between 0 and 1 and a float32
    pcd_colors_float: Float32[ndarray, "num_points 3"] = (
        pcd_colors_all.astype(np.float32) / 255.0
    )
    # Create an empty point cloud
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()

    # Ensure your positions and colors are of the appropriate type (typically float64 for points)
    pcd.points = o3d.utility.Vector3dVector(pcd_positions_all.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors_float.astype(np.float64))

    # downsample the point cloud
    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    # save point cloud to file
    o3d.io.write_point_cloud(str(ns_save_path / "sparse_pc.ply"), pcd)

    # use the last keyframe to get the focal and principal point
    focal: tuple[float, float]
    principal_point: tuple[float, float]
    focal, principal_point = frame_to_intir(keyframe)
    # save to nerfstudio format, assumes no distortion
    ns_data: NerfstudioData = NerfstudioData(
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
