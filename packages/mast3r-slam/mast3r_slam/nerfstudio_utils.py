"""Keyframes to a NerfStudio dataset, written by simplecv's ``save_to_nerfstudio``."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.conversion_utils import save_to_nerfstudio

from mast3r_slam.frame import Frame, SharedKeyframes
from mast3r_slam.mast3r_utils import frame_to_extrinsics, frame_to_intri


def save_kf_to_nerfstudio(
    ns_save_path: Path,
    keyframes: SharedKeyframes,
    confidence_thresh: int = 100,
) -> o3d.geometry.PointCloud:
    """Save keyframes to NerfStudio format and return the sparse point cloud.

    Exports images, camera poses (in OpenGL convention), and a merged sparse
    point cloud to the specified directory in NerfStudio's ``transforms.json``
    format. Every camera gets the intrinsics estimated from the last keyframe.

    Args:
        ns_save_path: Directory to write NerfStudio data into.
        keyframes: Shared keyframe buffer to read from.
        confidence_thresh: Minimum confidence for a point to be included in
            the sparse point cloud.

    Returns:
        An Open3D ``PointCloud`` of the merged, downsampled sparse reconstruction.
    """
    kf_list: list[Frame] = [keyframes[i] for i in range(len(keyframes))]
    intrinsics: Intrinsics = frame_to_intri(kf_list[-1], camera_conventions="RDF")

    bgr_list: list[UInt8[ndarray, "H W 3"]] = []
    pinholes: list[PinholeParameters] = []
    pcd_positions: list[Float32[ndarray, "n_valid 3"]] = []
    pcd_colors: list[UInt8[ndarray, "n_valid 3"]] = []
    for i, keyframe in enumerate(kf_list):
        rgb_uint8: UInt8[ndarray, "H W 3"] = (keyframe.rgb * 255).numpy().astype(np.uint8)
        bgr_list.append(cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
        # RDF (OpenCV); save_to_nerfstudio converts to RUB (OpenGL).
        extrinsics: Extrinsics = frame_to_extrinsics(keyframe, camera_conventions="RDF")
        pinholes.append(PinholeParameters(name=f"keyframe-{i}", extrinsics=extrinsics, intrinsics=intrinsics))

        assert keyframe.C is not None
        assert keyframe.X_canon is not None
        mask: Bool[ndarray, "hw"] = keyframe.C.cpu().numpy()[:, 0] > confidence_thresh
        positions: Float32[ndarray, "n_valid 3"] = keyframe.X_canon.cpu().numpy()[mask]
        world_T_cam: Float32[ndarray, "4 4"] = extrinsics.world_T_cam.astype(np.float32)
        pcd_positions.append(positions @ world_T_cam[:3, :3].T + world_T_cam[:3, 3])
        pcd_colors.append(rgb_uint8.reshape(-1, 3)[mask])

    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(pcd_positions).astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(pcd_colors).astype(np.float64) / 255.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    save_to_nerfstudio(ns_save_path, bgr_list, pinholes, pcd)
    return pcd
