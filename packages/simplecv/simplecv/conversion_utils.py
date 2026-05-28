from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import tqdm
from jaxtyping import Float32, UInt8
from serde import field as serde_field
from serde import serde
from serde.json import to_json

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters
from simplecv.image_types import BGRList
from simplecv.ops import conventions


@serde
class NSFrame:
    file_path: str
    world_T_cam_gl: Float32[np.ndarray, "4 4"] = serde_field(rename="transform_matrix")
    # 4x4 camera transformation matrix in OpenGL format
    colmap_im_id: int
    # # Add skip_if_default=True to optional fields
    # w: int | None = serde_field(default=None, skip_if_default=True)
    # h: int | None = serde_field(default=None, skip_if_default=True)
    # fl_x: float | None = serde_field(default=None, skip_if_default=True)
    # fl_y: float | None = serde_field(default=None, skip_if_default=True)
    # cx: float | None = serde_field(default=None, skip_if_default=True)
    # cy: float | None = serde_field(default=None, skip_if_default=True)
    # k1: float | None = serde_field(default=None, skip_if_default=True)
    # k2: float | None = serde_field(default=None, skip_if_default=True)
    # p1: float | None = serde_field(default=None, skip_if_default=True)
    # p2: float | None = serde_field(default=None, skip_if_default=True)


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


def save_to_nerfstudio(
    ns_save_path: Path,
    bgr_list: BGRList,
    pinhole_param_list: list[PinholeParameters],
    pointcloud: o3d.geometry.PointCloud,
    masks_list: list[UInt8[np.ndarray, "h w"]] | None = None,
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

    # create a directory for masks if they are provided
    if masks_list is not None:
        masks_dir = ns_save_path / "masks"
        masks_dir.mkdir(exist_ok=True)

    ns_frames_list = []

    assert len(bgr_list) == len(pinhole_param_list)
    for i in tqdm.tqdm(range(len(bgr_list)), desc="Saving frames", unit="frame"):
        bgr_img: np.ndarray = bgr_list[i]
        pinhole_param: PinholeParameters = pinhole_param_list[i]
        h, w, _ = bgr_img.shape
        extrinsics: Extrinsics = pinhole_param.extrinsics

        # Save the image with zero-padded numbering
        image_filename: str = f"frame_{i + 1:05d}.png"  # Format: frame_00001.png
        image_path: Path = images_dir / image_filename
        cv2.imwrite(str(image_path), bgr_img)
        relative_image_path = f"images/{image_filename}"

        if masks_list is not None:
            mask: np.ndarray = masks_list[i] * 255  # Convert boolean mask to uint8
            # Save the mask with zero-padded numbering, also make sure to use the same filename as the image
            # to ensure they are paired correctly (in particular with brush)
            mask_filename: str = f"frame_{i + 1:05d}.png"
            mask_path: Path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)

        # in RDF (OpenCV) Format
        world_T_cam_cv: Float32[np.ndarray, "4 4"] = extrinsics.world_T_cam.astype(np.float32)

        # in RUB (OpenGL) Format
        world_T_cam_gl: Float32[np.ndarray, "4 4"] = conventions.convert_pose(
            world_T_cam_cv,
            src_convention=conventions.CC.CV,
            dst_convention=conventions.CC.GL,
        )

        ns_frames_list.append(
            NSFrame(
                file_path=relative_image_path,
                world_T_cam_gl=world_T_cam_gl,
                colmap_im_id=i,
            )
        )

    # save point cloud to file
    o3d.io.write_point_cloud(str(ns_save_path / "sparse_pc.ply"), pointcloud)

    # make sure all cameras have the same intrinsics
    assert len(pinhole_param_list) > 0

    example_intri: Intrinsics = pinhole_param_list[0].intrinsics
    example_dist: BrownConradyDistortion | None = pinhole_param_list[0].distortion

    # save to nerfstudio format
    ns_data = NerfstudioData(
        w=w,
        h=h,
        fl_x=example_intri.fl_x,
        fl_y=example_intri.fl_y,
        cx=example_intri.cx,
        cy=example_intri.cy,
        k1=0.0 if example_dist is None else example_dist.k1,
        k2=0.0 if example_dist is None else example_dist.k2,
        p1=0.0 if example_dist is None else example_dist.p1,
        p2=0.0 if example_dist is None else example_dist.p2,
        camera_model="OPENCV",
        frames=ns_frames_list,
        applied_transform=np.eye(3, 4, dtype=np.float32),
        ply_file_path="sparse_pc.ply",
    )
    json_str: str = to_json(ns_data)
    with open(ns_save_path / "transforms.json", "w") as f:
        f.write(json_str)
