from mast3r_slam.api.inference import NerfstudioData
from serde.json import from_json
from pathlib import Path
from dataclasses import dataclass
import tyro
from simplecv.rerun_log_utils import RerunTyroConfig
import cv2
import rerun as rr
import open3d as o3d
import numpy as np
from simplecv.ops import conventions


@dataclass
class ViewNsDataConfig:
    rr_config: RerunTyroConfig
    transform_json_path: Path = Path(
        "/mnt/12tbdrive/data/hloc-glomap-data/lightglue-glomap-data/eval/6g-night-nov-29-2024/livingroom-nov-29-2024-train/merged/500/transforms.json"
    )


def view_ns_data(config: ViewNsDataConfig) -> None:
    with open(config.transform_json_path, "r") as f:
        ns_data: NerfstudioData = from_json(NerfstudioData, f.read())

    ply_path: Path = config.transform_json_path.parent / ns_data.ply_file_path
    pcd = o3d.io.read_point_cloud(str(ply_path))

    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    parent_log_path = Path("world")

    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(positions=points, colors=colors),
        static=True,
    )

    # # Apply the rotation to the root coordinate system
    # rr.log(
    #     f"{parent_log_path}",
    #     rr.Transform3D(
    #         rotation=rr.RotationAxisAngle(axis=(0, 0, 1), radians=-np.pi / 4)
    #     ),
    #     static=True,
    # )
    for idx, frame in enumerate(ns_data.frames):
        rr.set_time_sequence("sequence", idx)
        image_path: Path = config.transform_json_path.parent / frame.file_path
        assert image_path.exists(), f"Image path {image_path} does not exist"
        rgb = cv2.imread(str(image_path))
        mat4x4_gl = frame.transform_matrix

        world_T_cam_44_cv = conventions.convert_pose(
            mat4x4_gl,
            src_convention=conventions.CC.GL,
            dst_convention=conventions.CC.CV,
        )

        world_T_cam_44_gl = conventions.convert_pose(
            world_T_cam_44_cv,
            src_convention=conventions.CC.CV,
            dst_convention=conventions.CC.GL,
        )

        rotation = world_T_cam_44_gl[:3, :3]
        translation = world_T_cam_44_gl[:3, 3]

        cam_log_path: Path = parent_log_path / "camera"
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(translation=translation, mat3x3=rotation),
        )
        rr.log(
            f"{cam_log_path}/pinhole",
            rr.Pinhole(
                width=ns_data.w,
                height=ns_data.h,
                focal_length=(ns_data.fl_x, ns_data.fl_y),
                principal_point=(ns_data.cx, ns_data.cy),
                camera_xyz=rr.ViewCoordinates.RUB,
            ),
        )
        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(image=rgb, color_model=rr.ColorModel.BGR).compress(
                jpeg_quality=10
            ),
        )


if __name__ == "__main__":
    view_ns_data(tyro.cli(ViewNsDataConfig))
