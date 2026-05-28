from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8, UInt16
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.conversion_utils import save_to_nerfstudio
from simplecv.data.exoego.hocap import ExoCameraIDs, HOCapSequence, SubjectIDs
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)


def create_blueprint(exo_img_log_paths: list[Path], num_videos_to_log: Literal[4, 8] = 8) -> rrb.Blueprint:
    active_tab: int = 0  # 0 for video, 1 for images
    main_view = rrb.Vertical(
        contents=[
            rrb.Spatial3DView(
                origin="/",
            ),
            # take the first 4 video files
            rrb.Horizontal(
                contents=[
                    rrb.Tabs(
                        rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                        active_tab=active_tab,
                    )
                    for video_log_path in exo_img_log_paths[:4]
                ]
            ),
        ],
        row_shares=[3, 1],
    )
    additional_views = rrb.Vertical(
        contents=[
            rrb.Tabs(
                rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                active_tab=active_tab,
            )
            for video_log_path in exo_img_log_paths[4:]
        ]
    )
    # do the last 4 videos
    contents = [main_view]
    if num_videos_to_log == 8:
        contents.append(additional_views)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


@dataclass
class ConvertConfig:
    rr_config: RerunTyroConfig
    dataset: Literal["hocap", "assembly101"] = "hocap"
    root_directory: Path = Path("data/hocap/sample")
    subject_id: SubjectIDs | None = "8"
    sequence_name: str = "20231024_180733"
    num_videos_to_log: Literal[4, 8] = 8
    load_labels: bool = False


def convert_data_to_ns(config: ConvertConfig) -> None:
    print(f"Converting {config.dataset} data to Nerfstudio format")
    sequence: HOCapSequence = HOCapSequence(
        data_path=config.root_directory,
        sequence_name=config.sequence_name,
        subject_id=config.subject_id,
        load_labels=config.load_labels,
    )

    parent_log_path: Path = Path("world")

    exo_video_readers: MultiVideoReader = sequence.exo_video_readers
    exo_cam_list: list[PinholeParameters] = sequence.exo_cam_list
    depth_paths: dict[ExoCameraIDs, Path] = sequence.depth_paths[0]

    bgr_list: list[UInt8[ndarray, "H W 3"]] = exo_video_readers[0]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]

    cam_log_path_list: list[Path] = []
    fuser = Open3DFuser(fusion_resolution=0.01, max_fusion_depth=1.25)
    # log stationary exo cameras and video assets
    for exo_cam in exo_cam_list:
        cam_log_path: Path = parent_log_path / exo_cam.name
        cam_log_path_list.append(cam_log_path)
        image_plane_distance: float = 0.1 if config.dataset == "hocap" else 100.0
        log_pinhole(
            camera=exo_cam,
            cam_log_path=cam_log_path,
            image_plane_distance=image_plane_distance,
            static=True,
        )

    blueprint: rrb.Blueprint = create_blueprint(
        exo_img_log_paths=[cam_log_path / "pinhole" / "image" for cam_log_path in cam_log_path_list],
        num_videos_to_log=config.num_videos_to_log,
    )
    rr.send_blueprint(blueprint)

    for rgb, cam_log_path, exo_cam in zip(rgb_list, cam_log_path_list, exo_cam_list, strict=True):
        pinhole_log_path: Path = cam_log_path / "pinhole"
        rr.log(f"{pinhole_log_path}/video", rr.Image(rgb, color_model=rr.ColorModel.RGB), static=True)
        # rec.log(f"{pinhole_log_path}/depth", rr.DepthImage(depth_image, meter=1000))
        depth_path: Path = depth_paths[cam_log_path.name]
        depth_image: UInt16[np.ndarray, "480 640"] = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        fuser.fuse_frames(
            depth_image,
            exo_cam.intrinsics.k_matrix,
            exo_cam.extrinsics.cam_T_world,
            rgb,
        )

    mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
    mesh.compute_vertex_normals()

    pcd: o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(
        number_of_points=10_000,
    )

    rr.log(
        f"{parent_log_path}/mesh",
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.triangles,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=mesh.vertex_colors,
        ),
        static=True,
    )

    rr.log(
        f"{parent_log_path}/pc",
        rr.Points3D(
            positions=pcd.points,
            colors=pcd.colors,
        ),
        static=True,
    )

    save_to_nerfstudio(
        ns_save_path=Path("test_output"), bgr_list=bgr_list, pinhole_param_list=exo_cam_list, pointcloud=pcd
    )
