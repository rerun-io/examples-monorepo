from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8, UInt16
from tqdm import tqdm

from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_fused_mesh, log_pinhole


@dataclass
class PolyViewConfig:
    polycam_zip_path: Path
    rr_config: RerunTyroConfig
    log_incremental_mesh: bool = True
    rescale_factor: int = 4
    # logs the mesh incrementally, will take up more memory


def log_polycam_data(
    parent_path: Path,
    polycam_data: PolycamData,
    rescale_factor: int = 1,
) -> None:
    cam_path: Path = parent_path / "cam"
    pinhole_path: Path = cam_path / "pinhole"

    rgb: UInt8[np.ndarray, "h w 3"] = polycam_data.rgb_hw3
    depth: UInt16[np.ndarray, "h w"] = polycam_data.depth_hw
    confidence: UInt8[np.ndarray, "h w"] = polycam_data.confidence_hw

    # resize images to be half the size
    target_height: int = rgb.shape[0] // rescale_factor
    target_width: int = rgb.shape[1] // rescale_factor
    rgb_resized: UInt8[np.ndarray, "h w 3"] = cv2.resize(rgb, (target_width, target_height)).astype(
        np.uint8, copy=False
    )
    depth_resized: UInt16[np.ndarray, "h w"] = cv2.resize(depth, (target_width, target_height)).astype(
        np.uint16, copy=False
    )
    confidence_resized: UInt8[np.ndarray, "h w"] = cv2.resize(confidence, (target_width, target_height)).astype(
        np.uint8, copy=False
    )

    # rescale intrinsics to match the image size
    rescaled_intrinsics: Intrinsics = rescale_intri(
        camera_intrinsics=polycam_data.pinhole_params.intrinsics,
        target_height=target_height,
        target_width=target_width,
    )

    polycam_data.pinhole_params.intrinsics = rescaled_intrinsics

    log_pinhole(camera=polycam_data.pinhole_params, cam_log_path=cam_path)
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_resized).compress(jpeg_quality=75))
    rr.log(f"{pinhole_path}/confidence", rr.SegmentationImage(confidence_resized))
    rr.log(f"{pinhole_path}/gt_depth", rr.DepthImage(depth_resized, meter=1000))


def create_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{parent_log_path}/cam/pinhole/image"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/cam/pinhole/gt_depth"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/cam/pinhole/confidence"),
            ),
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    return blueprint


def view_polycam_data(config: PolyViewConfig) -> None:
    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)

    depth_fuser = Open3DFuser(fusion_resolution=0.04, max_fusion_depth=3.0)

    parent_path: Path = Path("world")
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    blueprint: rrb.Blueprint = create_blueprint(parent_log_path=parent_path)
    rr.send_blueprint(blueprint)

    pbar = tqdm(polycam_dataset, total=len(polycam_dataset))
    polycam_data: PolycamData
    for idx, (polycam_data) in enumerate(pbar):
        rr.set_time(timeline="timestep", sequence=idx)

        # filter depthmaps based on confidence, only keep with max confidence
        polycam_data.depth_hw[polycam_data.confidence_hw != DepthConfidenceLevel.HIGH] = 0

        k_matrix = polycam_data.pinhole_params.intrinsics.k_matrix
        if k_matrix is None:
            raise ValueError("Missing camera intrinsics for polycam sample.")
        depth_fuser.fuse_frames(
            polycam_data.depth_hw,
            k_matrix,
            polycam_data.pinhole_params.extrinsics.cam_T_world,
            polycam_data.rgb_hw3,
        )

        log_polycam_data(
            parent_path=parent_path,
            polycam_data=polycam_data,
            rescale_factor=config.rescale_factor,
        )

        # log mesh on every frame, this will take up more memory and computation, but looks way cooler
        if config.log_incremental_mesh:
            gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
            gt_mesh.compute_vertex_normals()

            log_fused_mesh(
                None,
                f"{parent_path}/gt_mesh",
                gt_mesh,
                static=False,
                face_rendering=rr.components.MeshFaceRendering.DoubleSided,
            )

    # export mesh
    gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
    gt_mesh.compute_vertex_normals()

    log_fused_mesh(
        None,
        f"{parent_path}/gt_mesh",
        gt_mesh,
        static=False,
        face_rendering=rr.components.MeshFaceRendering.DoubleSided,
    )
