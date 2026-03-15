"""Multi-view geometry prediction node.

Provides a self-contained API for running a multi-view geometry predictor
(currently VGGT, extensible to Pi3 or others) to produce oriented camera
poses, depths, confidences, and intrinsics from a list of RGB images.
This is the first node in the multi-view calibration pipeline.

The network genuinely takes a list — it processes all views jointly to
produce consistent multi-view geometry. This is the only node where list
input matches the network's actual contract.

Also provides a CLI entry point (``main``) for standalone usage with tyro.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from beartype.vale import Is
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.camera_parameters import Extrinsics
from simplecv.ops.conventions import CameraConventions, convert_pose
from simplecv.rerun_log_utils import RerunTyroConfig

from monopriors.models.multiview.vggt_model import MultiviewPred, VGGTPredictor, robust_filter_confidences

KeepTopPercent = Annotated[int | float, Is[lambda percent: 1 <= percent <= 100]]


@dataclass
class MultiviewGeometryConfig:
    """Configuration for VGGT multi-view geometry prediction."""

    preprocessing_mode: Literal["crop", "pad"] = "pad"
    """Image preprocessing strategy: 'crop' preserves aspect ratio, 'pad' adds white padding."""
    keep_top_percent: KeepTopPercent = 30.0
    """Fraction of high-confidence pixels retained after VGGT filtering.
    Value in [1, 100]; e.g. 30 → keep top 70%."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Execution backend for VGGT."""
    verbose: bool = False
    """Emit per-camera detail logging when True."""


@dataclass
class MultiviewGeometryResult:
    """Output of VGGT multi-view geometry prediction."""

    mv_pred_list: list[MultiviewPred]
    """Oriented multi-view predictions (poses, depths, confidences, intrinsics)."""
    depth_confidences: list[UInt8[ndarray, "H W"]]
    """Binary confidence masks after robust filtering (0 or 255)."""


def orient_mv_pred_list(
    mv_pred_list: list[MultiviewPred],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "none",
) -> list[MultiviewPred]:
    """Orient and optionally center multi-view predictions.

    Converts poses to OpenGL convention, applies orientation/centering,
    then converts back to OpenCV convention.

    Args:
        mv_pred_list: Raw multi-view predictions from VGGT.
        method: Orientation method for the up direction.
        center_method: How to center the poses.

    Returns:
        Reoriented multi-view predictions with updated extrinsics.
    """
    extri_list: list[Extrinsics] = [mv_pred.pinhole_param.extrinsics for mv_pred in mv_pred_list]

    world_T_cam_batch: Float[ndarray, "*num_poses 4 4"] = np.stack([extri.world_T_cam for extri in extri_list])
    assert len(set(mv_pred.pinhole_param.intrinsics.camera_conventions for mv_pred in mv_pred_list)) == 1
    if mv_pred_list[0].pinhole_param.intrinsics.camera_conventions == "RDF":
        world_T_cam_gl: Float[ndarray, "*num_poses 4 4"] = convert_pose(
            world_T_cam_batch, CameraConventions.CV, CameraConventions.GL
        )
    else:
        world_T_cam_gl = world_T_cam_batch

    oriented_world_T_cam_3x4_np, _ = auto_orient_and_center_poses(
        world_T_cam_gl.astype(np.float64), method=method, center_method=center_method
    )

    N: int = oriented_world_T_cam_3x4_np.shape[0]
    bottom_row: Float[ndarray, "N 1 4"] = np.broadcast_to(np.array([[0.0, 0.0, 0.0, 1.0]]), (N, 1, 4))
    oriented_world_T_cam_4x4_np: Float32[ndarray, "N 4 4"] = np.concatenate(
        [oriented_world_T_cam_3x4_np, bottom_row], axis=1
    ).astype(np.float32)
    oriented_world_T_cam_cv: Float[ndarray, "N 4 4"] = convert_pose(
        oriented_world_T_cam_4x4_np, CameraConventions.GL, CameraConventions.CV
    )

    oriented_mv_pred_list: list[MultiviewPred] = []
    for idx, mv_pred in enumerate(mv_pred_list):
        oriented_extri: Extrinsics = Extrinsics(
            world_R_cam=oriented_world_T_cam_cv[idx, :3, :3],
            world_t_cam=oriented_world_T_cam_cv[idx, :3, 3],
        )
        oriented_mv_pred_list.append(
            replace(mv_pred, pinhole_param=replace(mv_pred.pinhole_param, extrinsics=oriented_extri))
        )

    return oriented_mv_pred_list


def run_multiview_geometry(
    *,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    vggt_predictor: VGGTPredictor,
    config: MultiviewGeometryConfig,
) -> MultiviewGeometryResult:
    """Run VGGT multi-view geometry prediction.

    Runs the VGGT network on all views jointly, orients the resulting poses,
    and computes robust confidence masks.

    Args:
        rgb_list: Ordered RGB frames across cameras.
        vggt_predictor: Pre-initialised VGGT predictor (model already loaded).
        config: Geometry prediction configuration.

    Returns:
        MultiviewGeometryResult with oriented predictions and confidence masks.
    """
    mv_pred_list: list[MultiviewPred] = vggt_predictor(rgb_list)
    mv_pred_list = orient_mv_pred_list(mv_pred_list)

    depth_confidences: list[UInt8[ndarray, "H W"]] = [
        robust_filter_confidences(mv_pred.confidence_mask, keep_top_percent=config.keep_top_percent)
        for mv_pred in mv_pred_list
    ]

    return MultiviewGeometryResult(
        mv_pred_list=mv_pred_list,
        depth_confidences=depth_confidences,
    )


@dataclass
class MultiviewGeometryCLIConfig:
    """CLI configuration for VGGT geometry prediction."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_dir: Path = Path("data/examples/multiview/car_landscape_12")
    """Directory containing input images."""
    vggt_config: MultiviewGeometryConfig = field(default_factory=lambda: MultiviewGeometryConfig(verbose=True))
    """VGGT geometry prediction configuration."""


def main(config: MultiviewGeometryCLIConfig) -> None:
    """CLI entry point for VGGT geometry prediction with Rerun visualization."""
    import open3d as o3d
    import rerun as rr
    import rerun.blueprint as rrb
    from einops import rearrange
    from simplecv.ops.pc_utils import estimate_voxel_size
    from simplecv.rerun_log_utils import log_pinhole

    from monopriors.apis.multiview_calibration import (
        PARENT_LOG_PATH,
        SUPPORTED_IMAGE_EXTENSIONS,
        load_rgb_images,
        mv_pred_to_pointcloud,
    )

    # Load images
    image_paths: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_paths.extend(config.image_dir.glob(f"*{ext}"))
    image_paths = sorted(image_paths)
    assert len(image_paths) > 0, f"No images found in {config.image_dir}"
    rgb_list: list[UInt8[ndarray, "H W 3"]] = load_rgb_images(image_paths)

    # Init predictor
    vggt_predictor: VGGTPredictor = VGGTPredictor(
        device=config.vggt_config.device,
        preprocessing_mode=config.vggt_config.preprocessing_mode,
    )

    # Run geometry
    result: MultiviewGeometryResult = run_multiview_geometry(
        rgb_list=rgb_list,
        vggt_predictor=vggt_predictor,
        config=config.vggt_config,
    )

    # Setup Rerun blueprint
    from monopriors.gradio_ui.multiview_geometry_ui import create_multiview_blueprint

    final_view: rrb.ContainerLike = create_multiview_blueprint(parent_log_path=PARENT_LOG_PATH, num_images=len(rgb_list))
    blueprint: rrb.Blueprint = rrb.Blueprint(final_view, collapse_panels=True)
    rr.send_blueprint(blueprint=blueprint)
    rr.log(f"{PARENT_LOG_PATH}", rr.ViewCoordinates.RFU, static=True)

    # Log per-camera results
    for mv_pred, depth_conf in zip(result.mv_pred_list, result.depth_confidences, strict=True):
        cam_log_path: Path = PARENT_LOG_PATH / mv_pred.cam_name
        pinhole_log_path: Path = cam_log_path / "pinhole"
        log_pinhole(mv_pred.pinhole_param, cam_log_path=cam_log_path, image_plane_distance=0.05, static=True)
        rr.log(
            f"{pinhole_log_path}/image",
            rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB).compress(),
            static=True,
        )
        rr.log(
            f"{pinhole_log_path}/confidence", rr.Image(depth_conf, color_model=rr.ColorModel.L).compress(), static=True
        )
        filtered_depth: Float32[ndarray, "H W"] = np.where(depth_conf > 0, mv_pred.depth_map, 0)
        rr.log(f"{pinhole_log_path}/filtered_depth", rr.DepthImage(filtered_depth, meter=1), static=True)
        rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(mv_pred.depth_map, meter=1), static=True)

    # Log point cloud
    pointcloud: Float32[ndarray, "num_points 3"] = mv_pred_to_pointcloud(result.mv_pred_list)
    rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
        [rearrange(mv_pred.rgb_image, "h w c -> (h w) c") for mv_pred in result.mv_pred_list]
    )
    voxel_size: float = estimate_voxel_size(pointcloud.astype(np.float32), target_points=150_000)
    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(rgb_stack / 255.0)
    pcd_ds: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxel_size)
    rr.log(
        f"{PARENT_LOG_PATH}/point_cloud",
        rr.Points3D(np.asarray(pcd_ds.points, dtype=np.float32), colors=np.asarray(pcd_ds.colors, dtype=np.float32)),
        static=True,
    )
