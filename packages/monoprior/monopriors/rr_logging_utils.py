from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, UInt8

from monopriors.depth_utils import clip_disparity, depth_edges_mask, depth_to_disparity, depth_to_points
from monopriors.metric_depth_models import METRIC_PREDICTORS, MetricDepthPrediction
from monopriors.relative_depth_models import RELATIVE_PREDICTORS, RelativeDepthPrediction


def log_relative_pred(
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    remove_flying_pixels: bool = True,
    log_disparity: bool = False,
    jpeg_quality: int = 90,
    depth_edge_threshold: int | float = 1.1,
) -> None:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    # assume camera is at the origin
    cam_T_world_44: Float64[np.ndarray, "4 4"] = np.eye(4)

    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=relative_pred.K_33,
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality))

    depth_hw: Float32[np.ndarray, "h w"] = relative_pred.depth
    # filter out any inf/nan values
    depth_hw = np.nan_to_num(depth_hw, nan=0.0, posinf=0.0, neginf=0.0)

    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw * ~edges_mask

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw))

    confidence_hw: Bool[np.ndarray, "h w"] = relative_pred.confidence > 0.5
    rr.log(f"{pinhole_path}/confidence", rr.Image((confidence_hw * 255).astype(np.uint8)))

    if log_disparity:
        # removes outliers from disparity (sometimes we can get weirdly large values)
        clipped_disparity: UInt8[np.ndarray, "h w"] = clip_disparity(relative_pred.disparity)
        # log to cam_log_path to avoid backprojecting disparity
        rr.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, relative_pred.K_33)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=rgb_hw3.reshape(-1, 3),
        ),
    )


def create_compare_depth_blueprint(
    model_names: list[RELATIVE_PREDICTORS | METRIC_PREDICTORS],
) -> rrb.Blueprint:
    # model_names: list[str] = [model.__class__.__name__ for model in models]
    contents = [
        rrb.Spatial3DView(origin=f"{model_names[0]}"),
        rrb.Vertical(
            rrb.Spatial2DView(
                origin=f"{model_names[0]}/camera/pinhole/image",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[0]}/camera/pinhole/depth",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[0]}/camera/disparity",
            ),
        ),
        rrb.Spatial3DView(origin=f"{model_names[1]}"),
        rrb.Vertical(
            rrb.Spatial2DView(
                origin=f"{model_names[1]}/camera/pinhole/image",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[1]}/camera/pinhole/depth",
            ),
            rrb.Spatial2DView(
                origin=f"{model_names[1]}/camera/disparity",
            ),
        ),
    ]
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=contents,
            column_shares=(3, 1, 3, 1),
        ),
        collapse_panels=True,
    )
    return blueprint


def log_metric_pred(
    parent_log_path: Path,
    metric_pred: MetricDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    remove_flying_pixels: bool = True,
    jpeg_quality: int = 90,
    depth_edge_threshold: float = 1.1,
) -> None:
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    # assume camera is at the origin
    cam_T_world_44: Float64[np.ndarray, "4 4"] = np.eye(4)

    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=cam_T_world_44[:3, 3],
            mat3x3=cam_T_world_44[:3, :3],
            from_parent=True,
        ),
    )
    rr.log(
        f"{pinhole_path}",
        rr.Pinhole(
            image_from_camera=metric_pred.K_33,
            width=rgb_hw3.shape[1],
            height=rgb_hw3.shape[0],
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality))

    depth_hw: Float32[np.ndarray, "h w"] = metric_pred.depth_meters
    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        depth_hw: Float32[np.ndarray, "h w"] = depth_hw * ~edges_mask

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw, meter=1.0))

    # removes outliers from disparity (sometimes we can get weirdly large values)
    clipped_disparity: Float32[np.ndarray, "h w"] = depth_to_disparity(
        depth_hw, focal_length=int(metric_pred.K_33[0, 0]), baseline=1000.0
    )

    # log to cam_log_path to avoid backprojecting disparity
    rr.log(f"{cam_log_path}/disparity", rr.DepthImage(clipped_disparity))

    depth_1hw: Float32[np.ndarray, "1 h w"] = rearrange(depth_hw, "h w -> 1 h w")
    pts_3d: Float32[np.ndarray, "h w 3"] = depth_to_points(depth_1hw, metric_pred.K_33)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(
            positions=pts_3d.reshape(-1, 3),
            colors=rgb_hw3.reshape(-1, 3),
        ),
    )
