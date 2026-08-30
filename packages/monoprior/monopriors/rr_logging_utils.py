from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Bool, Float32, Float64, UInt8

from monopriors.depth_utils import clip_disparity, depth_edges_mask, depth_to_disparity, depth_to_points
from monopriors.models.metric_depth import METRIC_PREDICTORS, MetricDepthPrediction
from monopriors.models.relative_depth import RELATIVE_PREDICTORS, RelativeDepthPrediction
from monopriors.models.stereo_depth import StereoDepthPrediction
from monopriors.models.surface_normal.base_normal_model import SurfaceNormalPrediction

CONFIDENCE_THRESHOLD: float = 0.5
"""Confidence in [0, 1] above which a pixel is shown as ``confident`` in the semantic mask."""


def log_confidence(
    pinhole_path: Path,
    confidence_hw: Float32[np.ndarray, "h w"],
    threshold: float = CONFIDENCE_THRESHOLD,
    static: bool = False,
) -> None:
    """Log a confidence map twice: the full [0, 1] spectrum as grayscale (``confidence``), and its
    thresholded semantic version as a segmentation image (``confidence_mask``: confident / not confident)."""
    confidence_u8: UInt8[np.ndarray, "h w"] = (np.clip(confidence_hw, 0.0, 1.0) * 255).astype(np.uint8)
    rr.log(f"{pinhole_path}/confidence", rr.Image(confidence_u8, color_model=rr.ColorModel.L), static=static)
    classes: list[rr.AnnotationInfo] = [
        rr.AnnotationInfo(id=0, label="not confident", color=(220, 60, 60)),
        rr.AnnotationInfo(id=1, label="confident", color=(60, 200, 90)),
    ]
    rr.log(f"{pinhole_path}/confidence_mask", rr.AnnotationContext(classes), static=True)
    rr.log(f"{pinhole_path}/confidence_mask", rr.SegmentationImage((confidence_hw > threshold).astype(np.uint8)), static=static)


def create_relative_depth_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    """3D view beside image / depth / confidence (mask tab first, spectrum behind it).

    The confidence views stay in the layout even for predictors without a confidence head; an
    empty view is cheaper than a blueprint that depends on the prediction. Image-plane quads are
    excluded from the 3D view so only the point cloud and camera are drawn there.
    """
    pinhole_path: Path = parent_log_path / "camera" / "pinhole"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(contents=["$origin/**", f"- {pinhole_path}/image", f"- {pinhole_path}/confidence", f"- {pinhole_path}/confidence_mask"]),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{pinhole_path}/image"),
                rrb.Spatial2DView(origin=f"{pinhole_path}/depth"),
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{pinhole_path}/confidence_mask", name="confidence mask"),
                    rrb.Spatial2DView(origin=f"{pinhole_path}/confidence", name="confidence"),
                    active_tab=0,
                ),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def log_relative_pred(
    parent_log_path: Path,
    relative_pred: RelativeDepthPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    remove_flying_pixels: bool = True,
    log_disparity: bool = False,
    jpeg_quality: int = 90,
    depth_edge_threshold: int | float = 1.1,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
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
    depth_hw = np.asarray(np.nan_to_num(depth_hw, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)

    if remove_flying_pixels:
        edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        depth_hw = np.asarray(depth_hw * ~edges_mask, dtype=np.float32)

    rr.log(f"{pinhole_path}/depth", rr.DepthImage(depth_hw))

    if relative_pred.confidence is not None:
        log_confidence(pinhole_path, relative_pred.confidence, threshold=confidence_threshold)

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
        depth_hw = np.asarray(depth_hw * ~edges_mask, dtype=np.float32)

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


def log_normal_pred(
    parent_log_path: Path,
    normal_pred: SurfaceNormalPrediction,
    rgb_hw3: UInt8[np.ndarray, "h w 3"],
    K_33: Float32[np.ndarray, "3 3"] | None = None,
    jpeg_quality: int = 90,
) -> None:
    """Log surface normal prediction to Rerun.

    Args:
        parent_log_path: Root entity path for logging.
        normal_pred: Surface normal prediction to log.
        rgb_hw3: Input RGB image.
        K_33: Camera intrinsics. When provided, logs the real pinhole.
            When None, falls back to an estimated pinhole using max(h, w) as focal length.
        jpeg_quality: JPEG compression quality for images.
    """
    cam_log_path: Path = parent_log_path / "camera"
    pinhole_path: Path = cam_log_path / "pinhole"

    h: int
    w: int
    h, w, _ = rgb_hw3.shape

    if K_33 is not None:
        rr.log(
            f"{pinhole_path}",
            rr.Pinhole(
                image_from_camera=K_33,
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
    else:
        rr.log(
            f"{pinhole_path}",
            rr.Pinhole(
                focal_length=max(h, w),
                width=w,
                height=h,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=jpeg_quality))

    # normals are in [-1, 1] range, convert to [0, 255] for visualization
    normal_hw3: Float32[np.ndarray, "h w 3"] = normal_pred.normal_hw3
    normal_uint8: UInt8[np.ndarray, "h w 3"] = ((normal_hw3 + 1) / 2 * 255).astype(np.uint8)
    rr.log(f"{pinhole_path}/normals", rr.Image(normal_uint8).compress(jpeg_quality=jpeg_quality))

    confidence_hw1: Float32[np.ndarray, "h w 1"] = normal_pred.confidence_hw1
    rr.log(f"{pinhole_path}/confidence", rr.Image((confidence_hw1 * 255).astype(np.uint8)))


def create_stereo_depth_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    """3D rig + cloud beside the stereo pair, metric depth, and disparity (matches ``log_stereo_pred``)."""
    left_path: Path = parent_log_path / "rig_00" / "cam_00"
    right_path: Path = parent_log_path / "rig_00" / "cam_01"
    return rrb.Blueprint(
        rrb.Horizontal(
            # Images stay in the 3D view so each frustum shows its picture (exoego convention); disparity is 2D-only.
            rrb.Spatial3DView(origin=f"{parent_log_path}", contents=["$origin/**", f"- {left_path}/disparity"]),
            rrb.Vertical(
                rrb.Horizontal(rrb.Spatial2DView(origin=f"{left_path}/pinhole/image", name="left"), rrb.Spatial2DView(origin=f"{right_path}/pinhole/image", name="right")),
                rrb.Spatial2DView(origin=f"{left_path}/pinhole/depth", name="depth (m)"),
                rrb.Spatial2DView(origin=f"{left_path}/disparity", name="disparity (px)"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def log_stereo_pred(
    parent_log_path: Path,
    stereo_pred: StereoDepthPrediction,
    left_rgb: UInt8[np.ndarray, "h w 3"],
    right_rgb: UInt8[np.ndarray, "h w 3"],
    max_depth_m: float = 20.0,
    remove_flying_pixels: bool = True,
    depth_edge_threshold: float = 0.5,
    jpeg_quality: int = 90,
) -> None:
    """Log a calibrated stereo prediction as an exoego:v2 rig.

    ``rig_00/cam_00`` is the left (reference) camera, ``cam_01`` the right one at ``+baseline`` along x. Metric depth goes
    under the left pinhole (``cam_00/pinhole/depth``) so the viewer backprojects it; disparity is logged beside the pinhole
    (``cam_00/disparity``) so it is not. Depth beyond ``max_depth_m`` is dropped: sub-pixel disparities explode to kilometres.

    Args:
        parent_log_path: World entity path; the rig is logged at ``<parent>/rig_00``.
        stereo_pred: Prediction with ``K_33``, ``baseline_m``, and ``depth_meters`` filled.
        left_rgb: Left image, ``UInt8[ndarray, "h w 3"]``.
        right_rgb: Right image, ``UInt8[ndarray, "h w 3"]``.
        max_depth_m: Depth cut-off for the logged depth image.
        remove_flying_pixels: Zero depth on depth edges so the backprojected cloud has no streaks between surfaces.
        depth_edge_threshold: Depth-gradient magnitude (metres per pixel) above which a pixel counts as an edge.
        jpeg_quality: JPEG quality for the two images.
    """
    from simplecv.rerun_rig_logger import log_rig_static
    from simplecv.rig import Rig, stereo_rig_calibration

    height: int = left_rgb.shape[0]
    width: int = left_rgb.shape[1]
    rig: Rig = Rig(index=0, calibration=stereo_rig_calibration(stereo_pred.K_33, stereo_pred.baseline_m, width, height), image_plane_distance=0.5)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RDF, static=True)
    log_rig_static(rig, world_path=str(parent_log_path))
    left_path: Path = parent_log_path / "rig_00" / "cam_00"
    right_path: Path = parent_log_path / "rig_00" / "cam_01"
    rr.log(f"{left_path}/pinhole/image", rr.Image(left_rgb).compress(jpeg_quality=jpeg_quality), static=True)
    rr.log(f"{right_path}/pinhole/image", rr.Image(right_rgb).compress(jpeg_quality=jpeg_quality), static=True)
    depth_hw: Float32[np.ndarray, "h w"] = np.where(stereo_pred.depth_meters > max_depth_m, 0.0, stereo_pred.depth_meters).astype(np.float32)
    if remove_flying_pixels:
        edges_hw: Bool[np.ndarray, "h w"] = depth_edges_mask(depth_hw, threshold=depth_edge_threshold)
        depth_hw = np.asarray(depth_hw * ~edges_hw, dtype=np.float32)
    rr.log(f"{left_path}/pinhole/depth", rr.DepthImage(depth_hw, meter=1.0, depth_range=(0.0, max_depth_m)), static=True)
    rr.log(f"{left_path}/disparity", rr.DepthImage(stereo_pred.disparity), static=True)
