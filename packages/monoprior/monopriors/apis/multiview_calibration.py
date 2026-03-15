from dataclasses import dataclass, field, replace
from pathlib import Path
from timeit import default_timer as timer
from typing import Annotated, Literal

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from beartype.vale import Is
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray
from sam3_rerun.api.predictor import SAM3Config, SAM3Predictor, SAM3Results
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.camera_parameters import Extrinsics, PinholeParameters
from simplecv.ops.conventions import CameraConventions, convert_pose
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.ops.tsdf_depth_fuser import Open3DScaleInvariantFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_io import MultiVideoReader

from monopriors.depth_utils import depth_edges_mask, multidepth_to_points
from monopriors.models.multiview.vggt_model import MultiviewPred, VGGTPredictor, robust_filter_confidences
from monopriors.models.relative_depth import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor
from monopriors.scale_utils import compute_scale_and_shift

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
PARENT_LOG_PATH: Path = Path("world")
TIMELINE: str = "video_time"
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_depth_views(parent_log_path: Path, camera_index: int) -> rrb.Tabs:
    """
    Create depth visualization tabs for a specific camera.

    Args:
        parent_log_path: Parent log path for the camera views
        camera_index: Index of the camera to create depth views for

    Returns:
        Tabs blueprint containing depth and filtered depth views
    """
    depth_views: rrb.Tabs = rrb.Tabs(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/depth",
                contents=[
                    "+ $origin/**",
                ],
                name="Depth",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/filtered_depth",
                contents=[
                    "+ $origin/**",
                ],
                name="Filtered Depth",
            ),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/refined_depth",
                contents=[
                    "+ $origin/**",
                ],
                name="MoGe Depth",
            ),
        ],
        active_tab=2,
    )
    return depth_views


def create_camera_row(parent_log_path: Path, camera_index: int) -> rrb.Horizontal:
    """
    Create a single camera row with 3 views: content, depth, and confidence.

    Args:
        parent_log_path: Parent log path for the camera views
        camera_index: Index of the camera to create views for

    Returns:
        Horizontal blueprint containing pinhole content, depth views, and confidence map
    """
    camera_row: rrb.Horizontal = rrb.Horizontal(
        contents=[
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/image",
                contents=[
                    "+ $origin/**",
                ],
                name="Image Content",
            ),
            create_depth_views(parent_log_path, camera_index),
            rrb.Spatial2DView(
                origin=f"{parent_log_path}/camera_{camera_index}/pinhole/confidence",
                contents=[
                    "+ $origin/**",
                ],
                name="Confidence Map",
            ),
        ]
    )
    return camera_row


def chunk_cameras(num_cameras: int, chunk_size: int = 4) -> list[range]:
    """
    Group cameras into chunks of specified size.

    Args:
        num_cameras: Total number of cameras
        chunk_size: Maximum cameras per chunk (default 4)

    Returns:
        List of ranges representing camera chunks
    """
    chunks: list[range] = [range(i, min(i + chunk_size, num_cameras)) for i in range(0, num_cameras, chunk_size)]
    return chunks


def create_tabbed_camera_view(parent_log_path: Path, num_cameras: int) -> rrb.Tabs:
    """
    Create tabbed interface grouping cameras by 4s.

    Args:
        parent_log_path: Parent log path for the camera views
        num_cameras: Total number of cameras to display

    Returns:
        Tabs blueprint with each tab containing up to 4 camera rows
    """
    camera_chunks: list[range] = chunk_cameras(num_cameras)

    tabs: list[rrb.Vertical] = []
    for camera_range in camera_chunks:
        # Create camera rows for this chunk
        camera_rows: list[rrb.Horizontal] = [create_camera_row(parent_log_path, i) for i in camera_range]

        # Create tab name
        if camera_range.start + 1 == camera_range.stop:
            tab_name: str = f"Camera {camera_range.start + 1}"
        else:
            tab_name = f"Cameras {camera_range.start + 1}-{camera_range.stop}"

        # Create tab content
        tab_content: rrb.Vertical = rrb.Vertical(contents=camera_rows, name=tab_name)
        tabs.append(tab_content)

    tabbed_view: rrb.Tabs = rrb.Tabs(contents=tabs, name="Depths Tab")
    return tabbed_view


def create_final_view(parent_log_path: Path, num_images: int, show_videos: bool = False) -> rrb.ContainerLike:
    view3d = rrb.Spatial3DView(
        origin=f"{parent_log_path}",
        contents=[
            "+ $origin/**",
            f"- /{parent_log_path}/point_cloud",
            # don't include depths in the 3D view, as they can be very noisy
            *[f"- /{parent_log_path}/camera_{i}/pinhole/depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/filtered_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/refined_depth" for i in range(num_images)],
            *[f"- /{parent_log_path}/camera_{i}/pinhole/confidence" for i in range(num_images)],
            # *[f"- /{parent_log_path}/camera_{i}/pinhole/image" for i in range(num_images)],
        ],
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
    )

    # Create tabbed view that supports any number of cameras
    view_2d: rrb.Tabs = create_tabbed_camera_view(parent_log_path, num_images)
    if show_videos:
        view_2d_videos: rrb.Grid = rrb.Grid(
            contents=[
                rrb.Spatial2DView(origin=f"{parent_log_path}/camera_{i}/pinhole/video", name=f"Video {i + 1}")
                for i in range(num_images)
            ],
            name="Videos Tab",
        )
        view_2d = rrb.Tabs(view_2d, view_2d_videos)

    final_view: rrb.ContainerLike = rrb.Horizontal(contents=[view3d, view_2d], column_shares=[3, 2])

    return final_view


def orient_mv_pred_list(
    mv_pred_list: list[MultiviewPred],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "none",
) -> list[MultiviewPred]:
    extri_list: list[Extrinsics] = [mv_pred.pinhole_param.extrinsics for mv_pred in mv_pred_list]

    world_T_cam_batch: Float[ndarray, "*num_poses 4 4"] = np.stack([extri.world_T_cam for extri in extri_list])
    assert len(set(mv_pred.pinhole_param.intrinsics.camera_conventions for mv_pred in mv_pred_list)) == 1
    if mv_pred_list[0].pinhole_param.intrinsics.camera_conventions == "RDF":
        world_T_cam_gl: Float[ndarray, "*num_poses 4 4"] = convert_pose(
            world_T_cam_batch, CameraConventions.CV, CameraConventions.GL
        )
    else:
        world_T_cam_gl = world_T_cam_batch

    # NumPy-only orientation (returns (N,3,4))
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
    # put back into mv pred list using replace
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


def mv_pred_to_pointcloud(
    mv_pred_list: list[MultiviewPred], depth_list: list[Float32[ndarray, "H W"]] | None = None
) -> Float32[ndarray, "num_points 3"]:
    """
    Convert multiview predictions into a 3D point cloud in world coordinates.

    Args:
        mv_pred_list (list[MultiviewPred]): Sequence of multiview predictions containing
            per-view depth maps alongside calibrated camera intrinsics and extrinsics.
        depth_list (list[np.ndarray], optional): Optional override for depth maps, each of shape
            (H, W). When provided, these depths are used instead of the ones stored in
            ``mv_pred_list``.

    Returns:
        np.ndarray: A flattened array of shape (num_points, 3) holding 3D points expressed in the
        world reference frame.
    """
    # Select depth source: either the provided overrides or the depths stored on each prediction
    if depth_list is None:
        depth_maps: Float32[ndarray, "b h w 1"] = np.stack(
            [rearrange(mv_pred.depth_map, "h w -> h w 1") for mv_pred in mv_pred_list], axis=0
        ).astype(np.float32)
    else:
        depth_maps: Float32[ndarray, "b h w 1"] = np.stack(
            [rearrange(depth, "h w -> h w 1") for depth in depth_list], axis=0
        ).astype(np.float32)

    # Collect camera extrinsics (world_T_cam) for each view; multidepth_to_points expects this convention
    world_T_cam_b44: Float32[ndarray, "num_cams 4 4"] = np.stack(
        [mv_pred.pinhole_param.extrinsics.world_T_cam for mv_pred in mv_pred_list], axis=0
    ).astype(np.float32)

    # Gather intrinsics matrices so each depth map can be unprojected using its matching camera model
    K_b33: Float32[ndarray, "b 3 3"] = np.stack(
        [mv_pred.pinhole_param.intrinsics.k_matrix for mv_pred in mv_pred_list], axis=0
    ).astype(np.float32)

    # Lift each pixel into 3D world space, yielding one point per depth value
    world_points: Float32[ndarray, "b h w 3"] = multidepth_to_points(
        depth_maps=depth_maps, world_T_cam_batch=world_T_cam_b44, K_b33=K_b33
    )

    # Collapse the batch and image dimensions to obtain a flat point cloud
    pointcloud: Float32[ndarray, "num_points 3"] = world_points.reshape(-1, 3)
    return pointcloud


def segment_people(
    rgb: UInt8[ndarray, "H W 3"],
    *,
    seg_predictor: SAM3Predictor,
    text: str = "person",
    mask_threshold: float = 0.5,
    dilation: int = 0,
) -> Bool[np.ndarray, "h w"] | None:
    """Segment people using SAM3 text-conditioned instance segmentation.

    Args:
        rgb: Input image in RGB order with dtype uint8 and shape [H, W, 3].
        seg_predictor: SAM3Predictor instance for inference.
        text: Text prompt for SAM3 (default: "person").
        mask_threshold: Probability threshold to binarize masks.
        dilation: Kernel size for mask dilation (0 = no dilation).

    Returns:
        Boolean union mask of all detected people, or None if no detections.
    """
    sam3_results: SAM3Results = seg_predictor.predict_single_image(rgb_hw3=rgb, text=text)
    if len(sam3_results.scores) == 0:
        return None

    # Union all detected person masks into a single binary mask
    h: int = rgb.shape[0]
    w: int = rgb.shape[1]
    union_mask: Bool[np.ndarray, "h w"] = np.zeros((h, w), dtype=bool)
    for mask in sam3_results.masks:
        mask_bool: Bool[np.ndarray, "h w"] = mask >= mask_threshold
        union_mask = np.logical_or(union_mask, mask_bool)

    # Apply dilation to expand the mask boundaries
    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        union_mask = cv2.dilate(union_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return union_mask


KeepTopPercent = Annotated[int | float, Is[lambda percent: 1 <= percent <= 100]]


@dataclass
class MVCalibResults:
    depth_list: list[Float32[ndarray, "H W"]]
    pinhole_param_list: list[PinholeParameters]
    pcd: o3d.geometry.PointCloud


@dataclass
class MultiViewCalibratorConfig:
    """Configuration toggles for multi-view calibration pre- and post-processing."""

    keep_top_percent: KeepTopPercent = 30.0
    """Fraction of high-confidence pixels retained after VGGT filtering.
    Value must be in [1, 100] and controls how aggressively low-confidence pixels
    are discarded. The calibrator keeps (100 - keep_top_percent)% of pixels;
    e.g. 75 → keep top 25%, 30 → keep top 70%."""
    refine_depth_maps: bool = True
    """Run MoGe depth refinement on VGGT predictions before unprojection."""
    segment_people: bool = True
    """Enable SAM3 text-conditioned foreground removal for dynamic human actors."""
    preprocessing_mode: Literal["crop", "pad"] = "pad"
    """Mode for image preprocessing: 'crop' preserves aspect ratio, 'pad' adds white padding"""
    device: Literal["cuda", "cpu"] = "cuda"
    """Execution backend used when instantiating torch-based components."""
    verbose: bool = False
    """Emit detailed logging and per-stage timing when True."""


class MultiViewCalibrator:
    """Orchestrates multi-view calibration by fusing depth, segmentation, and refinement models.

    The calibrator runs VGGT to infer geometry per view, filters dynamic actors via
    SAM3 text-conditioned segmentation, and optionally refines the predicted depths with the
    Moge relative depth model before generating a consolidated Open3D point cloud.
    """

    def __init__(self, parent_log_path: Path, config: MultiViewCalibratorConfig) -> None:
        """Instantiate the detector, segmenter, and depth estimators needed for calibration."""
        self.config = config
        self.device = config.device
        self.parent_log_path = parent_log_path
        if self.config.segment_people:
            self.seg_predictor: SAM3Predictor = SAM3Predictor(SAM3Config(device=self.device))

        self.vggt_predictor: VGGTPredictor = VGGTPredictor(
            device=self.device,
            preprocessing_mode=self.config.preprocessing_mode,
        )

        if self.config.refine_depth_maps:
            self.moge_predictor: BaseRelativePredictor = get_relative_predictor("MogeV1Predictor")(device="cuda")

    def __call__(
        self,
        *,
        rgb_list: list[UInt8[ndarray, "H W 3"]],
    ) -> MVCalibResults:
        """Estimate calibrated pinhole parameters and a fused point cloud from RGB views.

        Args:
            rgb_list: Ordered list of RGB frames captured at the same timestamp across cameras.

        Returns:
            MVCalibResults containing per-camera pinhole parameters and a down-sampled point cloud
            reconstructed from high-confidence depth measurements (optionally refined by Moge).
        """
        mv_pred_list: list[MultiviewPred] = self.vggt_predictor(rgb_list)
        mv_pred_list: list[MultiviewPred] = orient_mv_pred_list(mv_pred_list)

        # Compute person segmentation masks per view. Keep for potential UI/analysis,
        # but do not alter confidences/depth with it here.
        if self.config.segment_people:
            segmask_list: list[Bool[np.ndarray, "H W"] | None] = []
            for rgb in rgb_list:
                people_masks: Bool[ndarray, "H W"] | None = segment_people(
                    rgb, seg_predictor=self.seg_predictor, dilation=50
                )
                segmask_list.append(people_masks)
        else:
            segmask_list = [None] * len(mv_pred_list)

        pointcloud: Float32[ndarray, "num_points 3"] = mv_pred_to_pointcloud(mv_pred_list)
        rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
            [rearrange(mv_pred.rgb_image, "h w c -> (h w) c") for mv_pred in mv_pred_list]
        )

        # create depth confidence values using robust filtering for top keep percentile
        depth_confidences: list[UInt8[ndarray, "H W"]] = [
            robust_filter_confidences(mv_pred.confidence_mask, keep_top_percent=self.config.keep_top_percent)
            for mv_pred in mv_pred_list
        ]

        # update depth_confidences to exclude people, create a totally new list so it doesn't modify the original
        new_depth_confidences = []
        for depth_conf, segmask in zip(depth_confidences, segmask_list, strict=True):
            if segmask is not None:
                new_depth_confidences.append(depth_conf * ~segmask)
            else:
                new_depth_confidences.append(depth_conf)

        depth_confidences = new_depth_confidences
        pc_conf_mask: Bool[ndarray, "num_points"] = np.concatenate(
            [rearrange(depth_conf, "h w -> (h w)") for depth_conf in depth_confidences]
        ).astype(bool)

        # Filter by confidence BEFORE downsampling for better quality and efficiency
        filtered_points_pre_ds: Float32[ndarray, "filtered_points 3"] = pointcloud[pc_conf_mask]
        filtered_colors_pre_ds: UInt8[ndarray, "filtered_points 3"] = rgb_stack[pc_conf_mask]

        # Create point cloud from high-confidence points only
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points_pre_ds)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors_pre_ds / 255.0)  # Open3D expects [0,1] range

        if self.config.refine_depth_maps:
            refined_depths_list: list[Float32[ndarray, "H W"]] = []

        for mv_pred in mv_pred_list:
            depth_map: Float32[ndarray, "H W"] = mv_pred.depth_map
            depth_conf: UInt8[ndarray, "H W"] = depth_confidences[mv_pred_list.index(mv_pred)]
            # Filter depth
            filtered_depth_map: Float32[ndarray, "H W"] = np.where(depth_conf > 0, depth_map, 0)

            if self.config.refine_depth_maps:
                relative_pred: RelativeDepthPrediction = self.moge_predictor.__call__(
                    rgb=mv_pred.rgb_image, K_33=mv_pred.pinhole_param.intrinsics.k_matrix
                )

                scale, shift = compute_scale_and_shift(
                    relative_pred.depth, filtered_depth_map, mask=depth_conf > 0, scale_only=False
                )
                metric_depth: Float32[np.ndarray, "h w"] = relative_pred.depth.copy() * scale + shift
                # filter depth
                edges_mask: Bool[np.ndarray, "h w"] = depth_edges_mask(metric_depth, threshold=0.01)
                metric_depth: Float32[np.ndarray, "h w"] = metric_depth * ~edges_mask
                # metric_depth = np.where(depth_conf > 0, metric_depth, 0)
                # remove people from metric depth
                if segmask_list[mv_pred_list.index(mv_pred)] is not None:
                    metric_depth: Float32[np.ndarray, "h w"] = metric_depth * ~segmask_list[mv_pred_list.index(mv_pred)]

                refined_depths_list.append(metric_depth)

            if self.config.verbose:
                cam_log_path: Path = self.parent_log_path / mv_pred.cam_name
                pinhole_log_path: Path = cam_log_path / "pinhole"
                log_pinhole(
                    mv_pred.pinhole_param,
                    cam_log_path=cam_log_path,
                    image_plane_distance=0.05,
                    static=True,
                )

                rr.log(
                    f"{pinhole_log_path}/image",
                    rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB).compress(),
                    static=True,
                )
                rr.log(
                    f"{pinhole_log_path}/confidence",
                    rr.Image(depth_conf, color_model=rr.ColorModel.L).compress(),
                    static=True,
                )
                rr.log(
                    f"{pinhole_log_path}/filtered_depth",
                    rr.DepthImage(filtered_depth_map, meter=1),
                    static=True,
                )
                rr.log(
                    f"{pinhole_log_path}/depth",
                    rr.DepthImage(depth_map, meter=1),
                    static=True,
                )
                if self.config.refine_depth_maps:
                    rr.log(
                        f"{pinhole_log_path}/refined_depth",
                        rr.DepthImage(metric_depth, meter=1),
                        static=True,
                    )

        if self.config.refine_depth_maps:
            moge_points: Float32[ndarray, "num_points 3"] = mv_pred_to_pointcloud(
                mv_pred_list, depth_list=refined_depths_list
            )
            new_pc: Float32[ndarray, "num_points 3"] = moge_points.reshape(-1, 3)
            rgb_stack: UInt8[ndarray, "num_points 3"] = np.concatenate(
                [rearrange(mv_pred.rgb_image, "h w c -> (h w) c") for mv_pred in mv_pred_list]
            )

            # Create point cloud from high-confidence points only
            pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(new_pc)
            pcd.colors = o3d.utility.Vector3dVector(rgb_stack / 255.0)  # Open3D expects [0,1] range

        mv_calib_results: MVCalibResults = MVCalibResults(
            pinhole_param_list=[mv_pred.pinhole_param for mv_pred in mv_pred_list],
            pcd=pcd,
            depth_list=refined_depths_list
            if self.config.refine_depth_maps
            else [mv_pred.depth_map for mv_pred in mv_pred_list],
        )
        return mv_calib_results


@dataclass
class MVInferenceConfig:
    """Runtime options for VGGT-based multi-view inference and calibration."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_dir: Path | None = None
    """Directory containing input images."""
    videos_dir: Path | None = None
    """Directory containing input videos."""
    ts_idx: int = 0
    """Timestep for video chosen frames."""
    mv_calibrator_config: MultiViewCalibratorConfig = field(default_factory=MultiViewCalibratorConfig)
    """Base calibrator configuration; `refine_depth_maps` overrides its refinement flag."""


def run_calibration_pipeline(
    *,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    mv_calibrator: MultiViewCalibrator,
    parent_log_path: Path,
    timeline: str,
    show_videos: bool = False,
) -> MVCalibResults:
    """Run the full calibration pipeline: blueprint, calibration, pointcloud, and TSDF mesh.

    All ``rr.log`` calls use the thread-local recording set by the caller
    (via ``with recording:`` in the UI, or the global recording in the CLI).

    Args:
        rgb_list: Ordered RGB frames across cameras.
        mv_calibrator: Pre-initialised calibrator (models already loaded).
        parent_log_path: Root Rerun entity path.
        timeline: Rerun timeline name.
        show_videos: Whether to include video views in the blueprint.

    Returns:
        MVCalibResults with per-camera pinholes and a fused point cloud.
    """
    start: float = timer()

    #####################################
    # 1. Setup Rerun related components #
    #####################################
    final_view: rrb.ContainerLike = create_final_view(
        parent_log_path=parent_log_path, num_images=len(rgb_list), show_videos=show_videos
    )
    blueprint: rrb.Blueprint = rrb.Blueprint(final_view, collapse_panels=True)
    rr.send_blueprint(blueprint=blueprint)
    rr.log(f"{parent_log_path}", rr.ViewCoordinates.RFU, static=True)
    rr.set_time(timeline, duration=0)

    ##############################
    # 2. Run MultiViewCalibrator #
    ##############################
    output: MVCalibResults = mv_calibrator(rgb_list=rgb_list)

    ###################################################
    # 3. Log Final Output (Not Verbose always logged) #
    ###################################################
    pcd: o3d.geometry.PointCloud = output.pcd

    # Automatically determine optimal voxel size based on point cloud characteristics
    voxel_size: float = estimate_voxel_size(np.asarray(pcd.points, dtype=np.float32), target_points=150_000)
    pcd_ds: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxel_size)

    filtered_points: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
    filtered_colors: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)

    rr.log(
        f"{parent_log_path}/point_cloud",
        rr.Points3D(filtered_points, colors=filtered_colors),
        static=True,
    )
    # Log camera intrinsics/extrinsics
    for cam_idx, pinhole_param in enumerate(output.pinhole_param_list):
        cam_log_path: Path = parent_log_path / f"camera_{cam_idx}"
        log_pinhole(
            pinhole_param,
            cam_log_path=cam_log_path,
            image_plane_distance=0.05,
            static=True,
        )

    #####################################
    # 4. Fuse Depths into TSDF Mesh     #
    #####################################
    if output.depth_list and output.pinhole_param_list:
        depth_fuser: Open3DScaleInvariantFuser = Open3DScaleInvariantFuser(grid_resolution=512)
        reference_points: Float32[ndarray, "num_points 3"] = np.asarray(pcd.points, dtype=np.float32)
        depth_fuser.initialise_from_points(reference_points)

        for depth_map, pinhole_param, rgb in zip(
            output.depth_list,
            output.pinhole_param_list,
            rgb_list,
            strict=True,
        ):
            depth_fuser.fuse_frame(depth_hw=depth_map, pinhole=pinhole_param, rgb_hw3=rgb)

        gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
        gt_mesh.compute_vertex_normals()

        vertex_positions: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertices, dtype=np.float32)
        triangle_indices: Int[ndarray, "num_faces 3"] = np.asarray(gt_mesh.triangles, dtype=np.int32)

        vertex_normals: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_normals, dtype=np.float32)
        vertex_colors: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_colors, dtype=np.float32)

        rr.log(
            str(parent_log_path / "gt_mesh"),
            rr.Mesh3D(
                vertex_positions=vertex_positions,
                triangle_indices=triangle_indices,
                vertex_normals=vertex_normals,
                vertex_colors=vertex_colors,
            ),
            static=True,
        )

    print(f"Inference completed in {timer() - start:.2f} seconds")
    return output


def load_rgb_images(image_paths: list[Path]) -> list[UInt8[ndarray, "H W 3"]]:
    """Load image files as RGB uint8 numpy arrays.

    Args:
        image_paths: Paths to image files to load.

    Returns:
        List of RGB images as uint8 numpy arrays.

    Raises:
        FileNotFoundError: If any image path cannot be read by OpenCV.
    """
    rgb_list: list[UInt8[ndarray, "H W 3"]] = []
    for image_path in image_paths:
        bgr: UInt8[ndarray, "H W 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image {image_path}")
        rgb: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_list.append(rgb)
    return rgb_list


def main(config: MVInferenceConfig) -> None:
    if config.image_dir is None and config.videos_dir is None:
        raise ValueError("Either image or videos directory must be specified")

    ####################################################
    # 0. Parse inputs to setup for MultiViewCalibrator #
    ####################################################
    if config.image_dir is not None:
        image_paths: list[Path] = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.extend(config.image_dir.glob(f"*{ext}"))
        image_paths = sorted(image_paths)
        assert len(image_paths) > 0, (
            f"No images found in {config.image_dir} in supported formats {SUPPORTED_IMAGE_EXTENSIONS}"
        )
        rgb_list: list[UInt8[ndarray, "H W 3"]] = load_rgb_images(image_paths)

    elif config.videos_dir is not None:
        video_path_list: list[Path] = sorted(config.videos_dir.glob("*.mp4"))
        assert len(video_path_list) > 0, f"No videos found in {config.videos_dir}"
        exo_timestamps: list[Int[ndarray, "num_frames"]] = []
        for i, video_path in enumerate(video_path_list):
            frame_timestamps_ns: Int[ndarray, "num_frames"] = log_video(
                video_source=video_path,
                video_log_path=PARENT_LOG_PATH / f"camera_{i}" / "pinhole" / "video",
                timeline=TIMELINE,
            )
            exo_timestamps.append(frame_timestamps_ns)

        mv_reader: MultiVideoReader = MultiVideoReader(video_path_list)
        bgr_list: list[UInt8[ndarray, "H W 3"]] = mv_reader[config.ts_idx]
        rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) for bgr in bgr_list]
    else:
        raise ValueError("Either image_dir or videos_dir must be specified")

    mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(PARENT_LOG_PATH, config=config.mv_calibrator_config)

    run_calibration_pipeline(
        rgb_list=rgb_list,
        mv_calibrator=mv_calibrator,
        parent_log_path=PARENT_LOG_PATH,
        timeline=TIMELINE,
        show_videos=config.videos_dir is not None,
    )
