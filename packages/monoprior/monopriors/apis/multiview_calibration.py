from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray
from sam3.api.predictor import SAM3Config, SAM3Predictor, SAM3Results
from simplecv.camera_parameters import PinholeParameters
from simplecv.ops.tsdf_depth_fuser import Open3DScaleInvariantFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.video_io import MultiVideoReader

from monopriors.apis.multiview_geometry import (
    MultiviewGeometryConfig,
    MultiviewGeometryResult,
    run_multiview_geometry,
)
from monopriors.models.multiview.multiview_model import MultiviewPred
from monopriors.models.multiview.multiview_pointcloud import mv_pred_to_filtered_pointcloud
from monopriors.models.multiview.multiview_predictor import (
    MultiviewPredictor,
    MultiviewPredictorConfig,
)
from monopriors.models.relative_depth import (
    RelativeDepthPrediction,
    get_relative_predictor,
)
from monopriors.models.relative_depth.base_relative_depth import BaseRelativePredictor

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
                name="Refined Depth",
            ),
        ],
        active_tab=1,
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


@dataclass
class MVCalibResults:
    depth_list: list[Float32[ndarray, "H W"]]
    pinhole_param_list: list[PinholeParameters]
    pcd: o3d.geometry.PointCloud


@dataclass(frozen=True, slots=True)
class MultiViewCalibratorConfig:
    """Configuration toggles for multi-view calibration pre- and post-processing."""

    predictor_config: MultiviewPredictorConfig = field(default_factory=MultiviewPredictorConfig)
    """Construction settings for the selected multi-view backend."""
    geometry_config: MultiviewGeometryConfig = field(default_factory=MultiviewGeometryConfig)
    """Per-run preprocessing, centering, confidence, and logging settings."""
    refine_depth_maps: bool = True
    """Run MoGe depth refinement on multi-view depth predictions before unprojection."""
    segment_people: bool = True
    """Enable SAM3 text-conditioned foreground removal for dynamic human actors."""


class MultiViewCalibrationPostprocessor:
    """Fuse predicted geometry with optional segmentation and depth refinement."""

    def __init__(
        self,
        parent_log_path: Path,
        config: MultiViewCalibratorConfig,
        *,
        seg_predictor: SAM3Predictor | None = None,
        moge_predictor: BaseRelativePredictor | None = None,
    ) -> None:
        """Instantiate only the optional post-processing dependencies."""
        self.config = config
        self.device = config.predictor_config.device
        self.parent_log_path = parent_log_path
        self.seg_predictor: SAM3Predictor | None = seg_predictor
        if self.config.segment_people and self.seg_predictor is None:
            self.seg_predictor = SAM3Predictor(SAM3Config(device=self.device))
        self.moge_predictor: BaseRelativePredictor | None = moge_predictor
        if self.config.refine_depth_maps and self.moge_predictor is None:
            self.moge_predictor = get_relative_predictor("MogeV1Predictor")(device=self.device)

    def __call__(
        self,
        *,
        rgb_list: list[UInt8[ndarray, "H W 3"]],
        geometry_result: MultiviewGeometryResult,
    ) -> MVCalibResults:
        """Apply optional segmentation and depth refinement to predicted geometry."""
        from monopriors.apis.depth_alignment import DepthAlignmentConfig, DepthAlignmentResult, run_depth_alignment

        mv_geo_result: MultiviewGeometryResult = geometry_result
        mv_pred_list: list[MultiviewPred] = mv_geo_result.mv_pred_list
        depth_confidences: list[UInt8[ndarray, "H W"]] = mv_geo_result.depth_confidences

        # 2. SAM3 segmentation: per-view person masks
        if self.config.segment_people:
            if self.seg_predictor is None:
                raise RuntimeError("Person segmentation was enabled without a SAM3 predictor.")
            segmask_list: list[Bool[np.ndarray, "H W"] | None] = []
            for rgb in rgb_list:
                people_masks: Bool[ndarray, "H W"] | None = segment_people(
                    rgb, seg_predictor=self.seg_predictor, dilation=50
                )
                segmask_list.append(people_masks)
        else:
            segmask_list = [None] * len(mv_pred_list)

        # Update depth confidences to exclude people
        updated_confidences: list[UInt8[ndarray, "H W"]] = []
        for depth_conf, segmask in zip(depth_confidences, segmask_list, strict=True):
            if segmask is not None:
                updated_confidences.append(depth_conf * ~segmask)
            else:
                updated_confidences.append(depth_conf)
        depth_confidences = updated_confidences

        # 3. Optional depth refinement: MoGe + depth alignment per view
        refined_depths_list: list[Float32[ndarray, "H W"]] = []
        if self.config.refine_depth_maps:
            if self.moge_predictor is None:
                raise RuntimeError("Depth refinement was enabled without a MoGe predictor.")
            alignment_config: DepthAlignmentConfig = DepthAlignmentConfig(edge_threshold=0.01, scale_only=False)

            for idx, mv_pred in enumerate(mv_pred_list):
                depth_conf: UInt8[ndarray, "H W"] = depth_confidences[idx]
                filtered_depth_map: Float32[ndarray, "H W"] = np.where(depth_conf > 0, mv_pred.depth_map, 0)

                # Run MoGe relative depth on this view (moge expects float32; simplecv k_matrix is float64)
                K_33_raw: Float[ndarray, "3 3"] | None = mv_pred.pinhole_param.intrinsics.k_matrix
                if K_33_raw is None:
                    raise ValueError("MoGe depth refinement requires camera intrinsics.")
                K_33: Float32[ndarray, "3 3"] = K_33_raw.astype(np.float32)
                relative_pred: RelativeDepthPrediction = self.moge_predictor(rgb=mv_pred.rgb_image, K_33=K_33)

                # Align MoGe depth to the multi-view coordinate frame using the decomposed alignment node
                alignment_result: DepthAlignmentResult = run_depth_alignment(
                    reference_depth=filtered_depth_map,
                    target_depth=relative_pred.depth,
                    confidence_mask=(depth_conf > 0),
                    exclusion_mask=segmask_list[idx],
                    config=alignment_config,
                )
                refined_depths_list.append(alignment_result.aligned_depth)

                if self.config.geometry_config.verbose:
                    cam_log_path: Path = self.parent_log_path / mv_pred.cam_name
                    pinhole_log_path: Path = cam_log_path / "pinhole"
                    rr.log(
                        f"{pinhole_log_path}/refined_depth",
                        rr.DepthImage(alignment_result.aligned_depth, meter=1),
                        static=True,
                    )

        # Verbose logging: per-camera detail
        if self.config.geometry_config.verbose:
            for idx, mv_pred in enumerate(mv_pred_list):
                depth_conf = depth_confidences[idx]
                filtered_depth_map = np.where(depth_conf > 0, mv_pred.depth_map, 0)
                cam_log_path = self.parent_log_path / mv_pred.cam_name
                pinhole_log_path = cam_log_path / "pinhole"
                log_pinhole(mv_pred.pinhole_param, cam_log_path=cam_log_path, image_plane_distance=0.05, static=True)
                # Log RAW images, NOT .compress(). The Rerun 0.33 native viewer
                # locks up (needs a force-quit) when rendering encoded images.
                # Root cause (traced through the Rerun source): `.compress()` turns
                # the image into an `EncodedImage` (PIL JPEG, media_type image/jpeg).
                # In 0.33 the viewer no longer uploads encoded images straight to a
                # texture — `EncodedImageVisualizer` routes them through the VIDEO
                # decode pipeline as `VideoCodec::ImageSequence` (re_view_spatial
                # visualizers/video/encoded_image.rs -> execute_video_stream_like ->
                # video_stream_cache + SyncImageDecoder + bounded re_quota_channel),
                # and that pipeline deadlocks. Reliably reproduced (compressed = hang,
                # raw = render, every trial) on BOTH Linux x86_64 and macOS arm64 with
                # `rerun file.rrd --screenshot-to`. The deadlock needs multiple encoded
                # images shown in a multi-view blueprint (a 3D view that includes them
                # + per-camera 2D views — exactly create_final_view); a single image or
                # a flat 2D grid does not reliably trigger it. Raw rr.Image and
                # DepthImage (which skip the video pipeline) always render. Format-
                # agnostic (PNG EncodedImage hangs too), not PIL-specific. Pre-0.32
                # didn't route encoded images through the video pipeline, so this "did
                # not used to happen". Only `--mv-calibrator-config.geometry-config.verbose` breaks
                # because it is the calibrator's sole source of encoded images. Raw is
                # larger on the wire but renders reliably.
                rr.log(
                    f"{pinhole_log_path}/image",
                    rr.Image(mv_pred.rgb_image, color_model=rr.ColorModel.RGB),
                    static=True,
                )
                rr.log(
                    f"{pinhole_log_path}/confidence",
                    rr.Image(depth_conf, color_model=rr.ColorModel.L),
                    static=True,
                )
                rr.log(f"{pinhole_log_path}/filtered_depth", rr.DepthImage(filtered_depth_map, meter=1), static=True)
                rr.log(f"{pinhole_log_path}/depth", rr.DepthImage(mv_pred.depth_map, meter=1), static=True)

        # The viewer consumes roughly 150k points. Raw and refined depths share
        # the same confidence/person filtering and point budget.
        pointcloud_depths: list[Float32[ndarray, "H W"]] | None = (
            refined_depths_list if self.config.refine_depth_maps else None
        )
        filtered_output: tuple[
            Float32[ndarray, "sampled_points 3"], UInt8[ndarray, "sampled_points 3"]
        ] = mv_pred_to_filtered_pointcloud(
            mv_pred_list,
            depth_confidences,
            depth_list=pointcloud_depths,
            target_points=150_000,
        )
        filtered_points: Float32[ndarray, "sampled_points 3"] = filtered_output[0]
        filtered_colors: UInt8[ndarray, "sampled_points 3"] = filtered_output[1]
        pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)

        mv_calib_results: MVCalibResults = MVCalibResults(
            pinhole_param_list=[mv_pred.pinhole_param for mv_pred in mv_pred_list],
            pcd=pcd,
            depth_list=refined_depths_list
            if self.config.refine_depth_maps
            else [mv_pred.depth_map for mv_pred in mv_pred_list],
        )
        return mv_calib_results


def run_multiview_calibration(
    *,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    multiview_predictor: MultiviewPredictor,
    config: MultiViewCalibratorConfig,
    parent_log_path: Path,
    seg_predictor: SAM3Predictor | None = None,
    moge_predictor: BaseRelativePredictor | None = None,
) -> MVCalibResults:
    """Run geometry and calibration post-processing with a caller-owned predictor.

    Args:
        rgb_list: Ordered RGB frames captured at the same timestamp across cameras.
        multiview_predictor: Predictor owned by the caller for the duration of this call.
        config: Geometry and optional post-processing configuration.
        parent_log_path: Root Rerun entity path for verbose per-camera output.
        seg_predictor: Optional preloaded SAM3 predictor.
        moge_predictor: Optional preloaded relative-depth predictor.

    Returns:
        Calibrated cameras, depths, and a confidence-filtered point cloud.
    """
    geometry_result: MultiviewGeometryResult = run_multiview_geometry(
        rgb_list=rgb_list,
        multiview_predictor=multiview_predictor,
        config=config.geometry_config,
    )
    postprocessor: MultiViewCalibrationPostprocessor = MultiViewCalibrationPostprocessor(
        parent_log_path,
        config,
        seg_predictor=seg_predictor,
        moge_predictor=moge_predictor,
    )
    return postprocessor(rgb_list=rgb_list, geometry_result=geometry_result)


@dataclass
class MVInferenceConfig:
    """Runtime options for multi-view inference and calibration."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    image_dir: Path | None = None
    """Directory containing input images."""
    videos_dir: Path | None = None
    """Directory containing input videos."""
    ts_idx: int = 0
    """Timestep for video chosen frames."""
    mv_calibrator_config: MultiViewCalibratorConfig = field(default_factory=MultiViewCalibratorConfig)
    """Multi-view predictor, geometry, and post-processing configuration."""


def log_calibration_results(
    *,
    rgb_list: list[UInt8[ndarray, "H W 3"]],
    output: MVCalibResults,
    parent_log_path: Path,
    timeline: str,
    show_videos: bool = False,
) -> MVCalibResults:
    """Log an already-computed calibration result and build its TSDF mesh.

    All ``rr.log`` calls use the thread-local recording set by the caller
    (via ``with recording:`` in the UI, or the global recording in the CLI).

    Args:
        rgb_list: Ordered RGB frames across cameras.
        output: Computed multi-view calibration result.
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

    ###################################################
    # 2. Log Final Output (Not Verbose always logged) #
    ###################################################
    pcd: o3d.geometry.PointCloud = output.pcd
    filtered_points: Float32[ndarray, "final_points 3"] = np.asarray(pcd.points, dtype=np.float32)
    filtered_colors: Float32[ndarray, "final_points 3"] = np.asarray(pcd.colors, dtype=np.float32)

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
    # 3. Fuse Depths into TSDF Mesh     #
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
    # 0. Parse calibration inputs                    #
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

    multiview_predictor: MultiviewPredictor = MultiviewPredictor(config.mv_calibrator_config.predictor_config)
    output: MVCalibResults = run_multiview_calibration(
        rgb_list=rgb_list,
        multiview_predictor=multiview_predictor,
        config=config.mv_calibrator_config,
        parent_log_path=PARENT_LOG_PATH,
    )
    log_calibration_results(
        rgb_list=rgb_list,
        output=output,
        parent_log_path=PARENT_LOG_PATH,
        timeline=TIMELINE,
        show_videos=config.videos_dir is not None,
    )
