"""Exo-only single-frame calibration and pose estimation.

Performs multiview calibration and body pose estimation using only exo cameras
from an RRD dataset on a single frame. Extracts GT keypoints from ExoEgoLabels
for later Umeyama alignment. Also logs ego cameras for visualization (but does
not use them for calibration).
"""

from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Literal, cast

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float, Float32, Int, UInt8
from monopriors.apis.multiview_calibration import MultiViewCalibrator, MultiViewCalibratorConfig, MVCalibResults
from numpy import ndarray
from simplecv.apis.view_exoego import LogPaths, SceneSetupResult, log_environment_mesh, log_exoego_batch, setup_scene
from simplecv.camera_parameters import Extrinsics, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    FACE_IDX,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.ops.tsdf_depth_fuser import Open3DScaleInvariantFuser
from simplecv.rerun_custom_types import Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from simplecv.video_io import MultiVideoReader, TorchCodecMultiVideoReader

from mv_api.coco133_layers import (
    COCO133_LAYER_COLORS,
    COCO133_LAYER_LABELS,
    Coco133AnnotationLayer,
)
from mv_api.multiview_pose_estimator import MultiviewBodyTracker, MultiviewBodyTrackerConfig, MVHistory

np.set_printoptions(suppress=True)

device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class UmeyamaResult:
    """Similarity transform mapping exo world to ego/quest world."""

    ego_T_exo: Float32[ndarray, "4 4"]
    """4x4 transform: ego_points = ego_T_exo @ exo_points."""
    n_matches: int
    """Number of valid keypoint correspondences used."""
    rms_error_m: float
    """Root mean square alignment error in meters."""


@dataclass(slots=True)
class ExoCalibResult:
    """Result of exo-only calibration and pose estimation."""

    exo_xyzc: Float32[ndarray, "133 4"]
    """Triangulated 3D COCO-133 keypoints from exo cameras (original, before alignment)."""
    aligned_xyzc: Float32[ndarray, "133 4"] | None
    """Triangulated keypoints after Umeyama alignment to ego world."""
    gt_xyzc: Float32[ndarray, "133 4"] | None
    """GT keypoints from ExoEgoLabels (if available)."""
    pinhole_list: list[PinholeParameters]
    """Calibrated exo cameras (aligned to ego world if align_to_ego=True)."""
    frame_idx: int
    """Frame index that was processed."""
    timestamp_ns: int
    """Timestamp of processed frame."""
    umeyama_result: UmeyamaResult | None
    """Umeyama alignment result (if alignment was performed)."""
    # TSDF fused mesh data
    mesh_vertex_positions: Float32[ndarray, "num_vertices 3"] | None = None
    """Vertex positions for TSDF fused mesh."""
    mesh_triangle_indices: Int[ndarray, "num_faces 3"] | None = None
    """Triangle indices for TSDF fused mesh."""
    mesh_vertex_normals: Float32[ndarray, "num_vertices 3"] | None = None
    """Per-vertex normals for TSDF fused mesh."""
    mesh_vertex_colors: Float32[ndarray, "num_vertices 3"] | None = None
    """Per-vertex colors for TSDF fused mesh (0-1 range)."""


@dataclass(slots=True)
class TimingReport:
    """Timing breakdown for pipeline stages."""

    dataset_load_s: float = 0.0
    scene_setup_s: float = 0.0
    frame_load_s: float = 0.0
    mv_calibration_s: float = 0.0
    tsdf_fusion_s: float = 0.0
    pose_estimation_s: float = 0.0
    logging_s: float = 0.0
    total_s: float = 0.0

    def print_report(self) -> None:
        """Print timing breakdown to console."""
        print("\n" + "=" * 50)
        print("[exo_only_calib] TIMING REPORT")
        print("=" * 50)
        print(f"  Dataset load:     {self.dataset_load_s:6.2f}s")
        print(f"  Scene setup:      {self.scene_setup_s:6.2f}s")
        print(f"  Frame loading:    {self.frame_load_s:6.2f}s")
        print(f"  MV Calibration:   {self.mv_calibration_s:6.2f}s")
        print(f"  TSDF Fusion:      {self.tsdf_fusion_s:6.2f}s")
        print(f"  Pose Estimation:  {self.pose_estimation_s:6.2f}s")
        print(f"  Logging:          {self.logging_s:6.2f}s")
        print("-" * 50)
        print(f"  TOTAL:            {self.total_s:6.2f}s")
        print("=" * 50 + "\n")


def _umeyama_transform(
    *,
    src_points: Float[ndarray, "n 3"],
    dst_points: Float[ndarray, "n 3"],
    allow_scaling: bool = False,
    eps: float = 1e-9,
) -> Float32[ndarray, "4 4"]:
    """Compute the similarity transform aligning src_points to dst_points.

    The returned transform follows the right-to-left convention:
        dst_points ≈ (dst_T_src[:3, :3] @ src_points.T).T + dst_T_src[:3, 3]

    Args:
        src_points: Source XYZ coordinates (e.g., triangulated exo keypoints).
        dst_points: Target XYZ coordinates (e.g., GT ego/quest keypoints).
        allow_scaling: If True, estimate an isotropic scale factor; otherwise fix scale=1.
        eps: Small value guarding against degenerate variance.

    Returns:
        dst_T_src: 4x4 homogeneous transform representing the similarity.
    """
    if src_points.shape != dst_points.shape:
        msg: str = f"Source and destination shapes must match; got {src_points.shape} vs {dst_points.shape}"
        raise ValueError(msg)

    n_points: int = int(src_points.shape[0])
    if n_points < 3:
        msg = f"Umeyama transform requires at least 3 correspondences; received {n_points}"
        raise ValueError(msg)

    src_f64: Float[ndarray, "n 3"] = src_points.astype(np.float64, copy=False)
    dst_f64: Float[ndarray, "n 3"] = dst_points.astype(np.float64, copy=False)

    src_mean: Float[ndarray, "3"] = np.mean(src_f64, axis=0)
    dst_mean: Float[ndarray, "3"] = np.mean(dst_f64, axis=0)

    src_centered: Float[ndarray, "n 3"] = src_f64 - src_mean
    dst_centered: Float[ndarray, "n 3"] = dst_f64 - dst_mean

    covariance: Float[ndarray, "3 3"] = (dst_centered.T @ src_centered) / float(n_points)

    u: Float[ndarray, "3 3"]
    singular_vals: Float[ndarray, "3"]
    vt: Float[ndarray, "3 3"]
    u, singular_vals, vt = np.linalg.svd(covariance)

    reflection: Float[ndarray, "3"] = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection = np.array([1.0, 1.0, -1.0], dtype=np.float64)

    dst_R_src: Float[ndarray, "3 3"] = u @ np.diag(reflection) @ vt

    src_var: float = float(np.mean(np.sum(src_centered**2, axis=1)))
    if src_var <= eps:
        msg = f"Source variance too small for stable Umeyama estimation (variance={src_var})"
        raise ValueError(msg)

    scale: float = 1.0
    if allow_scaling:
        scale = float(np.sum(singular_vals * reflection) / src_var)

    dst_t_src: Float[ndarray, "3"] = dst_mean - scale * (dst_R_src @ src_mean)

    dst_T_src: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    dst_T_src[:3, :3] = scale * dst_R_src
    dst_T_src[:3, 3] = dst_t_src

    return dst_T_src.astype(np.float32)


def get_target_frame_idx(
    frame_selection: Literal["middle", "first", "last"] | int,
    total_frames: int,
) -> int:
    """Compute target frame index from selection mode."""
    if isinstance(frame_selection, int):
        return min(max(0, frame_selection), total_frames - 1)
    if frame_selection == "middle":
        return total_frames // 2
    if frame_selection == "first":
        return 0
    if frame_selection == "last":
        return total_frames - 1
    msg: str = f"Unknown frame_selection: {frame_selection}"
    raise ValueError(msg)


def create_exo_ego_blueprint(
    exo_video_log_paths: list[Path] | None = None,
    ego_video_log_paths: list[Path] | None = None,
    max_exo_videos_to_log: int = 8,
) -> rrb.Blueprint:
    """Create Rerun blueprint matching view_exoego.py style.

    Layout: 3D view with ego tabs on right, exo tabs below.
    """
    main_view: rrb.ContainerLike = rrb.Spatial3DView(
        origin="/",
        name="3D View",
        line_grid=rrb.archetypes.LineGrid3D(visible=False),
    )

    # Ego views on the right (vertical stack of tabs)
    if ego_video_log_paths is not None:
        ego_view: rrb.Vertical = rrb.Vertical(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in ego_video_log_paths
            ]
        )
        main_view = rrb.Horizontal(
            contents=[main_view, ego_view],
            column_shares=[4, 1],
        )

    # Exo views below (horizontal row of tabs)
    if exo_video_log_paths is not None:
        exo_view: rrb.Horizontal = rrb.Horizontal(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in exo_video_log_paths[:max_exo_videos_to_log]
            ]
        )
        main_view = rrb.Vertical(
            contents=[main_view, exo_view],
            row_shares=[4, 1],
        )

    blueprint: rrb.Blueprint = rrb.Blueprint(main_view, collapse_panels=True)
    return blueprint


def set_annotation_context(*, recording: rr.RecordingStream | None) -> None:
    """Set annotation context with layer-specific colors for GT and estimated keypoints."""
    keypoint_infos: list[rr.AnnotationInfo] = [
        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
    ]
    class_descriptions: list[rr.ClassDescription] = []
    for layer in (
        Coco133AnnotationLayer.GT,
        Coco133AnnotationLayer.RAW_2D,
        Coco133AnnotationLayer.TRIANGULATED_3D,
    ):
        label: str = COCO133_LAYER_LABELS[layer]
        color: tuple[int, int, int] = COCO133_LAYER_COLORS[layer]
        class_descriptions.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=int(layer), label=label, color=color),
                keypoint_annotations=keypoint_infos,
                keypoint_connections=COCO_133_LINKS,
            )
        )
    rr.log(
        "/",
        rr.AnnotationContext(class_descriptions),
        static=True,
        recording=recording,
    )


def get_frame_timestamps_from_reader(video_reader: Any) -> Int[ndarray, "num_frames"]:
    """Compute frame timestamps (ns) from video reader metadata."""
    fps: float = video_reader.fps
    frame_cnt: int = video_reader.frame_cnt
    ns_per_frame: float = 1e9 / fps
    timestamps: Int[ndarray, "num_frames"] = (np.arange(frame_cnt) * ns_per_frame).astype(np.int64)
    return timestamps


# Pre-computed upper-body + hands + face filter indices for COCO-133
_UPPER_BODY_IDX: Int[ndarray, "_"] = np.array([5, 6, 7, 8, 9, 10])
WB_UPPER_BODY_IDS: Int[ndarray, "_"] = np.concatenate(
    [_UPPER_BODY_IDX, FACE_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX]
)
"""Upper body, face, and hand keypoint indices for filtering COCO-133."""


@dataclass
class ExoOnlyCalibServiceConfig:
    """Configuration for persistent calibration service with cached models.

    This config is tyro-compatible for CLI configuration with reasonable defaults.
    Note: filter_body_idxes is not configurable via CLI (uses WB_UPPER_BODY_IDS constant).
    """

    calib_config: MultiViewCalibratorConfig = field(default_factory=MultiViewCalibratorConfig)
    """Multi-view calibration parameters (VGGT + MoGe depth refinement)."""
    tracker_config: MultiviewBodyTrackerConfig = field(default_factory=MultiviewBodyTrackerConfig)
    """Body pose tracker configuration (YOLOX + RTMPose)."""
    parent_log_path: Path = field(default_factory=lambda: Path("world"))
    """Root path for Rerun logging entity hierarchy."""


@dataclass
class ExoOnlyCalibConfig:
    """Configuration for exo-only calibration CLI.

    Composes service config with dataset and Rerun options for CLI usage.
    """

    rr_config: RerunTyroConfig
    """Rerun logging configuration (spawn viewer, save RRD, etc.)."""
    dataset: AnnotatedExoEgoDatasetUnion
    """RRD dataset to load exo sequence from."""
    service_config: ExoOnlyCalibServiceConfig = field(default_factory=ExoOnlyCalibServiceConfig)
    """Service configuration for models and calibration parameters."""
    frame_selection: Literal["middle", "first", "last"] | int = "middle"
    """Which frame to process for calibration."""
    skip_tsdf_fusion: bool = False
    """Skip TSDF mesh fusion for faster processing (no mesh output)."""
    align_to_ego: bool = True
    """Align exo cameras to ego world using Umeyama on hand keypoints."""


class ExoOnlyCalibService:
    """Persistent service holding cached calibrator and tracker models.

    Instantiate once at application startup, then call repeatedly to process
    image batches without reloading models.

    The service focuses on calibration + pose estimation + alignment.
    Dataset loading and Rerun setup are the caller's responsibility.
    """

    config: ExoOnlyCalibServiceConfig
    """Service configuration."""
    mv_calibrator: MultiViewCalibrator
    """Cached multiview calibrator (VGGT + MoGe)."""
    pose_tracker: MultiviewBodyTracker
    """Cached body pose tracker (YOLOX + RTMPose)."""

    def __init__(self, config: ExoOnlyCalibServiceConfig) -> None:
        """Load models during construction."""
        self.config = config
        print("[ExoOnlyCalibService] Loading MultiViewCalibrator...")
        self.mv_calibrator = MultiViewCalibrator(
            parent_log_path=config.parent_log_path,
            config=config.calib_config,
        )
        print("[ExoOnlyCalibService] Loading MultiviewBodyTracker...")
        self.pose_tracker = MultiviewBodyTracker(
            config.tracker_config,
            filter_body_idxes=WB_UPPER_BODY_IDS,
        )
        print("[ExoOnlyCalibService] Models loaded and ready.")

    def __call__(
        self,
        *,
        bgr_list: list[UInt8[ndarray, "H W 3"]],
        gt_xyzc: Float32[ndarray, "133 4"] | None = None,
        skip_tsdf_fusion: bool = False,
        align_to_ego: bool = True,
        recording: rr.RecordingStream | None = None,
    ) -> ExoCalibResult:
        """Execute calibration and pose estimation on a batch of BGR images.

        Args:
            bgr_list: List of BGR images from exo cameras (same frame, different views).
            gt_xyzc: Optional GT keypoints for Umeyama alignment (133 COCO keypoints with confidence).
            skip_tsdf_fusion: Skip TSDF mesh fusion for faster processing.
            align_to_ego: Align exo cameras to ego world using Umeyama (requires gt_xyzc).
            recording: Optional Rerun recording stream for model-internal logging.

        Returns:
            ExoCalibResult containing calibrated cameras, triangulated keypoints,
            and optional aligned keypoints.
        """
        timing: TimingReport = TimingReport()
        pipeline_start: float = timer()

        # Convert BGR to RGB for calibrator
        t0: float = timer()
        rgb_list: list[UInt8[ndarray, "H W 3"]] = [
            np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8) for bgr in bgr_list
        ]
        timing.frame_load_s = timer() - t0

        # =============================================
        # 1. Run multiview calibration
        # =============================================
        t0 = timer()
        mv_calib_results: MVCalibResults = self.mv_calibrator(rgb_list=rgb_list)
        pinhole_param_list: list[PinholeParameters] = mv_calib_results.pinhole_param_list
        timing.mv_calibration_s = timer() - t0
        print(f"[ExoOnlyCalibService] MV Calibration in {timing.mv_calibration_s:.2f}s")

        # Assign camera names
        for i, cam in enumerate(pinhole_param_list):
            cam.name = f"camera_{i}"

        # Get pointcloud for TSDF initialization (needed for bounds)
        pcd: o3d.geometry.PointCloud = mv_calib_results.pcd

        # =============================================
        # 2. TSDF Fusion - generates mesh for visualization
        # =============================================
        mesh_vertex_positions: Float32[ndarray, "num_vertices 3"] | None = None
        mesh_triangle_indices: Int[ndarray, "num_faces 3"] | None = None
        mesh_vertex_normals: Float32[ndarray, "num_vertices 3"] | None = None
        mesh_vertex_colors: Float32[ndarray, "num_vertices 3"] | None = None

        if not skip_tsdf_fusion and mv_calib_results.depth_list and mv_calib_results.pinhole_param_list:
            t0 = timer()
            depth_fuser: Open3DScaleInvariantFuser = Open3DScaleInvariantFuser(grid_resolution=512)
            reference_points: Float32[ndarray, "num_points 3"] = np.asarray(pcd.points, dtype=np.float32)
            depth_fuser.initialise_from_points(reference_points)
            for depth_map, pinhole_param, rgb in zip(
                mv_calib_results.depth_list,
                mv_calib_results.pinhole_param_list,
                rgb_list,
                strict=True,
            ):
                depth_fuser.fuse_frame(depth_hw=depth_map, pinhole=pinhole_param, rgb_hw3=rgb)

            gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
            gt_mesh.compute_vertex_normals()
            mesh_vertex_positions = np.asarray(gt_mesh.vertices, dtype=np.float32)
            mesh_triangle_indices = np.asarray(gt_mesh.triangles, dtype=np.int32)
            mesh_vertex_normals = np.asarray(gt_mesh.vertex_normals, dtype=np.float32)
            mesh_vertex_colors = np.asarray(gt_mesh.vertex_colors, dtype=np.float32)
            timing.tsdf_fusion_s = timer() - t0
            print(f"[ExoOnlyCalibService] TSDF Fusion in {timing.tsdf_fusion_s:.2f}s")

        # =============================================
        # 3. Body pose estimation
        # =============================================
        t0 = timer()
        mv_output: MVHistory = self.pose_tracker(
            bgr_list=bgr_list,
            pinhole_list=pinhole_param_list,
            pred_state=MVHistory(),
            recording=recording,
        )
        timing.pose_estimation_s = timer() - t0
        print(f"[ExoOnlyCalibService] Pose estimation in {timing.pose_estimation_s:.2f}s")

        # Get triangulated keypoints
        exo_xyzc: Float32[ndarray, "133 4"] = (
            mv_output.xyzc_t.astype(np.float32, copy=True)
            if mv_output.xyzc_t is not None
            else np.full((133, 4), np.nan, dtype=np.float32)
        )

        # =============================================
        # 4. Umeyama alignment (exo -> ego using hands)
        # =============================================
        aligned_xyzc: Float32[ndarray, "133 4"] | None = None
        umeyama_result: UmeyamaResult | None = None

        if align_to_ego and gt_xyzc is not None:
            # Extract hand keypoints only
            hand_idx: Int[ndarray, "_"] = np.concatenate([LEFT_HAND_IDX, RIGHT_HAND_IDX])
            exo_hands: Float32[ndarray, "n_hands 4"] = exo_xyzc[hand_idx]
            gt_hands: Float32[ndarray, "n_hands 4"] = gt_xyzc[hand_idx]

            # Find valid correspondences
            valid_mask: Bool[ndarray, "n_hands"] = (
                np.isfinite(exo_hands[:, :3]).all(axis=1)
                & np.isfinite(gt_hands[:, :3]).all(axis=1)
                & (exo_hands[:, 3] > 0.0)
                & (gt_hands[:, 3] > 0.0)
            )
            n_valid: int = int(np.count_nonzero(valid_mask))

            if n_valid >= 3:
                src_points: Float[ndarray, "n 3"] = exo_hands[valid_mask, :3].astype(np.float64)
                dst_points: Float[ndarray, "n 3"] = gt_hands[valid_mask, :3].astype(np.float64)

                # Compute Umeyama transform
                ego_T_exo: Float32[ndarray, "4 4"] = _umeyama_transform(
                    src_points=src_points,
                    dst_points=dst_points,
                    allow_scaling=True,
                )

                # Compute alignment error
                src_h: Float[ndarray, "n 4"] = np.concatenate(
                    [src_points, np.ones((n_valid, 1), dtype=np.float64)], axis=1
                )
                aligned_src: Float[ndarray, "n 3"] = (ego_T_exo @ src_h.T).T[:, :3]
                residuals: Float[ndarray, "n"] = np.linalg.norm(aligned_src - dst_points, axis=1)
                rms_error: float = float(np.sqrt(np.mean(residuals**2)))

                umeyama_result = UmeyamaResult(
                    ego_T_exo=ego_T_exo,
                    n_matches=n_valid,
                    rms_error_m=rms_error,
                )
                print(f"[ExoOnlyCalibService] Umeyama alignment: {n_valid} matches, RMS = {rms_error * 1000:.2f}mm")

                # Extract scale
                scale: float = float(np.linalg.norm(ego_T_exo[:3, 0]))
                print(f"[ExoOnlyCalibService] Umeyama scale factor: {scale:.4f}")

                # Isometric transform for cameras
                ego_R_exo: Float32[ndarray, "3 3"] = (ego_T_exo[:3, :3] / scale).astype(np.float32)
                ego_t_exo: Float32[ndarray, "3"] = ego_T_exo[:3, 3].astype(np.float32)

                # Transform all keypoints to ego world
                exo_xyz_h: Float[ndarray, "133 4"] = np.concatenate(
                    [exo_xyzc[:, :3], np.ones((133, 1), dtype=np.float32)], axis=1
                )
                aligned_xyz: Float32[ndarray, "133 3"] = (ego_T_exo @ exo_xyz_h.T).T[:, :3].astype(np.float32)
                aligned_xyzc = np.concatenate([aligned_xyz, exo_xyzc[:, 3:4]], axis=1).astype(np.float32)

                # Transform exo camera extrinsics to ego world
                for i, cam in enumerate(pinhole_param_list):
                    world_T_cam_old: Float[ndarray, "4 4"] = cam.extrinsics.world_T_cam
                    world_R_cam_new: Float[ndarray, "3 3"] = ego_R_exo @ world_T_cam_old[:3, :3]
                    world_t_cam_new: Float[ndarray, "3"] = scale * (ego_R_exo @ world_T_cam_old[:3, 3]) + ego_t_exo
                    new_extrinsics: Extrinsics = Extrinsics(
                        world_R_cam=world_R_cam_new.astype(np.float32),
                        world_t_cam=world_t_cam_new.astype(np.float32),
                    )
                    pinhole_param_list[i] = PinholeParameters(
                        name=cam.name,
                        intrinsics=cam.intrinsics,
                        extrinsics=new_extrinsics,
                        distortion=cam.distortion,
                    )

                # Transform mesh vertices to ego world
                if mesh_vertex_positions is not None:
                    n_verts: int = mesh_vertex_positions.shape[0]
                    verts_h: Float[ndarray, "n_verts 4"] = np.concatenate(
                        [mesh_vertex_positions, np.ones((n_verts, 1), dtype=np.float32)], axis=1
                    )
                    mesh_vertex_positions = (ego_T_exo @ verts_h.T).T[:, :3].astype(np.float32)

                    # Transform normals (rotation only, no translation or scaling)
                    if mesh_vertex_normals is not None:
                        mesh_vertex_normals = (ego_R_exo @ mesh_vertex_normals.T).T.astype(np.float32)
            else:
                print(f"[ExoOnlyCalibService] Not enough hand correspondences ({n_valid}) for Umeyama")

        timing.total_s = timer() - pipeline_start
        timing.print_report()

        return ExoCalibResult(
            exo_xyzc=exo_xyzc,
            aligned_xyzc=aligned_xyzc,
            gt_xyzc=gt_xyzc,
            pinhole_list=pinhole_param_list,
            frame_idx=0,  # Caller should track this
            timestamp_ns=0,  # Caller should track this
            umeyama_result=umeyama_result,
            mesh_vertex_positions=mesh_vertex_positions,
            mesh_triangle_indices=mesh_triangle_indices,
            mesh_vertex_normals=mesh_vertex_normals,
            mesh_vertex_colors=mesh_vertex_colors,
        )


def main(config: ExoOnlyCalibConfig) -> None:
    """CLI entry point for exo-only calibration.

    Handles dataset loading, Rerun setup, and delegates calibration to
    ExoOnlyCalibService with pre-loaded data.
    """

    parent_log_path: Path = config.service_config.parent_log_path
    timeline: str = "video_time"

    # Create one-shot service using pre-configured service config
    service: ExoOnlyCalibService = ExoOnlyCalibService(config.service_config)

    # Load dataset
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()
    exo_sequence_obj: object | None = exoego_sequence.exo_sequence
    if exo_sequence_obj is None:
        msg: str = "Dataset setup failed to provide an exo sequence."
        raise ValueError(msg)
    exo_sequence: BaseExoSequence = cast(BaseExoSequence, exo_sequence_obj)

    # Setup Rerun
    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context(recording=None)

    # Setup scene
    scene_setup_result: SceneSetupResult = setup_scene(
        exoego_sequence,
        parent_log_path=parent_log_path,
        timeline=timeline,
        log_ego=True,
        log_exo=True,
    )
    log_paths: LogPaths = scene_setup_result.log_paths
    shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

    # Log GT data if available
    if exoego_sequence.exoego_labels is not None:
        log_exoego_batch(
            exoego_sequence,
            parent_log_path=parent_log_path,
            timeline=timeline,
            shortest_timestamp=shortest_timestamp,
            log_ego=True,
            log_exo=False,
            log_mano=True,
        )

    if exoego_sequence.environment_mesh is not None:
        log_environment_mesh(exoego_sequence, parent_log_path)

    # Send blueprint
    exo_video_log_paths: list[Path] = log_paths.exo_video_log_paths or []
    ego_video_log_paths: list[Path] = log_paths.ego_video_log_paths or []
    blueprint: rrb.Blueprint = create_exo_ego_blueprint(
        exo_video_log_paths=exo_video_log_paths if exo_video_log_paths else None,
        ego_video_log_paths=ego_video_log_paths if ego_video_log_paths else None,
    )
    rr.send_blueprint(blueprint)

    # Extract frames
    exo_mv_reader: MultiVideoReader | TorchCodecMultiVideoReader = exo_sequence.exo_video_readers
    exo_frame_timestamps_list: list[Int[ndarray, "num_frames"]] = [
        get_frame_timestamps_from_reader(reader) for reader in exo_mv_reader.video_readers
    ]
    min_exo_ts: Int[ndarray, "num_frames"] = min(
        exo_frame_timestamps_list, key=lambda arr: int(arr.shape[0])
    )
    total_frames: int = len(min_exo_ts)
    frame_idx: int = get_target_frame_idx(config.frame_selection, total_frames)
    timestamp_ns: int = int(min_exo_ts[frame_idx])

    # Load BGR frames
    bgr_list: list[UInt8[ndarray, "H W 3"]] = []
    for reader in exo_mv_reader.video_readers:
        frame_obj: Any = reader[frame_idx]
        if frame_obj is None:
            msg = f"Missing exo frame at index {frame_idx}."
            raise ValueError(msg)
        frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame_obj, dtype=np.uint8)
        bgr_list.append(frame_array)

    # Extract GT keypoints
    gt_xyzc: Float32[ndarray, "133 4"] | None = None
    if exoego_sequence.exoego_labels is not None:
        gt_xyzc_stack: Float[ndarray, "n_frames 133 4"] = exoego_sequence.exoego_labels.xyzc_stack
        if frame_idx < len(gt_xyzc_stack):
            gt_xyzc = gt_xyzc_stack[frame_idx].astype(np.float32, copy=True)

    rr.set_time(timeline=timeline, duration=np.timedelta64(timestamp_ns, "ns"))

    # Run calibration
    result: ExoCalibResult = service(
        bgr_list=bgr_list,
        gt_xyzc=gt_xyzc,
        skip_tsdf_fusion=config.skip_tsdf_fusion,
        align_to_ego=config.align_to_ego,
    )

    # Log calibration results to Rerun
    _log_calibration_results(
        result=result,
        parent_log_path=parent_log_path,
        exo_video_log_paths=exo_video_log_paths,
        filter_body_idxes=WB_UPPER_BODY_IDS,
        conf_thresh=service.pose_tracker.config.keypoint_threshold,
    )


def _log_calibration_results(
    result: ExoCalibResult,
    parent_log_path: Path,
    exo_video_log_paths: list[Path],
    filter_body_idxes: Int[ndarray, "_"],
    conf_thresh: float,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log calibration results to Rerun."""
    # Log camera poses
    for pinhole, exo_log_path in zip(result.pinhole_list, exo_video_log_paths, strict=True):
        cam_log_path: Path = exo_log_path.parent.parent
        log_pinhole(
            camera=pinhole,
            cam_log_path=cam_log_path,
            image_plane_distance=0.1,
            static=True,
            recording=recording,
        )

    # Log aligned keypoints if available
    if result.aligned_xyzc is not None:
        aligned_conf: Float32[ndarray, "n_kpts"] = result.aligned_xyzc[:, 3]
        aligned_rgb_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
            aligned_conf[np.newaxis, :, np.newaxis]
        )
        aligned_rgb: UInt8[ndarray, "n_kpts 3"] = aligned_rgb_stack[0]

        top_half_mask: Bool[ndarray, "_"] = np.isin(np.arange(133), filter_body_idxes)
        aligned_vis_xyz: Float32[ndarray, "n_kpts 3"] = result.aligned_xyzc[:, :3].copy()
        aligned_vis_xyz[~top_half_mask, :] = np.nan
        aligned_vis_xyz[aligned_conf < conf_thresh, :] = np.nan

        rr.log(
            str(parent_log_path / "exo_aligned" / "coco133_xyz"),
            Points3DWithConfidence(
                positions=aligned_vis_xyz,
                confidences=aligned_conf,
                class_ids=int(Coco133AnnotationLayer.TRIANGULATED_3D),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=aligned_rgb,
            ),
            static=True,
            recording=recording,
        )

    # Log TSDF fused mesh if available
    if result.mesh_vertex_positions is not None and result.mesh_triangle_indices is not None:
        mesh_colors: UInt8[ndarray, "n_verts 3"] | None = None
        if result.mesh_vertex_colors is not None:
            # Convert from [0,1] float to [0,255] uint8
            mesh_colors = (result.mesh_vertex_colors * 255).astype(np.uint8)

        rr.log(
            str(parent_log_path / "tsdf_mesh"),
            rr.Mesh3D(
                vertex_positions=result.mesh_vertex_positions,
                triangle_indices=result.mesh_triangle_indices,
                vertex_normals=result.mesh_vertex_normals,
                vertex_colors=mesh_colors,
            ),
            static=True,
            recording=recording,
        )


def run_exo_only_calibration(
    config: ExoOnlyCalibConfig,
    recording: rr.RecordingStream | None = None,
) -> ExoCalibResult:
    """Execute exo-only calibration and pose estimation on a single frame.

    .. deprecated::
        Use ExoOnlyCalibService for better performance. This function
        creates a new service instance on each call, loading models from scratch.

    Args:
        config: Pipeline configuration.
        recording: Optional Rerun recording stream.

    Returns:
        ExoCalibResult containing triangulated keypoints and GT labels.
    """
    # Create one-shot service using pre-configured service config
    service: ExoOnlyCalibService = ExoOnlyCalibService(config.service_config)

    # Load dataset
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()
    exo_sequence_obj: object | None = exoego_sequence.exo_sequence
    if exo_sequence_obj is None:
        msg: str = "Dataset setup failed to provide an exo sequence."
        raise ValueError(msg)
    exo_sequence: BaseExoSequence = cast(BaseExoSequence, exo_sequence_obj)

    # Extract frames
    exo_mv_reader: MultiVideoReader | TorchCodecMultiVideoReader = exo_sequence.exo_video_readers
    exo_frame_timestamps_list: list[Int[ndarray, "num_frames"]] = [
        get_frame_timestamps_from_reader(reader) for reader in exo_mv_reader.video_readers
    ]
    min_exo_ts: Int[ndarray, "num_frames"] = min(
        exo_frame_timestamps_list, key=lambda arr: int(arr.shape[0])
    )
    total_frames: int = len(min_exo_ts)
    frame_idx: int = get_target_frame_idx(config.frame_selection, total_frames)

    # Load BGR frames
    bgr_list: list[UInt8[ndarray, "H W 3"]] = []
    for reader in exo_mv_reader.video_readers:
        frame_obj: Any = reader[frame_idx]
        if frame_obj is None:
            msg = f"Missing exo frame at index {frame_idx}."
            raise ValueError(msg)
        frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame_obj, dtype=np.uint8)
        bgr_list.append(frame_array)

    # Extract GT keypoints
    gt_xyzc: Float32[ndarray, "133 4"] | None = None
    if exoego_sequence.exoego_labels is not None:
        gt_xyzc_stack: Float[ndarray, "n_frames 133 4"] = exoego_sequence.exoego_labels.xyzc_stack
        if frame_idx < len(gt_xyzc_stack):
            gt_xyzc = gt_xyzc_stack[frame_idx].astype(np.float32, copy=True)

    # Run calibration
    return service(
        bgr_list=bgr_list,
        gt_xyzc=gt_xyzc,
        skip_tsdf_fusion=config.skip_tsdf_fusion,
        align_to_ego=config.align_to_ego,
        recording=recording,
    )
