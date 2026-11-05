"""
Text-prompt SAM3 multiview body demo for HoCap/ExoEgo datasets.

This module provides multiview 3D body mesh fitting across exocentric cameras:
- Uses SAM3 video inference for person detection/tracking.
- Runs the SAM 3D Body role (posekit ``TopDownBodyPose3d``) on each camera to produce per-view body meshes.
- **Phase 2**: Uses multiview optimization to produce a single fused mesh.
- Logs results to Rerun for interactive 3D visualization.

Assumptions:
- Single person in the multiview capture (multi-person support in future).

Moved from ``packages/sam3d-body`` (``sam3d-body-mv-body-demo``).

Usage:
    pixi run -e posekit-body --frozen posekit-body3d-mv-body

This module holds the CLI configuration and the per-frame orchestration; the parts live in
``posekit.body_mv_sequence`` (exo rig), ``posekit.body_mv_inference`` (SAM3 + SAM 3D Body per camera),
``posekit.ops.multiview_body`` (optimizer), ``posekit.body_mv_fusion`` (fused body and reprojection checks) and
``posekit.body_mv_rerun`` (Rerun layout and logging).

See Also:
    - body3d_mv_video.py: Multiview SAM3 segmentation (no body fitting)
    - body3d.py: Single-image 3D body estimation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoSample
from simplecv.rerun_log_utils import RerunTyroConfig
from tqdm import tqdm

from posekit.body_mv_fusion import FusedFrame, fuse_frame, print_frame_summary
from posekit.body_mv_inference import CameraSegmentation, Sam3MVBodyModelConfig, Sam3VideoSegmenter, first_person_body
from posekit.body_mv_rerun import log_camera_outputs, log_exo_videos, log_fallback_mesh, log_fused_frame, log_keypoints_2d, log_static_scene
from posekit.body_mv_sequence import ExoRig, exo_camera_names, load_exo_rig
from posekit.models import AnnotatedBodyPose3dConfig
from posekit.models.sam3d_body import MhrBodyModel, Sam3dBodyConfig, Sam3dBodyPredictor
from posekit.ops.multiview_body import MultiviewBodyOptimizer, MultiviewOptimizerConfig
from sam3d_body.sam_3d_body_estimator import FinalPosePrediction


@dataclass(slots=True)
class Sam3MVBodyDemoConfig:
    """CLI options for multiview SAM3 body mesh fitting with Rerun."""

    dataset: AnnotatedExoEgoDatasetUnion
    """ExoEgo dataset configuration (Tyro-resolvable)."""
    prompt: str = "person"
    """Text concept to segment across all cameras."""
    max_frames: int | None = None
    """Optional cap on the number of frames to process."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/runtime options for Rerun (window layout, recording, etc.)."""
    model_config: Sam3MVBodyModelConfig = field(default_factory=Sam3MVBodyModelConfig)
    """Checkpoint, device, and dtype settings for the SAM3 video model."""
    body: AnnotatedBodyPose3dConfig = field(default_factory=Sam3dBodyConfig)
    """SAM 3D Body weights and mode."""


def _intrinsics(sample: ExoEgoSample, cam_idx: int) -> Float32[ndarray, "3 3"] | None:
    """``K`` of one exo camera of the sample, or None when the sample lacks it."""
    assert sample.exo_cam_params_list is not None
    cam_params = sample.exo_cam_params_list[cam_idx]
    if cam_params is None or cam_params.intrinsics.k_matrix is None:
        return None
    return cam_params.intrinsics.k_matrix.astype(np.float32)


def main(cfg: Sam3MVBodyDemoConfig) -> None:
    """
    Run multiview SAM3 body mesh fitting across exocentric cameras.

    This function:
    1. Loads the ExoEgo dataset and creates a shared SAM3 video session.
    2. Initializes the SAM3DBodyEstimator for per-camera body fitting.
    3. For each timestep, processes all exo cameras through detection + body estimation.
    4. Transforms meshes to world coordinates and logs visualizations to Rerun.

    Args:
        cfg: Demo configuration with dataset, prompt, and model settings.

    Raises:
        RuntimeError: If dataset yields zero frames.
    """
    sequence: BaseExoEgoSequence = cfg.dataset.setup()

    if len(sequence) == 0:
        raise RuntimeError("Dataset yielded zero canonical frames.")

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype: torch.dtype = dtype_map[cfg.model_config.dtype] if device.type == "cuda" else torch.float32

    segmenter: Sam3VideoSegmenter = Sam3VideoSegmenter(cfg.model_config, cfg.prompt, device, dtype)

    # Load SAM3DBody model for body mesh fitting
    body: Sam3dBodyPredictor = cfg.body.setup()
    body_faces: Int[ndarray, "n_faces 3"] = body.faces.cpu().numpy()

    parent_log_path: Path = Path("world")
    log_static_scene(sequence, cfg.prompt, exo_camera_names(sequence))
    video_timestamps: dict[str, Int[ndarray, "n_frames"]] = log_exo_videos(sequence, parent_log_path)

    total_frames: int = len(sequence)
    if cfg.max_frames is not None:
        total_frames = min(total_frames, cfg.max_frames)

    # Projection matrices for all valid cameras (once, since cameras are static)
    rig: ExoRig = load_exo_rig(sequence[0])
    if rig.num_views < 2:
        print("[warn] Need at least 2 valid cameras for multiview optimization. Falling back to per-camera mode.")
    n_views: int = rig.num_views

    # Initialize multiview optimizer (reused across frames for temporal smoothing)
    optimizer: MultiviewBodyOptimizer | None = None
    mhr_head: MhrBodyModel | None = None
    if n_views >= 2:
        mhr_head = body.mhr_model
        optimizer = MultiviewBodyOptimizer(
            projection_matrices=rig.projection_matrices,
            first_world_T_cam=rig.world_T_cams[0],
            mhr_head=mhr_head,
            config=MultiviewOptimizerConfig(num_iterations=30, learning_rate=0.01),
            device=str(device),
        )

    # Use no_grad for inference (allows enable_grad override for optimization)
    with torch.no_grad():
        # Timing accumulators
        time_inference: float = 0.0
        time_triangulation: float = 0.0
        time_optimization: float = 0.0
        time_validation: float = 0.0
        time_logging: float = 0.0

        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            sample: ExoEgoSample = sequence[frame_idx]

            if sample.exo_bgr_list is None or sample.exo_cam_params_list is None:
                continue

            # ===== Phase 1: Collect data from all cameras =====
            t_phase1_start: float = time.perf_counter()
            per_view_predictions: list[FinalPosePrediction] = []
            per_view_keypoints_2d: list[Float32[ndarray, "n_kpts 3"]] = []  # [u, v, conf]

            for cam_idx in rig.cam_indices:
                cam_params = sample.exo_cam_params_list[cam_idx]
                if cam_params is None:
                    continue

                frame_rgb: UInt8[ndarray, "h w 3"] = cv2.cvtColor(sample.exo_bgr_list[cam_idx], cv2.COLOR_BGR2RGB)
                segmentation: CameraSegmentation = segmenter.segment(frame_rgb)
                pinhole_path: Path = parent_log_path / "exo" / cam_params.name / "pinhole"

                # Log 2D outputs for each camera (video is logged as asset, just overlays here)
                cam_timestamp_ns: int | None = None
                if cam_params.name in video_timestamps and frame_idx < len(video_timestamps[cam_params.name]):
                    cam_timestamp_ns = int(video_timestamps[cam_params.name][frame_idx])
                log_camera_outputs(pinhole_path=str(pinhole_path), frame_idx=frame_idx, segmentation=segmentation, timestamp_ns=cam_timestamp_ns)

                # Run body estimation (single person assumption: take first detection)
                pred: FinalPosePrediction | None = first_person_body(body, frame_rgb, segmentation, cam_params.intrinsics.k_matrix)
                if pred is not None:
                    per_view_predictions.append(pred)

                    # Collect 2D keypoints with confidence=1 for detected keypoints
                    kpts_2d: Float32[ndarray, "n_kpts 2"] = pred.pred_keypoints_2d.astype(np.float32)
                    conf: Float32[ndarray, "n_kpts 1"] = np.ones((kpts_2d.shape[0], 1), dtype=np.float32)
                    per_view_keypoints_2d.append(np.concatenate([kpts_2d, conf], axis=-1))
                    log_keypoints_2d(pinhole_path, frame_idx, kpts_2d)

            # ===== Phase 2: Multiview optimization =====
            t_phase1_end: float = time.perf_counter()
            time_inference += t_phase1_end - t_phase1_start

            if optimizer is not None and mhr_head is not None and len(per_view_predictions) >= 2:
                # Stack 2D keypoints: [n_views, n_kpts, 3]
                keypoints_2d_stack: Float32[ndarray, "n_views n_kpts 3"] = np.stack(per_view_keypoints_2d, axis=0)

                # Triangulation timing
                t_triang_start: float = time.perf_counter()

                # Run optimizer (need to enable grad for optimization inside inference_mode context)
                t_opt_start: float = time.perf_counter()
                with torch.enable_grad():
                    result = optimizer.optimize_frame(
                        initial_predictions=per_view_predictions,
                        keypoints_2d_per_view=keypoints_2d_stack,
                    )
                t_opt_end: float = time.perf_counter()
                time_triangulation += t_opt_start - t_triang_start  # includes stacking
                time_optimization += t_opt_end - t_opt_start

                # ===== Phase 3: Fused mesh (first prediction as base + optimized pose) and validation =====
                view_intrinsics: list[Float32[ndarray, "3 3"] | None] = [
                    _intrinsics(sample, rig.cam_indices[view_idx]) for view_idx in range(len(per_view_predictions))
                ]
                fused: FusedFrame = fuse_frame(
                    mhr_head=mhr_head,
                    result=result,
                    per_view_predictions=per_view_predictions,
                    keypoints_2d_stack=keypoints_2d_stack,
                    view_intrinsics=view_intrinsics,
                    projection_matrices=rig.projection_matrices,
                    first_world_T_cam=rig.world_T_cams[0],
                    device=device,
                )
                log_fused_frame(frame_idx, fused, body_faces, rig.cam_names)

                # Print summary for this frame
                if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                    print_frame_summary(frame_idx, fused, n_views, per_view_predictions[0], view_intrinsics[0])

            elif len(per_view_predictions) > 0:
                # Fallback: log first camera's prediction if not enough views
                first: FinalPosePrediction = per_view_predictions[0]
                log_fallback_mesh(
                    frame_idx,
                    first.pred_vertices.astype(np.float32),
                    first.pred_cam_t.astype(np.float32),
                    rig.world_T_cams[0] if rig.world_T_cams else np.eye(4, dtype=np.float32),
                    body_faces,
                )

    # Print timing summary
    total_time: float = time_inference + time_triangulation + time_optimization + time_validation + time_logging
    print(f"\n[timing] Total: {total_time:.2f}s for {total_frames} frames ({total_frames/total_time:.1f} FPS)")
    print(f"  Inference:     {time_inference:.2f}s ({time_inference/total_time*100:.1f}%)")
    print(f"  Triangulation: {time_triangulation:.2f}s ({time_triangulation/total_time*100:.1f}%)")
    print(f"  Optimization:  {time_optimization:.2f}s ({time_optimization/total_time*100:.1f}%)")
    print(f"  Validation:    {time_validation:.2f}s ({time_validation/total_time*100:.1f}%)")
    print(f"  Logging:       {time_logging:.2f}s ({time_logging/total_time*100:.1f}%)")
    print(f"[done] Processed {total_frames} frames with multiview optimization across {n_views} cameras.")


__all__ = [
    "Sam3MVBodyModelConfig",
    "Sam3MVBodyDemoConfig",
    "main",
]
