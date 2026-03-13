"""Polycam CLI pipeline for Prompt Depth Anything.

This module adapts the upstream `prompt-da` example to the monorepo's shared
dependency stack. It loads a Polycam capture, runs PromptDA depth completion
frame-by-frame, logs both raw and completed depth to Rerun, and incrementally
fuses the predicted depths into a mesh.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8, UInt16
from monopriors.models.depth_completion.base_completion_depth import (
    CompletionDepthPrediction,
)
from monopriors.models.depth_completion.prompt_da import PromptDAPredictor
from numpy import ndarray
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm


@dataclass
class PDAPolycamConfig:
    """Runtime configuration for the Polycam PromptDA demo."""

    polycam_zip_path: Path
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    max_image_size: int = 1008
    max_depth_range_meter: float = 4.0
    depth_fusion_resolution: float = 0.04
    log_incremental_mesh: bool = True


def log_polycam_data(
    parent_path: Path,
    polycam_data: PolycamData,
    depth_pred: UInt16[ndarray, "h w"],
    rescale_factor: int = 1,
) -> None:
    """Log RGB, ARKit depth, confidence, and PromptDA depth for a frame.

    The incoming Polycam frame is rescaled for visualization only. Intrinsics
    are updated to match the logged image size so the 2D/3D Rerun views stay
    consistent.
    """

    cam_path: Path = parent_path / "cam"
    pinhole_path: Path = cam_path / "pinhole"

    rgb: UInt8[np.ndarray, "h w 3"] = polycam_data.rgb_hw3
    depth: UInt16[np.ndarray, "h w"] = polycam_data.depth_hw
    confidence: UInt8[np.ndarray, "h w"] = polycam_data.confidence_hw

    # Resize all logged images together so the pinhole stream stays aligned.
    target_height: int = rgb.shape[0] // rescale_factor
    target_width: int = rgb.shape[1] // rescale_factor
    rgb_resized = cv2.resize(rgb, (target_width, target_height))
    depth_resized = cv2.resize(depth, (target_width, target_height))
    confidence_resized = cv2.resize(confidence, (target_width, target_height))
    depth_pred_resized = cv2.resize(depth_pred, (target_width, target_height))

    # Rerun's pinhole view expects intrinsics that match the logged resolution.
    rescaled_intrinsics: Intrinsics = rescale_intri(
        camera_intrinsics=polycam_data.pinhole_params.intrinsics,
        target_height=target_height,
        target_width=target_width,
    )
    polycam_data.pinhole_params.intrinsics = rescaled_intrinsics

    log_pinhole(camera=polycam_data.pinhole_params, cam_log_path=cam_path)
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_resized).compress(jpeg_quality=75))
    rr.log(f"{pinhole_path}/confidence", rr.SegmentationImage(confidence_resized))
    rr.log(f"{pinhole_path}/arkit_depth", rr.DepthImage(depth_resized, meter=1000))
    rr.log(f"{pinhole_path}/pred_depth", rr.DepthImage(depth_pred_resized, meter=1000))


def filter_depth(
    depth_mm: UInt16[np.ndarray, "h w"],
    confidence: UInt8[np.ndarray, "h w"],
    confidence_threshold: DepthConfidenceLevel,
    max_depth_meter: float,
) -> UInt16[np.ndarray, "h w"]:
    """Remove unreliable or out-of-range depth values before TSDF fusion."""

    filtered_depth_mm: UInt16[np.ndarray, "h w"] = depth_mm.copy()
    filtered_depth_mm[confidence < confidence_threshold] = 0
    filtered_depth_mm[depth_mm > max_depth_meter * 1000] = 0
    return filtered_depth_mm


def create_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    """Create the default Rerun layout for the PromptDA Polycam demo."""

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.Spatial2DView(origin=parent_log_path / "cam" / "pinhole" / "arkit_depth"),
                rrb.Spatial2DView(origin=parent_log_path / "cam" / "pinhole" / "pred_depth"),
            ),
            column_shares=[20, 9],
        ),
        collapse_panels=True,
    )


def _log_mesh(parent_log_path: Path, pred_fuser: Open3DFuser) -> None:
    """Log the current fused mesh state to the Rerun scene."""

    pred_mesh = pred_fuser.get_mesh()
    pred_mesh.compute_vertex_normals()
    rr.log(
        f"{parent_log_path}/pred_mesh",
        rr.Mesh3D(
            vertex_positions=pred_mesh.vertices,
            triangle_indices=pred_mesh.triangles,
            vertex_normals=pred_mesh.vertex_normals,
            vertex_colors=pred_mesh.vertex_colors,
        ),
    )


def pda_polycam_inference(config: PDAPolycamConfig) -> None:
    """Run PromptDA over a Polycam capture and stream results to Rerun."""

    parent_log_path: Path = Path("world")
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    rr.send_blueprint(create_blueprint(parent_log_path))

    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)
    pred_fuser = Open3DFuser(
        fusion_resolution=config.depth_fusion_resolution,
        max_fusion_depth=config.max_depth_range_meter,
    )
    model = PromptDAPredictor(
        device="cuda",
        model_type="large",
        max_size=config.max_image_size,
    )

    polycam_data: PolycamData
    for frame_idx, polycam_data in enumerate(tqdm(polycam_dataset, desc="Inferring", total=len(polycam_dataset))):
        # Sequence time is what drives the frame scrubber in the viewer.
        rr.set_time("frame_idx", sequence=frame_idx)
        depth_pred: CompletionDepthPrediction = model(
            rgb=polycam_data.rgb_hw3,
            prompt_depth=polycam_data.original_depth_hw,
        )

        # The ARKit confidence map is a simple way to avoid fusing bad prompts.
        pred_filtered_depth_mm: UInt16[np.ndarray, "h w"] = filter_depth(
            depth_mm=depth_pred.depth_mm,
            confidence=polycam_data.confidence_hw,
            confidence_threshold=DepthConfidenceLevel.MEDIUM,
            max_depth_meter=config.max_depth_range_meter,
        )
        # We fuse the completed depth, but still log the unfiltered prediction.
        pred_fuser.fuse_frames(
            depth_hw=pred_filtered_depth_mm,
            K_33=polycam_data.pinhole_params.intrinsics.k_matrix,
            cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
            rgb_hw3=polycam_data.rgb_hw3,
        )
        log_polycam_data(
            parent_path=parent_log_path,
            polycam_data=polycam_data,
            depth_pred=depth_pred.depth_mm,
            rescale_factor=1,
        )
        if config.log_incremental_mesh:
            _log_mesh(parent_log_path=parent_log_path, pred_fuser=pred_fuser)

    # Always emit the final mesh, even when incremental mesh logging is disabled.
    _log_mesh(parent_log_path=parent_log_path, pred_fuser=pred_fuser)
