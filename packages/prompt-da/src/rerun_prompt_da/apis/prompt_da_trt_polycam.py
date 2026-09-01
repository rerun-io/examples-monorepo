"""Batched TensorRT Polycam pipeline for Prompt Depth Anything.

The TensorRT twin of :mod:`rerun_prompt_da.apis.prompt_da_polycam`: frames are
batched (default 8) through a dynamic-batch fp16 engine with all pre/post
processing on the GPU, then fused and logged to Rerun exactly like the torch
demo. Prints model and end-to-end throughput so speedups are measurable.
"""

import time
from dataclasses import dataclass, field
from itertools import batched, chain
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float32, UInt8, UInt16
from numpy import ndarray
from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from tqdm import tqdm

from rerun_prompt_da.apis.prompt_da_polycam import _log_mesh, create_blueprint, filter_depth, log_polycam_data
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor, network_image_hw


@dataclass
class PDATrtPolycamConfig:
    """Runtime configuration for the batched TensorRT Polycam demo."""

    polycam_zip_path: Path
    """Polycam capture zip (or extracted directory) to process."""
    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Rerun viewer/save/connect behavior."""
    batch_size: int = 8
    """Frames per TensorRT batch (engine profile optimum and max)."""
    max_image_size: int = 1008
    """Longest network image side; the actual size is 14-aligned from the capture resolution."""
    max_depth_range_meter: float = 4.0
    """Depth beyond this is dropped before TSDF fusion."""
    depth_fusion_resolution: float = 0.04
    """TSDF voxel size in meters."""
    log_incremental_mesh: bool = False
    """Log the fused mesh after every batch instead of only at the end."""




def pda_trt_polycam_inference(config: PDATrtPolycamConfig) -> None:
    """Run batched TensorRT PromptDA over a Polycam capture and stream to Rerun."""
    parent_log_path: Path = Path("world")
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    rr.send_blueprint(create_blueprint(parent_log_path))

    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=config.polycam_zip_path)
    pred_fuser = Open3DFuser(
        fusion_resolution=config.depth_fusion_resolution,
        max_fusion_depth=config.max_depth_range_meter,
    )

    # Consume the capture in batch-sized chunks so memory stays bounded on
    # long recordings; only the first batch is peeked to size the engine.
    batches = batched(polycam_dataset, config.batch_size)
    first_batch: tuple[PolycamData, ...] | None = next(batches, None)
    if first_batch is None:
        raise ValueError(f"Polycam capture {config.polycam_zip_path} contains no frames.")
    image_hw: tuple[int, int] = network_image_hw(first_batch[0].rgb_hw3.shape[:2], config.max_image_size)
    predictor = PromptDATrtPredictor(model_type="large", image_hw=image_hw, batch_size=config.batch_size)

    n_frames: int = 0
    model_seconds: float = 0.0
    wall_start: float = time.perf_counter()
    total_batches: int = -(-len(polycam_dataset) // config.batch_size)
    progress = tqdm(chain([first_batch], batches), desc="Inferring (TRT)", total=total_batches)
    batch: tuple[PolycamData, ...]
    for batch in progress:
        batch_start: int = n_frames
        n_frames += len(batch)
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack([data.rgb_hw3 for data in batch]))
        prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(
            np.stack([data.original_depth_hw for data in batch]).astype(np.float32) / 1000.0
        )

        torch.cuda.synchronize()
        model_start: float = time.perf_counter()
        depth_bhw: Float32[Tensor, "b h w"] = predictor(rgb_bhw3.cuda(), prompt_bhw.cuda())
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - model_start

        depth_mm_bhw: UInt16[ndarray, "b h w"] = (depth_bhw.cpu().numpy() * 1000).astype(np.uint16)
        for frame_offset, polycam_data in enumerate(batch):
            frame_idx: int = batch_start + frame_offset
            rr.set_time("frame_idx", sequence=frame_idx)
            depth_pred_mm: UInt16[ndarray, "h w"] = depth_mm_bhw[frame_offset]

            filtered_depth_mm: UInt16[ndarray, "h w"] = filter_depth(
                depth_mm=depth_pred_mm,
                confidence=polycam_data.confidence_hw,
                confidence_threshold=DepthConfidenceLevel.MEDIUM,
                max_depth_meter=config.max_depth_range_meter,
            )
            k_matrix: Float32[ndarray, "3 3"] | None = polycam_data.pinhole_params.intrinsics.k_matrix
            if k_matrix is None:
                raise ValueError("Polycam pinhole intrinsics must include a 3x3 k_matrix for TSDF fusion.")
            pred_fuser.fuse_frames(
                depth_hw=filtered_depth_mm,
                K_33=k_matrix,
                cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
                rgb_hw3=polycam_data.rgb_hw3,
            )
            log_polycam_data(
                parent_path=parent_log_path,
                polycam_data=polycam_data,
                depth_pred=depth_pred_mm,
                rescale_factor=1,
            )
        if config.log_incremental_mesh:
            _log_mesh(parent_log_path=parent_log_path, pred_fuser=pred_fuser)

    _log_mesh(parent_log_path=parent_log_path, pred_fuser=pred_fuser)

    wall_seconds: float = time.perf_counter() - wall_start
    print(
        f"[prompt-da] {n_frames} frames | model {model_seconds / n_frames * 1000:.1f} ms/frame "
        f"({n_frames / model_seconds:.1f} FPS) | pipeline {wall_seconds / n_frames * 1000:.1f} ms/frame "
        f"({n_frames / wall_seconds:.1f} FPS incl. fusion+logging)"
    )
