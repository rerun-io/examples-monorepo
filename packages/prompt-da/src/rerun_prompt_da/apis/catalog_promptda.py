"""Stream PromptDA depth completion over catalog segments, straight to the viewer.

The Rerun dataloader drives everything: one iterable dataset per segment carries
the AV1 video (NVDEC-decoded on the GPU), the LiDAR prompt depth, and the
pose/calibration columns; a stateless collate stacks them into batches for the
TensorRT engine. Nothing is written back to the catalog — the sibling tool
``prompt_da_arkitscenes`` owns the registered ``promptda`` layer.
"""

from dataclasses import dataclass, field

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torch.multiprocessing
from arkitscenes_download.ingest.blueprint import DEPTH_RANGE_MM
from arkitscenes_download.ingest.paths import (
    CAM_WIDE,
    DEPTH,
    PINHOLE_WIDE,
    PINHOLE_WIDE_LOWRES,
    RIG,
    TIMELINE,
    VIDEO_WIDE,
    WORLD,
)
from einops import rearrange
from jaxtyping import Float32, Float64, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rerun_prompt_da.apis.arkitscenes_shared import ARKITSCENES_DATASET
from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.promptda_stream import PromptDABatch, PromptDACollate, promptda_dataset
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor, preprocess_batch

CLOUD_CATALOG_URL: str = "rerun://api.latest.cloud.rerun.io:443"
PINHOLE_PROMPTDA: str = f"{CAM_WIDE}/pinhole_promptda"
"""Pinhole at the network's output resolution — viewer-only, so it stays out of ingest's paths.py.

The viewer back-projects a DepthImage with its parent pinhole's intrinsics, and
the prediction matches neither the native frame nor the LiDAR prompt.
"""
DEPTH_PROMPTDA_STREAM: str = f"{PINHOLE_PROMPTDA}/depth"
NATIVE_CAPTURE_HW: tuple[int, int] = (1440, 1920)
"""ARKitScenes wide-camera resolution; the engine wants one static network size for every segment."""


@dataclass(frozen=True)
class CatalogDataConfig:
    """ARKitScenes Rerun-catalog input configuration."""

    catalog_url: str = CLOUD_CATALOG_URL
    """Catalog server URL."""
    dataset_name: str = ARKITSCENES_DATASET
    """Catalog dataset entry name."""
    segments: tuple[str, ...] = ("42899799",)
    """Segment ids to process sequentially; empty runs every registered segment."""


@dataclass
class Config:
    """Streaming PromptDA inference configuration."""

    rr_config: RerunTyroConfig
    """Rerun viewer, connection, and save behavior."""
    data: CatalogDataConfig = field(default_factory=CatalogDataConfig)
    """Catalog server, dataset, and segment selection."""
    target_fps: float = 10.0
    """Inference rate; the dataloader's sampling grid is built at this rate."""
    batch_size: int = 8
    """Frames per TensorRT batch (the engine profile's max/opt batch)."""
    max_image_size: int = 1008
    """Largest image dimension accepted by the PromptDA predictor."""
    device: str = "cuda"
    """Device the decoded frames and the engine run on."""


def stream_blueprint(portrait: bool) -> rrb.Blueprint:
    """Lay out the RGB frame, the LiDAR prompt, and the prediction beside a 3D view."""
    depth_range = rr.DepthImage.from_fields(depth_range=DEPTH_RANGE_MM)
    return rrb.Blueprint(
        rrb.Horizontal(
            # Both depth images back-project into this view and land on top of each other.
            # Start with the coarse LiDAR prompt hidden so the prediction reads cleanly;
            # it stays in the view tree, one toggle away, for comparing the two.
            rrb.Spatial3DView(name="world", origin=WORLD, overrides={f"/{DEPTH}": rrb.EntityBehavior(visible=False)}),
            rrb.Vertical(
                rrb.Spatial2DView(name="RGB", origin=PINHOLE_WIDE, contents=["$origin/video"]),
                rrb.Spatial2DView(
                    name="prompt (ARKit LiDAR)", origin=PINHOLE_WIDE_LOWRES, overrides={f"/{DEPTH}": depth_range}
                ),
                rrb.Spatial2DView(
                    name="PromptDA", origin=PINHOLE_PROMPTDA, overrides={f"/{DEPTH_PROMPTDA_STREAM}": depth_range}
                ),
            ),
            # Portrait frames are tall, so the image column needs more width.
            column_shares=[2.0, 1.5] if portrait else [2.0, 1.0],
        ),
        collapse_panels=True,
    )


def log_pinhole(
    entity_path: str, K_native_33: Float32[ndarray, "3 3"], native_hw: tuple[int, int], target_hw: tuple[int, int]
) -> None:
    """Log intrinsics rescaled from the native frame to an image of ``target_hw``."""
    K_33: Float32[ndarray, "3 3"] = K_native_33.astype(np.float32).copy()
    K_33[0] *= target_hw[1] / native_hw[1]
    K_33[1] *= target_hw[0] / native_hw[0]
    rr.log(
        entity_path,
        rr.Pinhole(image_from_camera=K_33, width=target_hw[1], height=target_hw[0], camera_xyz=rr.ViewCoordinates.RDF),
    )


def process_segment(dataset: DatasetEntry, segment_id: str, config: Config, predictor: PromptDATrtPredictor) -> None:
    """Stream one segment through the dataloader, the engine, and the viewer."""
    device: torch.device = torch.device(config.device)
    samples, video_decoder = promptda_dataset(dataset, segment_id, config.target_fps, device)
    # num_workers stays 0: the NVDEC decoder is CUDA-stateful and the collate tracks
    # its position on the sampling grid, so both need natural order in this process.
    loader: DataLoader = DataLoader(
        samples, batch_size=config.batch_size, num_workers=0, collate_fn=PromptDACollate(samples, device)
    )
    grid_slots: int = samples.sample_index.total_samples
    first_batch: bool = True
    batch: PromptDABatch | None
    for batch in tqdm(
        loader, desc=segment_id, unit="batch", total=(grid_slots + config.batch_size - 1) // config.batch_size
    ):
        if batch is None:
            continue

        image_b3hw: Float32[Tensor, "b 3 net_h net_w"]
        prompt_b1hw: Float32[Tensor, "b 1 prompt_h=192 prompt_w=256"]
        image_b3hw, prompt_b1hw = preprocess_batch(batch.rgb_bhw3, batch.prompt_bhw, predictor.image_hw)
        depth_b1hw: Float32[Tensor, "b 1 net_h net_w"] = predictor.runtime({
            "image": image_b3hw,
            "prompt_depth": prompt_b1hw,
        })["depth"]
        # Undo the collate's landscape rotation so the prediction sits in the same frame
        # as the relayed video and the stored intrinsics.
        depth_bhw: Float32[Tensor, "b net_h net_w"] = rearrange(depth_b1hw, "b 1 h w -> b h w")
        depth_mm_bhw: UInt16[Tensor, "b pred_h pred_w"] = torch.rot90(
            (depth_bhw * 1000.0).to(torch.uint16), -batch.quarter_turns, dims=(1, 2)
        )
        predicted_mm_bhw: UInt16[ndarray, "b pred_h pred_w"] = depth_mm_bhw.cpu().numpy()

        if first_batch:
            first_batch = False
            assert video_decoder.samples is not None and video_decoder.keyframes is not None
            rr.log(WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.send_blueprint(stream_blueprint(portrait=batch.quarter_turns % 2 == 1))
            # The decoder already fetched every AV1 packet, so relay them once as a
            # VideoStream layer instead of re-encoding decoded frames.
            rr.log(VIDEO_WIDE, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
            rr.send_columns(
                VIDEO_WIDE,
                indexes=[rr.TimeColumn(TIMELINE, duration=video_decoder.times)],
                columns=rr.VideoStream.columns(sample=video_decoder.samples, is_keyframe=video_decoder.keyframes),
            )
        row: int
        for row, timestamp_ns in enumerate(batch.timestamps_ns):
            rr.set_time(TIMELINE, duration=np.timedelta64(timestamp_ns, "ns"))
            world_T_cam_44: Float64[ndarray, "4 4"] = batch.world_T_cam_b44[row]
            rr.log(RIG, rr.Transform3D(mat3x3=world_T_cam_44[:3, :3], translation=world_T_cam_44[:3, 3]))
            K_native_33: Float32[ndarray, "3 3"] = batch.K_native_b33[row]
            prompt_mm_hw: UInt16[ndarray, "stored_prompt_h stored_prompt_w"] = batch.prompt_mm_bhw[row]
            predicted_mm_hw: UInt16[ndarray, "pred_h pred_w"] = predicted_mm_bhw[row]
            log_pinhole(PINHOLE_WIDE, K_native_33, batch.stored_hw, batch.stored_hw)
            log_pinhole(PINHOLE_WIDE_LOWRES, K_native_33, batch.stored_hw, prompt_mm_hw.shape)
            log_pinhole(PINHOLE_PROMPTDA, K_native_33, batch.stored_hw, predicted_mm_hw.shape)
            rr.log(DEPTH, rr.DepthImage(prompt_mm_hw, meter=1000.0))
            rr.log(DEPTH_PROMPTDA_STREAM, rr.DepthImage(predicted_mm_hw, meter=1000.0))


def main(config: Config) -> None:
    """Stream every requested catalog segment through PromptDA and into the viewer."""
    # Rerun's tokio runtime is not fork-safe, so the dataloader warns unless the start
    # method is spawn — set before any dataset is built, as the upstream snippet does.
    # This pipeline keeps num_workers=0 regardless (see process_segment).
    torch.multiprocessing.set_start_method("spawn", force=True)
    client: CatalogClient = CatalogClient(config.data.catalog_url)
    dataset: DatasetEntry = client.get_dataset(name=config.data.dataset_name)
    segment_ids: list[str] = list(config.data.segments) or dataset.segment_ids()
    predictor: PromptDATrtPredictor = PromptDATrtPredictor(
        model_type="large",
        image_hw=network_image_hw(NATIVE_CAPTURE_HW, config.max_image_size),
        batch_size=config.batch_size,
    )

    segment_id: str
    for segment_id in segment_ids:
        process_segment(dataset, segment_id, config, predictor)
