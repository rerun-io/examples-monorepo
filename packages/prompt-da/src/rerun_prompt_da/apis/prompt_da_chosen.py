"""Run PromptDA only at frame-selection's chosen wide timestamps.

Unlike :mod:`prompt_da_arkitscenes`, which infers on a fixed ~10 Hz sampling
grid, this module reads the segment's registered ``frame_selection`` layer and
completes depth exactly at the chosen (sharpest-per-window) wide packet
timestamps. The layer therefore aligns one-to-one with the frames every
downstream gauss-surf stage trains on, and its timeline carries real packet
timestamps instead of synthesized grid slots. Everything after batch assembly
is shared with the grid tool (:func:`run_segment`).
"""

import functools
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import torch
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL
from arkitscenes_download.ingest.cells import component_instance, landscape_quarter_turns, portrait_from_segment_row
from arkitscenes_download.ingest.layer_batch import LayerBatchSummary, run_layer_batch, segment_ids_from_selection
from arkitscenes_download.ingest.paths import CONFIDENCE, DEPTH, FRAME_SELECTION_WIDE, PINHOLE_WIDE, RIG, TIMELINE, VIDEO_WIDE
from arkitscenes_download.ingest.timestamps import match_exact_timestamps, table_timestamps
from datafusion import col
from jaxtyping import Float64, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from simplecv.rerun_dataloader import open_segment_decoder
from torchcodec.decoders import VideoDecoder

from rerun_prompt_da.apis.arkitscenes_shared import NATIVE_FPS, connect_catalog, world_t_cam_from_pose
from rerun_prompt_da.apis.prompt_da_arkitscenes import (
    PROMPTDA_LAYER,
    SegmentResult,
    run_segment,
)
from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.promptda_stream import PromptDABatch, assemble_promptda_batch
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

CHOSEN_SHARPNESS_COLUMN: str = f"/{FRAME_SELECTION_WIDE}:sharpness"
"""Sparse chosen-wide-frame component written by gauss-surf frame selection."""
DEPTH_BLOB_COLUMN: str = f"/{DEPTH}:EncodedDepthImage:blob"
CONFIDENCE_COLUMN: str = f"/{CONFIDENCE}:SegmentationImage:buffer"
INTRINSICS_COLUMN: str = f"/{PINHOLE_WIDE}:Pinhole:image_from_camera"
POSE_TRANSLATION_COLUMN: str = f"/{RIG}:Transform3D:translation"
POSE_QUATERNION_COLUMN: str = f"/{RIG}:Transform3D:quaternion"


@dataclass(frozen=True, slots=True)
class PDAChosenConfig:
    """Configuration for chosen-frame PromptDA over one segment or a batch queue."""

    video_id: str | None = None
    """Single catalog segment to process."""
    video_ids_file: Path | None = None
    """Newline-separated segment identifiers for batch mode; mutually exclusive with ``video_id``."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = "arkitscenes-v2"
    """Catalog dataset to read segments from and register layers on."""
    output_dir: Path = Path("data/promptda")
    """Directory for generated layer RRDs (``<output_dir>/<video_id>.rrd``, corpus layer-major naming)."""
    force: bool = False
    """Regenerate and re-register segments whose output RRD already exists."""
    batch_size: int = 8
    """Frames per TensorRT batch (the engine profile's max/opt batch)."""
    max_image_size: int = 1008
    """Largest image dimension accepted by the PromptDA predictor."""
    max_depth_range_meter: float = 4.0
    """Maximum predicted depth retained for TSDF fusion, in metres."""
    depth_fusion_resolution: float = 0.015
    """TSDF voxel resolution in metres."""
    register: bool = True
    """Register each completed RRD as a replacement PromptDA layer."""


def chosen_batches(
    dataset: DatasetEntry,
    video_id: str,
    quarter_turns: int,
    batch_size: int,
) -> Iterator[PromptDABatch]:
    """Yield inference batches for exactly the segment's chosen wide packets.

    Reads the chosen timestamps with latest-at prompt depth, confidence,
    intrinsics, and pose in one catalog query, then decodes those packets
    (and only those) from the whole-segment NVDEC decoder.
    """
    view = dataset.filter_segments(video_id).filter_contents(
        [f"/{FRAME_SELECTION_WIDE}", f"/{DEPTH}", f"/{CONFIDENCE}", f"/{PINHOLE_WIDE}", f"/{RIG}"]
    )
    if CHOSEN_SHARPNESS_COLUMN not in set(view.arrow_schema().names):
        raise RuntimeError(f"segment {video_id} has no registered frame_selection layer (column {CHOSEN_SHARPNESS_COLUMN!r} absent)")
    table: pa.Table = (
        view.reader(index=TIMELINE, fill_latest_at=True)
        .filter(col(f'"{CHOSEN_SHARPNESS_COLUMN}"').is_not_null())
        .select(TIMELINE, DEPTH_BLOB_COLUMN, CONFIDENCE_COLUMN, INTRINSICS_COLUMN, POSE_TRANSLATION_COLUMN, POSE_QUATERNION_COLUMN)
        .sort(TIMELINE)
        .to_arrow_table()
    )
    if table.num_rows == 0:
        raise RuntimeError(f"segment {video_id} frame_selection layer contains no chosen wide rows")

    device: torch.device = torch.device("cuda")
    decoder: VideoDecoder
    packet_times_n, _, _, decoder = open_segment_decoder(dataset, video_id, VIDEO_WIDE, TIMELINE, device, int(NATIVE_FPS))
    packet_indices_n: ndarray = match_exact_timestamps(packet_times_n, table_timestamps(table))

    for batch_start in range(0, table.num_rows, batch_size):
        # Convert one batch at a time: the PNG blob cells are large as Python lists.
        batch_rows: list[dict[str, Any]] = table.slice(batch_start, batch_size).to_pylist()
        packet_indices: list[int] = [int(i) for i in packet_indices_n[batch_start : batch_start + batch_size]]
        prompts_hw: list[UInt16[ndarray, "ph pw"]] = []
        for row in batch_rows:
            blob_u8 = np.asarray(component_instance(row[DEPTH_BLOB_COLUMN], DEPTH_BLOB_COLUMN), dtype=np.uint8)
            prompt_hw = cv2.imdecode(blob_u8, cv2.IMREAD_UNCHANGED)
            if prompt_hw is None or prompt_hw.dtype != np.uint16:
                raise RuntimeError(f"segment {video_id}: prompt depth blob failed to decode as uint16 PNG")
            prompts_hw.append(prompt_hw)
        world_T_cam_44: list[Float64[ndarray, "4 4"]] = [
            world_t_cam_from_pose(
                np.asarray(component_instance(row[POSE_TRANSLATION_COLUMN], POSE_TRANSLATION_COLUMN), dtype=np.float64),
                np.asarray(component_instance(row[POSE_QUATERNION_COLUMN], POSE_QUATERNION_COLUMN), dtype=np.float64),
            )
            for row in batch_rows
        ]
        yield assemble_promptda_batch(
            packet_indices,
            [int(row[TIMELINE].value) for row in batch_rows],
            [decoder.get_frame_at(i).data for i in packet_indices],
            [torch.from_numpy(prompt_hw) for prompt_hw in prompts_hw],
            [
                torch.from_numpy(np.asarray(component_instance(row[CONFIDENCE_COLUMN], CONFIDENCE_COLUMN), dtype=np.uint8).reshape(prompt_hw.shape))
                for row, prompt_hw in zip(batch_rows, prompts_hw, strict=True)
            ],
            # Rerun stores Pinhole image_from_camera flattened column-major.
            [np.asarray(component_instance(row[INTRINSICS_COLUMN], INTRINSICS_COLUMN), dtype=np.float32).reshape(3, 3).T for row in batch_rows],
            world_T_cam_44,
            device=device,
            quarter_turns=quarter_turns,
        )


def main(config: PDAChosenConfig) -> None:
    """Run chosen-frame PromptDA for one segment or a batch queue with skip-if-exists semantics."""
    video_ids: list[str] = segment_ids_from_selection(config.video_id, config.video_ids_file)
    client: CatalogClient = connect_catalog(config.catalog_url, config.dataset_name)
    dataset: DatasetEntry = client.get_dataset(config.dataset_name)
    segment_table: pa.Table = pa.Table.from_batches(
        dataset.filter_segments(video_ids)
        .segment_table()
        .select(
            "rerun_segment_id",
            "property:capture:orientation",
            "property:capture:orientation_quarter_turns_ccw",
        )
        .collect()
    )
    row_by_id: dict[str, dict[str, Any]] = {str(row["rerun_segment_id"]): row for row in segment_table.to_pylist()}

    @functools.cache
    def predictor() -> PromptDATrtPredictor:
        # Built on first use so a fully-skipped queue never loads the engine.
        return PromptDATrtPredictor(model_type="large", image_hw=network_image_hw((1440, 1920), config.max_image_size), batch_size=config.batch_size)

    def generate(video_id: str, rrd_path: Path) -> str:
        row: dict[str, Any] = row_by_id[video_id]
        portrait: bool = portrait_from_segment_row(row)
        quarter_turns: int = landscape_quarter_turns(row)
        result: SegmentResult = run_segment(
            chosen_batches(dataset, video_id, quarter_turns, config.batch_size),
            video_id,
            portrait,
            rrd_path,
            predictor(),
            max_depth_range_meter=config.max_depth_range_meter,
            depth_fusion_resolution=config.depth_fusion_resolution,
        )
        return f"{result.inferred_frames} frames"

    def register(_video_id: str, rrd_path: Path) -> None:
        dataset.register([rrd_path.resolve().as_uri()], layer_name=PROMPTDA_LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()

    summary: LayerBatchSummary = run_layer_batch(
        video_ids,
        lambda video_id: config.output_dir / f"{video_id}.rrd",
        generate,
        register if config.register else None,
        force=config.force,
        label="promptda-chosen",
    )
    if summary.failed:
        raise SystemExit(1)
