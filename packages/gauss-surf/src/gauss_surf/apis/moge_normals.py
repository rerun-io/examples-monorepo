"""Infer MoGe v2 normals for chosen wide frames."""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
import torch.nn.functional as functional
from arkitscenes_download.ingest.paths import NORMALS_MOGE, PINHOLE_MOGE, TIMELINE, VIDEO_WIDE
from arkitscenes_download.ingest.recording import atomic_recording
from einops import rearrange
from jaxtyping import Float32, UInt8
from monopriors.models.moge_v2 import DEFAULT_IMAGE_HW, MoGeV2NormalOutput, MoGeV2TrtPredictor
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.camera_parameters import Intrinsics, rescale_intri
from torch import Tensor

from gauss_surf.catalog import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    SegmentReader,
    TimedeltaNs,
    _matrix_from_cell,
    _resolution_from_cell,
    register_layer,
    table_timestamps,
)
from gauss_surf.contracts import (
    FRAME_SELECTION_LAYER,
    MOGE_INFERENCE_BATCH_SIZE,
    MOGE_NORMALS_LAYER,
    WIDE_CHOSEN_SHARPNESS_COLUMN,
    WIDE_FPS,
    WIDE_INTRINSICS_COLUMN,
    WIDE_RESOLUTION_COLUMN,
)
from gauss_surf.normals_encoding import encode_normals_png, to_away_from_camera


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for one segment's MoGe normal layer."""

    video_id: str
    """ARKitScenes segment identifier whose chosen frames are inferred."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the gauss-surf development catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset to read and update."""
    output_dir: Path = Path("data/moge_normals")
    """Directory for generated per-segment MoGe normal RRDs."""


@dataclass(frozen=True, slots=True)
class ChosenFrameInputs:
    """Chosen timestamps and their per-frame MoGe-resolution intrinsics."""

    timestamps: TimedeltaNs
    """Chosen wide packet times as ``timedelta64[ns]``."""
    K_moge_n33: Float32[ndarray, "n_chosen 3 3"]
    """Wide intrinsics rescaled per axis from native size to 1008×756."""


def _load_chosen_frame_inputs(reader: SegmentReader) -> ChosenFrameInputs:
    """Read chosen times and latest-at temporal wide intrinsics from the catalog."""
    table: pa.Table = reader.chosen_table(
        WIDE_CHOSEN_SHARPNESS_COLUMN,
        (WIDE_INTRINSICS_COLUMN, WIDE_RESOLUTION_COLUMN),
    )

    timestamps_n: TimedeltaNs = table_timestamps(table)
    k_values: list[Any] = table.column(WIDE_INTRINSICS_COLUMN).to_pylist()
    resolution_values: list[Any] = table.column(WIDE_RESOLUTION_COLUMN).to_pylist()
    K_moge_n33: Float32[ndarray, "n_chosen 3 3"] = np.empty((table.num_rows, 3, 3), dtype=np.float32)
    for row_index, (k_value, resolution_value) in enumerate(zip(k_values, resolution_values, strict=True)):
        try:
            K_native_33: Float32[ndarray, "3 3"] = _matrix_from_cell(k_value, WIDE_INTRINSICS_COLUMN)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"segment {reader.video_id} chosen row {row_index} has no unambiguous wide intrinsic matrix") from error
        try:
            native_width: int
            native_height: int
            native_width, native_height = _resolution_from_cell(resolution_value, WIDE_RESOLUTION_COLUMN)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"segment {reader.video_id} chosen row {row_index} has no unambiguous wide resolution") from error
        native_intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=K_native_33,
            width=native_width,
            height=native_height,
        )
        moge_intrinsics: Intrinsics = rescale_intri(
            native_intrinsics,
            target_width=DEFAULT_IMAGE_HW[1],
            target_height=DEFAULT_IMAGE_HW[0],
        )
        assert moge_intrinsics.k_matrix is not None
        K_moge_n33[row_index] = moge_intrinsics.k_matrix.astype(np.float32, copy=False)
    return ChosenFrameInputs(timestamps=timestamps_n, K_moge_n33=K_moge_n33)


def _send_pinhole_columns(recording: rr.RecordingStream, inputs: ChosenFrameInputs) -> None:
    """Write one temporal MoGe-resolution pinhole per chosen frame."""
    frame_count: int = len(inputs.timestamps)
    resolutions_n2: Float32[ndarray, "n_chosen 2"] = np.tile(
        np.array([[DEFAULT_IMAGE_HW[1], DEFAULT_IMAGE_HW[0]]], dtype=np.float32),
        (frame_count, 1),
    )
    rr.send_columns(
        PINHOLE_MOGE,
        indexes=[rr.TimeColumn(TIMELINE, duration=inputs.timestamps)],
        columns=rr.Pinhole.columns(
            image_from_camera=inputs.K_moge_n33,
            resolution=resolutions_n2,
            camera_xyz=[rr.ViewCoordinates.RDF] * frame_count,
        ),
        recording=recording,
    )


def _infer_and_write_normals(
    reader: SegmentReader,
    inputs: ChosenFrameInputs,
    predictor: MoGeV2TrtPredictor,
    recording: rr.RecordingStream,
) -> int:
    """Decode exact chosen frames, infer batches, and stream PNG columns."""
    decoded_frames: Iterator[UInt8[Tensor, "3 h w"]] = reader.decode_frames(
        VIDEO_WIDE,
        inputs.timestamps,
        fps=WIDE_FPS,
        device=torch.device("cuda"),
    )
    inferred_frames: int = 0
    with torch.inference_mode():
        for start in range(0, len(inputs.timestamps), MOGE_INFERENCE_BATCH_SIZE):
            batch_timestamps_n: TimedeltaNs = inputs.timestamps[start : start + MOGE_INFERENCE_BATCH_SIZE]
            frames_hwc: list[UInt8[Tensor, "h w 3"]] = []
            for _timestamp in batch_timestamps_n:
                frame_chw: UInt8[Tensor, "3 h w"] = next(decoded_frames)
                frame_hwc: UInt8[Tensor, "h w 3"] = rearrange(frame_chw, "c h w -> h w c")  # pyrefly: ignore  # bad-argument-type — einops stub
                frames_hwc.append(frame_hwc)

            frames_bhw3: UInt8[Tensor, "batch h w 3"] = torch.stack(frames_hwc)
            frames_b3hw: Float32[Tensor, "batch 3 h w"] = rearrange(frames_bhw3, "b h w c -> b c h w").to(torch.float32)  # pyrefly: ignore  # bad-argument-type — einops stub
            resized_b3hw: Float32[Tensor, "batch 3 net_h=756 net_w=1008"] = functional.interpolate(
                frames_b3hw,
                size=DEFAULT_IMAGE_HW,
                mode="bilinear",
                antialias=True,
            )
            resized_bhw3: UInt8[Tensor, "batch net_h=756 net_w=1008 3"] = rearrange(
                resized_b3hw.round().clamp(0.0, 255.0).to(torch.uint8),
                "b c h w -> b h w c",
            )  # pyrefly: ignore  # bad-argument-type — einops stub
            prediction: MoGeV2NormalOutput = predictor.predict_normals(resized_bhw3)
            # MoGe emits toward-camera RDF normals; the layer stores the negated
            # away-from-camera convention — the gaussurf training target and the
            # colorization the mvsanywhere fork's figures use.
            normals_toward_bhw3: Float32[ndarray, "batch net_h=756 net_w=1008 3"] = (
                prediction.normals_bhw3.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            normals_bhw3: Float32[ndarray, "batch net_h=756 net_w=1008 3"] = np.stack(
                [to_away_from_camera(normals_toward_hw3) for normals_toward_hw3 in normals_toward_bhw3]
            )
            png_blobs: list[bytes] = [encode_normals_png(normals_hw3) for normals_hw3 in normals_bhw3]
            batch_count: int = len(png_blobs)
            rr.send_columns(
                NORMALS_MOGE,
                indexes=[rr.TimeColumn(TIMELINE, duration=batch_timestamps_n)],
                columns=rr.EncodedImage.columns(blob=png_blobs, media_type=["image/png"] * batch_count),
                recording=recording,
            )
            inferred_frames += batch_count
            print(f"inferred {inferred_frames}/{len(inputs.timestamps)} frames")
    return inferred_frames


def main(config: Config) -> None:
    """Infer chosen wide frames, publish the MoGe normals RRD, and register it."""
    started_at: float = time.perf_counter()
    if not torch.cuda.is_available():
        raise SystemExit("MoGe normal inference requires CUDA for NVDEC and TensorRT")
    reader: SegmentReader = SegmentReader.open(config.catalog_url, config.dataset_name, config.video_id)
    reader.require_layers((FRAME_SELECTION_LAYER,))
    reader.require_zero_orientation("MoGe normal inference")
    dataset: DatasetEntry = reader.dataset
    inputs: ChosenFrameInputs = _load_chosen_frame_inputs(reader)
    predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(
        heads="normal",
        network_hw_options=(DEFAULT_IMAGE_HW,),
        batch_size=MOGE_INFERENCE_BATCH_SIZE,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path: Path = config.output_dir / f"{config.video_id}.rrd"
    with atomic_recording(rrd_path, config.video_id, send_properties=False) as recording:
        _send_pinhole_columns(recording, inputs)
        inferred_frames: int = _infer_and_write_normals(reader, inputs, predictor, recording)
    if inferred_frames != len(inputs.timestamps):
        raise RuntimeError(f"inferred {inferred_frames} normal maps for {len(inputs.timestamps)} chosen timestamps")
    register_layer(dataset, rrd_path, MOGE_NORMALS_LAYER)
    wall_seconds: float = time.perf_counter() - started_at
    throughput: float = inferred_frames / wall_seconds
    print(f"rrd: {rrd_path}")
    print(f"registered layer: {MOGE_NORMALS_LAYER}")
    print(f"inferred {inferred_frames} frames in {wall_seconds:.2f}s ({throughput:.2f} img/s)")
