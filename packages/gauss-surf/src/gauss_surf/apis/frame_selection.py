"""Select sharp ARKitScenes frames and publish a catalog layer."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import rerun as rr
import torch
from arkitscenes_download.ingest.paths import (
    FRAME_SELECTION_ULTRAWIDE,
    FRAME_SELECTION_WIDE,
    SHARPNESS_ULTRAWIDE,
    SHARPNESS_WIDE,
    TIMELINE,
    VIDEO_ULTRAWIDE,
    VIDEO_WIDE,
)
from arkitscenes_download.ingest.recording import atomic_recording
from beartype.roar import BeartypeException
from einops import rearrange
from jaxtyping import Float32, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.rerun_dataloader import open_segment_decoder
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from gauss_surf.catalog import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, connect_catalog, register_layer
from gauss_surf.contracts import FRAME_SELECTION_LAYER, ULTRAWIDE_FPS, WIDE_FPS
from gauss_surf.frame_selection import FrameScores, SelectionResult, TimedeltaNs, score_frames, select_ultrawide_frames, select_wide_frames

ScoreArrays: TypeAlias = tuple[Float32[ndarray, "n"], Float32[ndarray, "n"]]


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for one or many segments' frame-selection layers."""

    video_id: str | None = None
    """Single ARKitScenes segment identifier to score."""
    video_ids_file: Path | None = None
    """Newline-separated segment identifiers for batch mode; mutually exclusive with ``video_id``."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the gauss-surf development catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset to read and update."""
    output_dir: Path = Path("data/frame_selection")
    """Directory for generated per-segment frame-selection RRDs (``<output_dir>/<video_id>.rrd``)."""
    force: bool = False
    """Regenerate and re-register segments whose output RRD already exists."""
    window_size: int = 6
    """Number of consecutive wide frames in each strict non-overlapping window."""
    min_chosen: int = 200
    """Minimum wide winners required to publish the segment layer."""
    score_batch_size: int = 32
    """Number of decoded CUDA frames scored together."""


@dataclass(frozen=True, slots=True)
class CameraMeasurements:
    """Dense measurements decoded from one camera stream."""

    timestamps: TimedeltaNs
    """Packet timestamp per scored frame, as ``timedelta64[ns]``."""
    sharpness: Float32[ndarray, "n_frames"]
    """Normalized sharpness score per frame."""
    texture: Float32[ndarray, "n_frames"]
    """Grayscale standard deviation per frame."""


def _score_frame_batch(frames_hwc: list[UInt8[Tensor, "h w 3"]]) -> ScoreArrays:
    """Stack and score one same-resolution decoded frame batch."""
    frames_bhw3: UInt8[Tensor, "b h w 3"] = torch.stack(frames_hwc)
    scores: FrameScores = score_frames(frames_bhw3)
    sharpness_b: Float32[ndarray, "b"] = scores.sharpness.detach().cpu().numpy().astype(np.float32, copy=False)
    texture_b: Float32[ndarray, "b"] = scores.texture.detach().cpu().numpy().astype(np.float32, copy=False)
    return sharpness_b, texture_b


def _score_camera(
    dataset: DatasetEntry,
    video_id: str,
    entity_path: str,
    fps: float,
    device: torch.device,
    batch_size: int,
) -> CameraMeasurements:
    """Decode and score every video packet from one camera entity in packet order.

    The packet timeline itself drives iteration, so every packet is scored exactly
    once and ``timestamps`` matches ``sharpness`` by construction — unlike the
    previous fixed-rate sampling grid, whose latest-at resampling could duplicate
    or skip packets whenever the stream's packet spacing was irregular.
    """
    if batch_size < 1:
        raise ValueError("score_batch_size must be positive")
    timestamps_n: TimedeltaNs
    decoder: VideoDecoder
    timestamps_n, _samples, _keyframes, decoder = open_segment_decoder(dataset, video_id, entity_path, TIMELINE, device, int(fps))
    if len(timestamps_n) == 0:
        raise RuntimeError(f"{entity_path} has no video packets for segment {video_id}")
    if np.any(np.diff(timestamps_n.astype(np.int64)) <= 0):
        raise RuntimeError(f"{entity_path} packet timestamps are not strictly increasing for segment {video_id}")

    pending_frames_hwc: list[UInt8[Tensor, "h w 3"]] = []
    sharpness_parts: list[Float32[ndarray, "batch"]] = []
    texture_parts: list[Float32[ndarray, "batch"]] = []
    with torch.inference_mode():
        for frame_index in range(len(timestamps_n)):
            frame_chw: UInt8[Tensor, "3 h w"] = decoder.get_frame_at(frame_index).data
            frame_hwc: UInt8[Tensor, "h w 3"] = rearrange(frame_chw, "c h w -> h w c")  # pyrefly: ignore  # bad-argument-type — einops stub
            pending_frames_hwc.append(frame_hwc)
            if len(pending_frames_hwc) < batch_size:
                continue
            batch_scores: ScoreArrays = _score_frame_batch(pending_frames_hwc)
            sharpness_parts.append(batch_scores[0])
            texture_parts.append(batch_scores[1])
            pending_frames_hwc = []
        if pending_frames_hwc:
            final_scores: ScoreArrays = _score_frame_batch(pending_frames_hwc)
            sharpness_parts.append(final_scores[0])
            texture_parts.append(final_scores[1])

    sharpness_n: Float32[ndarray, "n_frames"] = np.concatenate(sharpness_parts).astype(np.float32, copy=False)
    texture_n: Float32[ndarray, "n_frames"] = np.concatenate(texture_parts).astype(np.float32, copy=False)
    assert len(sharpness_n) == len(timestamps_n), f"{entity_path} scored {len(sharpness_n)} of {len(timestamps_n)} packets"
    return CameraMeasurements(timestamps=timestamps_n, sharpness=sharpness_n, texture=texture_n)


def _send_camera_columns(
    recording: rr.RecordingStream,
    sharpness_entity: str,
    chosen_entity: str,
    result: SelectionResult,
) -> None:
    """Write one camera's dense scores and sparse selection rows."""
    rr.send_columns(
        sharpness_entity,
        indexes=[rr.TimeColumn(TIMELINE, duration=result.timestamps)],
        columns=rr.Scalars.columns(scalars=result.sharpness),
        recording=recording,
    )
    chosen_sharpness_n: Float32[ndarray, "n_chosen"] = result.sharpness[result.chosen_indices]
    rr.send_columns(
        chosen_entity,
        indexes=[rr.TimeColumn(TIMELINE, duration=result.chosen_timestamps)],
        columns=rr.AnyValues.columns(
            sharpness=chosen_sharpness_n,
            window=result.windows,
            rule=list(result.rules),
        ),
        recording=recording,
    )


def _print_summary(camera_name: str, result: SelectionResult) -> None:
    """Print one camera's scored/chosen counts and sharpness distribution."""
    score_min: float = float(np.min(result.sharpness))
    score_median: float = float(np.median(result.sharpness))
    score_max: float = float(np.max(result.sharpness))
    print(
        f"{camera_name}: scored={len(result.sharpness)} chosen={len(result.chosen_indices)} "
        f"sharpness[min/median/max]={score_min:.2f}/{score_median:.2f}/{score_max:.2f}"
    )


def process_segment(dataset: DatasetEntry, config: Config, video_id: str, device: torch.device) -> Path:
    """Score both cameras of one segment, write its RRD, and register the layer.

    Returns:
        The written layer RRD path.
    """
    wide_measurements: CameraMeasurements = _score_camera(
        dataset,
        video_id,
        VIDEO_WIDE,
        WIDE_FPS,
        device,
        config.score_batch_size,
    )
    ultrawide_measurements: CameraMeasurements = _score_camera(
        dataset,
        video_id,
        VIDEO_ULTRAWIDE,
        ULTRAWIDE_FPS,
        device,
        config.score_batch_size,
    )
    wide_result: SelectionResult = select_wide_frames(
        wide_measurements.sharpness,
        wide_measurements.timestamps,
        window_size=config.window_size,
        min_chosen=config.min_chosen,
    )
    ultrawide_result: SelectionResult = select_ultrawide_frames(
        ultrawide_measurements.sharpness,
        ultrawide_measurements.timestamps,
    )
    _print_summary("wide", wide_result)
    _print_summary("ultrawide", ultrawide_result)
    if wide_result.rejected:
        raise RuntimeError(f"segment {video_id} rejected: {wide_result.rejection_reason}; no layer written")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path: Path = config.output_dir / f"{video_id}.rrd"
    with atomic_recording(rrd_path, video_id, send_properties=False) as recording:
        _send_camera_columns(recording, SHARPNESS_WIDE, FRAME_SELECTION_WIDE, wide_result)
        _send_camera_columns(recording, SHARPNESS_ULTRAWIDE, FRAME_SELECTION_ULTRAWIDE, ultrawide_result)
    register_layer(dataset, rrd_path, FRAME_SELECTION_LAYER)
    return rrd_path


def main(config: Config) -> None:
    """Run frame selection for one segment or a batch queue with skip-if-exists semantics."""
    if (config.video_id is None) == (config.video_ids_file is None):
        raise SystemExit("provide exactly one of --video-id or --video-ids-file")
    if not torch.cuda.is_available():
        raise SystemExit("frame selection requires a CUDA device for whole-segment NVDEC decoding")
    if config.video_id is not None:
        video_ids: list[str] = [config.video_id]
    else:
        assert config.video_ids_file is not None  # enforced by the exclusive-arg check above
        video_ids = [line.strip() for line in config.video_ids_file.read_text().splitlines() if line.strip()]
    client: CatalogClient = connect_catalog(config.catalog_url, config.dataset_name)
    dataset: DatasetEntry = client.get_dataset(config.dataset_name)
    device: torch.device = torch.device("cuda")

    completed: int = 0
    skipped: int = 0
    failed: list[str] = []
    for video_id in video_ids:
        rrd_path: Path = config.output_dir / f"{video_id}.rrd"
        if rrd_path.is_file() and not config.force:
            skipped += 1
            print(f"SKIP {video_id}: {rrd_path} exists (use --force to regenerate)", flush=True)
            continue
        start: float = time.perf_counter()
        try:
            process_segment(dataset, config, video_id, device)
        except BeartypeException:
            raise
        except Exception as error:
            failed.append(video_id)
            print(f"FAIL {video_id}: {type(error).__name__}: {error}", flush=True)
            if config.video_id is not None:
                raise
            continue
        completed += 1
        print(f"DONE {video_id}: {time.perf_counter() - start:.1f}s ({completed} done, {skipped} skipped, {len(failed)} failed)", flush=True)

    print(f"frame-selection batch: {completed} done, {skipped} skipped, {len(failed)} failed of {len(video_ids)}")
    if failed:
        print("failed segments: " + ", ".join(failed))
