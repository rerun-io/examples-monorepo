"""Select sharp ARKitScenes frames and publish a catalog layer."""

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
from einops import rearrange
from jaxtyping import Float32, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import DataSource, Field, FixedRateSampling, NoShuffle, RerunIterableDataset
from simplecv.rerun_dataloader import SegmentNvdecDecoder
from torch import Tensor
from torch.utils.data import DataLoader

from gauss_surf.catalog import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, SegmentReader, register_layer
from gauss_surf.contracts import FRAME_SELECTION_LAYER, ULTRAWIDE_FPS, WIDE_FPS
from gauss_surf.frame_selection import FrameScores, SelectionResult, TimedeltaNs, score_frames, select_ultrawide_frames, select_wide_frames

FETCH_SIZE: int = 1024
"""Samples fetched per catalog request."""

ScoreArrays: TypeAlias = tuple[Float32[ndarray, "n"], Float32[ndarray, "n"]]


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for one segment's frame-selection layer."""

    video_id: str
    """ARKitScenes segment identifier to score."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the gauss-surf development catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset to read and update."""
    output_dir: Path = Path("data/frame_selection")
    """Directory for generated per-segment frame-selection RRDs."""
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


def _camera_dataset(
    dataset: DatasetEntry,
    video_id: str,
    entity_path: str,
    fps: float,
    device: torch.device,
) -> tuple[RerunIterableDataset, SegmentNvdecDecoder]:
    """Build one natural-order dataset for one camera's video packets."""
    decoder: SegmentNvdecDecoder = SegmentNvdecDecoder(dataset, entity_path, TIMELINE, device, int(fps))
    fields: dict[str, Field] = {"video": Field(f"/{entity_path}:VideoStream:sample", decode=decoder)}
    samples: RerunIterableDataset = RerunIterableDataset(
        DataSource(dataset=dataset, segments=[video_id]),
        index=TIMELINE,
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=fps),
        shuffle_strategy=NoShuffle(),  # pyrefly: ignore  # unexpected-keyword — gauss-surf runs the 0.36 prerelease API
        fetch_size=FETCH_SIZE,
    )
    return samples, decoder


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
    """Decode and score every packet from one camera entity in natural order."""
    if batch_size < 1:
        raise ValueError("score_batch_size must be positive")
    samples: RerunIterableDataset
    decoder: SegmentNvdecDecoder
    samples, decoder = _camera_dataset(dataset, video_id, entity_path, fps, device)
    loader: DataLoader = DataLoader(samples, batch_size=None, num_workers=0)
    pending_frames_hwc: list[UInt8[Tensor, "h w 3"]] = []
    sharpness_parts: list[Float32[ndarray, "batch"]] = []
    texture_parts: list[Float32[ndarray, "batch"]] = []

    row: dict[str, Tensor | None]
    with torch.inference_mode():
        for row in loader:
            video: Tensor | None = row["video"]
            if video is None:
                continue
            frame_chw: UInt8[Tensor, "3 h w"] = video
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

    if not sharpness_parts:
        raise RuntimeError(f"{entity_path} yielded no decodable frames for segment {video_id}")
    sharpness_n: Float32[ndarray, "n_frames"] = np.concatenate(sharpness_parts).astype(np.float32, copy=False)
    texture_n: Float32[ndarray, "n_frames"] = np.concatenate(texture_parts).astype(np.float32, copy=False)
    timestamps_n: TimedeltaNs = np.asarray(decoder.times, dtype="timedelta64[ns]")
    if len(sharpness_n) != len(timestamps_n):
        raise RuntimeError(
            f"{entity_path} decoded {len(sharpness_n)} frames but exposed {len(timestamps_n)} packet timestamps; "
            "the native-rate sampling grid is not one-to-one"
        )
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


def main(config: Config) -> None:
    """Score both cameras, publish an accepted selection RRD, and register it."""
    if not torch.cuda.is_available():
        raise SystemExit("frame selection requires a CUDA device for whole-segment NVDEC decoding")
    reader: SegmentReader = SegmentReader.open(config.catalog_url, config.dataset_name, config.video_id)
    reader.row()
    dataset: DatasetEntry = reader.dataset
    device: torch.device = torch.device("cuda")

    wide_measurements: CameraMeasurements = _score_camera(
        dataset,
        config.video_id,
        VIDEO_WIDE,
        WIDE_FPS,
        device,
        config.score_batch_size,
    )
    ultrawide_measurements: CameraMeasurements = _score_camera(
        dataset,
        config.video_id,
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
        raise SystemExit(f"segment {config.video_id} rejected: {wide_result.rejection_reason}; no layer written")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path: Path = config.output_dir / f"{config.video_id}.rrd"
    with atomic_recording(rrd_path, config.video_id, send_properties=False) as recording:
        _send_camera_columns(recording, SHARPNESS_WIDE, FRAME_SELECTION_WIDE, wide_result)
        _send_camera_columns(recording, SHARPNESS_ULTRAWIDE, FRAME_SELECTION_ULTRAWIDE, ultrawide_result)
    register_layer(dataset, rrd_path, FRAME_SELECTION_LAYER)
    print(f"rrd: {rrd_path}")
    print(f"registered layer: {FRAME_SELECTION_LAYER}")
