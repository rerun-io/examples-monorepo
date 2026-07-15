"""Alignment of MOV PTS with the ARKit uptime clock."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64

from arkitscenes_download.ingest.mov import MetadataPacket


@dataclass(frozen=True, slots=True)
class ClockAlignment:
    """Recovered constant mapping from MOV PTS to ARKit uptime."""

    offset_seconds: float
    """Constant added to MOV PTS."""
    max_depth_residual_seconds: float
    """Largest nearest-frame residual for low-resolution depth timestamps."""
    within_two_ms_fraction: float
    """Fraction of low-resolution depth timestamps within two milliseconds."""
    dispersion_seconds: float
    """Robust p99-p1 spread of wide per-frame clock offsets."""
    drift_seconds_per_second: float
    """Linear offset drift against MOV PTS."""
    ultrawide_offset_seconds: float
    """Offset independently recovered from ultrawide metadata."""
    ultrawide_agreement_seconds: float
    """Absolute wide/ultrawide offset difference."""


MAX_DISPERSION_SECONDS: float = 0.004
MAX_DRIFT_SECONDS_PER_SECOND: float = 5e-6
MAX_ULTRAWIDE_AGREEMENT_SECONDS: float = 0.002
MAX_DATASET_CAMERA_PHASE_SECONDS: float = 0.11
CLOCK_ACCEPTANCE_FRACTION: float = 0.99
MAX_FIRST_SAMPLE_CLAMP_SECONDS: float = 0.25


def shared_epoch(
    wide_source_times: Float64[np.ndarray, "n"], ultrawide_source_times: Float64[np.ndarray, "m"]
) -> tuple[float, int, int]:
    """Locate the first instant BOTH cameras have a frame.

    Every consumer of a sequence (ingest and the raw PromptDA tool) must agree
    on this epoch, or their layers drift apart on the shared timeline.

    Returns:
        The epoch in uptime seconds, plus the count of pre-epoch leading frames
        to drop from the wide and ultrawide streams respectively.
    """
    epoch: float = max(float(wide_source_times[0]), float(ultrawide_source_times[0]))
    wide_drop: int = int(np.count_nonzero(wide_source_times < epoch))
    ultrawide_drop: int = int(np.count_nonzero(ultrawide_source_times < epoch))
    return epoch, wide_drop, ultrawide_drop


def clamped_to_epoch(timestamps: Float64[np.ndarray, "n"], epoch: float) -> Float64[np.ndarray, "n"]:
    """Snap a stream's first post-epoch sample onto the epoch itself.

    Different sensors tick at different phases, so the first sample of a
    stream lands up to one sample period after t=0; snapping it back makes
    every view populated at the very start of the timeline. Streams starting
    genuinely late are left honest.
    """
    if len(timestamps) == 0 or timestamps[0] - epoch >= MAX_FIRST_SAMPLE_CLAMP_SECONDS:
        return timestamps
    clamped: Float64[np.ndarray, "n"] = timestamps.copy()
    clamped[0] = epoch
    return clamped


def nearest_index(timestamps: Float64[np.ndarray, "n"], timestamp: float) -> int:
    """Return the index of the sorted timestamp nearest to a query timestamp."""
    if len(timestamps) == 0:
        raise ValueError("cannot select from an empty timestamp sequence")
    insertion: int = int(np.searchsorted(timestamps, timestamp))
    candidates: list[int] = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    return min(candidates, key=lambda index: abs(float(timestamps[index]) - timestamp))


def timestamp_from_path(path: Path) -> float:
    """Parse the trailing uptime timestamp from an extracted asset filename."""
    return float(path.stem.rsplit("_", 1)[-1])


def path_timestamps(paths: list[Path]) -> Float64[np.ndarray, "n"]:
    """Parse uptime timestamps from extracted asset filenames."""
    return np.asarray([timestamp_from_path(path) for path in paths], dtype=np.float64)


def _correspondences(packets: list[MetadataPacket]) -> tuple[Float64[np.ndarray, "n"], Float64[np.ndarray, "n"]]:
    """Decode MOV PTS and embedded ARKit timestamps for one camera track."""
    metadata_times: list[float] = []
    movie_times: list[float] = []
    for packet in packets:
        if not packet.items:
            continue
        value: dict[str, object] = json.loads(packet.items[0])
        timestamp_value: dict[str, object] = value["OriginalTimestampWhenWrittenToFile"]  # type: ignore[assignment]
        metadata_times.append(float(timestamp_value["value"]) / float(timestamp_value["timescale"]))  # pyrefly: ignore  # bad-argument-type
        movie_times.append(packet.pts_seconds)
    return np.asarray(movie_times, dtype=np.float64), np.asarray(metadata_times, dtype=np.float64)


def analyze_clock_correspondences(
    wide_pts: Float64[np.ndarray, "n"],
    wide_arkit: Float64[np.ndarray, "n"],
    ultrawide_pts: Float64[np.ndarray, "m"],
    ultrawide_arkit: Float64[np.ndarray, "m"],
    max_agreement_seconds: float = MAX_ULTRAWIDE_AGREEMENT_SECONDS,
) -> tuple[float, float, float, float, float]:
    """Measure and gate two independently observed camera clock mappings."""
    offsets: Float64[np.ndarray, "n"] = wide_arkit - wide_pts
    offset_seconds: float = float(np.median(offsets))
    dispersion_seconds: float = float(np.percentile(offsets, 99.0) - np.percentile(offsets, 1.0))
    drift_seconds_per_second: float = float(np.polyfit(wide_pts - wide_pts[0], offsets, 1)[0])
    ultrawide_offset_seconds: float = float(np.median(ultrawide_arkit - ultrawide_pts))
    agreement_seconds: float = abs(ultrawide_offset_seconds - offset_seconds)
    if dispersion_seconds > MAX_DISPERSION_SECONDS:
        raise ValueError(f"wide clock offset dispersion {dispersion_seconds * 1e3:.3f} ms exceeds {MAX_DISPERSION_SECONDS * 1e3:.1f} ms")
    if abs(drift_seconds_per_second) > MAX_DRIFT_SECONDS_PER_SECOND:
        raise ValueError(f"wide clock drift {drift_seconds_per_second * 1e6:.3f} us/s exceeds {MAX_DRIFT_SECONDS_PER_SECOND * 1e6:.1f} us/s")
    if agreement_seconds > max_agreement_seconds:
        raise ValueError(f"camera clock offsets disagree by {agreement_seconds * 1e3:.3f} ms")
    return offset_seconds, dispersion_seconds, drift_seconds_per_second, ultrawide_offset_seconds, agreement_seconds


def recover_clock_from_packets(
    wide_packets: list[MetadataPacket], ultrawide_packets: list[MetadataPacket], depth_paths: list[Path]
) -> ClockAlignment:
    """Recover and gate independent wide and ultrawide clock mappings."""
    wide_pts, wide_arkit = _correspondences(wide_packets)
    ultrawide_pts, ultrawide_arkit = _correspondences(ultrawide_packets)
    offset_seconds, dispersion, drift, ultrawide_offset, agreement = analyze_clock_correspondences(
        wide_pts, wide_arkit, ultrawide_pts, ultrawide_arkit, MAX_DATASET_CAMERA_PHASE_SECONDS
    )
    frame_times: Float64[np.ndarray, "n"] = wide_pts + offset_seconds
    depth_times: Float64[np.ndarray, "m"] = path_timestamps(depth_paths)
    insertion: np.ndarray = np.searchsorted(frame_times, depth_times)
    insertion = np.clip(insertion, 1, len(frame_times) - 1)
    residuals: Float64[np.ndarray, "m"] = np.minimum(abs(depth_times - frame_times[insertion - 1]), abs(depth_times - frame_times[insertion]))
    return ClockAlignment(offset_seconds, float(residuals.max()), float(np.mean(residuals <= 0.002)), dispersion, drift, ultrawide_offset, agreement)
