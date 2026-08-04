"""One-time VRS → AV1 MP4 conversion for HOT3D Aria sequences.

Reads each VRS file with pyvrs, decodes JPEG frames per camera stream,
re-encodes to AV1 MP4 using PyAV, and extracts calibration from the
MPS online_calibration.jsonl.

Usage:
    pixi run preprocess-hot3d --root /mnt/nas/datasets/hot3d/aria
    pixi run preprocess-hot3d --root /mnt/nas/datasets/hot3d/aria --sequence P0003_c701bd11
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvrs
from tqdm import tqdm
from turbojpeg import TurboJPEG

from simplecv.configs.dataset_paths import HOT3D_ROOT
from simplecv.data.hot3d_utils import (
    ARIA_STREAM_ID_TO_LABEL,
    QUEST_STREAM_ID_TO_LABEL,
    Hot3dSequenceCalibration,
    detect_headset,
    parse_camera_models_json,
    parse_online_calibration_first,
    save_calibration,
)
from simplecv.video_encoder import MP4Writer, VideoCodecChoice

# Shared TurboJPEG instance (thread-safe)
_TJ: TurboJPEG = TurboJPEG()

# Default VRS stream IDs per headset type
ARIA_STREAM_IDS: list[str] = ["214-1", "1201-1", "1201-2"]
QUEST_STREAM_IDS: list[str] = ["1201-1", "1201-2"]

# Output filenames per stream label (shared across headsets)
STREAM_LABEL_TO_FILENAME: dict[str, str] = {
    "camera-rgb": "rgb.mp4",
    "camera-slam-left": "slam_left.mp4",
    "camera-slam-right": "slam_right.mp4",
}

# Combined stream ID → label mapping
ALL_STREAM_ID_TO_LABEL: dict[str, str] = {**ARIA_STREAM_ID_TO_LABEL, **QUEST_STREAM_ID_TO_LABEL}

OUTPUT_DIR_NAME: str = "_simplecv"


def _default_streams_for_headset(headset: str) -> list[str]:
    """Return default VRS stream IDs for the given headset type."""
    return QUEST_STREAM_IDS if headset == "Quest3" else ARIA_STREAM_IDS


@dataclass
class PreprocessConfig:
    """Configuration for HOT3D VRS preprocessing."""

    root: Path = HOT3D_ROOT / "aria"
    """Root directory containing sequence folders."""
    sequence: str = ""
    """Process a single sequence (empty = all sequences with recording.vrs)."""
    num_decode_workers: int = 8
    """Number of parallel JPEG decode threads."""
    skip_existing: bool = True
    """Skip sequences that already have _simplecv/ output."""
    streams: list[str] = field(default_factory=list)
    """VRS stream IDs to extract. Empty = auto-detect from metadata.json."""


def decode_jpeg_to_yuv(jpeg_bytes: bytes) -> list[np.ndarray]:
    """Decode JPEG bytes to YUV420 planes using TurboJPEG (fastest path)."""
    return _TJ.decode_to_yuv_planes(jpeg_bytes)


def extract_stream_to_mp4(
    vrs_path: Path,
    stream_id: str,
    output_path: Path,
    num_workers: int,
) -> list[int]:
    """Extract a single VRS image stream to AV1 MP4.

    Returns list of frame timestamps in nanoseconds.
    """
    # pyvrs API follows rerun-io/examples-monorepo/packages/pyvrs-viewer patterns
    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))

    assert stream_id in reader.stream_ids, f"Stream {stream_id} not found in {vrs_path}. Available: {reader.stream_ids}"
    assert reader.might_contain_images(stream_id), f"Stream {stream_id} does not contain images"

    info: dict = reader.get_stream_info(stream_id)
    n_frames: int = info["data_records_count"]
    label: str = ALL_STREAM_ID_TO_LABEL.get(stream_id, stream_id)

    # Filter to data records for this stream (pyvrs filtered iteration pattern)
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")

    # ── Phase 1: Read JPEG frames from VRS ──────────────────────────────
    t_read_start: float = time.perf_counter()
    jpeg_frames: list[bytes] = []
    timestamps_ns: list[int] = []

    for record in tqdm(filtered, total=n_frames, desc=f"Reading {label}", leave=False):
        if record.n_image_blocks == 0:
            continue

        # image_blocks[0] is a 1D uint8 ndarray of raw JPEG bytes (pyvrs convention)
        jpeg_bytes: bytes = record.image_blocks[0].tobytes()
        timestamp_sec: float = float(record.timestamp)
        timestamp_ns: int = int(timestamp_sec * 1e9)

        jpeg_frames.append(jpeg_bytes)
        timestamps_ns.append(timestamp_ns)

    t_read_elapsed: float = time.perf_counter() - t_read_start

    if not jpeg_frames:
        print(f"  [WARN] No image frames found in stream {stream_id}")
        return []

    # ── Phase 2+3: Overlapped decode + encode ───────────────────────────
    # Parallel JPEG→YUV decode feeds directly into NVENC encode.
    # ThreadPoolExecutor.map() returns a lazy iterator — decode runs ahead
    # while the main thread encodes, overlapping CPU decode with GPU encode.
    t_pipeline_start: float = time.perf_counter()

    # Calculate FPS from timestamps
    if len(timestamps_ns) > 1:
        dt_ns: float = float(timestamps_ns[-1] - timestamps_ns[0]) / (len(timestamps_ns) - 1)
        fps: float = 1e9 / dt_ns
    else:
        fps = 30.0

    n_frames: int = len(jpeg_frames)
    width: int = 0
    height: int = 0
    writer: MP4Writer = MP4Writer(output_path, codec=VideoCodecChoice.AV1, fps=fps)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for planes in tqdm(pool.map(decode_jpeg_to_yuv, jpeg_frames), total=n_frames, desc=f"Decode+Encode {label}", leave=False):
            if width == 0:
                height, width = planes[0].shape
            if len(planes) >= 3:
                writer.write_yuv_planes(planes[0], planes[1], planes[2])
            else:
                writer.write_yuv_planes(planes[0])
    writer.close()

    t_pipeline_elapsed: float = time.perf_counter() - t_pipeline_start
    t_total: float = t_read_elapsed + t_pipeline_elapsed

    encoder_name: str = writer.encoder_name
    print(
        f"  {label}: {n_frames} frames, {width}x{height}, {fps:.1f}fps → {output_path.name} "
        f"[{encoder_name}] ({t_total:.1f}s total: read {t_read_elapsed:.1f}s, decode+encode {t_pipeline_elapsed:.1f}s)"
    )
    return timestamps_ns


def preprocess_sequence(seq_dir: Path, config: PreprocessConfig) -> None:
    """Preprocess a single HOT3D sequence (Aria or Quest 3)."""
    vrs_path: Path = seq_dir / "recording.vrs"
    if not vrs_path.exists():
        print(f"  [SKIP] No recording.vrs in {seq_dir}")
        return

    # Auto-detect headset type and select streams
    headset: str = detect_headset(seq_dir)
    streams: list[str] = config.streams if config.streams else _default_streams_for_headset(headset)

    output_dir: Path = seq_dir / OUTPUT_DIR_NAME
    if config.skip_existing and output_dir.exists():
        expected_files: list[str] = ["calibration.json", "timestamps_ns.json"]
        for sid in streams:
            label: str = ALL_STREAM_ID_TO_LABEL.get(sid, sid)
            expected_files.append(STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4"))
        if all((output_dir / f).exists() for f in expected_files):
            print(f"  [SKIP] Already preprocessed: {seq_dir.name}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    t_seq_start: float = time.perf_counter()
    print(f"  Headset: {headset}, streams: {streams}")

    # Extract calibration — prefer MPS online_calibration (Aria), fall back to camera_models.json (Quest)
    cal_jsonl: Path = seq_dir / "mps" / "slam" / "online_calibration.jsonl"
    cam_models_path: Path = seq_dir / "camera_models.json"
    if cal_jsonl.exists():
        # Aria: MPS-refined per-timestamp intrinsics (most accurate source)
        cal: Hot3dSequenceCalibration = parse_online_calibration_first(cal_jsonl)
    elif cam_models_path.exists():
        # Quest 3 (or Aria without MPS): factory calibration
        cal = parse_camera_models_json(cam_models_path)
    else:
        print("  [WARN] No calibration source found, skipping")
        return
    save_calibration(cal, output_dir / "calibration.json")
    print(f"  Calibration: {len(cal.streams)} streams extracted")

    # Extract video streams
    all_timestamps: dict[str, list[int]] = {}
    for stream_id in streams:
        label: str = ALL_STREAM_ID_TO_LABEL.get(stream_id, stream_id)
        filename: str = STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4")
        output_path: Path = output_dir / filename

        timestamps: list[int] = extract_stream_to_mp4(
            vrs_path=vrs_path,
            stream_id=stream_id,
            output_path=output_path,
            num_workers=config.num_decode_workers,
        )
        all_timestamps[label] = timestamps

    # Save timestamps
    ts_path: Path = output_dir / "timestamps_ns.json"
    ts_path.write_text(json.dumps(all_timestamps))

    t_seq_elapsed: float = time.perf_counter() - t_seq_start
    print(f"  Done in {t_seq_elapsed:.1f}s ({len(streams)} streams)")


def main(config: PreprocessConfig) -> None:
    """Preprocess HOT3D VRS files."""
    root: Path = config.root
    assert root.exists(), f"Root directory not found: {root}"

    if config.sequence:
        # Single sequence
        seq_dir: Path = root / config.sequence
        assert seq_dir.exists(), f"Sequence not found: {seq_dir}"
        print(f"Processing: {config.sequence}")
        preprocess_sequence(seq_dir, config)
    else:
        # All sequences with recording.vrs
        seq_dirs: list[Path] = sorted([d for d in root.iterdir() if d.is_dir() and (d / "recording.vrs").exists()])
        print(f"Found {len(seq_dirs)} sequences with recording.vrs")
        for i, seq_dir in enumerate(seq_dirs):
            print(f"\n[{i + 1}/{len(seq_dirs)}] {seq_dir.name}")
            preprocess_sequence(seq_dir, config)

    print("\nPreprocessing complete.")
