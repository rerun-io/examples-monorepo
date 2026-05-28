"""One-time VRS → AV1 MP4 conversion for Aria Gen2 Pilot sequences.

Aria Gen2 encodes video streams as H.265 inside VRS (unlike Gen1 which uses
JPEG). All streams (RGB + SLAM) use monochrome (gray8) H.265 Rext profile
which neither Rerun nor NVDEC can decode, so we transcode to yuv420p AV1
via ffmpeg + NVENC (~4s per 10k-frame SLAM stream, ~8s for RGB).

Usage:
    pixi run preprocess-aria-gen2-pilot --root /mnt/8tb/data/aria-gen2-pilot
    pixi run preprocess-aria-gen2-pilot --root /mnt/8tb/data/aria-gen2-pilot --sequence walk_1
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyvrs
from tqdm import tqdm

from simplecv.data.hot3d_utils import (
    AriaSequenceCalibration,
    parse_online_calibration_first,
    save_calibration,
)
from simplecv.video_encoder import ffmpeg_transcode, pick_ffmpeg_encoder

# ── Aria Gen2 stream mapping ──────────────────────────────────────────── #
# Gen2 has 5 cameras vs Gen1's 3, with different SLAM camera labels.

ARIA_GEN2_STREAM_ID_TO_LABEL: dict[str, str] = {
    "214-1": "camera-rgb",
    "1201-1": "slam-front-left",
    "1201-2": "slam-front-right",
    "1201-3": "slam-side-left",
    "1201-4": "slam-side-right",
}

ARIA_GEN2_STREAM_LABEL_TO_FILENAME: dict[str, str] = {
    "camera-rgb": "rgb.mp4",
    "slam-front-left": "slam_front_left.mp4",
    "slam-front-right": "slam_front_right.mp4",
    "slam-side-left": "slam_side_left.mp4",
    "slam-side-right": "slam_side_right.mp4",
}

ARIA_GEN2_STREAM_IDS: list[str] = ["214-1", "1201-1", "1201-2", "1201-3", "1201-4"]

OUTPUT_DIR_NAME: str = "_simplecv"
VRS_FILENAME: str = "video.vrs"


@dataclass
class PreprocessConfig:
    """Configuration for Aria Gen2 Pilot VRS preprocessing."""

    root: Path = Path("/mnt/8tb/data/aria-gen2-pilot")
    """Root directory containing sequence folders."""
    sequence: str = ""
    """Process a single sequence (empty = all sequences with video.vrs)."""
    skip_existing: bool = True
    """Skip sequences that already have _simplecv/ output."""
    streams: list[str] = field(default_factory=list)
    """VRS stream IDs to extract. Empty = default Aria Gen2 streams."""


# ── VRS helpers ───────────────────────────────────────────────────────── #


def _stream_vrs_to_file(
    vrs_path: Path, stream_id: str, label: str, dest: Path,
) -> list[int]:
    """Stream raw image blocks from VRS directly to a file, collecting timestamps.

    Writes encoded blocks as they're read instead of buffering in memory,
    keeping peak RAM bounded regardless of stream size.
    """
    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))
    info: dict = reader.get_stream_info(stream_id)
    n_frames: int = info["data_records_count"]
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")

    timestamps_ns: list[int] = []
    with open(dest, "wb") as f:
        for rec in tqdm(filtered, total=n_frames, desc=f"Reading {label}", leave=False):
            if rec.n_image_blocks > 0:
                f.write(rec.image_blocks[0].tobytes())
                timestamps_ns.append(int(rec.timestamp * 1e9))
    return timestamps_ns


def _get_vrs_stream_dimensions(vrs_path: Path, stream_id: str) -> tuple[int, int]:
    """Read image (width, height) from VRS stream image_spec."""
    reader: pyvrs.SyncVRSReader = pyvrs.SyncVRSReader(str(vrs_path))
    filtered = reader.filtered_by_fields(stream_ids=stream_id, record_types="data")
    for rec in filtered:
        if rec.image_specs:
            spec = rec.image_specs[0]
            return spec.width, spec.height
    raise ValueError(f"Could not determine dimensions for stream {stream_id}")


# ── Stream processing ─────────────────────────────────────────────────── #


def _fps_from_timestamps(timestamps_ns: list[int]) -> int:
    """Compute integer FPS from nanosecond timestamps."""
    if len(timestamps_ns) > 1:
        dt_ns: float = float(timestamps_ns[-1] - timestamps_ns[0]) / (len(timestamps_ns) - 1)
        return max(1, round(1e9 / dt_ns))
    return 30


# NVENC CQ for SLAM streams (512x512 grayscale).  Higher = smaller file.
# SLAM frames have low entropy; CQ 40 gives ~17MB vs ~85MB at NVENC default.
# RGB uses None (NVENC default) which already produces good quality at ~88MB.
_CQ_SLAM: int = 40


def transcode_h265_stream_to_mp4(
    vrs_path: Path,
    stream_id: str,
    output_path: Path,
    encoder: str,
) -> list[int]:
    """Extract H.265 NAL units from VRS and transcode to yuv420p AV1 MP4.

    The VRS stores monochrome (gray8) H.265 which Rerun cannot decode
    (H.265 Rext profile) and NVDEC cannot decode (no Rext support).
    CPU H.265 decode + NVENC AV1 encode via :func:`ffmpeg_transcode`.

    Returns list of VRS timestamps in nanoseconds.
    """
    label: str = ARIA_GEN2_STREAM_ID_TO_LABEL.get(stream_id, stream_id)
    cq: int | None = None if label == "camera-rgb" else _CQ_SLAM

    # Stream VRS directly to temp file (no in-memory buffering)
    t0: float = time.perf_counter()
    tmp_h265: Path = Path(tempfile.mktemp(suffix=".h265"))
    timestamps_ns: list[int] = _stream_vrs_to_file(vrs_path, stream_id, label, tmp_h265)
    t_read: float = time.perf_counter() - t0

    if not timestamps_ns:
        print(f"  [WARN] No frames in stream {stream_id}")
        tmp_h265.unlink(missing_ok=True)
        return []

    # Transcode via ffmpeg
    t1: float = time.perf_counter()
    ffmpeg_transcode(
        input_path=tmp_h265,
        output_path=output_path,
        input_format="hevc",
        fps=_fps_from_timestamps(timestamps_ns),
        encoder=encoder,
        cq=cq,
    )
    tmp_h265.unlink()

    t_transcode: float = time.perf_counter() - t1
    print(
        f"  {label}: {len(timestamps_ns)} frames → {output_path.name} "
        f"[{encoder}, cq={cq}] ({t_read + t_transcode:.1f}s: read {t_read:.1f}s, transcode {t_transcode:.1f}s)"
    )
    return timestamps_ns


# ── Sequence-level orchestration ──────────────────────────────────────── #


def preprocess_sequence(seq_dir: Path, config: PreprocessConfig) -> None:
    """Preprocess a single Aria Gen2 Pilot sequence."""
    vrs_path: Path = seq_dir / VRS_FILENAME
    if not vrs_path.exists():
        print(f"  [SKIP] No {VRS_FILENAME} in {seq_dir}")
        return

    streams: list[str] = config.streams if config.streams else ARIA_GEN2_STREAM_IDS

    output_dir: Path = seq_dir / OUTPUT_DIR_NAME
    if config.skip_existing and output_dir.exists():
        expected_files: list[str] = ["calibration.json", "timestamps_ns.json"]
        for sid in streams:
            label: str = ARIA_GEN2_STREAM_ID_TO_LABEL.get(sid, sid)
            expected_files.append(ARIA_GEN2_STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4"))
        if all((output_dir / f).exists() for f in expected_files):
            print(f"  [SKIP] Already preprocessed: {seq_dir.name}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    t_seq_start: float = time.perf_counter()
    print(f"  Streams: {streams}")

    # ── Calibration (patch dimensions from VRS image specs) ───────────────
    cal_jsonl: Path = seq_dir / "mps" / "slam" / "online_calibration.jsonl"
    if not cal_jsonl.exists():
        print("  [WARN] No online_calibration.jsonl found, skipping")
        return

    cal: AriaSequenceCalibration = parse_online_calibration_first(cal_jsonl)
    for sid in streams:
        label = ARIA_GEN2_STREAM_ID_TO_LABEL.get(sid, sid)
        try:
            w, h = _get_vrs_stream_dimensions(vrs_path, sid)
        except ValueError:
            continue
        for stream_cal in cal.streams:
            if stream_cal.stream_label == label:
                stream_cal.width, stream_cal.height = w, h

    save_calibration(cal, output_dir / "calibration.json")
    print(f"  Calibration: {len(cal.streams)} streams, dims patched from VRS")

    # ── Video streams ─────────────────────────────────────────────────────
    encoder: str = pick_ffmpeg_encoder()
    all_timestamps: dict[str, list[int]] = {}

    for stream_id in streams:
        label = ARIA_GEN2_STREAM_ID_TO_LABEL.get(stream_id, stream_id)
        filename: str = ARIA_GEN2_STREAM_LABEL_TO_FILENAME.get(label, f"{label}.mp4")
        output_path: Path = output_dir / filename

        all_timestamps[label] = transcode_h265_stream_to_mp4(
            vrs_path=vrs_path, stream_id=stream_id, output_path=output_path, encoder=encoder,
        )

    # ── Save timestamps ───────────────────────────────────────────────────
    (output_dir / "timestamps_ns.json").write_text(json.dumps(all_timestamps))

    print(f"  Done in {time.perf_counter() - t_seq_start:.1f}s ({len(streams)} streams)")


def main(config: PreprocessConfig) -> None:
    """Preprocess Aria Gen2 Pilot VRS files."""
    root: Path = config.root
    assert root.exists(), f"Root directory not found: {root}"

    if config.sequence:
        seq_dir: Path = root / config.sequence
        assert seq_dir.exists(), f"Sequence not found: {seq_dir}"
        print(f"Processing: {config.sequence}")
        preprocess_sequence(seq_dir, config)
    else:
        seq_dirs: list[Path] = sorted([d for d in root.iterdir() if d.is_dir() and (d / VRS_FILENAME).exists()])
        print(f"Found {len(seq_dirs)} sequences with {VRS_FILENAME}")
        for i, seq_dir in enumerate(seq_dirs):
            print(f"\n[{i + 1}/{len(seq_dirs)}] {seq_dir.name}")
            preprocess_sequence(seq_dir, config)

    print("\nPreprocessing complete.")
