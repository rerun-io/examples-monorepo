"""Download Aria Gen2 Pilot sequences from CDN URL JSON.

Reads AriaGen2PilotDataset_download_urls.json and fetches selected data
types for each sequence.

Usage:
    pixi run download-aria-gen2-pilot \
        --urls-json /mnt/nas/datasets/aria-gen2-pilot/AriaGen2PilotDataset_download_urls.json \
        --output-dir /mnt/nas/datasets/aria-gen2-pilot \
        --max-sequences 2
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from serde import serde
from serde.json import from_json
from tqdm import tqdm

from simplecv.apis.download_utils import download_file, extract_zip, verify_sha1
from simplecv.configs.dataset_paths import ARIA_GEN2_PILOT_ROOT

# ── URL JSON schema (pyserde) ─────────────────────────────────────────── #


@serde
class DataTypeEntry:
    """A single downloadable asset (VRS file, ZIP archive, etc.)."""

    filename: str
    """CDN filename, e.g. ``AriaGen2PilotDataset_v1.0_walk_1_main_recording.vrs``."""
    sha1sum: str
    """Expected SHA-1 hex digest for integrity verification."""
    file_size_bytes: int
    """Expected file size in bytes (used for skip-if-complete check)."""
    download_url: str
    """CDN download URL."""


@serde
class SequenceUrls:
    """All downloadable assets for a single sequence."""

    main_vrs: DataTypeEntry | None = None
    video_main_rgb: DataTypeEntry | None = None
    mps_slam_trajectories: DataTypeEntry | None = None
    mps_slam_calibration: DataTypeEntry | None = None
    mps_slam_points: DataTypeEntry | None = None
    mps_slam_summary: DataTypeEntry | None = None
    mps_hand_tracking: DataTypeEntry | None = None
    mps_artifacts: DataTypeEntry | None = None
    depth: DataTypeEntry | None = None
    scene: DataTypeEntry | None = None
    heart_rate: DataTypeEntry | None = None
    diarization: DataTypeEntry | None = None
    hand_object_interaction: DataTypeEntry | None = None

    def get(self, data_type: str) -> DataTypeEntry | None:
        """Look up a data type entry by name."""
        return getattr(self, data_type, None)


@serde
class AriaGen2PilotUrlJson:
    """Top-level schema for ``AriaGen2PilotDataset_download_urls.json``."""

    sequences: dict[str, SequenceUrls]
    """Mapping from sequence name (e.g. ``walk_1``) to its downloadable assets."""


# ── Download configuration ────────────────────────────────────────────── #

# Data types needed for simplecv integration (video + MPS trajectory/calibration + hand tracking).
# Skip mps_artifacts (huge ~3.5GB), mps_slam_points (huge ~3.4GB), depth (large ~2GB),
# scene, heart_rate, diarization, hand_object_interaction, and mps_slam_summary.
DESIRED_DATA_TYPES: list[str] = [
    "main_vrs",
    "mps_slam_trajectories",
    "mps_slam_calibration",
    "mps_hand_tracking",
]

# Mapping from data type to subdirectory inside the sequence folder.
# MPS data lives under mps/ following projectaria conventions.
DATA_TYPE_SAVE_PATHS: dict[str, str] = {
    "main_vrs": ".",
    "video_main_rgb": ".",
    "mps_slam_trajectories": "mps/slam",
    "mps_slam_calibration": "mps/slam",
    "mps_hand_tracking": "mps/hand_tracking",
    "mps_slam_points": "mps/slam",
    "mps_slam_summary": "mps/slam",
    "mps_artifacts": "mps",
    "depth": "depth",
    "scene": "scene",
    "heart_rate": "heart_rate",
    "diarization": "diarization",
    "hand_object_interaction": "hand_object_interaction",
}

# Aria Gen2 Pilot VRS canonical name (per sequence_config.main.recording).
VRS_CANONICAL_NAME: str = "video.vrs"


@dataclass
class DownloadConfig:
    """Configuration for Aria Gen2 Pilot download."""

    urls_json: Path = ARIA_GEN2_PILOT_ROOT / "AriaGen2PilotDataset_download_urls.json"
    """Path to the AriaGen2PilotDataset_download_urls.json file."""
    output_dir: Path = ARIA_GEN2_PILOT_ROOT
    """Output directory for downloaded sequences."""
    max_sequences: int | None = None
    """Maximum number of sequences to download (None = all)."""
    data_types: list[str] = field(default_factory=lambda: DESIRED_DATA_TYPES)
    """Data types to download per sequence."""
    verify_sha1: bool = True
    """Verify SHA1 checksums after download."""


# ── Download logic ────────────────────────────────────────────────────── #


def download_sequence(
    sequence_name: str,
    sequence_urls: SequenceUrls,
    output_dir: Path,
    data_types: list[str],
    verify: bool,
) -> bool:
    """Download all requested data types for a single sequence."""
    seq_dir: Path = output_dir / sequence_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    success: bool = True
    for dtype in data_types:
        entry: DataTypeEntry | None = sequence_urls.get(dtype)
        if entry is None:
            continue

        # Determine save location
        save_subdir: str = DATA_TYPE_SAVE_PATHS.get(dtype, ".")
        save_dir: Path = seq_dir / save_subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        dest_path: Path = save_dir / entry.filename

        # Skip if the final output already exists (idempotent reruns).
        # ZIPs are deleted after extraction, and main_vrs is renamed to
        # video.vrs, so checking dest_path alone would re-download.
        if dtype == "main_vrs" and (seq_dir / VRS_CANONICAL_NAME).exists():
            continue
        if entry.filename.endswith(".zip") and not dest_path.exists() and any(save_dir.iterdir()):
            # ZIP was already extracted + deleted on a prior run.
            continue

        # Download
        try:
            download_file(entry.download_url, dest_path, entry.file_size_bytes)
        except Exception as exc:
            print(f"  [FAIL] {dtype}: {exc}")
            success = False
            continue

        # Verify checksum
        if verify and not verify_sha1(dest_path, entry.sha1sum):
            print(f"  [FAIL] {dtype}: SHA1 mismatch for {entry.filename}")
            dest_path.unlink(missing_ok=True)
            success = False
            continue

        # Extract ZIPs
        if entry.filename.endswith(".zip") and dest_path.exists():
            extract_zip(dest_path, save_dir)

        # Rename VRS to canonical name
        if dtype == "main_vrs" and dest_path.exists():
            canonical: Path = seq_dir / VRS_CANONICAL_NAME
            if not canonical.exists():
                shutil.move(str(dest_path), str(canonical))

    return success


def main(config: DownloadConfig) -> None:
    """Download Aria Gen2 Pilot sequences."""
    assert config.urls_json.exists(), f"URL JSON not found: {config.urls_json}"

    url_json: AriaGen2PilotUrlJson = from_json(AriaGen2PilotUrlJson, config.urls_json.read_text())

    seq_names: list[str] = list(url_json.sequences.keys())
    if config.max_sequences is not None:
        seq_names = seq_names[: config.max_sequences]

    print(f"Data types: {config.data_types}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for seq_name in tqdm(seq_names, desc="Downloading sequences"):
        ok: bool = download_sequence(
            sequence_name=seq_name,
            sequence_urls=url_json.sequences[seq_name],
            output_dir=config.output_dir,
            data_types=config.data_types,
            verify=config.verify_sha1,
        )
        status: str = "OK" if ok else "PARTIAL"
        tqdm.write(f"  [{status}] {seq_name}")
