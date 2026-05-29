"""Download HOT3D Aria sequences from CDN URL JSON.

Reads the download URL JSON (e.g. Hot3DAria_download_urls.json) obtained from
projectaria.com and fetches selected data types for each sequence.

Usage:
    pixi run download-hot3d \\
        --urls-json /mnt/8tb/data/hot3d/Hot3DAria_download_urls.json \\
        --output-dir /mnt/8tb/data/hot3d/aria \\
        --max-sequences 20
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from simplecv.apis.download_utils import download_file, extract_zip, verify_sha1

# Re-export for backwards compatibility
__all__ = ["download_file", "extract_zip", "verify_sha1"]

# Data types we want for simplecv integration. Skip mps_slam_points (huge, 400MB+)
# and mps_artifacts (huge, 460MB+) and mps_slam_summary (trivial).
DESIRED_DATA_TYPES: list[str] = [
    "main_vrs",
    "hand_data",
    "mps_slam_trajectories",
    "mps_slam_calibration",
    "ground_truth",
    "video_main_rgb",
    "mps_eye_gaze",
]

# Mapping from data type to subdirectory inside the sequence folder.
# Mirrors HOT3D's DATA_TYPE_TO_SAVE_PATH convention.
DATA_TYPE_SAVE_PATHS: dict[str, str] = {
    "main_vrs": ".",
    "video_main_rgb": ".",
    "hand_data": ".",
    "ground_truth": ".",
    "mps_slam_trajectories": "mps/slam",
    "mps_slam_calibration": "mps/slam",
    "mps_slam_points": "mps/slam",
    "mps_slam_summary": "mps/slam",
    "mps_eye_gaze": "mps/eye_gaze",
    "mps_artifacts": "mps",
}

# VRS files get renamed to this canonical name after download.
VRS_CANONICAL_NAME: str = "recording.vrs"


@dataclass
class DownloadConfig:
    """Configuration for HOT3D download."""

    urls_json: Path
    """Path to the Hot3DAria_download_urls.json file."""
    output_dir: Path = Path("/mnt/8tb/data/hot3d/aria")
    """Output directory for downloaded sequences."""
    max_sequences: int | None = 20
    """Maximum number of sequences to download (None = all)."""
    data_types: list[str] = field(default_factory=lambda: DESIRED_DATA_TYPES)
    """Data types to download per sequence."""
    verify_sha1: bool = True
    """Verify SHA1 checksums after download."""


def download_sequence(
    sequence_name: str,
    sequence_data: dict[str, dict],
    output_dir: Path,
    data_types: list[str],
    verify: bool,
) -> bool:
    """Download all requested data types for a single sequence."""
    seq_dir: Path = output_dir / sequence_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    success: bool = True
    for dtype in data_types:
        if dtype not in sequence_data:
            continue

        info: dict = sequence_data[dtype]
        filename: str = info["filename"]
        url: str = info["download_url"]
        expected_size: int = info["file_size_bytes"]
        expected_sha1: str = info["sha1sum"]

        # Determine save location
        save_subdir: str = DATA_TYPE_SAVE_PATHS.get(dtype, ".")
        save_dir: Path = seq_dir / save_subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        dest_path: Path = save_dir / filename

        # Download
        try:
            download_file(url, dest_path, expected_size)
        except Exception as exc:
            print(f"  [FAIL] {dtype}: {exc}")
            success = False
            continue

        # Verify checksum
        if verify and not verify_sha1(dest_path, expected_sha1):
            print(f"  [FAIL] {dtype}: SHA1 mismatch for {filename}")
            dest_path.unlink(missing_ok=True)
            success = False
            continue

        # Extract ZIPs
        if filename.endswith(".zip"):
            extract_zip(dest_path, save_dir)

        # Rename VRS to canonical name
        if dtype == "main_vrs" and dest_path.exists():
            canonical: Path = seq_dir / VRS_CANONICAL_NAME
            if not canonical.exists():
                shutil.move(str(dest_path), str(canonical))

    return success


def main(config: DownloadConfig) -> None:
    """Download HOT3D Aria sequences."""
    assert config.urls_json.exists(), f"URL JSON not found: {config.urls_json}"

    with open(config.urls_json) as f:
        data: dict = json.load(f)

    sequences: dict[str, dict] = data["sequences"]
    seq_names: list[str] = list(sequences.keys())

    if config.max_sequences is not None:
        seq_names = seq_names[: config.max_sequences]

    print(f"Downloading {len(seq_names)} sequences to {config.output_dir}")
    print(f"Data types: {config.data_types}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    for i, seq_name in enumerate(seq_names):
        print(f"\n[{i + 1}/{len(seq_names)}] {seq_name}")
        ok: bool = download_sequence(
            sequence_name=seq_name,
            sequence_data=sequences[seq_name],
            output_dir=config.output_dir,
            data_types=config.data_types,
            verify=config.verify_sha1,
        )
        if ok:
            print(f"  [OK] {seq_name}")
        else:
            print(f"  [PARTIAL] {seq_name}")

    print(f"\nDone. Downloaded {len(seq_names)} sequences to {config.output_dir}")
