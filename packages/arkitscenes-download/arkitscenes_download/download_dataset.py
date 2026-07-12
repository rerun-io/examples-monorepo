"""Core download logic for the ARKitScenes raw dataset.

Adapted from the official Apple downloader
(https://github.com/apple/ARKitScenes/blob/main/download_data.py) and the
Rerun example port, using only the Python standard library at runtime (CSV via
`csv`, extraction via `zipfile`, transfers shelled out to `curl`).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

ARKITSCENES_URL: Final = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
"""Base URL for all v1 ARKitScenes assets."""
POINT_CLOUDS_FOLDER: Final = "laser_scanner_point_clouds"
"""Sub-folder name (both remote and local) holding Faro laser-scan point clouds."""

RawAsset: TypeAlias = str
"""One asset kind of the raw dataset (see ``ALL_ASSETS`` for the full set)."""

# The 11 per-frame / per-sequence assets that ship as zip archives. Each extracts
# into a folder of the same name (minus ``.zip``) under the video directory.
ZIP_ASSETS: Final[frozenset[str]] = frozenset(
    {
        "confidence",
        "highres_depth",
        "lowres_depth",
        "lowres_wide",
        "lowres_wide_intrinsics",
        "ultrawide",
        "ultrawide_intrinsics",
        "vga_wide",
        "vga_wide_intrinsics",
        "wide",
        "wide_intrinsics",
    }
)
"""Assets delivered as ``<asset>.zip`` and extracted into ``<asset>/``."""

# High-resolution assets that only exist for the ~2,257 upsampling sequences.
UPSAMPLING_ONLY_ASSETS: Final[frozenset[str]] = frozenset({"highres_depth", "wide", "wide_intrinsics"})
"""Assets present only when ``metadata.is_in_upsampling`` is true."""

ALL_ASSETS: Final[tuple[str, ...]] = (
    "mov",
    "annotation",
    "mesh",
    "lowres_wide.traj",
    "confidence",
    "lowres_depth",
    "lowres_wide",
    "lowres_wide_intrinsics",
    "ultrawide",
    "ultrawide_intrinsics",
    "vga_wide",
    "vga_wide_intrinsics",
    "highres_depth",
    "wide",
    "wide_intrinsics",
)
"""Every downloadable raw asset — the superset that means "everything"."""

# Sequences for which the 3dod-derived assets (mesh / annotation / trajectory)
# were never released. Copied verbatim from Apple's downloader.
MISSING_3DOD_VIDEO_IDS: Final[frozenset[str]] = frozenset(
    {
        "47334522",
        "47334523",
        "42897421",
        "45261582",
        "47333152",
        "47333155",
        "48458535",
        "48018733",
        "47429677",
        "48458541",
        "42897848",
        "47895482",
        "47333960",
        "47430089",
        "42899148",
        "42897612",
        "42899153",
        "42446164",
        "48018149",
        "47332198",
        "47334515",
        "45663223",
        "45663226",
        "45663227",
    }
)
"""Video ids missing the mesh / annotation / ``lowres_wide.traj`` assets."""


@dataclass(slots=True, frozen=True)
class VideoMetadata:
    """One row of the raw dataset ``metadata.csv``."""

    video_id: str
    """8-digit sequence id, used verbatim in asset URLs and local paths."""
    visit_id: str
    """Venue id shared by sequences of the same room; ``"NA"`` when absent."""
    fold: str
    """Split the sequence belongs to — ``"Training"`` or ``"Validation"``."""
    sky_direction: str
    """Gravity/sky direction tag (``Up``/``Down``/``Left``/``Right``)."""
    has_laser_scanner_point_clouds: bool
    """Whether Faro laser-scan point clouds exist for this sequence's venue."""
    is_in_upsampling: bool
    """Whether the high-res (``highres_depth``/``wide``) assets are available."""
    is_in_threedod: bool
    """Whether the sequence is part of the 3D object detection subset."""


def _parse_bool(value: str) -> bool:
    """Parse a CSV truthiness cell (``True``/``False``/``1``/``0``) into a bool."""
    return value.strip().lower() in {"true", "1", "1.0", "yes"}


def _parse_visit_id(visit_id: str) -> int | None:
    """Return the integer visit id, or None when it is missing/non-integer."""
    raw: str = visit_id.strip()
    if not raw or raw.upper() == "NA":
        return None
    try:
        as_float: float = float(raw)
    except ValueError:
        return None
    if as_float != as_float or not as_float.is_integer():  # NaN or fractional
        return None
    return int(as_float)


def human_bytes(num_bytes: int) -> str:
    """Format a byte count as a human-readable IEC string (e.g. ``1.4 GiB``)."""
    size: float = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def download_file(url: str, dst_path: Path) -> bool:
    """Download ``url`` to ``dst_path`` with curl (resumable, retrying).

    Args:
        url: Fully-qualified source URL.
        dst_path: Destination file path; parent directories are created.

    Returns:
        True on success (or if the file already exists); False if the asset is
        unavailable (HTTP 4xx) or the transfer failed after retries.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return True

    part_path: Path = dst_path.parent / (dst_path.name + ".part")
    command: list[str] = [
        "curl",
        "-L",
        "--fail",
        "--no-progress-meter",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--retry-connrefused",
        "-C",
        "-",
        "-o",
        str(part_path),
        url,
    ]
    result: subprocess.CompletedProcess[bytes] = subprocess.run(command, check=False)
    if result.returncode == 0:
        part_path.rename(dst_path)
        return True
    if result.returncode == 22:  # curl --fail: server returned HTTP >= 400
        part_path.unlink(missing_ok=True)
        print(f"    unavailable (HTTP error): {url}")
        return False
    print(f"    transfer failed (curl exit {result.returncode}; keeping .part for resume): {url}")
    return False


def extract_zip(zip_path: Path, video_dir: Path, folder_name: str) -> None:
    """Extract an asset archive into ``video_dir/<folder_name>/`` atomically.

    Every ARKitScenes asset zip wraps its files in a single top-level directory
    named after the asset. Extraction goes to a hidden staging directory first,
    then that directory is renamed into place, so an interrupted extraction can
    never leave a partial ``<folder_name>/`` that later looks complete.

    Args:
        zip_path: Path to the downloaded ``.zip``.
        video_dir: The sequence directory that should contain ``<folder_name>/``.
        folder_name: Final folder name (the asset name, i.e. zip name minus ``.zip``).
    """
    staging: Path = video_dir / f".{folder_name}.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(staging)

    final_dir: Path = video_dir / folder_name
    wrapped: Path = staging / folder_name
    # Move the extracted tree into its final location, then drop the staging dir.
    (wrapped if wrapped.is_dir() else staging).rename(final_dir)
    shutil.rmtree(staging, ignore_errors=True)


def asset_to_filename(video_id: str, asset: RawAsset, metadata: VideoMetadata) -> str | None:
    """Resolve the remote filename for ``asset``, or None if not applicable.

    Handles the two gating rules from Apple's downloader: high-res assets only
    exist for upsampling sequences, and mesh/annotation/trajectory are missing
    for a fixed list of sequences.
    """
    if asset in UPSAMPLING_ONLY_ASSETS and not metadata.is_in_upsampling:
        return None
    if asset in ZIP_ASSETS:
        return f"{asset}.zip"
    if asset == "mov":
        return f"{video_id}.mov"
    if asset == "mesh":
        return None if video_id in MISSING_3DOD_VIDEO_IDS else f"{video_id}_3dod_mesh.ply"
    if asset == "annotation":
        return None if video_id in MISSING_3DOD_VIDEO_IDS else f"{video_id}_3dod_annotation.json"
    if asset == "lowres_wide.traj":
        return None if video_id in MISSING_3DOD_VIDEO_IDS else "lowres_wide.traj"
    raise ValueError(f"Unknown raw asset: {asset!r}")


def load_metadata(download_dir: Path) -> dict[str, VideoMetadata]:
    """Download (if needed) and parse the raw dataset ``metadata.csv``.

    Args:
        download_dir: Dataset root; the CSV is cached at ``raw/metadata.csv``.

    Returns:
        Mapping from ``video_id`` to its parsed :class:`VideoMetadata`.
    """
    csv_path: Path = download_dir / "raw" / "metadata.csv"
    if not download_file(f"{ARKITSCENES_URL}/raw/metadata.csv", csv_path):
        raise RuntimeError("Failed to download raw metadata.csv")

    metadata: dict[str, VideoMetadata] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_id: str = row["video_id"].strip()
            metadata[video_id] = VideoMetadata(
                video_id=video_id,
                visit_id=row["visit_id"].strip(),
                fold=row["fold"].strip(),
                sky_direction=row.get("sky_direction", "").strip(),
                has_laser_scanner_point_clouds=_parse_bool(row["has_laser_scanner_point_clouds"]),
                is_in_upsampling=_parse_bool(row["is_in_upsampling"]),
                is_in_threedod=_parse_bool(row["is_in_threedod"]),
            )
    return metadata


def _point_cloud_ids_for_visit(visit_id: int, download_dir: Path) -> list[str]:
    """Return the laser-scan point-cloud ids belonging to ``visit_id``."""
    mapping_path: Path = download_dir / POINT_CLOUDS_FOLDER / "laser_scanner_point_clouds_mapping.csv"
    mapping_url: str = f"{ARKITSCENES_URL}/raw/{POINT_CLOUDS_FOLDER}/laser_scanner_point_clouds_mapping.csv"
    if not mapping_path.exists() and not download_file(mapping_url, mapping_path):
        return []

    point_cloud_ids: list[str] = []
    with mapping_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["visit_id"].strip() == str(visit_id):
                point_cloud_ids.append(row["laser_scanner_point_clouds_id"].strip())
    return point_cloud_ids


def download_point_clouds(metadata: VideoMetadata, download_dir: Path) -> None:
    """Download the Faro laser-scan point clouds for a sequence's venue."""
    visit_id: int | None = _parse_visit_id(metadata.visit_id)
    if visit_id is None:
        print(f"    skipping point clouds for {metadata.video_id}: bad visit id {metadata.visit_id!r}")
        return

    point_cloud_ids: list[str] = _point_cloud_ids_for_visit(visit_id, download_dir)
    visit_dir: Path = download_dir / POINT_CLOUDS_FOLDER / str(visit_id)
    for point_cloud_id in point_cloud_ids:
        for extension in (".ply", "_pose.txt"):
            filename: str = f"{point_cloud_id}{extension}"
            url: str = f"{ARKITSCENES_URL}/raw/{POINT_CLOUDS_FOLDER}/{visit_id}/{filename}"
            download_file(url, visit_dir / filename)


def download_video(
    metadata: VideoMetadata,
    assets: tuple[RawAsset, ...],
    download_dir: Path,
    keep_zip: bool,
    include_point_clouds: bool,
) -> None:
    """Download all requested ``assets`` for a single sequence.

    Assets already present on disk (extracted folder for zips, file for the
    rest) are skipped, so the call is idempotent and resumable.

    Args:
        metadata: Parsed metadata row for the sequence.
        assets: Ordered asset kinds to fetch (see ``ALL_ASSETS``).
        download_dir: Dataset root (writes to ``raw/<fold>/<video_id>/``).
        keep_zip: Keep the downloaded ``.zip`` after extraction when True.
        include_point_clouds: Also fetch laser-scan point clouds when available.
    """
    video_dir: Path = download_dir / "raw" / metadata.fold / metadata.video_id
    url_prefix: str = f"{ARKITSCENES_URL}/raw/{metadata.fold}/{metadata.video_id}"

    for asset in assets:
        filename: str | None = asset_to_filename(metadata.video_id, asset, metadata)
        if filename is None:
            continue

        is_zip: bool = filename.endswith(".zip")
        if is_zip and (video_dir / filename[: -len(".zip")]).is_dir():
            continue  # already extracted

        dst_path: Path = video_dir / filename
        if not is_zip and dst_path.exists():
            continue

        print(f"    {asset} -> {filename}")
        if not download_file(f"{url_prefix}/{filename}", dst_path):
            continue
        if is_zip:
            extract_zip(dst_path, video_dir, filename[: -len(".zip")])
            if not keep_zip:
                dst_path.unlink(missing_ok=True)

    if include_point_clouds and metadata.has_laser_scanner_point_clouds:
        print("    laser_scanner_point_clouds")
        download_point_clouds(metadata, download_dir)
