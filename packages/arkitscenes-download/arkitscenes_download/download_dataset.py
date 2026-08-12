"""Core download logic for the ARKitScenes raw dataset.

Adapted from the official Apple downloader
(https://github.com/apple/ARKitScenes/blob/main/download_data.py) and the
Rerun example port. Kept deliberately light at import time (CSV via `csv`,
extraction via `zipfile`, transfers shelled out to `curl`, rich for output,
and only import-cheap simplecv modules) because batch wrappers spawn this
once per sequence — keep numpy/torch-weight imports out of this path.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from rich.console import Console

CONSOLE: Final[Console] = Console(markup=False)

ARKITSCENES_URL: Final = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
"""Base URL for all v1 ARKitScenes assets."""
POINT_CLOUDS_FOLDER: Final = "laser_scanner_point_clouds"
"""Sub-folder name (both remote and local) holding Faro laser-scan point clouds."""

RawAsset: TypeAlias = str
"""One asset kind of the raw dataset (see ``ALL_ASSETS`` for the full set)."""

DEFAULT_ASSETS: Final[tuple[str, ...]] = (
    "mov",
    "annotation",
    "mesh",
    "lowres_wide.traj",
    "confidence",
    "lowres_depth",
    "lowres_wide_intrinsics",
    "ultrawide_intrinsics",
)
"""Raw assets required by the layered ARKitScenes ingest."""

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


@dataclass(slots=True, frozen=True)
class PlannedDownload:
    """One pending raw asset download."""

    video_id: str
    """Sequence identifier owning the asset."""
    asset: RawAsset
    """Requested raw asset kind."""
    filename: str
    """Remote and local asset filename."""
    url: str
    """Fully-qualified asset URL."""
    dst_path: Path
    """Local destination path."""
    is_zip: bool
    """Whether the downloaded file must be extracted."""


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


def parse_content_length(head_output: str) -> int | None:
    """Parse the last Content-Length header from curl output."""
    last_value: str | None = None
    for line in head_output.splitlines():
        name: str
        separator: str
        value: str
        name, separator, value = line.partition(":")
        if separator and name.strip().lower() == "content-length":
            last_value = value.strip()
    if last_value is None:
        return None
    try:
        return int(last_value)
    except ValueError:
        return None


def head_content_length(url: str) -> int | None:
    """Fetch an asset's HTTP content length."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["curl", "-sIL", "--fail", "--max-time", "30", url],
        check=False,
        capture_output=True,
        text=True,
    )
    return parse_content_length(result.stdout) if result.returncode == 0 else None


def prefetch_sizes(plans: list[PlannedDownload], workers: int = 16) -> dict[str, int]:
    """Fetch content lengths concurrently for planned downloads; URLs with unknown size are absent."""
    urls: list[str] = list(dict.fromkeys(plan.url for plan in plans))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes: list[int | None] = list(pool.map(head_content_length, urls))
    return {url: size for url, size in zip(urls, sizes, strict=True) if size is not None}


def download_file(url: str, dst_path: Path, on_bytes: Callable[[int], None] | None = None) -> int | None:
    """Download ``url`` to ``dst_path`` with curl (resumable, retrying).

    Args:
        url: Fully-qualified source URL.
        dst_path: Destination file path; parent directories are created.
        on_bytes: Callback periodically receiving the resumable ``.part`` size
            while curl is polled with ``Popen.wait(timeout=0.25)``.

    Returns:
        The downloaded size in bytes on success (or the on-disk size if the
        file already exists); None if the asset is unavailable (HTTP 4xx) or
        the transfer failed after retries. Test ``is None``, not truthiness —
        a zero-byte file is a success.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        return dst_path.stat().st_size

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
    def part_bytes() -> int:
        """Return the current transfer size; the .part file may not exist yet (or just got renamed)."""
        try:
            return part_path.stat().st_size
        except FileNotFoundError:
            return 0

    process: subprocess.Popen[bytes] = subprocess.Popen(command)
    while True:
        try:
            # wait() returns the instant curl exits; sleeping instead would add
            # up to 0.25 s of dead time per asset (hours across a full split).
            process.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if on_bytes is not None:
                on_bytes(part_bytes())
    returncode: int = process.returncode
    if returncode == 0:
        size: int = part_bytes()
        part_path.rename(dst_path)
        return size
    if returncode == 22:  # curl --fail: server returned HTTP >= 400
        part_path.unlink(missing_ok=True)
        CONSOLE.print(f"    unavailable (HTTP error): {url}")
        return None
    CONSOLE.print(f"    transfer failed (curl exit {returncode}; keeping .part for resume): {url}")
    return None


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


def plan_video_downloads(metadata: VideoMetadata, assets: tuple[RawAsset, ...], download_dir: Path) -> list[PlannedDownload]:
    """Plan the still-pending applicable assets for one sequence."""
    video_dir: Path = download_dir / "raw" / metadata.fold / metadata.video_id
    url_prefix: str = f"{ARKITSCENES_URL}/raw/{metadata.fold}/{metadata.video_id}"
    plans: list[PlannedDownload] = []
    for asset in dict.fromkeys(assets):
        filename: str | None = asset_to_filename(metadata.video_id, asset, metadata)
        if filename is None:
            continue
        is_zip: bool = filename.endswith(".zip")
        dst_path: Path = video_dir / filename
        extracted_path: Path = video_dir / filename.removesuffix(".zip")
        if (is_zip and extracted_path.is_dir()) or (not is_zip and dst_path.exists()):
            continue
        plans.append(PlannedDownload(metadata.video_id, asset, filename, f"{url_prefix}/{filename}", dst_path, is_zip))
    return plans


def load_metadata(download_dir: Path) -> dict[str, VideoMetadata]:
    """Download (if needed) and parse the raw dataset ``metadata.csv``.

    Args:
        download_dir: Dataset root; the CSV is cached at ``raw/metadata.csv``.

    Returns:
        Mapping from ``video_id`` to its parsed :class:`VideoMetadata`.
    """
    csv_path: Path = download_dir / "raw" / "metadata.csv"
    if download_file(f"{ARKITSCENES_URL}/raw/metadata.csv", csv_path) is None:
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
    if not mapping_path.exists() and download_file(mapping_url, mapping_path) is None:
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
        CONSOLE.log(f"    skipping point clouds for {metadata.video_id}: bad visit id {metadata.visit_id!r}")
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
    plans: list[PlannedDownload],
    download_dir: Path,
    keep_zip: bool,
    include_point_clouds: bool,
    on_bytes: Callable[[int], None] | None = None,
    on_asset_complete: Callable[[PlannedDownload, int], None] | None = None,
) -> None:
    """Download an already-filtered list of pending assets for one sequence.

    Assets already present on disk (extracted folder for zips, file for the
    rest) are skipped, so the call is idempotent and resumable.

    Args:
        metadata: Parsed metadata row for the sequence.
        plans: Pending assets produced by :func:`plan_video_downloads`.
        download_dir: Dataset root used for point-cloud downloads.
        keep_zip: Keep the downloaded ``.zip`` after extraction when True.
        include_point_clouds: Also fetch laser-scan point clouds when available.
        on_bytes: Callback receiving the in-flight asset's current ``.part`` size.
        on_asset_complete: Callback receiving each completed plan and its downloaded byte size.
    """
    video_dir: Path = download_dir / "raw" / metadata.fold / metadata.video_id
    for plan in plans:
        # Plans snapshot disk state at planning time; re-check just before the
        # transfer so an asset landed since then (or planned twice) is skipped
        # instead of re-downloaded and re-extracted onto the existing directory.
        extracted_dir: Path = video_dir / plan.filename.removesuffix(".zip")
        if (plan.is_zip and extracted_dir.is_dir()) or (not plan.is_zip and plan.dst_path.exists()):
            continue
        size: int | None = download_file(plan.url, plan.dst_path, on_bytes)
        if size is None:
            continue
        if plan.is_zip:
            extract_zip(plan.dst_path, video_dir, plan.filename.removesuffix(".zip"))
            if not keep_zip:
                plan.dst_path.unlink(missing_ok=True)
        if on_asset_complete is not None:
            on_asset_complete(plan, size)

    if include_point_clouds and metadata.has_laser_scanner_point_clouds:
        CONSOLE.log(f"[{metadata.video_id}] laser_scanner_point_clouds")
        download_point_clouds(metadata, download_dir)
