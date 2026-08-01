"""Command-line entry point for downloading ARKitScenes sequences."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, DownloadColumn, Progress, TaskID, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from simplecv.print_utils import format_bytes

from arkitscenes_download.download_dataset import (
    ALL_ASSETS,
    CONSOLE,
    PlannedDownload,
    VideoMetadata,
    download_video,
    load_metadata,
    plan_video_downloads,
    prefetch_sizes,
)
from arkitscenes_download.fs import directory_size


@dataclass
class Config:
    """Download raw ARKitScenes sequences to a local directory."""

    download_dir: Path
    """Dataset root; assets land under ``raw/<fold>/<video_id>/``."""
    num_random: int | None = 10
    """Sample this many random sequences (seeded). Ignored if ``video_ids`` is set; use None to take the whole split."""
    seed: int = 0
    """RNG seed for the random sample, so the selection is reproducible."""
    split: Literal["Training", "Validation", "both"] = "both"
    """Restrict the random sample to a single fold, or draw from both."""
    video_ids: tuple[str, ...] = ()
    """Explicit sequence ids to download; overrides random sampling when non-empty."""
    assets: tuple[str, ...] = ALL_ASSETS
    """Which raw assets to fetch. Defaults to everything (see ``ALL_ASSETS``)."""
    include_point_clouds: bool = True
    """Also download Faro laser-scan point clouds when available."""
    keep_zip: bool = False
    """Keep the downloaded ``.zip`` archives after extraction."""
    prefetch: bool = True
    """HEAD-request asset sizes up front for an accurate overall byte total. Disable when a wrapper (e.g. the Modal worker) calls this once per sequence — the extra round-trips buy nothing there."""


def _select_video_ids(config: Config, metadata: dict[str, VideoMetadata]) -> list[str]:
    """Resolve the final ordered list of sequence ids to download."""
    if config.video_ids:
        unknown: list[str] = [vid for vid in config.video_ids if vid not in metadata]
        if unknown:
            raise SystemExit(f"Unknown video_ids not in metadata: {unknown}")
        return list(config.video_ids)

    pool: list[str] = sorted(vid for vid, meta in metadata.items() if config.split == "both" or meta.fold == config.split)
    if config.num_random is None:
        return pool

    count: int = min(config.num_random, len(pool))
    return random.Random(config.seed).sample(pool, count)


def main(config: Config) -> None:
    """Download the selected ARKitScenes sequences."""
    # Share download_dataset's console: its failure prints would otherwise
    # bypass this function's Live region and garble the bars.
    console: Console = CONSOLE

    console.print(f"Loading raw metadata into {config.download_dir} ...")
    metadata: dict[str, VideoMetadata] = load_metadata(config.download_dir)
    console.print(f"  {len(metadata)} sequences in metadata")

    video_ids: list[str] = _select_video_ids(config, metadata)
    plans_by_video: dict[str, list[PlannedDownload]] = {
        video_id: plan_video_downloads(metadata[video_id], config.assets, config.download_dir) for video_id in video_ids
    }
    plans: list[PlannedDownload] = [plan for video_id in video_ids for plan in plans_by_video[video_id]]
    if config.prefetch:
        with console.status("Checking remote asset sizes..."):
            sizes: dict[str, int] = prefetch_sizes(plans)
    else:
        sizes = {}
    known_total: int = sum(sizes.values())
    unknown_sizes: int = sum(plan.url not in sizes for plan in plans)
    unknown_note: str = f" + {unknown_sizes} unknown-size assets" if unknown_sizes else ""
    console.print(f"Plan: {len(video_ids)} sequences, {len(plans)} assets, {format_bytes(known_total)} to fetch{unknown_note}")
    console.print(f"Point clouds: {config.include_point_clouds} | keep zips: {config.keep_zip}")

    start: float = time.monotonic()
    bytes_done: int = 0
    # Two Progress renderables in one Live so each task row gets fitting columns
    # (bytes/speed for the transfer, plain counts for sequences).
    progress: Progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    sequences_progress: Progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    bytes_task: TaskID = progress.add_task("download bytes", total=known_total)
    sequences_task: TaskID = sequences_progress.add_task(f"0/{len(video_ids)} sequences, {len(video_ids)} remaining", total=len(video_ids))

    def on_bytes(current: int) -> None:
        """Reflect the in-flight asset's current part-file size in overall progress."""
        progress.update(bytes_task, completed=bytes_done + current)

    def on_asset_complete(plan: PlannedDownload, actual_bytes: int) -> None:
        """Commit one completed asset's bytes to overall progress."""
        nonlocal bytes_done, known_total
        expected_bytes: int | None = sizes.get(plan.url)
        if expected_bytes is None:
            known_total += actual_bytes
            progress.update(bytes_task, total=known_total)
        bytes_done += actual_bytes if expected_bytes is None else expected_bytes
        progress.update(bytes_task, completed=bytes_done)
        if not console.is_terminal:
            console.print(f"asset complete: {plan.video_id} {plan.asset} ({format_bytes(actual_bytes)})")

    with Live(Group(progress, sequences_progress), console=console, refresh_per_second=10.0):
        for index, video_id in enumerate(video_ids, start=1):
            meta: VideoMetadata = metadata[video_id]
            download_video(
                metadata=meta,
                plans=plans_by_video[video_id],
                download_dir=config.download_dir,
                keep_zip=config.keep_zip,
                include_point_clouds=config.include_point_clouds,
                on_bytes=on_bytes,
                on_asset_complete=on_asset_complete,
            )
            video_dir: Path = config.download_dir / "raw" / meta.fold / video_id
            remaining: int = len(video_ids) - index
            sequences_progress.advance(sequences_task)
            sequences_progress.update(sequences_task, description=f"{index}/{len(video_ids)} sequences, {remaining} remaining")
            if not console.is_terminal:
                console.print(f"sequence complete: {video_id} ({format_bytes(directory_size(video_dir))} on disk)")

    elapsed: float = time.monotonic() - start
    total_bytes: int = directory_size(config.download_dir)
    console.print(f"Finished {len(video_ids)} sequence(s) in {elapsed / 60:.1f} min")
    console.print(f"Total on disk under {config.download_dir}: {format_bytes(total_bytes)}")
