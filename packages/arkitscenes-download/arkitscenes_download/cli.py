"""Command-line entry point for downloading ARKitScenes sequences."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from arkitscenes_download.download_dataset import (
    ALL_ASSETS,
    VideoMetadata,
    download_video,
    human_bytes,
    load_metadata,
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


def main() -> None:
    """Parse CLI args and download the selected ARKitScenes sequences."""
    config: Config = tyro.cli(Config)

    print(f"Loading raw metadata into {config.download_dir} ...")
    metadata: dict[str, VideoMetadata] = load_metadata(config.download_dir)
    print(f"  {len(metadata)} sequences in metadata")

    video_ids: list[str] = _select_video_ids(config, metadata)
    print(f"\nSelected {len(video_ids)} sequence(s): {', '.join(video_ids)}")
    print(f"Assets ({len(config.assets)}): {', '.join(config.assets)}")
    print(f"Point clouds: {config.include_point_clouds} | keep zips: {config.keep_zip}\n")

    start: float = time.monotonic()
    for index, video_id in enumerate(video_ids, start=1):
        meta: VideoMetadata = metadata[video_id]
        tags: str = f"{meta.fold}, upsampling={meta.is_in_upsampling}, laser={meta.has_laser_scanner_point_clouds}"
        print(f"[{index}/{len(video_ids)}] {video_id} ({tags})")
        download_video(
            metadata=meta,
            assets=config.assets,
            download_dir=config.download_dir,
            keep_zip=config.keep_zip,
            include_point_clouds=config.include_point_clouds,
        )
        video_dir: Path = config.download_dir / "raw" / meta.fold / video_id
        print(f"    done — {human_bytes(directory_size(video_dir))} on disk")

    elapsed: float = time.monotonic() - start
    total_bytes: int = directory_size(config.download_dir)
    print(f"\nFinished {len(video_ids)} sequence(s) in {elapsed / 60:.1f} min")
    print(f"Total on disk under {config.download_dir}: {human_bytes(total_bytes)}")


if __name__ == "__main__":
    main()
