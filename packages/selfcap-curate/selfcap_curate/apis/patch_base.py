"""Write migrated copies of explicitly selected served SelfCap bases."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dataforge import paths

from selfcap_curate.patch_base import patch
from selfcap_curate.records import TimeSync, episode_keys, read_frozen_sync


@dataclass
class Config:
    """Read-only inputs and the one directory this command may write."""

    output_dir: Path
    """Local directory for the new RRDs; never the served base directory."""
    sync_dir: Path
    """Read-only served sync record directory."""
    keys: tuple[str, ...] = ()
    """Episode keys date__sid8__epNNN (optional selfcap__ prefix)."""
    key_file: Path | None = None
    """Optional text file containing one episode key per line."""
    served_dir: Path = Path("/mnt/nas/datasets/dataforge-selfcap-rrd/base")
    """Read-only source base directory."""
    raw_root: Path = Path("/mnt/nas/datasets/exoego-self-collected/main/cut")
    """Read-only cut corpus root."""
    sync_name: str = "selfcap__{key}.json"
    """Sync filename template; use {key}.json explicitly for legacy served records."""
    mode: Literal["migrate-keep-clock", "migrate-realign"] = "migrate-keep-clock"
    """Restore served clocks, or rebuild video_time from the supplied record."""
    quest_video_delta_ms: float | None = None
    """Optional extra Quest video_time shift, applied to the selected episodes."""


def main(config: Config) -> None:
    """Patch named episodes into output_dir; never register or replace a base."""
    paths.require_outside([config.output_dir], roots=[paths.NAS_ROOT, config.raw_root, config.served_dir, config.sync_dir])
    keys = episode_keys(config.keys, config.key_file)
    for key in keys:
        sync: TimeSync = read_frozen_sync(config.sync_dir / config.sync_name.format(key=key.key))
        if sync.key != key.key:
            raise ValueError(f"{key}: sync record key is {sync.key}")
        raw: Path = config.raw_root / sync.date / sync.session / "episodes" / f"episode-{sync.episode.removeprefix('ep')}"
        filename: str = key.rrd_name
        result: Path = patch(
            config.served_dir / filename,
            raw,
            sync,
            output_path=config.output_dir / filename,
            quest_video_delta_ms=config.quest_video_delta_ms,
            mode=config.mode,
        )
        print(f"patched {key} → {result}")
