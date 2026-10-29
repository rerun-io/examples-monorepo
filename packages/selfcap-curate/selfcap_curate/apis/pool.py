"""Catalog-only exo calibration and focal-profile pooling Tyro entry points."""

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from dataforge import paths
from dataforge.records import read_json
from serde.json import to_json

from selfcap_curate.apis import open_session
from selfcap_curate.catalog_io import CATALOG_URL, open_segment, read_phone_detections, read_time_sync
from selfcap_curate.device_tables import FocalProfile, pool, pool_rotations
from selfcap_curate.exo_calibration import profile_session
from selfcap_curate.quest_fit import read_session_episode
from selfcap_curate.records import GyroFit, OakFit


@dataclass
class Config:
    """Explicit session inputs; patched bases must already be in the selected catalog."""

    output: Path
    """Local output machine JSON; no NAS or catalog writes."""
    segment_ids: tuple[str, ...] = ()
    """Explicit episodes of one session."""
    key_file: Path | None = None
    """Optional newline-separated segment ids or unprefixed keys."""
    catalog_url: str = CATALOG_URL
    """Read-only endpoint containing the patched bases."""


def main(config: Config) -> None:
    """Solve exactly one session from its explicitly named episodes."""
    session, keys, dataset = open_session(config.segment_ids, config.key_file, config.catalog_url, output=config.output, protected=())
    episodes = []
    for key in keys:
        segment = open_segment(dataset, key.segment)
        record = read_time_sync(dataset, key.segment)
        episode = read_session_episode(segment, record)
        detections = read_phone_detections(segment)
        if set(episode.layout.phones) - detections.keys():
            raise ValueError(f"{key}: Sapiens2 layer missing phone detections; run selfcap-align and register its sapiens2 layer first")
        episodes.append(replace(episode, detections=detections))
    result = profile_session(session, episodes)
    config.output.write_text(to_json(result))


@dataclass
class PoolConfig:
    """Rebuild the pooled device table from focal-profile results."""

    profiles: tuple[Path, ...]
    """Explicit scratch or newly produced focal-profile JSON paths."""
    output: Path
    """Local machine JSON destination."""
    generated_date: str = field(default_factory=lambda: date.today().isoformat())
    """ISO date recorded beside each device."""


def pool_main(config: PoolConfig) -> None:
    """Pool typed profiles; source schemas are checked before fitting."""
    if config.output.exists():
        raise FileExistsError(config.output)
    paths.require_outside([config.output], roots=[paths.NAS_ROOT])
    profiles = [read_json(path, FocalProfile) for path in config.profiles]
    config.output.write_text(to_json(pool(profiles, config.generated_date)))


@dataclass
class RotationConfig:
    free_paths: tuple[Path, ...]
    """Explicit frozen oak_free hand-fit JSON records."""
    gyro_dir: Path
    """Read-only gyro_mount results, named by session."""
    output: Path
    """Local machine table output; no NAS writes."""
    generated_date: str = field(default_factory=lambda: date.today().isoformat())
    """ISO date recorded with each device's provenance."""


def rotation_main(config: RotationConfig) -> None:
    """Rebuild the device table from typed free-hand and gyro session records."""
    if config.output.exists():
        raise FileExistsError(config.output)
    paths.require_outside([config.output], roots=[paths.NAS_ROOT, config.gyro_dir])
    free = [read_json(p, OakFit) for p in config.free_paths]
    gyros = {f.session: read_json(path, GyroFit) for f in free if (path := config.gyro_dir / f"{f.session}.json").exists()}
    config.output.write_text(to_json(pool_rotations(free, gyros, config.generated_date)))
