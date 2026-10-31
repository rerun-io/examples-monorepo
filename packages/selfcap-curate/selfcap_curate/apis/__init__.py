"""Session output publication shared by local command interfaces."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from dataforge import paths
from rerun.catalog import CatalogClient, DatasetEntry

from selfcap_curate.catalog_io import DATASET_NAME
from selfcap_curate.records import EpisodeKey, episode_keys, one_session


@contextmanager
def session_output(destination: Path) -> Iterator[Path]:
    """Publish a completed session by one sibling-directory rename; failures leave no output."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        staging: Path = Path(temporary)
        yield staging
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        staging.rename(destination)


def open_session(
    ids: tuple[str, ...],
    key_file: Path | None,
    catalog_url: str,
    *,
    output: Path,
    protected: tuple[Path, ...],
    source: Path | None = None,
    new_directory: bool = True,
    rate: float | None = None,
    dataset: str = DATASET_NAME,
) -> tuple[str, tuple[EpisodeKey, ...], DatasetEntry]:
    """Shared selection/output preflight, retaining session and per-episode publication guards."""
    keys = episode_keys(ids, key_file)
    session = one_session(keys)
    if rate is not None and (not math.isfinite(rate) or rate <= 0.0):
        raise ValueError("rate must be finite and positive")
    paths.require_outside([output], roots=[paths.NAS_ROOT, *protected])
    if source is not None:
        for key in keys:
            if not (source / key.rrd_name).is_file():
                raise FileNotFoundError(source / key.rrd_name)
            if not new_directory and (output / key.rrd_name).exists():
                raise FileExistsError(output / key.rrd_name)
    if new_directory and (output.exists() or output.is_symlink()):
        raise FileExistsError(output)
    return session, keys, CatalogClient(catalog_url).get_dataset(dataset)
