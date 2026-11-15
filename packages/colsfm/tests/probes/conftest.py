"""Shared fixtures for the pycolmap probes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pycolmap
import pytest

from probes.synthetic import SyntheticScene, build_scene


@pytest.fixture(autouse=True, scope="session")
def _quiet_glog() -> None:
    """Silence COLMAP's glog chatter so probe output stays readable.

    COLMAP logs at INFO by default and `triangulate_points` alone emits a few
    hundred lines. `minloglevel = 2` keeps warnings and errors.
    """
    pycolmap.logging.minloglevel = 2


@pytest.fixture
def scene() -> SyntheticScene:
    """The default five-frame, two-camera synthetic rig sequence."""
    return build_scene()


@pytest.fixture
def scene_on_disk(scene: SyntheticScene, tmp_path: Path) -> Iterator[tuple[SyntheticScene, Path, Path]]:
    """The synthetic scene materialised as a COLMAP database plus placeholder images.

    Yields:
        A tuple of (scene, database path, image directory).
    """
    from probes.synthetic import write_database, write_placeholder_images

    database_path: Path = tmp_path / "database.db"
    image_dir: Path = tmp_path / "images"
    write_database(scene, database_path)
    write_placeholder_images(scene, image_dir)
    yield scene, database_path, image_dir
