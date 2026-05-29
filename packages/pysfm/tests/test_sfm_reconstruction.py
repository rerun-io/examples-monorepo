"""Tests for SfM reconstruction public helpers."""

from __future__ import annotations

from pathlib import Path

import rerun.blueprint as rrb

from pysfm.apis.sfm_reconstruction import create_sfm_blueprint
from pysfm.gradio_ui.nodes.sfm_reconstruction_ui import FOUNTAIN_IMAGES_DIR, _fountain_example_files


def test_create_sfm_blueprint_works_with_beartype_dev_mode() -> None:
    """Blueprint helper returns a Rerun container when beartype is active."""
    blueprint: rrb.ContainerLike = create_sfm_blueprint(parent_log_path=Path("world"))

    assert isinstance(blueprint, rrb.Horizontal)


def test_fountain_example_files_populates_directory_upload() -> None:
    """Fountain example provides one directory-upload example row."""
    examples: list[list[list[str]]] = _fountain_example_files()

    assert len(examples) == 1
    assert len(examples[0]) == 1
    assert len(examples[0][0]) > 0
    assert all(Path(image_path).parent == FOUNTAIN_IMAGES_DIR for image_path in examples[0][0])
