"""``dataforge-view``: a recording is opened as all of its layers, not just its base one.

The rrds here are empty files. What is under test is which paths the verb hands
to the viewer, and ``log_file_from_path`` is the seam that says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from dataforge import paths
from dataforge.apis import view
from dataforge.apis.view import Config
from dataforge.datasets.msd import MsdConfig
from dataforge.identity import SequenceIdentity

IDENTITY: SequenceIdentity = SequenceIdentity(dataset="msd-index", parts=("MIO_others", "MIO09_short_1_updown"))
"""A two-part sequence key, so the verb really has to rebuild the identity from the filename."""


@pytest.fixture
def opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Point the output root at ``tmp_path`` and collect what the viewer is handed."""
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    loaded: list[Path] = []
    monkeypatch.setattr(rr, "log_file_from_path", loaded.append)
    return loaded


def write_layer(layer: str) -> Path:
    """Put an empty rrd of ``IDENTITY`` in one layer directory."""
    target: Path = paths.rrd_path(paths.output_root(), layer=layer, identity=IDENTITY)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    return target


def view_index() -> None:
    """Run the verb over the msd-index dataset, without spawning a viewer."""
    view.main(Config(rr_config=RerunTyroConfig(headless=True), dataset=MsdConfig(device="index")))


def test_every_layer_of_the_selected_recording_is_opened(opened: list[Path]) -> None:
    """The layers share a recording id, so handing the viewer all of them merges them."""
    base_target: Path = write_layer(paths.BASE_LAYER)
    gt_target: Path = write_layer(paths.GT_LAYER)

    view_index()

    assert opened == [base_target, gt_target]


def test_a_recording_with_no_derived_layer_opens_its_base_alone(opened: list[Path]) -> None:
    base_target: Path = write_layer(paths.BASE_LAYER)

    view_index()

    assert opened == [base_target]
