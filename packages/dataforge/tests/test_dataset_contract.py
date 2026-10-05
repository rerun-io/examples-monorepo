"""The contract every registered dataset keeps, checked once over the whole registry.

The per-dataset modules test what a dataset does with its own data. This module
tests what the verbs expect from every dataset: ``setup()`` builds it, its layers
land one rrd each under the output root (a preview in its own tree), register
can build both blueprints without the raw tree, and the table card decodes one
camera. A new dataset gets these checks by registering; nothing here names a
dataset, so the suite holds at every level of the review stack. What needs a
dataset's own source type (the raw-root guard, ``conftest.assert_raw_root_guarded``)
lives in that dataset's test module.
"""

from dataclasses import fields, replace
from pathlib import Path

import pytest
import rerun.blueprint as rrb

from dataforge import paths
from dataforge.datasets import dataset_defaults
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity
from dataforge.writing import blueprint_views

CORPUS_DERIVED: frozenset[str] = frozenset({"wildcap"})
"""Commands whose default blueprint derives from the captures on disk, so an empty raw tree raises instead."""

FRAME_LIMITED: list[str] = [command for command, config in dataset_defaults.items() if "frame_limit" in {field.name for field in fields(config)}]
"""Commands whose config takes ``--frame-limit``."""

every_dataset = pytest.mark.parametrize("command", list(dataset_defaults))


def sequence(dataset: DataforgeDataset) -> SequenceIdentity:
    """A sequence of ``dataset`` that discovery never saw; ``targets()`` must place it without a raw tree."""
    return SequenceIdentity(dataset.config.name, ("subject", "sequence"))


def without_raw_tree(command: str, tmp_path: Path) -> DataforgeDatasetConfig:
    """The dataset's default config pointed at a raw root that does not exist."""
    return replace(dataset_defaults[command], root=tmp_path / "absent")  # pyrefly: ignore[unexpected-keyword]


@every_dataset
def test_setup_builds_the_dataset_its_registry_key_names(command: str) -> None:
    config: DataforgeDatasetConfig = dataset_defaults[command]
    assert config.command == command
    # setup() instantiates the ABC, so a dataset missing an abstract verb or blueprint hook raises TypeError here.
    dataset: DataforgeDataset = config.setup()
    assert isinstance(dataset, DataforgeDataset)
    assert dataset.config is config
    assert config.name == command or config.name.startswith(f"{command}-")


@every_dataset
def test_layers_start_with_base_and_never_repeat(command: str) -> None:
    layers: tuple[str, ...] = dataset_defaults[command].setup().layers
    assert layers[0] == paths.BASE_LAYER
    assert len(set(layers)) == len(layers)


@every_dataset
def test_targets_are_one_rrd_per_layer_under_the_output_root(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    dataset: DataforgeDataset = dataset_defaults[command].setup()
    identity: SequenceIdentity = sequence(dataset)
    targets: dict[str, Path] = dataset.targets(identity)
    assert list(targets) == list(dataset.layers)
    assert targets == {layer: paths.rrd_path(tmp_path, layer=layer, identity=identity) for layer in dataset.layers}


@pytest.mark.parametrize("command", FRAME_LIMITED)
def test_a_preview_lands_in_its_own_tree(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    full_dataset: DataforgeDataset = dataset_defaults[command].setup()
    preview_dataset: DataforgeDataset = replace(dataset_defaults[command], frame_limit=2).setup()  # pyrefly: ignore[unexpected-keyword]
    full: dict[str, Path] = full_dataset.targets(sequence(full_dataset))
    preview: dict[str, Path] = preview_dataset.targets(sequence(preview_dataset))
    assert list(preview) == list(full)
    assert not set(preview.values()) & set(full.values())
    assert all(path.is_relative_to(tmp_path / "preview-first2") for path in preview.values())


@pytest.mark.parametrize("frame_limit", [0, -1])
@pytest.mark.parametrize("command", FRAME_LIMITED)
def test_a_non_positive_frame_limit_is_refused(command: str, frame_limit: int) -> None:
    dataset: DataforgeDataset = replace(dataset_defaults[command], frame_limit=frame_limit).setup()  # pyrefly: ignore[unexpected-keyword]
    with pytest.raises(ValueError, match="frame_limit must be positive"):
        dataset.targets(sequence(dataset))


@every_dataset
def test_both_blueprints_build_and_save_without_the_raw_tree(command: str, tmp_path: Path) -> None:
    config: DataforgeDatasetConfig = without_raw_tree(command, tmp_path)
    dataset: DataforgeDataset = config.setup()
    blueprints: dict[str, rrb.Blueprint] = {"table": dataset.table_blueprint()}
    if command in CORPUS_DERIVED:
        with pytest.raises(FileNotFoundError, match=config.name):
            dataset.default_blueprint()
    else:
        blueprints["default"] = dataset.default_blueprint()
    for kind, blueprint in blueprints.items():
        saved: Path = tmp_path / f"{kind}.rbl"
        blueprint.save(config.name, str(saved))
        assert saved.stat().st_size > 0, kind


@every_dataset
def test_the_table_card_decodes_one_camera(command: str, tmp_path: Path) -> None:
    views: list[rrb.View] = blueprint_views(without_raw_tree(command, tmp_path).setup().table_blueprint())
    assert [type(view) for view in views].count(rrb.Spatial2DView) == 1


@pytest.mark.parametrize("command", [command for command, config in dataset_defaults.items() if type(config.setup()).remote_sequences is DataforgeDataset.remote_sequences])
def test_a_dataset_without_a_source_listing_says_so(command: str) -> None:
    dataset: DataforgeDataset = dataset_defaults[command].setup()
    with pytest.raises(NotImplementedError, match=type(dataset).__name__):
        dataset.remote_sequences()
