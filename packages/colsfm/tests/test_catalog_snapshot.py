"""Catalog inputs are complete snapshots: one directory per full specification, published whole or not at all."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from serde.json import from_json

from colsfm.alignment import GROUND_TRUTH_FILE_NAME
from colsfm.catalog import CATALOG_INPUT_FILE, CatalogInputConfig, CatalogSnapshot, write_snapshot
from colsfm.frame_manifest import FRAME_SOURCE_FILE, FrameSourceManifest, StreamKeyframes, read_manifest


def _config(tmp_path: Path, **changes: Any) -> CatalogInputConfig:
    return dataclasses.replace(CatalogInputConfig(dataset="msd-g2", segment="seg", output_root=tmp_path), **changes)


def _manifest(names: list[str]) -> FrameSourceManifest:
    t_ns = [int(name.split("/")[1]) for name in names]
    return FrameSourceManifest(
        catalog_url="rerun+http://127.0.0.1:51235",
        dataset="msd-g2",
        segment="seg",
        timeline="timestamp",
        streams=[StreamKeyframes(entity="world/rig_00/cam_00/pinhole/video", names=names, t_ns=t_ns)],
    )


def _meta(names: list[str]) -> dict[str, object]:
    return {"keyframes_metadata": [{"image_name": name} for name in names]}


@pytest.mark.parametrize(
    "change",
    [
        {"catalog_url": "rerun+http://other:51235"},
        {"dataset": "robocap"},
        {"cameras": ("cam0",)},
        {"trajectory_entity": "/world/rig_01"},
        {"ground_truth_entity": "/world/rig_00"},
        {"trim_start_seconds": 1.0},
        {"timeline": "frame"},
        {"stereo_pairs": (("cam0", "cam1"),)},
    ],
    ids=lambda change: next(iter(change)),
)
def test_specifications_that_differ_anywhere_never_share_a_directory(tmp_path: Path, change: dict[str, object]) -> None:
    """The directory used to be keyed by segment, stride and cap only; any other change reused the old files."""
    assert _config(tmp_path).input_dir == _config(tmp_path).input_dir
    assert _config(tmp_path, **change).input_dir != _config(tmp_path).input_dir


def test_a_failed_prepare_leaves_the_previous_snapshot_whole(tmp_path: Path) -> None:
    """A ground-truth write that fails after the manifest used to leave a new manifest beside old metadata."""
    input_dir = _config(tmp_path).input_dir
    write_snapshot(input_dir, CatalogSnapshot(spec="a", ground_truth=False), _manifest(["cam0/1"]), _meta(["cam0/1"]), None)

    def failing_truth(path: Path) -> object:
        raise OSError("catalog went away")

    with pytest.raises(OSError, match="went away"):
        write_snapshot(input_dir, CatalogSnapshot(spec="b", ground_truth=True), _manifest(["cam0/2"]), _meta(["cam0/2"]), failing_truth)
    assert read_manifest(input_dir / FRAME_SOURCE_FILE) == _manifest(["cam0/1"])
    assert from_json(CatalogSnapshot, (input_dir / CATALOG_INPUT_FILE).read_text()).spec == "a"
    assert sorted(path.name for path in tmp_path.iterdir()) == [input_dir.name], "no staging directory is left behind"


def test_an_unscored_snapshot_cannot_inherit_ground_truth(tmp_path: Path) -> None:
    """Preparing without ground truth over a scored snapshot leaves no `ground_truth.txt`, and says so."""
    input_dir = _config(tmp_path).input_dir
    write_snapshot(
        input_dir,
        CatalogSnapshot(spec="a", ground_truth=True),
        _manifest(["cam0/1"]),
        _meta(["cam0/1"]),
        lambda path: path.write_text("1 0 0 0 0 0 0 1\n"),
    )
    assert (input_dir / GROUND_TRUTH_FILE_NAME).is_file()
    write_snapshot(input_dir, CatalogSnapshot(spec="a", ground_truth=False), _manifest(["cam0/1"]), _meta(["cam0/1"]), None)
    assert not (input_dir / GROUND_TRUTH_FILE_NAME).exists()
    assert from_json(CatalogSnapshot, (input_dir / CATALOG_INPUT_FILE).read_text()).ground_truth is False


def test_metadata_naming_other_images_than_the_manifest_is_refused(tmp_path: Path) -> None:
    """The run decodes the manifest's names and maps the metadata's; a mismatch publishes nothing."""
    input_dir = _config(tmp_path).input_dir
    with pytest.raises(ValueError, match="same set"):
        write_snapshot(input_dir, CatalogSnapshot(spec="a", ground_truth=False), _manifest(["cam0/1"]), _meta(["cam0/2"]), None)
    assert not input_dir.exists()


def test_the_default_keeps_every_second_frame(tmp_path: Path) -> None:
    """`frame_stride` defaults to 2 for every catalog path (`run catalog`, `overview`)."""
    config: CatalogInputConfig = CatalogInputConfig(dataset="msd-g2", segment="seg", output_root=tmp_path)
    assert config.frame_stride == 2
    assert config.input_dir.name.startswith("seg_s2_")
    assert dataclasses.replace(config, frame_stride=1).input_dir.name.startswith("seg_s1_")
