"""A catalog input from local files: a base rrd and named layer rrds, served on a private in-process catalog."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import tyro

from colsfm.apis.run import CatalogCommand, Command
from colsfm.catalog import CatalogInputConfig, read_catalog_trajectory, serve_catalog_input


def _recording(path: Path, entity: str, translations: list[list[float]] | None = None) -> Path:
    import rerun as rr

    recording = rr.RecordingStream("colsfm-test", recording_id="seg")
    recording.save(path)
    recording.log(entity, rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    for index, translation in enumerate(translations or []):
        recording.set_time("video_time", duration=np.timedelta64(1_000 * (index + 1), "ns"))
        recording.log(entity, rr.Transform3D(translation=translation, quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])))
    recording.flush()
    recording.disconnect()
    return path


def _local(tmp_path: Path, segment: str = "") -> CatalogInputConfig:
    base: Path = _recording(tmp_path / "base.rrd", "/world/rig_00")
    poses: Path = _recording(tmp_path / "slam.rrd", "/world/runs/slam_rs/rig", [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    return CatalogInputConfig(dataset="insta360", segment=segment, rrd=base, layers={"slam_rs": poses}, output_root=tmp_path / "inputs")


@pytest.mark.integration
def test_a_local_base_and_layer_serve_the_segment_the_base_names(tmp_path: Path) -> None:
    """Needs a catalog server (an in-process one, on an OS port): the trajectory comes from the layer."""
    from rerun.catalog import CatalogClient

    config: CatalogInputConfig = _local(tmp_path)
    with serve_catalog_input(config) as served:
        assert served.segment == "seg"
        assert served.catalog_url != config.catalog_url
        dataset = CatalogClient(served.catalog_url).get_dataset(name="insta360")
        trajectory = read_catalog_trajectory(dataset, served.segment, "/world/runs/slam_rs/rig", "video_time")
        first_dir: Path = served.input_dir
    np.testing.assert_allclose(trajectory.translations[:, 0], [1.0, 2.0])
    with serve_catalog_input(config) as again:
        assert again.input_dir == first_dir  # the private server's port is not part of the snapshot's name


@pytest.mark.integration
def test_a_segment_that_is_not_the_base_s_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="other"), serve_catalog_input(_local(tmp_path, segment="other")):
        pass


def test_a_remote_input_keeps_its_snapshot_name_and_a_local_one_is_named_by_its_files(tmp_path: Path) -> None:
    remote = CatalogInputConfig(dataset="insta360", segment="seg", output_root=tmp_path)
    assert "rrd" not in remote.spec() and "layers" not in remote.spec()
    local = CatalogInputConfig(dataset="insta360", rrd=tmp_path / "base.rrd", layers={"slam_rs": tmp_path / "a.rrd"}, output_root=tmp_path)
    other = dataclasses.replace(local, layers={"slam_rs": tmp_path / "b.rrd"})
    assert "catalog_url" not in local.spec()
    assert local.spec() != other.spec()


@pytest.mark.parametrize(
    ("changes", "message"),
    [({}, "segment"), ({"layers": {"slam_rs": Path("a.rrd")}, "segment": "seg"}, "rrd")],
)
def test_an_input_without_a_segment_or_with_layers_but_no_base_is_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CatalogInputConfig(dataset="insta360", **changes)  # pyrefly: ignore[bad-argument-type]


def test_the_cli_takes_a_base_rrd_and_named_layers() -> None:
    command = tyro.cli(
        Command,
        args=["catalog", "--source.dataset", "insta360", "--source.rrd", "base.rrd", "--source.layers", "cube", "c.rrd", "slam_rs", "s.rrd"],
    )
    assert isinstance(command, CatalogCommand)
    assert command.source.rrd == Path("base.rrd")
    assert command.source.layers == {"cube": Path("c.rrd"), "slam_rs": Path("s.rrd")}
