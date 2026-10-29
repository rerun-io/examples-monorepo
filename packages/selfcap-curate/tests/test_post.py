"""One-process post: calibrate and keypoints read align's local output, donors stay on the source catalog."""

from pathlib import Path

import pytest
import rerun as rr
from rerun.catalog import CatalogClient

import selfcap_curate.apis.post as post
from selfcap_curate.catalog_io import DATASET_NAME
from selfcap_curate.records import episode_keys

SEGMENTS = ("selfcap__2025-12-11__6d3740b8__ep001", "selfcap__2025-12-11__6d3740b8__ep005")


def write_rrd(path: Path, segment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream("post-test", recording_id=segment) as recording:
        recording.save(path)
        recording.log("/world/marker", rr.AnyValues(value=1), static=True)


@pytest.mark.integration
def test_post_serves_align_output_to_calibrate_and_keypoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def align(config) -> None:
        assert (config.base_dir, config.catalog_url, config.segment_ids) == (tmp_path / "base", "rerun+http://source:1", SEGMENTS)
        for key in episode_keys(config.segment_ids, None):
            write_rrd(config.output_dir / key.rrd_name, key.segment)
            write_rrd(config.output_dir / "sapiens2" / key.rrd_name, key.segment)
        calls.append("align")

    def calibrate(config) -> None:
        dataset = CatalogClient(config.catalog_url).get_dataset(DATASET_NAME)
        assert sorted(dataset.segment_ids()) == sorted(SEGMENTS)
        assert all(layers == ["base", "sapiens2"] for layers in dataset.segment_table().to_arrow_table()["rerun_layer_names"].to_pylist())
        assert config.donor_catalog_url == "rerun+http://source:1"
        assert config.output_dir == tmp_path / "out" / "calib"
        calls.append("calibrate")

    def keypoints(config) -> None:
        assert sorted(CatalogClient(config.catalog_url).get_dataset(DATASET_NAME).segment_ids()) == sorted(SEGMENTS)
        assert (config.calib_dir, config.output_dir, config.rate) == (tmp_path / "out" / "calib", tmp_path / "out" / "keypoints", 30.0)
        calls.append("keypoints")

    monkeypatch.setattr(post.align, "main", align)
    monkeypatch.setattr(post.calibrate, "main", calibrate)
    monkeypatch.setattr(post.keypoints, "main", keypoints)
    post.main(
        post.Config(base_dir=tmp_path / "base", output_dir=tmp_path / "out", rate=30.0, segment_ids=SEGMENTS, catalog_url="rerun+http://source:1")
    )
    assert calls == ["align", "calibrate", "keypoints"]


def test_post_refuses_an_existing_output(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        post.main(post.Config(base_dir=tmp_path / "base", output_dir=tmp_path / "out", rate=2.0, segment_ids=SEGMENTS))
