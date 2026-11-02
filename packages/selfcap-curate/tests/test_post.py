"""One-process post: calibrate and keypoints read align's local output, donors stay on the source or donor catalog."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import rerun as rr
from dataforge.schema import video_path
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_video import CatalogVideo

import selfcap_curate.apis.align as align_api
import selfcap_curate.apis.post as post
from selfcap_curate.catalog_io import DATASET_NAME, Segment, SessionVideos
from selfcap_curate.records import SelfcapLayout, episode_keys

SEGMENTS = ("selfcap__2025-12-11__6d3740b8__ep001", "selfcap__2025-12-11__6d3740b8__ep005")


def write_rrd(path: Path, segment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rr.RecordingStream("post-test", recording_id=segment) as recording:
        recording.save(path)
        recording.log("/world/marker", rr.AnyValues(value=1), static=True)


def write_aligned(config: align_api.Config, session_videos: SessionVideos) -> None:
    """What align leaves for the later stages: cached packets, the local bases and their sapiens2 layers."""
    for key in episode_keys(config.segment_ids, None):
        video: CatalogVideo = CatalogVideo(
            np.array([0, 33_333_333, 66_666_666], dtype="timedelta64[ns]"),
            [np.array([1], dtype=np.uint8)] * 3,
            [True, False, False],
            "av1",
        )
        session_videos[key.segment] = Segment(
            Mock(spec=DatasetEntry), key.segment, SelfcapLayout(2, 3, ("A", "B")), {video_path(0, 0): video, video_path(1, 0): video}
        )
        write_rrd(config.output_dir / key.rrd_name, key.segment)
        write_rrd(config.output_dir / "sapiens2" / key.rrd_name, key.segment)


@pytest.mark.integration
@pytest.mark.parametrize("rate", [2.0, None])
def test_post_serves_align_output_to_calibrate_and_keypoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rate: float | None) -> None:
    calls: list[str] = []

    shared: SessionVideos | None = None

    def align(config, *, session_videos) -> None:
        nonlocal shared
        shared = session_videos
        assert (config.base_dir, config.catalog_url, config.segment_ids) == (tmp_path / "base", "rerun+http://source:1", SEGMENTS)
        write_aligned(config, session_videos)
        calls.append("align")

    def calibrate(config, *, session_videos) -> None:
        assert session_videos is shared
        dataset = CatalogClient(config.catalog_url).get_dataset(DATASET_NAME)
        assert sorted(dataset.segment_ids()) == sorted(SEGMENTS)
        assert all(layers == ["base", "sapiens2"] for layers in dataset.segment_table().to_arrow_table()["rerun_layer_names"].to_pylist())
        assert config.donor_catalog_url == "rerun+http://source:1"
        assert config.output_dir == tmp_path / "out" / "calib"
        calls.append("calibrate")

    def keypoints(config, *, session_videos) -> None:
        assert session_videos is shared
        assert sorted(CatalogClient(config.catalog_url).get_dataset(DATASET_NAME).segment_ids()) == sorted(SEGMENTS)
        assert (config.calib_dir, config.output_dir, config.rate) == (tmp_path / "out" / "calib", tmp_path / "out" / "keypoints", rate)
        calls.append("keypoints")

    monkeypatch.setattr(post.align, "main", align)
    monkeypatch.setattr(post.calibrate, "main", calibrate)
    monkeypatch.setattr(post.keypoints, "main", keypoints)
    post.main(
        post.Config(base_dir=tmp_path / "base", output_dir=tmp_path / "out", rate=rate, segment_ids=SEGMENTS, catalog_url="rerun+http://source:1")
    )
    assert calls == ["align", "calibrate", "keypoints"]


@pytest.mark.integration
def test_post_gives_each_stage_the_values_of_the_hand_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage-by-stage command lines: --exo-method head --oak-egohand-assets --donor-catalog-url on calibrate only."""
    segment: tuple[str, ...] = SEGMENTS[:1]
    seen: dict[str, align_api.Config | post.calibrate.Config | post.keypoints.Config] = {}

    def align(config, *, session_videos) -> None:
        seen["align"] = config
        write_aligned(config, session_videos)

    monkeypatch.setattr(post.align, "main", align)
    monkeypatch.setattr(post.calibrate, "main", lambda config, *, session_videos: seen.update(calibrate=config))
    monkeypatch.setattr(post.keypoints, "main", lambda config, *, session_videos: seen.update(keypoints=config))
    post.main(
        post.Config(
            base_dir=tmp_path / "base",
            output_dir=tmp_path / "out",
            segment_ids=segment,
            catalog_url="rerun+http://local:1",
            dataset="selfcap-candidate",
            donor_catalog_url="rerun+http://donor:2",
            exo_method="head",
            oak_egohand_assets=tmp_path / "egohand",
        )
    )
    aligned, calibrated, keypointed = seen["align"], seen["calibrate"], seen["keypoints"]
    assert (aligned.base_dir, aligned.catalog_url, aligned.dataset, aligned.segment_ids) == (
        tmp_path / "base",
        "rerun+http://local:1",
        "selfcap-candidate",
        segment,
    )
    assert calibrated.donor_catalog_url == "rerun+http://donor:2"
    assert (calibrated.exo_method, calibrated.oak_egohand_assets, calibrated.segment_ids) == ("head", tmp_path / "egohand", segment)
    # Align's output is served privately under the default name, so the later stages never see the source dataset.
    assert calibrated.catalog_url == keypointed.catalog_url != "rerun+http://local:1"
    assert calibrated.dataset == keypointed.dataset == DATASET_NAME
    assert keypointed.segment_ids == segment


@pytest.mark.integration
def test_post_reads_landmark_labels_from_the_catalog_calibrate_gates_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_server) -> None:
    write_rrd(tmp_path / "reference.rrd", SEGMENTS[0])
    read_from: list[str] = []

    def read_landmarks(dataset: DatasetEntry, segment: str) -> None:
        read_from.append(dataset.name)
        assert segment in dataset.segment_ids()

    monkeypatch.setattr(post.align, "main", lambda config, *, session_videos: write_aligned(config, session_videos))
    monkeypatch.setattr(post, "read_landmarks", read_landmarks)
    monkeypatch.setattr(post.calibrate, "main", lambda config, *, session_videos: None)
    monkeypatch.setattr(post.keypoints, "main", lambda config, *, session_videos: None)
    with private_server({"reference": [tmp_path / "reference.rrd"]}) as donor:
        post.main(
            post.Config(
                base_dir=tmp_path / "base",
                output_dir=tmp_path / "out",
                segment_ids=SEGMENTS[:1],
                catalog_url="rerun+http://unreachable:1",
                donor_catalog_url=donor.url(),
                exo_refine=True,
                exo_reference_dataset="reference",
            )
        )
    assert read_from == ["reference"]


def test_post_refuses_an_existing_output(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        post.main(post.Config(base_dir=tmp_path / "base", output_dir=tmp_path / "out", rate=2.0, segment_ids=SEGMENTS))


def test_the_egohand_assets_are_an_optional_override(tmp_path: Path) -> None:
    assert post.Config(tmp_path / "base", tmp_path / "out").oak_egohand_assets is None
