"""Dataset selection and two-camera output use SHOW3D's and HOT3D's catalog paths."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from dataforge.writing import blueprint_views
from rerun.catalog import DatasetEntry
from test_results import _track

from handtrack.apis import run_pipeline
from handtrack.apis.run_pipeline import RunConfig, select_segments
from handtrack.blueprint import handtrack_blueprint
from handtrack.data.catalog import HOT3D_QUEST3, SHOW3D, UMETRACK, CatalogDataError, SegmentInfo
from handtrack.results import load_track, save_track


def test_show3d_reader_keeps_native_pixels_and_exports_headset_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import dataclass

    import pyarrow as pa
    from dataforge.writing import atomic_recording
    from rerun.chunk import RrdReader
    from simplecv.catalog_video import CatalogVideo
    from test_catalog import _statics
    from test_segment_labels import _timeline

    from handtrack import pipeline, rerun_layers
    from handtrack.apis.export_layers import read_ground_truth
    from handtrack.data import catalog
    from handtrack.eval.segment import score_track
    from handtrack.oracle import GroundTruthViews, OracleDetector, OracleKeypoints
    from handtrack.results import TrackMetadata
    from handtrack.tracker import Tracker

    info = SegmentInfo(SHOW3D, "show3d_demo", "show3d", "grab", "train", "HLU829", 6, 60)
    timeline = _timeline(left_confidence=0.9, right_translation=(0.04, 0.0, 0.4))
    statics = _statics(fisheye=False)
    video = CatalogVideo(timeline.video_time_ns, [], [], "av1")
    paths: list[str] = []

    def videos(entry, segment, cameras, clock):
        paths.extend(cameras)
        return video, video

    monkeypatch.setattr(pipeline, "read_statics", lambda entry, selected: statics)
    monkeypatch.setattr(pipeline, "read_hand_timeline", lambda entry, selected, _static: timeline)
    monkeypatch.setattr(pipeline, "read_catalog_videos", videos)
    entry = MagicMock(spec=DatasetEntry)
    data = pipeline.read_segment(entry, info)
    assert paths == ["/world/rig_01/cam_00/pinhole/video", "/world/rig_01/cam_01/pinhole/video"]
    assert data.camera_angles == (0.0, 0.0)  # UmeTrack sees the native image with its own camera model (measured: -90 degrees broke the right hand)

    @dataclass
    class Batch:
        data: torch.Tensor

    class Decoder:
        def get_frames_at(self, indices: list[int]) -> Batch:
            pixels = torch.zeros((len(indices), 1, 1280, 1024), dtype=torch.uint8)
            pixels[:, :, 100:200, 300:400] = 231
            return Batch(pixels)

    monkeypatch.setattr(pipeline, "open_nvdec_decoder", lambda video, fps, device: Decoder())
    chunk = next(pipeline.decoded_frames(data, 3, torch.device("cpu")))
    assert len(chunk.native) == 2 and chunk.native[0].shape == (3, 1280, 1024)
    assert int(chunk.native[0][0, 150, 350]) == 231
    torch.testing.assert_close(chunk.net[:, 0], data.letterboxes[0].apply(chunk.native[0]))
    truth_views = GroundTruthViews.from_labels(data.labels)
    tracker = Tracker(data.rig, data.letterboxes, timeline.hand_model, 1.0, OracleDetector(truth_views), OracleKeypoints(truth_views, 1.0, 0.0, 0.0))
    run = pipeline.run_tracker(data, tracker, 3, torch.device("cpu"))
    track = pipeline.segment_track(data, run, TrackMetadata(info.segment_id, "oracle", "oracle", "known", 1.0, {}, dataset=SHOW3D))
    scores = score_track(track, data.labels, data.letterboxes)
    assert scores[0].mkpe_mm is not None and scores[0].mkpe_mm < 3.0
    assert len(scores[2]) == 2

    # Query the rig-1 GT boundary, with SHOW3D confidence and invalid-headset rules.
    timeline = replace(timeline, confidence=timeline.confidence.clone(), headset_valid=timeline.headset_valid.clone())
    timeline.confidence[1, 0] = 0.05
    timeline.headset_valid[2] = False
    monkeypatch.setattr(catalog, "read_statics", lambda entry, selected: statics)
    monkeypatch.setattr(catalog, "hand_timeline", lambda table, _static, selected: timeline)
    entry.filter_segments.return_value.filter_contents.return_value.reader.return_value.to_arrow_table.return_value = pa.table({
        "video_time": pa.array(timeline.video_time_ns, type=pa.duration("ns")), "frame_index": np.arange(6, dtype=np.int64)})
    truth = read_ground_truth(entry, info)
    selected_entities = entry.filter_segments.return_value.filter_contents.call_args.args[0]
    assert selected_entities[0] == "/world/rig_01"
    assert truth.present[0].all() and not truth.present[1, 0] and not truth.present[2].any()
    path = tmp_path / "show3d.rrd"
    with atomic_recording(path, recording_id=info.segment_id, send_properties=False) as recording:
        rerun_layers.write_handtrack_layer(recording, track, truth, truth.model)
    reader = RrdReader(path)
    entities = {str(chunk.entity_path) for chunk in reader.stream(store=reader.recordings()[0]).to_chunks()}
    assert "/world/rig_01/cam_01/pinhole/handtrack/keypoints/left" in entities
    assert not any(entity.startswith("/world/rig_00/") for entity in entities)


def test_show3d_selection_uses_heldout_subjects_and_explicit_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heldout = SegmentInfo(SHOW3D, "heldout", "show3d", "grab", "train", "HLU829", 3, 60)
    training = replace(heldout, segment_id="training", subject="OTHER")
    calls: list[str] = []

    def listed(entry: DatasetEntry, name: str) -> tuple[SegmentInfo, ...]:
        calls.append(name)
        return heldout, training

    monkeypatch.setattr(run_pipeline, "list_segments", listed)
    entry = MagicMock(spec=DatasetEntry)
    config = RunConfig(dataset=SHOW3D, split="val", domain="synthetic", output_root=tmp_path)
    assert select_segments(config, entry) == (heldout,)
    assert select_segments(replace(config, segments=("training", "heldout")), entry) == (training, heldout)
    assert calls == [SHOW3D, SHOW3D]
    with pytest.raises(CatalogDataError, match="no hand labels"):
        select_segments(replace(config, split="test"), entry)
    with pytest.raises(ValueError, match=SHOW3D):
        select_segments(replace(config, segments=("missing",)), entry)


def test_umetrack_synthetic_selection_keeps_testing_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic = SegmentInfo(UMETRACK, "synthetic", "synthetic", "hand_hand", "testing", "user_1", 3, 30)
    monkeypatch.setattr(run_pipeline, "list_segments", lambda entry, name: (synthetic, replace(synthetic, segment_id="real", domain="real")))
    assert select_segments(RunConfig(domain="synthetic", output_root=tmp_path), MagicMock(spec=DatasetEntry)) == (synthetic,)


def test_hot3d_selection_takes_every_labelled_scene_whatever_the_domain_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenes = tuple(SegmentInfo(HOT3D_QUEST3, f"hot3d-quest3__P000{i}_x", "hot3d", "", "train", f"P000{i}", 30, 30) for i in range(3))
    monkeypatch.setattr(run_pipeline, "list_segments", lambda entry, name: scenes)
    assert select_segments(RunConfig(dataset=HOT3D_QUEST3, domain="real", output_root=tmp_path), MagicMock(spec=DatasetEntry)) == scenes
    with pytest.raises(CatalogDataError, match="unseen test set"):
        select_segments(RunConfig(dataset=HOT3D_QUEST3, split="val", output_root=tmp_path), MagicMock(spec=DatasetEntry))


def test_hot3d_blueprint_uses_its_two_cameras_on_rig_0() -> None:
    origins = [str(view.origin) for view in blueprint_views(handtrack_blueprint(HOT3D_QUEST3))]
    assert origins[:3] == ["/world/rig_00", "/world/rig_00/cam_00/pinhole", "/world/rig_00/cam_01/pinhole"]
    assert len(origins) == 6


def test_show3d_track_roundtrip_and_blueprint_use_two_headset_cameras(tmp_path: Path) -> None:
    original = _track(3)
    track = replace(original, meta=replace(original.meta, dataset=SHOW3D, keypoints="umetrack"),
                    box=original.box[:, :2], box_source=original.box_source[:, :2],
                    keypoints_2d=original.keypoints_2d[:, :2], presence=original.presence[:, :2])
    loaded = load_track(save_track(track, tmp_path))
    np.testing.assert_array_equal(loaded.keypoints_2d, track.keypoints_2d)
    origins = [str(view.origin) for view in blueprint_views(handtrack_blueprint(SHOW3D))]
    assert origins[:3] == ["/world/rig_01", "/world/rig_01/cam_00/pinhole", "/world/rig_01/cam_01/pinhole"]
    assert len(origins) == 6


def test_umetrack_loads_no_keynet_and_records_pretrained_weights_and_calibration(tmp_path: Path) -> None:
    from serde.json import to_json

    from handtrack.reference.results import Calibration
    from handtrack.tracker import TrackerConfig

    weights = tmp_path / "pretrained.torch"
    weights.write_bytes(b"pretrained pose weights")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(to_json(Calibration(["umetrack__synthetic__hand_hand__training__user_10__recording_00"],
                                              "source", 10, 0.87, 0.8, 0.9, 1.0, 0.5, 1.5, 1)))
    config = RunConfig(dataset=SHOW3D, keypoints="umetrack", detector="oracle", umetrack_weights=weights,
                       umetrack_calibration=calibration, umetrack_root=tmp_path, umetrack_shim=tmp_path, checkpoints=tmp_path / "no-keynet",
                       output_root=tmp_path)
    networks = run_pipeline.load_networks(config, torch.device("cpu"))
    assert networks.keynet is None and networks.keynet_sha256 == run_pipeline.file_sha256(weights)
    identity = run_pipeline.ensure_run_identity(config, networks)
    assert len(identity) == 64
    with pytest.raises(ValueError, match="dataset"):
        run_pipeline.ensure_run_identity(replace(config, dataset=UMETRACK), networks)
    calibration.unlink()
    with pytest.raises(ValueError, match="--umetrack-calibration"):
        run_pipeline.ensure_run_identity(config, networks)
    with pytest.raises(ValueError, match="once per hand/frame"):
        replace(config, tracker=TrackerConfig(refine_shift=0.5))


def test_show3d_standalone_export_merges_without_a_projections_layer(tmp_path: Path) -> None:
    import pyarrow as pa
    import rerun as rr
    from dataforge.writing import atomic_recording
    from rerun.chunk import RrdReader

    from handtrack.apis.export_layers import export_clip

    segment = "show3d_demo"
    for layer in ("base", "hand_pose", "handtrack"):
        with atomic_recording(tmp_path / f"{layer}.rrd", recording_id=segment, send_properties=False) as recording:
            recording.log(f"/world/{layer}", rr.Points3D([[0.0, 0.0, 0.5]]))
    table = pa.table({"rerun_segment_id": [segment], "rerun_layer_names": [["base", "hand_pose"]],
                      "rerun_storage_urls": [[(tmp_path / f"{layer}.rrd").as_uri() for layer in ("base", "hand_pose")]]})
    target = tmp_path / "standalone.rrd"
    export_clip(table, segment, tmp_path / "handtrack.rrd", target, SHOW3D)
    reader = RrdReader(target)
    assert reader.blueprints() and reader.recordings()[0].recording_id == segment
    entities = {str(chunk.entity_path) for chunk in reader.stream(store=reader.recordings()[0]).to_chunks()}
    assert {"/world/base", "/world/hand_pose", "/world/handtrack"} <= entities
