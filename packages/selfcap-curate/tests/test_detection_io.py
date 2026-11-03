"""Sapiens2 layers preserve sparse detections and both clocks."""

from dataclasses import replace

import numpy as np
import pytest
from dataforge.catalog_swap import swap_layer
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from simplecv.rerun_custom_types import confidence_scores_to_rgb

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_phone_detections, read_time_sync
from selfcap_curate.detection_io import write_detections_layer, write_oak_egohand_layer
from selfcap_curate.records import OakDetections, PhoneDetections, SelfcapLayout


def test_egohand_layer_retains_sparse_video_times_and_confidence(tmp_path):
    path = tmp_path / "egohand.rrd"
    xy = np.full((2, 133, 2), np.nan)
    xy[:, 91:112] = 10.0
    confidence = np.zeros((2, 133))
    confidence[:, 91:112] = 0.8
    detections = OakDetections(np.array([1.0, 4.0]), xy, confidence)
    write_oak_egohand_layer(path, "episode", 4, {"rgb": detections, "left": detections, "right": detections})
    reader = RrdReader(path)
    assert reader.recordings()[0].recording_id == "episode"
    chunks = list(reader.stream())
    assert all("/__properties" not in str(chunk.entity_path) for chunk in chunks)
    data = next(chunk.to_record_batch() for chunk in chunks if not chunk.is_static)
    assert data.column("video_time").cast("int64").to_pylist() == [1_000_000_000, 4_000_000_000]
    assert data.column("confidence").to_pylist() == confidence.tolist()
    assert {str(c.entity_path) for c in chunks if not c.is_static} == {f"/world/rig_04/cam_{cam:02d}/pinhole/pred/egohand_calib/coco133_uv" for cam in range(3)}


def test_layer_preserves_all_joints_clocks_and_has_no_properties(tmp_path, fixture_sync):
    sync = replace(fixture_sync(), offsets_ms={"exo/A": 0.0000025})
    xy = np.arange(532, dtype=np.float32).reshape(2, 133, 2).astype(np.float64)
    xy[0, 10] = np.nan
    scores = np.full((2, 133), float(np.float16(0.7)))
    scores[0, 10] = 0.0
    hand = PhoneDetections(np.array([0.040000001, 0.080000003]), xy, scores)
    path = tmp_path / sync.episode_key.rrd_name
    write_detections_layer(path, sync, SelfcapLayout(1, 2, ("A",)), {"A": hand})
    reader = RrdReader(path)
    assert reader.recordings()[0].application_id == "dataforge"
    assert reader.recordings()[0].recording_id == sync.episode_key.segment
    chunks = list(reader.stream())
    assert {str(c.entity_path) for c in chunks} == {
        "/world/rig_00/cam_00/pinhole/pred/sapiens2",
        "/world/rig_00/cam_00/pinhole/pred/sapiens2/coco133_uv",
        "/world/rig_00/cam_00/pinhole/pred/sapiens2/coco133_uv_min70",
    }
    batch = next(c.to_record_batch() for c in chunks if not c.is_static and str(c.entity_path).endswith("coco133_uv"))
    assert batch.column("capture_time").cast("int64").to_pylist() == [40_000_001, 80_000_003]
    assert batch.column("video_time").cast("int64").to_pylist() == [40_000_003, 80_000_005]
    np.testing.assert_array_equal(np.asarray(batch.column("Points2D:positions").to_pylist()), xy)
    # Points2DWithConfidence: the raw score (float16 values are exact in float32) and the shared score colours.
    np.testing.assert_array_equal(np.asarray(batch.column("simplecv.KeypointConfidence2D:confidences").to_pylist()), scores)
    colors = np.asarray(batch.column("Points2D:colors").to_pylist(), dtype=np.uint32).reshape(2, 133)
    expected = confidence_scores_to_rgb(scores.reshape(1, -1, 1))[0].astype(np.uint32)
    np.testing.assert_array_equal(colors, ((expected[:, 0] << 24) | (expected[:, 1] << 16) | (expected[:, 2] << 8) | 255).reshape(2, 133))
    assert colors[0, 10] != colors[0, 11]  # the missed joint is coloured as zero confidence
    assert batch.column("Points2D:keypoint_ids")[0].as_py() == list(range(133))
    # Points the viewer can see: 3 px dots without labels, as dataforge.hands draws COCO-133 keypoints.
    style = next(c.to_record_batch() for c in chunks if c.is_static and str(c.entity_path).endswith("coco133_uv"))
    assert style.column("Points2D:radii").to_pylist() == [[3.0]]
    assert style.column("Points2D:show_labels").to_pylist() == [[False]]


def test_layer_shows_only_points_at_or_above_0_7(tmp_path, fixture_sync):
    # The full set stays at coco133_uv; the layout draws coco133_uv_min70, which drops the points under 0.7.
    sync = replace(fixture_sync(), offsets_ms={"exo/A": 0.0})
    xy = np.arange(532, dtype=np.float32).reshape(2, 133, 2).astype(np.float64)
    scores = np.full((2, 133), 0.9)
    scores[0, :3] = [0.69, 0.7, 0.0]
    scores[1, 5] = 0.2
    path = tmp_path / sync.episode_key.rrd_name
    write_detections_layer(path, sync, SelfcapLayout(1, 2, ("A",)), {"A": PhoneDetections(np.array([0.04, 0.08]), xy, scores)})
    chunks = list(RrdReader(path).stream())
    shown = next(c.to_record_batch() for c in chunks if not c.is_static and str(c.entity_path).endswith("coco133_uv_min70"))
    ids = [row.as_py() for row in shown.column("Points2D:keypoint_ids")]
    assert ids == [[1, *range(3, 133)], [*range(5), *range(6, 133)]]  # each kept point keeps its id, so links still join
    positions = [np.asarray(row.as_py()) for row in shown.column("Points2D:positions")]
    np.testing.assert_array_equal(positions[0], xy[0, ids[0]])
    np.testing.assert_array_equal(positions[1], xy[1, ids[1]])
    confidences = [row.as_py() for row in shown.column("simplecv.KeypointConfidence2D:confidences")]
    np.testing.assert_allclose(confidences[0], scores[0, ids[0]], rtol=1e-6)
    style = next(c.to_record_batch() for c in chunks if c.is_static and str(c.entity_path).endswith("coco133_uv_min70"))
    assert style.column("Points2D:radii").to_pylist() == [[3.0]]
    assert style.column("Points2D:show_labels").to_pylist() == [[False]]


@pytest.mark.integration
def test_catalog_reader_uses_segment_layout_and_preserves_sparse_frames(tmp_path, fixture_sync, synthetic_base, private_server):
    sync = replace(fixture_sync(), offsets_ms={"exo/Z": -0.0000035})
    layout = SelfcapLayout(3, 4, ("Z", "A", "B"))
    hand = PhoneDetections(np.array([0.040000001, 0.080000003]), np.ones((2, 133, 2)), np.full((2, 133), float(np.float16(0.7))))
    base = synthetic_base(tmp_path / "base.rrd", segment=sync.episode_key.segment, phones=layout.phones)
    layer = tmp_path / sync.episode_key.rrd_name
    write_detections_layer(layer, sync, layout, {"Z": hand})
    other = synthetic_base(tmp_path / "other.rrd", segment=sync.episode_key.segment[:-3] + "003", phones=("Q", "R", "S", "T"))
    with private_server({DATASET_NAME: [base, other]}) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        dataset.register([layer.resolve().as_uri()], layer_name="sapiens2").wait()
        result = read_phone_detections(open_segment(dataset, sync.episode_key.segment))
        assert tuple(result) == ("Z",)
        np.testing.assert_allclose(result["Z"].times_s, hand.times_s, rtol=0.0, atol=1e-9)
        np.testing.assert_array_equal(result["Z"].xy, hand.xy)
        np.testing.assert_array_equal(result["Z"].scores, hand.scores)
        assert read_phone_detections(open_segment(dataset, sync.episode_key.segment[:-3] + "003")) == {}


@pytest.mark.integration
def test_align_and_calibrate_read_the_full_set_with_real_scores(tmp_path, fixture_sync, synthetic_base, private_server):
    # read_phone_detections feeds align and calibrate, which apply their own gates: 0.69 must arrive as 0.69.
    sync = replace(fixture_sync(), offsets_ms={"exo/Z": 0.0})
    layout = SelfcapLayout(3, 4, ("Z", "A", "B"))
    xy = np.arange(532, dtype=np.float64).reshape(2, 133, 2)
    scores = np.full((2, 133), float(np.float16(0.9)))
    scores[0, :2] = [float(np.float16(0.69)), float(np.float16(0.71))]
    base = synthetic_base(tmp_path / "base.rrd", segment=sync.episode_key.segment, phones=layout.phones)
    layer = tmp_path / sync.episode_key.rrd_name
    write_detections_layer(layer, sync, layout, {"Z": PhoneDetections(np.array([0.04, 0.08]), xy, scores)})
    with private_server({DATASET_NAME: [base]}) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        dataset.register([layer.resolve().as_uri()], layer_name="sapiens2").wait()
        result = read_phone_detections(open_segment(dataset, sync.episode_key.segment))
    np.testing.assert_array_equal(result["Z"].xy, xy)
    np.testing.assert_array_equal(result["Z"].scores, scores)


@pytest.mark.integration
def test_catalog_reader_still_reads_layers_written_with_plain_confidence(tmp_path, fixture_sync, synthetic_base, private_server):
    # Layers written before Points2DWithConfidence keep their score in a plain "confidence" AnyValues column.
    import rerun as rr
    from dataforge.logging_toolkit import capture_column, time_column

    sync = fixture_sync()
    layout = SelfcapLayout(3, 4, ("A", "B", "C"))
    base = synthetic_base(tmp_path / "base.rrd", segment=sync.episode_key.segment, phones=layout.phones)
    stamps = np.array([40_000_001], dtype=np.int64)
    scores = np.full(133, 0.5)
    entity = "/world/rig_00/cam_00/pinhole/pred/sapiens2/coco133_uv"
    columns = rr.Points2D.columns(positions=np.ones((133, 2), np.float32), keypoint_ids=np.arange(133, dtype=np.uint16)).partition([133])
    plain = rr.AnyValues.columns(confidence=[scores.tolist()])
    old = Chunk.from_columns(entity, indexes=[capture_column(stamps), time_column(stamps)], columns=[*columns, *plain])
    layer = tmp_path / sync.episode_key.rrd_name
    LazyChunkStream.from_iter([old]).write_rrd(layer, application_id="dataforge", recording_id=sync.episode_key.segment)
    with private_server({DATASET_NAME: [base]}) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        dataset.register([layer.resolve().as_uri()], layer_name="sapiens2").wait()
        np.testing.assert_array_equal(read_phone_detections(open_segment(dataset, sync.episode_key.segment))["A"].scores, scores[None])


@pytest.mark.integration
def test_swap_layer_can_register_a_previously_absent_layer(tmp_path, fixture_sync, synthetic_base, private_server):
    sync = fixture_sync()
    phones = tuple(sync.exo_fit)
    layout = SelfcapLayout(len(phones), len(phones) + 1, phones)
    hand = PhoneDetections(np.array([0.04]), np.ones((1, 133, 2)), np.ones((1, 133)))
    base = synthetic_base(tmp_path / "base.rrd", segment=sync.episode_key.segment, phones=phones, sync=sync)
    layer = tmp_path / sync.episode_key.rrd_name
    write_detections_layer(layer, sync, layout, {phones[0]: hand})
    with private_server({DATASET_NAME: [base]}) as server:
        entry = server.client().get_dataset(DATASET_NAME)
        segment = open_segment(entry, sync.episode_key.segment)
        assert read_phone_detections(segment) == {}
        swap_layer(entry, "sapiens2", [layer], idle_probe=lambda: True)
        np.testing.assert_array_equal(read_phone_detections(segment)[phones[0]].xy, hand.xy)
        assert read_time_sync(entry, sync.episode_key.segment).key == sync.key
