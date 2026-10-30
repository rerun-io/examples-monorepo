"""Sapiens2 layers preserve sparse detections and both clocks."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dataforge.catalog_swap import swap_layer
from rerun.chunk import RrdReader

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_phone_detections, read_time_sync
from selfcap_curate.detection_io import read_oak_observations, write_detections_layer
from selfcap_curate.records import PhoneDetections, SelfcapLayout


def test_oak_observations_preserve_pixels_and_reject_another_segment(tmp_path):
    path = tmp_path / "hands.parquet"
    xy = np.full((3, 133, 2), np.nan)
    xy[:, 91:133] = 200.0
    scores = np.zeros((3, 133))
    scores[:, 91:133] = 0.9
    pq.write_table(pa.table({
        "segment_id": ["segment-a"] * 3,
        "video_time": pa.array([100_000_000, 150_000_000, 200_000_000], type=pa.int64()),
        "xy": xy.tolist(), "scores": scores.tolist(),
    }), path)
    result = read_oak_observations(path, "segment-a")
    np.testing.assert_allclose(result.video_time, [0.1, 0.15, 0.2])
    np.testing.assert_equal(result.xy, xy)
    np.testing.assert_equal(result.scores, scores)
    with pytest.raises(ValueError, match="segment-a"):
        read_oak_observations(path, "segment-b")


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
    }
    batch = next(c.to_record_batch() for c in chunks if not c.is_static)
    assert batch.column("capture_time").cast("int64").to_pylist() == [40_000_001, 80_000_003]
    assert batch.column("video_time").cast("int64").to_pylist() == [40_000_003, 80_000_005]
    np.testing.assert_array_equal(np.asarray(batch.column("Points2D:positions").to_pylist()), xy)
    np.testing.assert_array_equal(np.asarray(batch.column("confidence").to_pylist()), scores)
    assert batch.column("Points2D:keypoint_ids")[0].as_py() == list(range(133))


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
