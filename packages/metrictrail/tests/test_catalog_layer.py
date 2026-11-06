"""The catalog layer schema (entity paths, timeline, properties), its refusals, and registration (fake dataset)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from rerun.catalog import OnDuplicateSegmentLayer
from rerun.chunk import RrdReader

from metrictrail.artifacts import ColoredCloud
from metrictrail.catalog.layer import LayerInputs, LayerProperties, backproject_keyframes, layer_entities, segment_id, write_layer
from metrictrail.catalog.register import layer_uris, register_layer_run, register_layers


def layer_inputs(count: int = 20) -> LayerInputs:
    world_T_cam: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (count, 1, 1))
    world_T_cam[:, 0, 3] = np.linspace(0.0, 1.0, count)
    depth: Float32[ndarray, "8 10"] = np.full((8, 10), 2.0, dtype=np.float32)
    return LayerInputs(
        world_T_cam=world_T_cam,
        times_ns=(1_000_000_000 + np.arange(count) * 33_333_333).astype(np.int64),
        intrinsics_px=np.array([10.0, 10.0, 5.0, 4.0]),
        raster_hw=(8, 10),
        keyframe_rows=np.array([0, 10], dtype=np.int64),
        keyframe_depth_m=[depth, depth],
        points=ColoredCloud(points_xyz=np.zeros((3, 3), dtype=np.float32), colors_rgb=np.zeros((3, 3), dtype=np.uint8)),
        gt_T_est_se3=np.eye(4),
    )


def layer_properties() -> LayerProperties:
    return LayerProperties(
        producer="unit test (synthetic)",
        vipe_commit="ead2e00",
        repo_commit=None,
        env="env",
        host="host",
        utc="2026-11-06T00:00:00Z",
        dataset_path="frames",
        run_output_dir="run",
        run_record="run/vipe/desk_metrictrail.json",
        upstream_case="desk",
        seed=1234,
        seeds=[1234, 1235, 1236],
        slam_preset="metrictrail_gba_refine",
        raster_policy="unchanged",
        tracking_stride=1,
        raster_hw=[8, 10],
        intrinsics_source="sharedk",
        calibration_status="stable_gt_free",
        k_fx_fy_cx_cy=[10.0, 10.0, 5.0, 4.0],
        num_frames=20,
        num_tracked=20,
        num_keyframes=2,
        num_frames_not_in_base=0,
        time_source="rgb.csv",
        display_alignment="se3",
        se3_translation=[0.0, 0.0, 0.0],
        se3_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        alignment_gt_match="frame index",
        sim3_scale=1.0,
        sim3_translation=[0.0, 0.0, 0.0],
        sim3_quaternion_xyzw=[0.0, 0.0, 0.0, 1.0],
        paper_ate_m=0.01,
        paper_num_pairs=20,
        se3_ate_m=0.02,
        vslamlab_ate_m=None,
        vslamlab_num_evaluated=None,
        paper_ate_m_per_seed=[0.01, 0.011, 0.012],
        paper_ate_m_mean=0.011,
        paper_ate_m_std=0.001,
        paper_table="Table 9",
        paper_dataset_value=4.4,
        paper_units="cm",
        depth_source="depth zip",
        points_voxel_m=0.01,
        runtime_s=None,
        tracking_fps=None,
        max_memory_allocated_gib=None,
    )


def test_layer_schema(tmp_path: Path) -> None:
    target: Path = write_layer(tmp_path / "layer.rrd", "vslam-lab-rgbdtum__desk", "metrictrail", layer_inputs(), layer_properties())
    reader = RrdReader(target)
    assert [entry.recording_id for entry in reader.recordings()] == ["vslam-lab-rgbdtum__desk"]
    assert [entry.application_id for entry in reader.recordings()] == ["dataforge"]
    schema = reader.store().schema()
    assert set(layer_entities("metrictrail").values()) <= set(schema.entity_paths())
    assert [index.name for index in schema.index_columns()] == ["video_time"]
    columns: set[str] = set(schema.column_names())
    assert {
        "property:metrictrail:paper_ate_m",
        "property:metrictrail:sim3_scale",
        "property:metrictrail:producer",
        "property:metrictrail:seeds",
    } <= columns
    assert "property:metrictrail:vslamlab_ate_m" not in columns  # null fields are left out
    assert "/world/runs/metrictrail/cam/pinhole/depth:DepthImage:buffer" in columns
    assert not any(name.startswith("property:RecordingInfo") for name in columns)  # a derived layer adds no RecordingInfo


def test_layer_is_never_overwritten(tmp_path: Path) -> None:
    write_layer(tmp_path / "layer.rrd", "s", "metrictrail_upstream", layer_inputs(), layer_properties())
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_layer(tmp_path / "layer.rrd", "s", "metrictrail_upstream", layer_inputs(), layer_properties())


def test_layer_refuses_unsorted_time(tmp_path: Path) -> None:
    inputs: LayerInputs = layer_inputs()
    inputs.times_ns[3] = inputs.times_ns[2]
    with pytest.raises(ValueError, match="non-increasing"):
        write_layer(tmp_path / "layer.rrd", "s", "metrictrail", inputs, layer_properties())
    assert not (tmp_path / "layer.rrd").exists()


def test_segment_id_matches_dataforge_vslam_lab() -> None:
    assert segment_id("RGBDTUM", "rgbd_dataset_freiburg1_desk") == "vslam-lab-rgbdtum__rgbd_dataset_freiburg1_desk"


def test_backproject_keyframes_puts_a_pixel_at_its_depth() -> None:
    world_T_cam: Float64[ndarray, "1 4 4"] = np.eye(4)[None].copy()
    world_T_cam[0, :3, 3] = [1.0, 0.0, 0.0]
    depth: Float32[ndarray, "4 4"] = np.full((4, 4), 2.0, dtype=np.float32)
    rgb: UInt8[ndarray, "4 4 3"] = np.full((4, 4, 3), 9, dtype=np.uint8)
    cloud: ColoredCloud = backproject_keyframes(
        world_T_cam,
        np.array([0], dtype=np.int64),
        [depth],
        [rgb],
        np.array([2.0, 2.0, 0.0, 0.0]),
        stride_px=4,
        depth_range_m=(0.2, 10.0),
        voxel_m=0.01,
    )
    np.testing.assert_allclose(cloud.points_xyz, [[1.0, 0.0, 2.0]])  # pixel (0, 0) on the optical axis, 2 m deep
    assert cloud.colors_rgb.tolist() == [[9, 9, 9]]


class FakeOperation:
    def __init__(self) -> None:
        self.waited: bool = False

    def wait(self) -> None:
        self.waited = True


class FakeTable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows: list[dict[str, object]] = rows

    def to_arrow_table(self) -> pa.Table:
        return pa.Table.from_pylist(self.rows)


class FakeDataset:
    """Holds base + gt for one segment; registration appends or replaces a layer, like the server."""

    def __init__(self, *segments: str) -> None:
        self.layers: dict[str, dict[str, str]] = {segment: {"base": "file:///base.rrd", "gt": "file:///gt.rrd"} for segment in segments}
        self.duplicates: list[OnDuplicateSegmentLayer] = []
        self.operation = FakeOperation()

    def register(
        self,
        recording_uri: list[str],
        *,
        layer_name: str | Sequence[str] = "base",
        on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.ERROR,
    ) -> FakeOperation:
        self.duplicates.append(on_duplicate)
        for uri in recording_uri:
            segment: str = Path(uri.removeprefix("file://")).stem  # the layer file is named after its segment
            if str(layer_name) in self.layers[segment] and on_duplicate == OnDuplicateSegmentLayer.ERROR:
                raise RuntimeError(f"{segment} already holds {layer_name}")
            self.layers[segment][str(layer_name)] = uri
        return self.operation

    def segment_ids(self) -> list[str]:
        return list(self.layers)

    def segment_table(self) -> FakeTable:
        return FakeTable(
            [
                {"rerun_segment_id": segment, "rerun_layer_names": list(layers), "rerun_storage_urls": list(layers.values())}
                for segment, layers in self.layers.items()
            ]
        )


def test_register_layers_registers_checks_and_never_creates_segments(tmp_path: Path) -> None:
    layer: Path = tmp_path / "vslam-lab-rgbdtum__desk.rrd"
    layer.write_bytes(b"")
    dataset = FakeDataset("vslam-lab-rgbdtum__desk")
    register_layers(dataset, [layer], ["vslam-lab-rgbdtum__desk"], "metrictrail", replace=False)
    assert dataset.layers["vslam-lab-rgbdtum__desk"]["metrictrail"] == layer.as_uri()
    assert dataset.duplicates == [OnDuplicateSegmentLayer.ERROR] and dataset.operation.waited
    with pytest.raises(ValueError, match="register dataforge's base and gt layers first"):
        register_layers(dataset, [layer], ["vslam-lab-rgbdtum__xyz"], "metrictrail", replace=True)
    with pytest.raises(FileNotFoundError):
        layer_uris([tmp_path / "missing.rrd"])


def test_register_layer_run_replaces_only_segments_that_already_hold_the_layer(tmp_path: Path) -> None:
    old_run, new_run = tmp_path / "old", tmp_path / "new"
    for run, segments in ((old_run, ("vslam-lab-rgbdtum__desk",)), (new_run, ("vslam-lab-rgbdtum__desk", "vslam-lab-rgbdtum__xyz"))):
        run.mkdir()
        for segment in segments:
            (run / f"{segment}.rrd").write_bytes(b"")
    dataset = FakeDataset("vslam-lab-rgbdtum__desk", "vslam-lab-rgbdtum__xyz")
    assert register_layer_run(dataset, sorted(old_run.glob("*.rrd")), "metrictrail") == {"vslam-lab-rgbdtum__desk": False}
    assert register_layer_run(dataset, sorted(new_run.glob("*.rrd")), "metrictrail") == {
        "vslam-lab-rgbdtum__xyz": False,
        "vslam-lab-rgbdtum__desk": True,
    }
    assert dataset.duplicates == [OnDuplicateSegmentLayer.ERROR, OnDuplicateSegmentLayer.ERROR, OnDuplicateSegmentLayer.REPLACE]
    assert {segment: layers["metrictrail"] for segment, layers in dataset.layers.items()} == {
        "vslam-lab-rgbdtum__desk": (new_run / "vslam-lab-rgbdtum__desk.rrd").as_uri(),
        "vslam-lab-rgbdtum__xyz": (new_run / "vslam-lab-rgbdtum__xyz.rrd").as_uri(),
    }
