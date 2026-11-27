"""The slam-rs result layer writes its runs under /world/runs on camera time, and animates the base rig only without a gt layer."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
import rerun.chunk as rrc
from dataforge import schema
from fixture_types import Rows, RowsReader
from rerun.catalog import DatasetEntry
from simplecv.catalog_layers import atomic_recording
from simplecv.rerun_walk_path import TRAIL, TRAJECTORY

from slam_rs import catalog_layer
from slam_rs.apis import catalog_layer as catalog_layer_api
from slam_rs.catalog_feed import CatalogSegment
from slam_rs.catalog_layer import write_layer
from slam_rs.follow_frame import follow_entity
from slam_rs.tracking import DroppedFrameset, DropReason, SegmentRun
from slam_rs.trajectory import Trajectory, empty_trajectory


@pytest.fixture
def run() -> SegmentRun:
    """Two finite poses on an absolute inertial clock."""
    return SegmentRun(
        estimate=Trajectory(
            t_ns=np.array([8_014_902_432, 9_014_902_432], dtype=np.int64),
            position_m=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
            quaternion_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        ),
        framesets=2, lost=0, wall_s=0.1, ground_truth=empty_trajectory(), median_tracker_ms=10.0, config_sha256="test-config", calibration_sha256="test-calibration",
    )


@pytest.mark.parametrize("run_root", ["/world/runs/slam_rs", "/world/runs/slam_rs_vio"])
def test_layer_writes_both_runs_on_camera_time(tmp_path: Path, read_rows: RowsReader, run: SegmentRun, run_root: str) -> None:
    """Absolute inertial poses return to camera time under the final and the raw VIO run; with the back end off (no ``final``) they are equal."""
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", run, clock_offset_ns=7_014_902_432, profile="fast", backend="cpu")
    rows: Rows = read_rows(path)
    assert [row.t_ns for row in rows[f"{run_root}/rig"]] == [1_000_000_000, 2_000_000_000]
    np.testing.assert_allclose(rows[f"{run_root}/rig"][-1].values["Transform3D:translation"], [[1, 2, 3]])
    np.testing.assert_allclose(rows[f"{run_root}/rig"][-1].values["Transform3D:quaternion"], [[0, 0, 0, 1]])
    assert [row.t_ns for row in rows[f"{run_root}/trail"]] == [1_000_000_000, 2_000_000_000], "a trail row per pose"


def test_the_final_run_is_the_re_anchored_trajectory_and_the_raw_run_the_vio(tmp_path: Path, read_rows: RowsReader, run: SegmentRun) -> None:
    """``slam_rs`` holds :attr:`SegmentRun.final`, ``slam_rs_vio`` the raw estimate; a base rig without ``gt`` follows the final run."""
    final: Trajectory = replace(run.estimate, position_m=run.estimate.position_m + np.array([0.0, 0.0, 0.5]))
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", replace(run, final=final), clock_offset_ns=0, profile="fast", backend="cpu", animate_base_rig=True)
    rows: Rows = read_rows(path)
    np.testing.assert_allclose(rows["/world/runs/slam_rs/rig"][-1].values["Transform3D:translation"], [[1, 2, 3.5]])
    np.testing.assert_allclose(rows["/world/runs/slam_rs_vio/rig"][-1].values["Transform3D:translation"], [[1, 2, 3]])
    assert rows["/world/rig_00"] == rows["/world/runs/slam_rs/rig"]
    with pytest.raises(ValueError, match="timestamps"):
        write_layer(tmp_path / "bad.rrd", "session", replace(run, final=replace(final, t_ns=final.t_ns + 1)), clock_offset_ns=0, profile="fast", backend="cpu")


def test_layer_leaves_the_base_rig_and_its_sensors_to_the_dataset(tmp_path: Path, run: SegmentRun) -> None:
    """The ground-truth layer owns ``/world/rig_00``: the result writes only under its two runs, no camera or IMU static."""
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", run, clock_offset_ns=0, profile="fast", backend="cpu")
    entities: set[str] = {chunk.entity_path for chunk in rrc.RrdReader(path).stream().collect().stream()}
    parts: tuple[str, ...] = ("rig", "trajectory", "trail", "endpoints/start", "endpoints/start_ring", "endpoints/end", "follow")
    assert entities == {"/", "/world/runs/slam_rs", *(f"/world/runs/{name}/{part}" for name in ("slam_rs", "slam_rs_vio") for part in parts)}


def test_without_ground_truth_the_final_run_also_animates_the_base_rig(tmp_path: Path, read_rows: RowsReader, run: SegmentRun) -> None:
    """A dataset with no ``gt`` layer (RoboCap) has no clash: the final run moves the rig its cameras and Follow view hang from."""
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", run, clock_offset_ns=7_014_902_432, profile="fast", backend="cpu", animate_base_rig=True)
    rows: Rows = read_rows(path)
    assert rows["/world/rig_00"] == rows["/world/runs/slam_rs/rig"]


def test_each_run_gets_its_level_follow_frame(tmp_path: Path, read_rows: RowsReader, run: SegmentRun) -> None:
    """``/world/runs/<run>/follow`` rides each run's position and turns about the world's up axis only, towards the walk;
    run-scoped, so it is written with or without the base rig. The final run's frame follows the final poses."""
    final: Trajectory = replace(run.estimate, position_m=run.estimate.position_m + np.array([0.0, 0.0, 0.5]))
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", replace(run, final=final), clock_offset_ns=0, profile="fast", backend="cpu")
    rows: Rows = read_rows(path)
    for name, top in (("slam_rs", 3.5), ("slam_rs_vio", 3.0)):
        follow: str = f"/world/runs/{name}/follow"
        assert [row.t_ns for row in rows[follow]] == [row.t_ns for row in rows[f"/world/runs/{name}/rig"]]
        np.testing.assert_allclose(rows[follow][-1].values["Transform3D:translation"], [[1, 2, top]])
        # 1 m along x and 2 m along y in the second between the poses: a heading of atan2(2, 1) = 63.43 deg about z.
        np.testing.assert_allclose(rows[follow][-1].values["Transform3D:quaternion"], [[0.0, 0.0, 0.52573, 0.85065]], atol=1e-5)


def test_the_run_scoped_names_are_the_ones_the_dataforge_layouts_read() -> None:
    """dataforge's layouts root follow views at ``schema.follow_path(source)`` and draw ``trail_path`` / ``trajectory_path``."""
    assert follow_entity("/world/runs/slam_rs") == schema.follow_path(schema.SLAM_RS_SOURCE) == "/world/runs/slam_rs/follow"
    # The replay's core restates the path names (it runs without dataforge): one vocabulary for both writers.
    assert f"/world/runs/slam_rs/{TRAIL}" == schema.trail_path(schema.SLAM_RS_SOURCE)
    assert f"/world/runs/slam_rs_vio/{TRAJECTORY}" == schema.trajectory_path(schema.SLAM_RS_VIO_SOURCE)


class _FakeCalibration:
    """Stands in for the compiled calibration; the tool only passes it on."""

    @staticmethod
    def from_catalog(*_args: object) -> "_FakeCalibration":
        return _FakeCalibration()


class _FakeVioConfig:
    """Stands in for the compiled VIO config."""

    @staticmethod
    def from_json(_text: str) -> "_FakeVioConfig":
        return _FakeVioConfig()


class _FakeVio:
    """A CPU estimator; the run itself comes from the patched ``drive``."""

    gpu: bool = False

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


@pytest.mark.parametrize(("segment_id", "ground_truth_uri", "writes_base_rig"), [
    ("msd-g2__MGO07", "file:///gt/MGO07.rrd", False),
    ("robocap__session", None, True),
])
def test_the_shipped_tool_moves_the_base_rig_only_without_ground_truth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run: SegmentRun, segment_id: str, ground_truth_uri: str | None, writes_base_rig: bool,
) -> None:
    """``slam-rs-catalog-layer`` decides the clash from the catalog: an MSD segment with a ``gt`` layer keeps
    ``/world/rig_00`` for the ground truth; a RoboCap session (no ``gt``) still has its rig animated by the final run.
    The masks are read from the segment's layers, by the entity name the tool is given."""
    dataset: MagicMock = MagicMock(spec=DatasetEntry)
    resolved: CatalogSegment = CatalogSegment("rerun+http://catalog", segment_id.split("__")[0], segment_id, dataset, ground_truth_uri)
    feed: SimpleNamespace = SimpleNamespace(
        export_offset_ns=0, video_time_offset_ns=0, imu=SimpleNamespace(cam_time_offset_ns=0), cameras=(), frame_t_ns=run.estimate.t_ns, camera_positions=(0, 1),
    )
    monkeypatch.setattr(catalog_layer_api, "resolve_catalog_segments", lambda _sources: (resolved,))
    monkeypatch.setattr(catalog_layer_api, "select_decode_device", lambda _request: "cpu")
    opened: dict[str, object] = {}

    def open_segment(*_args: object, **kwargs: object) -> nullcontext[SimpleNamespace]:
        opened.update(kwargs)
        return nullcontext(feed)

    monkeypatch.setattr(catalog_layer_api, "open_segment", open_segment)
    monkeypatch.setattr(catalog_layer_api, "_core", SimpleNamespace(Calibration=_FakeCalibration, VioConfig=_FakeVioConfig, Vio=_FakeVio))

    def tracked(rec: object, _feed: object, _vio: object, *, animate_base_rig: bool, **_kwargs: object) -> catalog_layer.LayerRun:
        """The driver's stand-in: the layer's poses as ``track_layer`` writes them, with the tool's base-rig decision."""
        catalog_layer.log_layer(rec, run, clock_offset_ns=0, profile="fast", backend="cpu", animate_base_rig=animate_base_rig)  # type: ignore[arg-type]
        return catalog_layer.LayerRun(run=run, stage=None)

    monkeypatch.setattr(catalog_layer_api, "track_layer", tracked)

    catalog_layer_api.main(
        catalog_layer_api.Config(segment=segment_id, output_dir=tmp_path, catalog="rerun+http://catalog", detail="poses", mask_entity="people_mask")
    )
    assert opened["mask_entity"] == "people_mask"

    output: Path = tmp_path / f"{segment_id}.rrd"
    entities: set[str] = {chunk.entity_path for chunk in rrc.RrdReader(output).stream().collect().stream()}
    assert ("/world/rig_00" in entities) is writes_base_rig
    assert {"/world/runs/slam_rs/rig", "/world/runs/slam_rs_vio/rig"} <= entities
    dataset.register.assert_called_once()
    assert dataset.register.call_args.args == ([output.resolve().as_uri()],)


def test_incomplete_run_does_not_replace_a_previous_result(tmp_path: Path, run: SegmentRun) -> None:
    path: Path = tmp_path / "result.rrd"
    write_layer(path, "session", run, clock_offset_ns=0, profile="fast", backend="cpu")
    previous: bytes = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        write_layer(path, "session", replace(run, lost=1), clock_offset_ns=0, profile="fast", backend="cpu")
    assert path.read_bytes() == previous


@pytest.mark.parametrize(("stamp", "reason"), [
    (1, DropReason.BEFORE_INITIALIZATION),
    (10_000_000_000, DropReason.IMU_ENDED),
])
def test_explained_boundary_drop_is_written(
    tmp_path: Path, run: SegmentRun, stamp: int,
    reason: DropReason,
) -> None:
    output: Path = tmp_path / "boundary.rrd"
    dropped: DroppedFrameset = DroppedFrameset(stamp, reason)
    write_layer(output, "session", replace(run, framesets=3, lost=1, dropped=(dropped,)), clock_offset_ns=0, profile="fast", backend="cpu")
    drop_batches: list[pa.RecordBatch] = [
        chunk.to_record_batch() for chunk in rrc.RrdReader(output).stream().filter(content="/world/runs/slam_rs/dropped").to_chunks()
    ]
    assert len(drop_batches) == 1
    assert drop_batches[0].column("t_ns").to_pylist() == [[stamp]]
    assert drop_batches[0].column("reasons").to_pylist() == [[reason]]


@pytest.mark.parametrize("masks", [(), ("/world/rig_00/cam_00/pinhole/people_mask", "/world/rig_00/cam_01/pinhole/people_mask")])
def test_the_layer_records_the_masks_its_run_tracked_with(tmp_path: Path, run: SegmentRun, masks: tuple[str, ...]) -> None:
    """A masked run replaces an unmasked one under the same layer name, so the run's metadata says which it was."""
    output: Path = tmp_path / "masks.rrd"
    with atomic_recording(output, application_id="dataforge", recording_id="session", send_properties=False) as rec:
        catalog_layer.log_layer(rec, run, clock_offset_ns=0, profile="fast", backend="cpu", mask_entities=masks)
    batches: list[pa.RecordBatch] = [
        batch for chunk in rrc.RrdReader(output).stream().filter(content="/world/runs/slam_rs").to_chunks()
        if "masked" in (batch := chunk.to_record_batch()).schema.names
    ]
    assert len(batches) == 1
    assert batches[0].column("masked").to_pylist() == [[bool(masks)]]
    listed: list[object] = batches[0].column("mask_entities").to_pylist() if "mask_entities" in batches[0].schema.names else [None]
    assert listed == ([list(masks)] if masks else [None])


def test_explained_interior_gap_still_refuses_the_layer(tmp_path: Path, run: SegmentRun) -> None:
    dropped: DroppedFrameset = DroppedFrameset(8_500_000_000, DropReason.IMU_ENDED)
    with pytest.raises(ValueError, match="incomplete"):
        write_layer(tmp_path / "gap.rrd", "session", replace(run, framesets=3, lost=1, dropped=(dropped,)), clock_offset_ns=0, profile="fast", backend="cpu")


def test_drop_reasons_keep_their_serialized_strings() -> None:
    from dataclasses import dataclass

    from serde import serde
    from serde.json import from_json, to_json

    @serde
    @dataclass(frozen=True, slots=True)
    class Record:
        reason: DropReason
        """Persisted cause of a missing pose."""

    for reason, text in (
        (DropReason.BEFORE_INITIALIZATION, "insufficient stereo support before initialization"),
        (DropReason.IMU_ENDED, "end of stream before IMU coverage"),
    ):
        encoded = to_json(Record(reason))
        assert encoded == '{"reason":"' + text + '"}'
        assert from_json(Record, encoded).reason is reason
