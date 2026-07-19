import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from mv_api.api.catalog_prediction_layer import (
    CatalogPredictionLayerConfig,
    CatalogSegment,
    ExoCameraStream,
    PredictionRecordingInfo,
    ViewerScreenshotTarget,
    _log_prediction_frame,
    build_native_viewer_screenshot_command,
    build_prediction_recording_info,
    build_prediction_rrd_path,
    build_viewer_screenshot_targets,
    capture_native_viewer_screenshots,
    catalog_segments_from_dataset,
    detect_uniform_native_fps,
    discover_exo_camera_streams,
    index_value_to_time_ns,
    native_fps_from_packet_ns,
    none_decoded_exo_stream_names,
    prediction_visualization_colors,
    register_prediction_layer,
    save_exo_viewer_blueprint,
    select_catalog_segment,
    validate_exo_camera_calibration,
    write_viewer_validation_notes,
)


class _Descriptor:
    def __init__(self, entity_path: str, component: str) -> None:
        self.entity_path: str = entity_path
        self.component: str = component


class _Schema:
    def __init__(self, descriptors: list[_Descriptor]) -> None:
        self._descriptors: list[_Descriptor] = descriptors

    def component_columns(self) -> dict[_Descriptor, Any]:
        return {descriptor: object() for descriptor in self._descriptors}


class _SegmentDataFrame:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def collect(self) -> list[pa.RecordBatch]:
        return self._table.to_batches()


class _DatasetEntry:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def segment_table(self) -> _SegmentDataFrame:
        return _SegmentDataFrame(self._table)

    def segment_url(self, recording_id: str) -> str:
        return f"rerun+http://127.0.0.1:9988/recording/{recording_id}"


class _RegistrationHandle:
    def __init__(self) -> None:
        self.wait_called: bool = False

    def wait(self) -> None:
        self.wait_called = True


class _LayerDatasetEntry:
    def __init__(self) -> None:
        self.registration_handle: _RegistrationHandle = _RegistrationHandle()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def register(self, recording_uri: list[str], **kwargs: Any) -> _RegistrationHandle:
        self.calls.append((recording_uri, kwargs))
        return self.registration_handle


class _PacketReader:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def select(self, *_columns: str) -> "_PacketReader":
        return self

    def limit(self, _count: int) -> "_PacketReader":
        return self

    def to_arrow_table(self) -> pa.Table:
        return self._table


class _EntityView:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def reader(self, index: str) -> _PacketReader:
        return _PacketReader(self._table)


class _NativeFpsDatasetEntry:
    def __init__(self, packet_ns_by_entity: dict[str, list[int]]) -> None:
        self._by_entity: dict[str, list[int]] = packet_ns_by_entity

    def filter_segments(self, _recording_id: str) -> "_NativeFpsDatasetEntry":
        return self

    def filter_contents(self, entity_path: str) -> _EntityView:
        return _EntityView(pa.table({"video_time": pa.array(self._by_entity[entity_path], type=pa.int64())}))


def test_select_catalog_segment_defaults_to_first_sorted_assembly101_row() -> None:
    rows: list[CatalogSegment] = [
        CatalogSegment(
            dataset="assembly101",
            sequence_key="all/z-sequence",
            recording_id="assembly101__all__z-sequence",
            recording_uri="rerun+http://127.0.0.1:9988/recording/z",
            path=Path("/catalog/assembly101/all/z-sequence.rrd"),
        ),
        CatalogSegment(
            dataset="assembly101",
            sequence_key="all/a-sequence",
            recording_id="assembly101__all__a-sequence",
            recording_uri="rerun+http://127.0.0.1:9988/recording/a",
            path=Path("/catalog/assembly101/all/a-sequence.rrd"),
        ),
    ]

    selected: CatalogSegment = select_catalog_segment(rows)

    assert selected.sequence_key == "all/a-sequence"
    assert selected.recording_id == "assembly101__all__a-sequence"


def test_catalog_segments_from_dataset_preserves_registered_recording_ids_and_metadata() -> None:
    segment_table: pa.Table = pa.table(
        {
            "rerun_segment_id": [
                "assembly101__all__fallback",
                "assembly101__all__metadata",
            ],
            "property:info:sequence_key": [
                None,
                ["all/from-metadata"],
            ],
        }
    )
    dataset_entry: _DatasetEntry = _DatasetEntry(segment_table)

    rows: list[CatalogSegment] = catalog_segments_from_dataset(
        dataset_entry,
        dataset_dir=Path("/catalog/assembly101"),
        dataset_name="assembly101",
    )

    assert [row.sequence_key for row in rows] == ["all/fallback", "all/from-metadata"]
    assert [row.recording_id for row in rows] == ["assembly101__all__fallback", "assembly101__all__metadata"]
    assert rows[1].recording_uri == "rerun+http://127.0.0.1:9988/recording/assembly101__all__metadata"
    assert rows[1].path == Path("/catalog/assembly101/all/from-metadata.rrd")


def test_build_prediction_recording_info_uses_source_segment_recording_id() -> None:
    config: CatalogPredictionLayerConfig = CatalogPredictionLayerConfig(application_id="catalog-app")
    segment: CatalogSegment = CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )

    recording_info: PredictionRecordingInfo = build_prediction_recording_info(config=config, segment=segment)

    assert recording_info.application_id == "catalog-app"
    assert recording_info.recording_id == "assembly101__all__a-sequence"


def test_register_prediction_layer_replaces_duplicate_layers() -> None:
    from rerun.catalog import OnDuplicateSegmentLayer

    dataset_entry: _LayerDatasetEntry = _LayerDatasetEntry()

    handle: _RegistrationHandle = register_prediction_layer(
        dataset_entry,
        rrd_path=Path("/tmp/prediction.rrd"),
        layer_name="mvapi_coco133_upper_body_v1",
    )

    assert handle.wait_called is True
    assert len(dataset_entry.calls) == 1
    recording_uri, kwargs = dataset_entry.calls[0]
    assert recording_uri == [Path("/tmp/prediction.rrd").resolve().as_uri()]
    assert kwargs == {
        "layer_name": "mvapi_coco133_upper_body_v1",
        "on_duplicate": OnDuplicateSegmentLayer.REPLACE,
    }


def test_native_fps_from_packet_ns_detects_uniform_rate() -> None:
    step_ns: int = 1_000_000_000 // 60
    packet_ns: np.ndarray = np.arange(0, 20 * step_ns, step_ns, dtype=np.int64)

    assert round(native_fps_from_packet_ns(packet_ns)) == 60


def test_native_fps_from_packet_ns_rejects_single_packet() -> None:
    with pytest.raises(ValueError, match="Need >=2"):
        native_fps_from_packet_ns(np.array([0], dtype=np.int64))


def test_native_fps_from_packet_ns_rejects_irregular_spacing() -> None:
    packet_ns: np.ndarray = np.array([0, 10, 20, 30, 200], dtype=np.int64)

    with pytest.raises(ValueError, match="too irregular"):
        native_fps_from_packet_ns(packet_ns)


def _exo_stream(name: str) -> ExoCameraStream:
    entity: str = f"/world/exo/{name}/pinhole/video"
    return ExoCameraStream(
        name=name,
        video_entity=entity,
        field_path=f"{entity}:VideoStream:sample",
        pinhole_entity=f"/world/exo/{name}/pinhole",
        transform_entity=f"/world/exo/{name}",
    )


def _packet_ns_at_fps(fps: float, count: int = 30) -> list[int]:
    step_ns: int = int(1_000_000_000 / fps)
    return [i * step_ns for i in range(count)]


def _native_fps_segment() -> CatalogSegment:
    return CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )


def test_detect_uniform_native_fps_returns_shared_exo_rate() -> None:
    streams: list[ExoCameraStream] = [_exo_stream("C10095"), _exo_stream("C10115")]
    entry: _NativeFpsDatasetEntry = _NativeFpsDatasetEntry({stream.video_entity: _packet_ns_at_fps(60.0) for stream in streams})

    assert round(detect_uniform_native_fps(dataset_entry=entry, segment=_native_fps_segment(), streams=streams)) == 60


def test_detect_uniform_native_fps_rejects_mixed_rate_exo_streams() -> None:
    streams: list[ExoCameraStream] = [_exo_stream("C10095"), _exo_stream("C10115")]
    entry: _NativeFpsDatasetEntry = _NativeFpsDatasetEntry(
        {
            streams[0].video_entity: _packet_ns_at_fps(60.0),
            streams[1].video_entity: _packet_ns_at_fps(30.0),
        }
    )

    with pytest.raises(ValueError, match="do not share one native fps"):
        detect_uniform_native_fps(dataset_entry=entry, segment=_native_fps_segment(), streams=streams)


def test_discover_exo_camera_streams_uses_sorted_video_stream_fields() -> None:
    schema: _Schema = _Schema(
        [
            _Descriptor("/world/ego/e1/pinhole/video", "rerun.archetypes.VideoStream:sample"),
            _Descriptor("/world/exo/C10115/pinhole/video", "rerun.archetypes.VideoStream:sample"),
            _Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.VideoStream:sample"),
        ]
    )

    streams = discover_exo_camera_streams(schema)

    assert [stream.name for stream in streams] == ["C10095", "C10115"]
    assert [stream.field_path for stream in streams] == [
        "/world/exo/C10095/pinhole/video:VideoStream:sample",
        "/world/exo/C10115/pinhole/video:VideoStream:sample",
    ]


def test_discover_exo_camera_streams_fails_loudly_for_asset_video_only() -> None:
    schema: _Schema = _Schema(
        [
            _Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.AssetVideo:blob"),
        ]
    )

    with pytest.raises(ValueError, match="AssetVideo.*VideoStream:sample"):
        discover_exo_camera_streams(schema)


def test_none_decoded_exo_stream_names_skips_pre_keyframe_samples_but_requires_keys() -> None:
    streams = discover_exo_camera_streams(
        _Schema(
            [
                _Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.VideoStream:sample"),
                _Descriptor("/world/exo/C10115/pinhole/video", "rerun.archetypes.VideoStream:sample"),
            ]
        )
    )

    none_streams: list[str] = none_decoded_exo_stream_names({"C10095": None, "C10115": object()}, streams)

    assert none_streams == ["C10095"]
    with pytest.raises(ValueError, match="missing required exo camera key 'C10115'"):
        none_decoded_exo_stream_names({"C10095": object()}, streams)


def test_validate_exo_camera_calibration_fails_when_pinhole_is_missing() -> None:
    schema: _Schema = _Schema(
        [
            _Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.VideoStream:sample"),
            _Descriptor("/world/exo/C10095", "rerun.archetypes.Transform3D:translation"),
            _Descriptor("/world/exo/C10095", "rerun.archetypes.Transform3D:mat3x3"),
        ]
    )
    streams = discover_exo_camera_streams(schema)

    with pytest.raises(ValueError, match="C10095.*Pinhole:image_from_camera"):
        validate_exo_camera_calibration(schema, streams)


def test_validate_exo_camera_calibration_fails_when_transform_is_missing() -> None:
    schema: _Schema = _Schema(
        [
            _Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.VideoStream:sample"),
            _Descriptor("/world/exo/C10095/pinhole", "rerun.archetypes.Pinhole:image_from_camera"),
            _Descriptor("/world/exo/C10095/pinhole", "rerun.archetypes.Pinhole:camera_xyz"),
            _Descriptor("/world/exo/C10095/pinhole", "rerun.archetypes.Pinhole:resolution"),
        ]
    )
    streams = discover_exo_camera_streams(schema)

    with pytest.raises(ValueError, match="C10095.*Transform3D:translation"):
        validate_exo_camera_calibration(schema, streams)


def test_build_prediction_rrd_path_uses_dataset_sequence_and_layer_name() -> None:
    segment: CatalogSegment = CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )

    rrd_path: Path = build_prediction_rrd_path(
        output_root=Path("packages/mv-api/artifacts/catalog_layers"),
        segment=segment,
        layer_name="mvapi_coco133_upper_body_v1",
    )

    assert rrd_path == Path(
        "packages/mv-api/artifacts/catalog_layers/assembly101/all/a-sequence/mvapi_coco133_upper_body_v1.rrd"
    )


def test_build_viewer_screenshot_targets_creates_one_target_per_exo_camera() -> None:
    schema: _Schema = _Schema(
        [
            _Descriptor("/world/exo/C10115/pinhole/video", "rerun.archetypes.VideoStream:sample"),
            _Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.VideoStream:sample"),
        ]
    )
    streams = discover_exo_camera_streams(schema)

    targets = build_viewer_screenshot_targets(
        run_dir=Path("packages/mv-api/artifacts/rerun-viewer-validation/run-1"),
        streams=streams,
    )

    assert [target.camera_name for target in targets] == ["C10095", "C10115"]
    assert [target.screenshot_path for target in targets] == [
        Path("packages/mv-api/artifacts/rerun-viewer-validation/run-1/exo_C10095.png"),
        Path("packages/mv-api/artifacts/rerun-viewer-validation/run-1/exo_C10115.png"),
    ]
    assert [target.blueprint_path for target in targets] == [
        Path("packages/mv-api/artifacts/rerun-viewer-validation/run-1/exo_C10095.rbl"),
        Path("packages/mv-api/artifacts/rerun-viewer-validation/run-1/exo_C10115.rbl"),
    ]
    assert [target.overlay_entity for target in targets] == [
        "/world/exo/C10095/pinhole/pred/mvapi/coco133_uv",
        "/world/exo/C10115/pinhole/pred/mvapi/coco133_uv",
    ]


def test_native_viewer_screenshot_command_uses_new_viewer_instance() -> None:
    segment: CatalogSegment = CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )
    target: ViewerScreenshotTarget = build_viewer_screenshot_targets(
        run_dir=Path("packages/mv-api/artifacts/rerun-viewer-validation/run-1"),
        streams=discover_exo_camera_streams(
            _Schema([_Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.VideoStream:sample")])
        ),
    )[0]

    command: list[str] = build_native_viewer_screenshot_command(
        target=target,
        segment=segment,
        window_size="1920x1080",
    )

    assert "rerun" in command
    assert "--new" in command
    assert command[command.index("--window-size") + 1] == "1920x1080"
    assert "--screenshot-to" in command
    assert "rerun+http://127.0.0.1:9988/recording/a" in command


def test_catalog_prediction_layer_config_defaults_match_spec() -> None:
    config: CatalogPredictionLayerConfig = CatalogPredictionLayerConfig()

    assert config.rrd_root == Path("/mnt/8tb/data/exoego-forge-catalog")
    assert config.catalog_url == "rerun+http://127.0.0.1:9988"
    assert config.assembly101_row_id == 120
    assert config.max_frames == 10
    assert config.video_codec == "av1"
    assert config.keyframe_interval == 300
    assert config.native_fps_override is None
    assert config.output_root == Path("artifacts/catalog_layers")
    assert config.layer_name == "mvapi_coco133_upper_body_v1"
    assert config.register_layer is True
    assert config.capture_native_viewer_screenshots is True


def test_pixi_catalog_environment_and_task_are_wired_for_dataloader_lane() -> None:
    repo_root: Path = Path(__file__).resolve().parents[3]
    pixi_data: dict[str, Any] = tomllib.loads((repo_root / "pixi.toml").read_text())

    rerun_sdk: dict[str, Any] = pixi_data["feature"]["rerun-prerelease"]["pypi-dependencies"]["rerun-sdk"]
    catalog_task: dict[str, Any] = pixi_data["feature"]["mv-api-catalog"]["tasks"]["mv-api-catalog-prediction-layer"]
    catalog_dev_env: dict[str, Any] = pixi_data["environments"]["mv-api-catalog-dev"]

    assert set(rerun_sdk["extras"]) >= {"datafusion", "dataloader"}
    assert catalog_task["cmd"] == "python tools/apps/catalog_prediction_layer.py"
    assert catalog_task["cwd"] == "packages/mv-api"
    assert {"rerun-prerelease", "mv-api-catalog", "dev"}.issubset(set(catalog_dev_env["features"]))


def test_index_value_to_time_ns_accepts_dataloader_datetime64_and_integer_values() -> None:
    assert index_value_to_time_ns(np.datetime64(123456789, "ns")) == 123456789
    assert index_value_to_time_ns(987654321) == 987654321


def test_write_viewer_validation_notes_records_every_required_exo_screenshot(tmp_path: Path) -> None:
    segment: CatalogSegment = CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )
    targets = [
        *build_viewer_screenshot_targets(
            run_dir=tmp_path,
            streams=[
                # Construct directly to keep this test focused on notes output.
                discover_exo_camera_streams(
                    _Schema([_Descriptor("/world/exo/C10095/pinhole/video", "rerun.archetypes.VideoStream:sample")])
                )[0],
                discover_exo_camera_streams(
                    _Schema([_Descriptor("/world/exo/C10115/pinhole/video", "rerun.archetypes.VideoStream:sample")])
                )[0],
            ],
        )
    ]

    notes_path: Path = write_viewer_validation_notes(
        run_dir=tmp_path,
        command="pixi run -e mv-api-catalog --frozen rerun ...",
        catalog_url="rerun+http://127.0.0.1:9988",
        segment=segment,
        rrd_path=Path("artifacts/catalog_layers/assembly101/all/a-sequence/mvapi_coco133_upper_body_v1.rrd"),
        layer_name="mvapi_coco133_upper_body_v1",
        targets=targets,
    )

    notes_text: str = notes_path.read_text()
    assert "exo_C10095.png" in notes_text
    assert "exo_C10115.png" in notes_text
    assert "exo_C10095.rbl" in notes_text
    assert "exo_C10115.rbl" in notes_text
    assert "/world/exo/C10095/pinhole/pred/mvapi/coco133_uv" in notes_text
    assert "/world/exo/C10115/pinhole/pred/mvapi/coco133_uv" in notes_text


def test_save_exo_viewer_blueprint_fixes_2d_visual_bounds(tmp_path: Path) -> None:
    from rerun.experimental import RrdReader

    target: ViewerScreenshotTarget = ViewerScreenshotTarget(
        camera_name="C10095",
        overlay_entity="/world/exo/C10095/pinhole/pred/mvapi/coco133_uv",
        screenshot_path=tmp_path / "exo_C10095.png",
        blueprint_path=tmp_path / "exo_C10095.rbl",
    )

    blueprint_path: Path = save_exo_viewer_blueprint(target)

    blueprint_store: Any = RrdReader(blueprint_path).blueprints()[0]
    summary: str = str(RrdReader(blueprint_path).stream(store=blueprint_store).collect().summary())
    assert "VisualBounds2D:range" in summary


def test_capture_native_viewer_screenshots_uses_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    segment: CatalogSegment = CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )
    target: ViewerScreenshotTarget = ViewerScreenshotTarget(
        camera_name="C10095",
        overlay_entity="/world/exo/C10095/pinhole/pred/mvapi/coco133_uv",
        screenshot_path=tmp_path / "exo_C10095.png",
        blueprint_path=tmp_path / "exo_C10095.rbl",
    )
    observed_timeout: dict[str, float | None] = {}

    def save_blueprint(_: ViewerScreenshotTarget, **__: Any) -> Path:
        target.blueprint_path.write_bytes(b"blueprint")
        return target.blueprint_path

    def run_command(*_: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed_timeout["value"] = kwargs.get("timeout")
        target.screenshot_path.write_bytes(b"png")
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr("mv_api.api.catalog_prediction_layer.save_exo_viewer_blueprint", save_blueprint)
    monkeypatch.setattr("mv_api.api.catalog_prediction_layer.subprocess.run", run_command)

    capture_native_viewer_screenshots(
        targets=[target],
        segment=segment,
        window_size="1920x1080",
        timeout_seconds=7.5,
    )

    assert observed_timeout["value"] == 7.5


def test_prediction_layer_rows_are_column_logged_on_video_time_only(tmp_path: Path) -> None:
    import rerun as rr
    from rerun.experimental import RrdReader

    stream: ExoCameraStream = ExoCameraStream(
        name="C10095",
        video_entity="/world/exo/C10095/pinhole/video",
        field_path="/world/exo/C10095/pinhole/video:VideoStream:sample",
        pinhole_entity="/world/exo/C10095/pinhole",
        transform_entity="/world/exo/C10095",
    )
    positions_3d: np.ndarray = np.zeros((133, 4), dtype=np.float32)
    positions_3d[:, 0:3] = np.arange(133, dtype=np.float32)[:, np.newaxis]
    positions_3d[:, 3] = 0.95
    positions_2d: np.ndarray = np.zeros((1, 133, 3), dtype=np.float32)
    positions_2d[0, :, 0:2] = np.arange(133, dtype=np.float32)[:, np.newaxis]
    positions_2d[0, :, 2] = 0.95
    mv_state: SimpleNamespace = SimpleNamespace(xyzc_t=positions_3d, uvc_t=positions_2d)
    rrd_path: Path = tmp_path / "predictions.rrd"
    recording: rr.RecordingStream = rr.RecordingStream(application_id="test", recording_id="rec")
    recording.save(rrd_path)

    _log_prediction_frame(
        mv_state=mv_state,
        streams=[stream],
        top_half_mask=np.ones(133, dtype=bool),
        keypoint_threshold=0.5,
        timestamp_ns=123_456_789,
        recording=recording,
    )
    recording.flush()

    summary: str = str(RrdReader(rrd_path).stream().collect().summary())
    assert "/world/pred/mvapi/coco133_xyz rows=1 static=False timelines=['video_time']" in summary
    assert "/world/exo/C10095/pinhole/pred/mvapi/coco133_uv rows=1 static=False timelines=['video_time']" in summary
    assert "timelines=['log_tick', 'log_time', 'video_time']" not in summary


def test_prediction_visualization_colors_are_red_for_catalog_overlays() -> None:
    colors: np.ndarray = prediction_visualization_colors(num_keypoints=4)

    assert colors.dtype == np.uint8
    assert colors.shape == (4, 3)
    np.testing.assert_array_equal(colors, np.full((4, 3), [255, 0, 0], dtype=np.uint8))
