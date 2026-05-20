import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest

from mv_api.api.catalog_prediction_layer import (
    CatalogPredictionLayerConfig,
    CatalogSegment,
    CatalogTimeRange,
    ExoCameraStream,
    PredictionRecordingInfo,
    ViewerScreenshotTarget,
    _log_prediction_frame,
    _open_catalog,
    align_time_range_to_sample_grid,
    build_duration_video_time_sample_index,
    build_native_viewer_screenshot_command,
    build_prediction_recording_info,
    build_prediction_rrd_path,
    build_viewer_screenshot_targets,
    capture_native_viewer_screenshots,
    catalog_segments_from_dataset,
    discover_exo_camera_streams,
    index_value_to_time_ns,
    intersect_time_ranges,
    none_decoded_exo_stream_names,
    prediction_visualization_colors,
    register_prediction_layer,
    rgb_chw_to_bgr_hwc,
    save_exo_viewer_blueprint,
    select_catalog_segment,
    select_video_time_target_values,
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


def test_register_prediction_layer_fails_on_duplicates_by_default() -> None:
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
    assert kwargs == {"layer_name": "mvapi_coco133_upper_body_v1"}


def test_intersect_time_ranges_uses_common_exo_overlap() -> None:
    time_range: CatalogTimeRange = intersect_time_ranges(
        [
            CatalogTimeRange(start_ns=0, end_ns=100),
            CatalogTimeRange(start_ns=20, end_ns=80),
            CatalogTimeRange(start_ns=10, end_ns=90),
        ]
    )

    assert time_range == CatalogTimeRange(start_ns=20, end_ns=80)


def test_align_time_range_to_sample_grid_uses_first_valid_grid_point() -> None:
    aligned_range: CatalogTimeRange = align_time_range_to_sample_grid(
        segment_start_ns=0,
        segment_end_ns=99,
        ns_per_sample=10,
        time_range=CatalogTimeRange(start_ns=25, end_ns=64),
    )

    assert aligned_range == CatalogTimeRange(start_ns=30, end_ns=60)


def test_duration_video_time_sample_index_uses_int_nanosecond_grid_for_dataloader() -> None:
    segment: CatalogSegment = CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )

    sample_index = build_duration_video_time_sample_index(
        segment=segment,
        segment_time_range=CatalogTimeRange(start_ns=0, end_ns=99),
        time_range=CatalogTimeRange(start_ns=25, end_ns=64),
        sample_rate_hz=100_000_000.0,
    )

    segment_metadata, index_value = sample_index.global_to_local(2)

    assert sample_index.is_timestamp is False
    assert sample_index.ns_per_sample == 10
    assert sample_index.total_samples == 4
    assert segment_metadata.segment_id == "assembly101__all__a-sequence"
    assert index_value == np.timedelta64(50, "ns")
    assert sorted(sample_index.indices_in_range(30, 60)) == [30, 40, 50, 60]

    context_sample_index = build_duration_video_time_sample_index(
        segment=segment,
        segment_time_range=CatalogTimeRange(start_ns=0, end_ns=99),
        time_range=CatalogTimeRange(start_ns=25, end_ns=64),
        sample_rate_hz=100_000_000.0,
        context_index_values=np.array([0, 25, 50, 75], dtype=np.int64),
    )
    assert sorted(context_sample_index.indices_in_range(30, 60)) == [50]

    bootstrap_sample_index = build_duration_video_time_sample_index(
        segment=segment,
        segment_time_range=CatalogTimeRange(start_ns=0, end_ns=99),
        time_range=CatalogTimeRange(start_ns=0, end_ns=64),
        sample_rate_hz=100_000_000.0,
        bootstrap_window_ns=25,
    )
    _bootstrap_segment, bootstrap_index_value = bootstrap_sample_index.global_to_local(0)
    assert bootstrap_index_value == np.timedelta64(30, "ns")


def test_duration_video_time_sample_index_can_use_exact_video_packet_targets() -> None:
    segment: CatalogSegment = CatalogSegment(
        dataset="assembly101",
        sequence_key="all/a-sequence",
        recording_id="assembly101__all__a-sequence",
        recording_uri="rerun+http://127.0.0.1:9988/recording/a",
        path=Path("/catalog/assembly101/all/a-sequence.rrd"),
    )
    candidate_values: np.ndarray = np.array([0, 17, 33, 50, 67, 83], dtype=np.int64)
    target_values: np.ndarray = select_video_time_target_values(
        candidate_values=candidate_values,
        segment_time_range=CatalogTimeRange(start_ns=0, end_ns=83),
        time_range=CatalogTimeRange(start_ns=0, end_ns=83),
        sample_rate_hz=100_000_000.0,
        bootstrap_window_ns=25,
    )

    np.testing.assert_array_equal(target_values, np.array([33, 50, 67, 83], dtype=np.int64))
    assert set(target_values.tolist()).issubset(set(candidate_values.tolist()))

    sample_index = build_duration_video_time_sample_index(
        segment=segment,
        segment_time_range=CatalogTimeRange(start_ns=0, end_ns=83),
        time_range=CatalogTimeRange(start_ns=0, end_ns=83),
        sample_rate_hz=100_000_000.0,
        bootstrap_window_ns=25,
        context_index_values=candidate_values,
        target_index_values=target_values,
    )

    assert sample_index.total_samples == 4
    _first_segment, first_index_value = sample_index.global_to_local(0)
    _last_segment, last_index_value = sample_index.global_to_local(3)
    assert first_index_value == np.timedelta64(33, "ns")
    assert last_index_value == np.timedelta64(83, "ns")
    assert sorted(sample_index.indices_in_range(30, 55)) == [33, 50]


def test_video_time_target_selection_can_start_at_first_keyframe_packet() -> None:
    candidate_values: np.ndarray = np.array([0, 17, 33, 50, 67, 83], dtype=np.int64)
    target_values: np.ndarray = select_video_time_target_values(
        candidate_values=candidate_values,
        segment_time_range=CatalogTimeRange(start_ns=0, end_ns=83),
        time_range=CatalogTimeRange(start_ns=0, end_ns=83),
        sample_rate_hz=100_000_000.0,
        bootstrap_window_ns=0,
    )

    np.testing.assert_array_equal(target_values, np.array([0, 17, 33, 50, 67, 83], dtype=np.int64))


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


def test_rgb_chw_to_bgr_hwc_converts_channel_order_and_layout() -> None:
    rgb_chw: np.ndarray = np.array(
        [
            [[1, 2], [3, 4]],
            [[10, 20], [30, 40]],
            [[100, 110], [120, 130]],
        ],
        dtype=np.uint8,
    )

    bgr_hwc = rgb_chw_to_bgr_hwc(rgb_chw)

    expected_bgr_hwc: np.ndarray = np.array(
        [
            [[100, 10, 1], [110, 20, 2]],
            [[120, 30, 3], [130, 40, 4]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(bgr_hwc, expected_bgr_hwc)


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
    assert config.catalog_port == 9988
    assert config.catalog_optimize_for_catalog is True
    assert config.catalog_optimize_datasets == ("assembly101",)
    assert config.assembly101_row_id == 120
    assert config.max_frames == 10
    assert config.video_codec == "av1"
    assert config.keyframe_interval == 300
    assert config.output_root == Path("artifacts/catalog_layers")
    assert config.layer_name == "mvapi_coco133_upper_body_v1"
    assert config.register_layer is True
    assert config.capture_native_viewer_screenshots is True


def test_open_catalog_mounts_full_catalog_but_only_preoptimizes_target_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    client: object = object()

    class _Server:
        def client(self) -> object:
            return client

        def url(self) -> str:
            return "rerun+http://127.0.0.1:1234"

    def mount_catalog(
        rrd_root: Path,
        *,
        datasets: tuple[str, ...],
        port: int,
        application_id: str,
        show_progress: bool,
        optimize_for_catalog: bool,
        optimize_datasets: tuple[str, ...],
    ) -> _Server:
        calls.update(
            {
                "rrd_root": rrd_root,
                "datasets": datasets,
                "port": port,
                "application_id": application_id,
                "show_progress": show_progress,
                "optimize_for_catalog": optimize_for_catalog,
                "optimize_datasets": optimize_datasets,
            }
        )
        return _Server()

    simplecv_module: ModuleType = ModuleType("simplecv")
    apis_module: ModuleType = ModuleType("simplecv.apis")
    catalog_module: ModuleType = ModuleType("simplecv.apis.exoego_forge_catalog")
    cast(Any, catalog_module).mount_catalog = mount_catalog
    cast(Any, simplecv_module).apis = apis_module
    cast(Any, apis_module).exoego_forge_catalog = catalog_module
    monkeypatch.setitem(sys.modules, "simplecv", simplecv_module)
    monkeypatch.setitem(sys.modules, "simplecv.apis", apis_module)
    monkeypatch.setitem(sys.modules, "simplecv.apis.exoego_forge_catalog", catalog_module)
    config: CatalogPredictionLayerConfig = CatalogPredictionLayerConfig(
        rrd_root=tmp_path,
        catalog_port=1234,
        dataset_name="assembly101",
        application_id="catalog-app",
    )

    opened_client: Any
    server: Any
    catalog_url: str
    opened_client, server, catalog_url = _open_catalog(config)

    assert opened_client is client
    assert isinstance(server, _Server)
    assert catalog_url == "rerun+http://127.0.0.1:1234"
    assert calls["rrd_root"] == tmp_path.resolve()
    assert calls["datasets"] == ()
    assert calls["optimize_for_catalog"] is True
    assert calls["optimize_datasets"] == ("assembly101",)


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
