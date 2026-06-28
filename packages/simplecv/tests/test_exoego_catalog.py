from __future__ import annotations

import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import tomllib

import simplecv.apis.exoego_forge_catalog as catalog_module
from simplecv.apis.exoego_forge_catalog import (
    CATALOG_CAMERA_NAMES,
    DEFAULT_CATALOG_DATASETS,
    DEFAULT_CATALOG_OPTIMIZE_DATASETS,
    DEFAULT_CATALOG_RRD_CACHE_DIR,
    MARKER_FLAG_COLUMN,
    TABLE_BLUEPRINT_METADATA_KEY,
    TABLE_CARD_PREVIEW_END_SECONDS,
    TABLE_CARD_PREVIEW_START_SECONDS,
    CatalogConfig,
    RRDIndexRow,
    _optimize_rrd_for_catalog,
    _register_default_dataset_blueprint,
    _table_preview_camera,
    build_exoego_catalog_blueprint,
    build_rrd_index_rows_from_dataset,
    build_rrd_index_rows_from_paths,
    build_rrd_index_table_blueprint,
    build_rrd_index_table_schema,
    build_table_card_blueprint,
    discover_rrd_paths,
    discover_rrd_uris,
    mount_catalog,
    table_name_for_dataset,
)
from simplecv.data.exoego.aria_gen2_pilot import AriaGen2PilotConfig, AriaGen2PilotSequence
from simplecv.data.exoego.assembly101 import Assembly101Config, Assembly101Sequence
from simplecv.data.exoego.ego_dex import EgoDexConfig, EgoDexSequence
from simplecv.data.exoego.epfl_smart_kitchen import EpflSmartKitchenConfig, EpflSmartKitchenSequence
from simplecv.data.exoego.hocap import HocapConfig, HocapSequence
from simplecv.data.exoego.hot3d import Hot3dConfig, Hot3dSequence
from simplecv.data.exoego.sequence_identity import SequenceIdentity
from simplecv.data.exoego.umetrack import UmeTrackConfig, UmeTrackSequence
from simplecv.rig import entity_id

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MONOREPO_ROOT: Path = PROJECT_ROOT.parents[1]


def test_sequence_identity_paths_and_recording_id() -> None:
    identity = SequenceIdentity(
        dataset="umetrack",
        parts=("real", "hand_hand", "testing", "user_05", "recording_09"),
    )

    assert identity.sequence_key == "real/hand_hand/testing/user_05/recording_09"
    assert identity.recording_id == "umetrack__real__hand_hand__testing__user_05__recording_09"
    assert identity.rrd_path(Path("data/exoego-forge-catalog")) == Path(
        "data/exoego-forge-catalog/umetrack/real/hand_hand/testing/user_05/recording_09.rrd"
    )


@pytest.mark.parametrize(
    ("dataset", "parts"),
    [
        ("bad__dataset", ("sequence",)),
        ("hocap", ("subject__1", "recording")),
        ("hocap", ("subject_1/bad__recording",)),
    ],
)
def test_sequence_identity_rejects_recording_id_separator(
    dataset: str,
    parts: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="cannot contain '__'"):
        SequenceIdentity(dataset=dataset, parts=parts)


@pytest.mark.parametrize(
    ("identity", "dataset", "sequence_key", "recording_id"),
    [
        (
            AriaGen2PilotSequence.sequence_identity_for_config(AriaGen2PilotConfig(sequence_name="cook_0")),
            "aria-gen2",
            "cook_0",
            "aria-gen2__cook_0",
        ),
        (
            Assembly101Sequence.sequence_identity_for_config(Assembly101Config(split=None, sequence_name="seq_01")),
            "assembly101",
            "all/seq_01",
            "assembly101__all__seq_01",
        ),
        (
            HocapSequence.sequence_identity_for_config(HocapConfig(subject_id="8", sequence_name="20231024_180733")),
            "hocap",
            "subject_8/20231024_180733",
            "hocap__subject_8__20231024_180733",
        ),
        (
            Hot3dSequence.sequence_identity_for_config(Hot3dConfig(headset="quest3", sequence_name="P0001_10a27bf7")),
            "hot3d-quest3",
            "P0001_10a27bf7",
            "hot3d-quest3__P0001_10a27bf7",
        ),
        (
            UmeTrackSequence.sequence_identity_for_config(
                UmeTrackConfig(
                    data_type="real",
                    hand_interaction="hand_hand",
                    split="testing",
                    user=5,
                    recording_id=9,
                )
            ),
            "umetrack",
            "real/hand_hand/testing/user_05/recording_09",
            "umetrack__real__hand_hand__testing__user_05__recording_09",
        ),
        (
            EgoDexSequence.sequence_identity_for_config(EgoDexConfig(split="test", sequence_name="add_remove_lid", episode=3)),
            "ego-dex",
            "test/add_remove_lid/episode_0003",
            "ego-dex__test__add_remove_lid__episode_0003",
        ),
        (
            EpflSmartKitchenSequence.sequence_identity_for_config(
                EpflSmartKitchenConfig(
                    split="train",
                    participant_id="YH2002",
                    session_name="2023_12_04_10_15_23",
                )
            ),
            "epfl-smart-kitchen",
            "train/YH2002/2023_12_04_10_15_23",
            "epfl-smart-kitchen__train__YH2002__2023_12_04_10_15_23",
        ),
    ],
)
def test_dataset_sequence_identity_for_config(
    identity: SequenceIdentity,
    dataset: str,
    sequence_key: str,
    recording_id: str,
) -> None:
    assert identity.dataset == dataset
    assert identity.sequence_key == sequence_key
    assert identity.recording_id == recording_id


def test_discover_rrd_paths_and_uris_group_by_dataset(tmp_path: Path) -> None:
    aria_rrd: Path = tmp_path / "aria-gen2" / "cook_0.rrd"
    hocap_rrd: Path = tmp_path / "hocap" / "subject_8" / "20231024_180733.rrd"
    hot3d_aria_rrd: Path = tmp_path / "hot3d-aria" / "P0001_4bf4e21a.rrd"
    skipped_rrd: Path = tmp_path / "ego100k" / "video.rrd"
    legacy_hot3d_rrd: Path = tmp_path / "hot3d" / "aria" / "P0001_4bf4e21a.rrd"
    for path in (aria_rrd, hocap_rrd, hot3d_aria_rrd, skipped_rrd, legacy_hot3d_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    paths: dict[str, list[Path]] = discover_rrd_paths(
        tmp_path,
        datasets=("aria-gen2", "hocap", "hot3d-aria", "missing"),
    )
    uris: dict[str, list[str]] = discover_rrd_uris(
        tmp_path,
        datasets=("aria-gen2", "hocap", "hot3d-aria", "missing"),
    )

    assert set(paths) == {"aria-gen2", "hocap", "hot3d-aria"}
    assert paths["aria-gen2"] == [aria_rrd.resolve()]
    assert paths["hocap"] == [hocap_rrd.resolve()]
    assert paths["hot3d-aria"] == [hot3d_aria_rrd.resolve()]
    assert set(uris) == {"aria-gen2", "hocap", "hot3d-aria"}
    assert uris["aria-gen2"] == [aria_rrd.resolve().as_uri()]
    assert uris["hocap"] == [hocap_rrd.resolve().as_uri()]
    assert uris["hot3d-aria"] == [hot3d_aria_rrd.resolve().as_uri()]


def test_discover_rrd_paths_raises_when_no_requested_dataset_has_rrds(tmp_path: Path) -> None:
    skipped_rrd: Path = tmp_path / "ego100k" / "video.rrd"
    skipped_rrd.parent.mkdir(parents=True)
    skipped_rrd.write_bytes(b"not a real rrd")

    with pytest.raises(FileNotFoundError):
        discover_rrd_paths(tmp_path, datasets=("aria-gen2", "hocap"))


def test_catalog_config_defaults_to_general_catalog_index() -> None:
    config: CatalogConfig = CatalogConfig()

    assert [field.name for field in fields(CatalogConfig)] == [
        "rrd_root",
        "datasets",
        "port",
        "application_id",
        "optimize_for_catalog",
        "catalog_rrd_cache_dir",
        "optimize_datasets",
        "open_browser",
        "web_port",
    ]
    assert config.rrd_root == Path("data/exoego-forge-catalog")
    assert config.datasets == DEFAULT_CATALOG_DATASETS
    assert config.datasets == (
        "aria-gen2",
        "assembly101",
        "epfl-smart-kitchen",
        "hocap",
        "hot3d-aria",
        "hot3d-quest3",
        "umetrack",
        "ego-dex",
    )
    assert config.optimize_for_catalog is True
    assert config.optimize_datasets == DEFAULT_CATALOG_OPTIMIZE_DATASETS
    assert set(config.optimize_datasets) == set(config.datasets)
    assert config.catalog_rrd_cache_dir == DEFAULT_CATALOG_RRD_CACHE_DIR


def test_pixi_catalog_task_skips_preoptimization_for_interactive_startup() -> None:
    pixi_text: str = (MONOREPO_ROOT / "pixi.toml").read_text()
    pixi_data: dict[str, Any] = tomllib.loads(pixi_text)
    catalog_task: dict[str, Any] = pixi_data["feature"]["simplecv"]["tasks"]["simplecv-catalog"]
    catalog_cmd: str = catalog_task["cmd"]

    assert "[feature.simplecv.tasks.simplecv-catalog]" in pixi_text
    assert catalog_cmd == "python tools/catalog.py --rrd-root /mnt/8tb/data/exoego-forge-catalog --no-optimize-for-catalog"
    assert "pre-optimizes missing RRD cache copies before Rerun registration" in pixi_text


@pytest.mark.parametrize(
    ("dataset_name", "table_name"),
    [
        ("aria-gen2", "aria_gen2_table"),
        ("assembly101", "assembly101_table"),
        ("epfl-smart-kitchen", "epfl_smart_kitchen_table"),
        ("hocap", "hocap_table"),
        ("hot3d-aria", "hot3d_aria_table"),
        ("hot3d-quest3", "hot3d_quest3_table"),
        ("umetrack", "umetrack_table"),
        ("ego-dex", "ego_dex_table"),
    ],
)
def test_table_name_for_dataset(dataset_name: str, table_name: str) -> None:
    assert table_name_for_dataset(dataset_name) == table_name


def test_hot3d_has_hand_labels_respects_no_gt_metadata(tmp_path: Path) -> None:
    seq_dir: Path = tmp_path / "P0016_0ca96b7b"
    seq_dir.mkdir()
    (seq_dir / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": False}))

    assert Hot3dSequence._has_hand_labels(seq_dir) is False


def test_hot3d_has_hand_labels_rejects_empty_files_when_gt_available(tmp_path: Path) -> None:
    seq_dir: Path = tmp_path / "P0015_e7458eb3"
    seq_dir.mkdir()
    (seq_dir / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": True}))
    (seq_dir / "umetrack_hand_user_profile.json").touch()
    (seq_dir / "umetrack_hand_pose_trajectory.jsonl").touch()

    with pytest.raises(AssertionError, match="Hand GT is marked available"):
        Hot3dSequence._has_hand_labels(seq_dir)


def test_hot3d_has_hand_labels_accepts_nonempty_gt_files(tmp_path: Path) -> None:
    seq_dir: Path = tmp_path / "P0015_e7458eb3"
    seq_dir.mkdir()
    (seq_dir / "metadata.json").write_text(json.dumps({"have_hand_object_pose_gt": True}))
    (seq_dir / "umetrack_hand_user_profile.json").write_text("{}")
    (seq_dir / "umetrack_hand_pose_trajectory.jsonl").write_text("{}\n")

    assert Hot3dSequence._has_hand_labels(seq_dir) is True


class _FakeSegmentTable:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def collect(self) -> list[pa.RecordBatch]:
        return self._table.to_batches()


class _FakeDatasetEntry:
    def __init__(self, table: pa.Table) -> None:
        self._table: pa.Table = table

    def segment_table(self) -> _FakeSegmentTable:
        return _FakeSegmentTable(self._table)

    def segment_url(self, recording_id: str) -> str:
        return f"rerun+http://127.0.0.1:9988/dataset/fake?segment_id={recording_id}"


class _FakeBlueprintDatasetEntry:
    def __init__(self) -> None:
        self.registered_blueprints: list[tuple[str, bool]] = []

    def register_blueprint(self, blueprint_uri: str, *, set_default: bool) -> None:
        self.registered_blueprints.append((blueprint_uri, set_default))


class _FakeServer:
    pass


def test_build_rrd_index_rows_from_paths(tmp_path: Path) -> None:
    first_rrd: Path = tmp_path / "assembly101" / "all" / "seq_01.rrd"
    second_rrd: Path = tmp_path / "assembly101" / "all" / "nested" / "seq_02.rrd"
    for path in (first_rrd, second_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    rows: list[RRDIndexRow] = build_rrd_index_rows_from_paths(tmp_path, dataset_name="assembly101")

    assert [row.id for row in rows] == [0, 1]
    assert [row.dataset for row in rows] == ["assembly101", "assembly101"]
    assert [row.sequence_key for row in rows] == ["all/nested/seq_02", "all/seq_01"]
    assert rows[0].recording_uri == str(second_rrd.resolve())
    assert rows[1].path == str(first_rrd.resolve())
    assert [row.size_bytes for row in rows] == [len(b"not a real rrd"), len(b"not a real rrd")]


def test_mount_catalog_python_server_preserves_recursive_file_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd_root: Path = tmp_path / "rrds"
    first_rrd: Path = rrd_root / "assembly101" / "all" / "seq_01.rrd"
    second_rrd: Path = rrd_root / "assembly101" / "all" / "nested" / "seq_02.rrd"
    for path in (first_rrd, second_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")
    captured_datasets: dict[str, list[Path]] = {}

    class _FakeClient:
        def get_dataset(self, dataset_name: str) -> _FakeBlueprintDatasetEntry:
            assert dataset_name == "assembly101"
            return _FakeBlueprintDatasetEntry()

    class _FakeRerunServer:
        def __init__(self, *, datasets: dict[str, list[Path]], port: int | None) -> None:
            assert port is None
            captured_datasets.update(datasets)

        def url(self) -> str:
            return "rerun+http://127.0.0.1:9999"

        def client(self) -> _FakeClient:
            return _FakeClient()

        def is_running(self) -> bool:
            return False

        def shutdown(self) -> None:
            pass

        def __enter__(self) -> _FakeRerunServer:
            return self

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: Any | None,
        ) -> None:
            pass

    monkeypatch.setattr(catalog_module.rr.server, "Server", _FakeRerunServer)
    monkeypatch.setattr(catalog_module, "_register_default_dataset_blueprint", lambda *_args, **_kwargs: Path("noop.rbl"))

    mount_catalog(
        rrd_root,
        datasets=("assembly101",),
        optimize_for_catalog=False,
        show_progress=False,
    )

    assert captured_datasets == {
        "assembly101": [
            second_rrd.resolve(),
            first_rrd.resolve(),
        ]
    }


def test_catalog_main_shutdowns_server_directly_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd_root: Path = tmp_path / "rrds"
    source_rrd: Path = rrd_root / "assembly101" / "all" / "seq_01.rrd"
    source_rrd.parent.mkdir(parents=True, exist_ok=True)
    source_rrd.write_bytes(b"not a real rrd")
    shutdown_calls: list[str] = []

    class _FakeTable:
        id: str = "table_id"

    class _FakeClient:
        def get_dataset(self, dataset_name: str) -> _FakeBlueprintDatasetEntry:
            assert dataset_name == "assembly101"
            return _FakeBlueprintDatasetEntry()

    class _FakeServer:
        def url(self) -> str:
            return "rerun+http://127.0.0.1:9988"

        def client(self) -> _FakeClient:
            return _FakeClient()

        def shutdown(self) -> None:
            shutdown_calls.append("shutdown")

        def __enter__(self) -> _FakeServer:
            raise AssertionError("catalog main should not rely on Rerun Server.__enter__")

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_value: BaseException | None,
            _traceback: Any | None,
        ) -> None:
            raise AssertionError("catalog main should not rely on Rerun Server.__exit__")

    fake_server = _FakeServer()

    monkeypatch.setattr(catalog_module, "mount_catalog", lambda *_args, **_kwargs: fake_server)
    monkeypatch.setattr(
        catalog_module,
        "build_rrd_index_rows_from_dataset",
        lambda *_args, **_kwargs: [
            RRDIndexRow(
                id=0,
                dataset="assembly101",
                sequence_key="all/seq_01",
                recording_uri="rerun+http://127.0.0.1:9988/dataset/assembly101?segment_id=assembly101__all__seq_01",
                path=str(source_rrd),
                size_bytes=source_rrd.stat().st_size,
            )
        ],
    )
    monkeypatch.setattr(catalog_module, "create_rrd_index_table", lambda *_args, **_kwargs: _FakeTable())
    monkeypatch.setattr(catalog_module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt))

    catalog_module.main(
        CatalogConfig(
            rrd_root=rrd_root,
            datasets=("assembly101",),
            optimize_for_catalog=False,
        )
    )

    assert shutdown_calls == ["shutdown"]


def test_register_default_dataset_blueprint_registers_full_segment_blueprint() -> None:
    dataset_entry = _FakeBlueprintDatasetEntry()
    server = _FakeServer()

    blueprint_path: Path = _register_default_dataset_blueprint(
        server,  # type: ignore[arg-type]
        dataset_entry,
        dataset_name="assembly101",
    )

    assert blueprint_path.is_file()
    assert dataset_entry.registered_blueprints == [(blueprint_path.resolve().as_uri(), True)]


def test_optimize_rrd_for_catalog_reuses_fresh_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rrd_root: Path = tmp_path / "rrds"
    source_path: Path = rrd_root / "hocap" / "subject_1" / "recording.rrd"
    cache_root: Path = tmp_path / "cache"
    optimized_path: Path = cache_root / "hocap" / "subject_1" / "recording.rrd"
    source_path.parent.mkdir(parents=True)
    optimized_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"raw")
    optimized_path.write_bytes(b"optimized")
    source_mtime_ns: int = source_path.stat().st_mtime_ns
    fresh_mtime_ns: int = source_mtime_ns + 1_000_000_000
    os.utime(optimized_path, ns=(fresh_mtime_ns, fresh_mtime_ns))

    def fail_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("fresh cache should avoid rerun rrd optimize")

    monkeypatch.setattr(subprocess, "run", fail_run)

    result: Path = _optimize_rrd_for_catalog(source_path, rrd_root=rrd_root, cache_root=cache_root)

    assert result == optimized_path.resolve()
    assert result.read_bytes() == b"optimized"


def test_optimize_rrd_for_catalog_writes_tmp_then_renames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd_root: Path = tmp_path / "rrds"
    source_path: Path = rrd_root / "hocap" / "subject_1" / "recording.rrd"
    cache_root: Path = tmp_path / "cache"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"raw")
    run_calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        run_calls.append(args)
        tmp_output_path: Path = Path(args[-1])
        tmp_output_path.write_bytes(b"optimized")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result: Path = _optimize_rrd_for_catalog(source_path, rrd_root=rrd_root, cache_root=cache_root)
    expected_path: Path = cache_root / "hocap" / "subject_1" / "recording.rrd"
    expected_tmp_path: Path = expected_path.with_suffix(".rrd.tmp")

    assert result == expected_path.resolve()
    assert result.read_bytes() == b"optimized"
    assert not expected_tmp_path.exists()
    assert run_calls == [
        ["rerun", "rrd", "optimize", str(source_path.resolve()), "-o", str(expected_tmp_path.resolve())],
    ]


def test_optimize_rrd_for_catalog_reports_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rrd_root: Path = tmp_path / "rrds"
    source_path: Path = rrd_root / "hocap" / "subject_1" / "recording.rrd"
    cache_root: Path = tmp_path / "cache"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"raw")

    def fake_run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=2, stdout="", stderr="bad optimize")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="bad optimize"):
        _optimize_rrd_for_catalog(source_path, rrd_root=rrd_root, cache_root=cache_root)


def test_optimize_rrd_for_catalog_rejects_paths_outside_root(tmp_path: Path) -> None:
    rrd_root: Path = tmp_path / "rrds"
    source_path: Path = tmp_path / "external" / "recording.rrd"
    cache_root: Path = tmp_path / "cache"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"raw")

    with pytest.raises(ValueError, match="not under RRD root"):
        _optimize_rrd_for_catalog(source_path, rrd_root=rrd_root, cache_root=cache_root)


def test_build_rrd_index_rows_from_registered_dataset_segments(tmp_path: Path) -> None:
    first_rrd: Path = tmp_path / "assembly101" / "all" / "seq_01.rrd"
    second_rrd: Path = tmp_path / "assembly101" / "all" / "nested" / "seq_02.rrd"
    for path in (first_rrd, second_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real rrd")

    segment_table: pa.Table = pa.table(
        {
            "rerun_segment_id": [
                "assembly101__all__seq_01",
                "assembly101__all__nested__seq_02",
            ],
            "property:info:sequence_key": [
                ["all/seq_01"],
                ["all/nested/seq_02"],
            ],
        }
    )
    rows: list[RRDIndexRow] = build_rrd_index_rows_from_dataset(
        _FakeDatasetEntry(segment_table),
        dataset_dir=tmp_path / "assembly101",
        dataset_name="assembly101",
    )

    assert [row.id for row in rows] == [0, 1]
    assert [row.dataset for row in rows] == ["assembly101", "assembly101"]
    assert [row.sequence_key for row in rows] == ["all/nested/seq_02", "all/seq_01"]
    assert rows[0].recording_uri.endswith("segment_id=assembly101__all__nested__seq_02")
    assert rows[1].path == str(first_rrd.resolve())
    assert [row.size_bytes for row in rows] == [len(b"not a real rrd"), len(b"not a real rrd")]


def test_registered_dataset_rows_preserve_metadata_sequence_key(tmp_path: Path) -> None:
    rrd_path: Path = tmp_path / "hocap" / "subject_8" / "20231024_180733.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    rrd_path.write_bytes(b"not a real rrd")

    segment_table: pa.Table = pa.table(
        {
            "rerun_segment_id": ["hocap__fallback__recording_id"],
            "property:info:sequence_key": [["subject_8/20231024_180733"]],
        }
    )
    rows: list[RRDIndexRow] = build_rrd_index_rows_from_dataset(
        _FakeDatasetEntry(segment_table),
        dataset_dir=tmp_path / "hocap",
        dataset_name="hocap",
    )

    assert rows[0].sequence_key == "subject_8/20231024_180733"
    assert rows[0].path == str(rrd_path.resolve())
    assert rows[0].size_bytes == len(b"not a real rrd")


def test_rrd_index_table_schema_includes_dataset_and_table_blueprint() -> None:
    schema: pa.Schema = build_rrd_index_table_schema("base64:test-blueprint")

    assert schema.names == ["id", "dataset", "sequence_key", "recording_uri", "path", "size_bytes", MARKER_FLAG_COLUMN]
    assert schema.metadata == {TABLE_BLUEPRINT_METADATA_KEY: b"base64:test-blueprint"}


def test_assembly101_table_preview_camera_preserves_reference_artifact_camera() -> None:
    preview_camera: tuple[str, str] | None = _table_preview_camera(
        "assembly101",
        {"ego": ("e1", "e2", "e3", "e4"), "exo": ("C10095",)},
    )

    assert preview_camera == ("ego", "e3")


def test_table_preview_camera_falls_back_when_override_camera_is_missing() -> None:
    preview_camera: tuple[str, str] | None = _table_preview_camera(
        "assembly101",
        {"ego": ("e1", "e2"), "exo": ("C10095",)},
    )

    assert preview_camera == ("ego", "e1")


def test_table_preview_window_constants_cover_first_ten_seconds() -> None:
    assert TABLE_CARD_PREVIEW_START_SECONDS == 0.0
    assert TABLE_CARD_PREVIEW_END_SECONDS == 10.0


def test_epfl_smart_kitchen_catalog_uses_hololens_and_all_nine_exo_cameras() -> None:
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES["epfl-smart-kitchen"]

    assert camera_names["ego"] == ("hololens",)
    assert camera_names["exo"] == (
        "output0",
        "Aoutput0",
        "Aoutput1",
        "Aoutput2",
        "Aoutput3",
        "Boutput0",
        "Boutput1",
        "Boutput2",
        "Boutput3",
    )


def test_video_exclusion_queries_remove_hocap_videos_from_3d_view() -> None:
    blueprint = build_table_card_blueprint("hocap", timeline="video_time")
    root_contents: list[Any] = list(blueprint.root_container.contents)
    scene_view = root_contents[0]
    camera_names: dict[str, tuple[str, ...]] = CATALOG_CAMERA_NAMES["hocap"]
    # exoego:v2 rig layout: exo cam i -> rig_i/cam_00, ego cam j -> rig_<num_exo>/cam_j.
    exo_names: tuple[str, ...] = camera_names["exo"]
    expected_contents: list[str] = ["+ /**"]
    for kind in ("ego", "exo"):
        for camera_name in camera_names[kind]:
            if kind == "exo":
                node: str = f"/world/{entity_id('rig', exo_names.index(camera_name))}/{entity_id('cam', 0)}"
            else:
                node = f"/world/{entity_id('rig', len(exo_names))}/{entity_id('cam', camera_names['ego'].index(camera_name))}"
            video_entity_path: str = f"{node}/pinhole/video"
            expected_contents.append(f"- {video_entity_path}")
            expected_contents.append(f"- {video_entity_path}/**")

    assert len(scene_view.contents) == 1 + 2 * (len(camera_names["ego"]) + len(camera_names["exo"]))
    assert scene_view.contents == expected_contents


def test_table_card_blueprint_builds() -> None:
    blueprint = build_table_card_blueprint("assembly101", timeline="video_time")

    assert blueprint is not None


def test_table_card_blueprints_play_uniform_selection_without_3d_range_override() -> None:
    for dataset_name in DEFAULT_CATALOG_DATASETS:
        blueprint = build_table_card_blueprint(dataset_name, timeline="video_time")
        time_panel = blueprint.time_panel
        root_contents: list[Any] = list(blueprint.root_container.contents)
        scene_view = root_contents[0]
        time_selection = time_panel.time_selection
        assert time_selection is not None

        assert time_panel.timeline == "video_time"
        assert time_panel.play_state == "playing"
        assert time_panel.loop_mode == "selection"
        assert time_selection.min.value == 0
        assert time_selection.max.value == 10_000_000_000
        assert scene_view.visualizer_overrides == {}

        encoded_blueprint: str = build_rrd_index_table_blueprint(dataset_name)
        assert encoded_blueprint.startswith("base64:")


def test_catalog_blueprints_exist_for_default_datasets() -> None:
    for dataset_name in DEFAULT_CATALOG_DATASETS:
        blueprint = build_exoego_catalog_blueprint(dataset_name)
        root_container = blueprint.root_container
        root_contents: list[Any] = list(root_container.contents)
        column_shares: list[float] | None = getattr(root_container, "column_shares", None)
        row_shares: list[float] | None = getattr(root_container, "row_shares", None)

        assert blueprint is not None
        if column_shares is not None:
            assert len(column_shares) == len(root_contents)
        if row_shares is not None:
            assert len(row_shares) == len(root_contents)
