import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import tyro

SCRIPT_PATH: Path = Path(__file__).parents[1] / "tools" / "apps" / "register_catalog_with_predictions.py"
PIXI_TOML_PATH: Path = Path(__file__).parents[3] / "pixi.toml"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("register_catalog_with_predictions", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registration_defaults_to_full_catalog_and_discovers_prediction_rrds(tmp_path: Path) -> None:
    module: Any = _load_script_module()
    prediction_root: Path = tmp_path / "predictions"
    first_rrd: Path = prediction_root / "assembly101" / "all" / "seq_a" / "mvapi_coco133_upper_body_v1_full_red.rrd"
    second_rrd: Path = prediction_root / "hocap" / "subject_1" / "mvapi_other_layer.rrd"
    for path in (second_rrd, first_rrd):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rrd")

    config: Any = module.CatalogRegistrationConfig(prediction_root=prediction_root)

    discovered_paths: list[Path] = module.discover_prediction_rrd_paths(config)

    assert config.datasets == ()
    assert config.optimize_for_catalog is False
    assert discovered_paths == [first_rrd.resolve(), second_rrd.resolve()]


def test_cli_parser_uses_tyro_instead_of_argparse() -> None:
    script_text: str = SCRIPT_PATH.read_text()

    assert "import argparse" not in script_text
    assert "def parse_args" not in script_text
    assert "tyro.cli" in script_text
    assert "def main(config: CatalogRegistrationConfig)" in script_text


def test_tyro_config_keeps_expected_minimal_cli_flags(tmp_path: Path) -> None:
    module: Any = _load_script_module()
    rrd_root: Path = tmp_path / "catalog"
    prediction_root: Path = tmp_path / "predictions"
    prediction_rrd: Path = prediction_root / "assembly101" / "seq.rrd"

    config: Any = tyro.cli(
        module.CatalogRegistrationConfig,
        args=[
            "--rrd-root",
            str(rrd_root),
            "--prediction-root",
            str(prediction_root),
            "--prediction-rrd",
            str(prediction_rrd),
            "--dataset",
            "assembly101",
            "hocap",
            "--prediction-dataset",
            "assembly101",
            "--optimize-dataset",
            "assembly101",
            "hocap",
            "--no-progress",
            "--exit-after-register",
        ],
    )

    assert config.rrd_root == rrd_root
    assert config.prediction_root == prediction_root
    assert config.prediction_rrds == (prediction_rrd,)
    assert config.datasets == ("assembly101", "hocap")
    assert config.prediction_dataset_name == "assembly101"
    assert config.optimize_datasets == ("assembly101", "hocap")
    assert config.show_progress is False
    assert config.exit_after_register is True


def test_pixi_mac_registration_env_is_mac_only_and_lean() -> None:
    pixi_data: dict[str, Any] = tomllib.loads(PIXI_TOML_PATH.read_text())
    feature: dict[str, Any] = pixi_data["feature"]["mv-api-catalog-register-mac"]
    environment: dict[str, Any] = pixi_data["environments"]["mv-api-catalog-register-mac"]
    task: dict[str, Any] = feature["tasks"]["mv-api-catalog-register"]

    assert feature["platforms"] == ["osx-arm64"]
    assert feature["dependencies"] == {"python": "3.12.*"}
    assert set(feature["pypi-dependencies"]) == {"rerun-sdk", "simplecv", "tyro"}
    assert feature["pypi-dependencies"]["simplecv"]["rev"] == "178479b53d14b9dd4a79b212dd2c5a3be52b4de8"
    assert feature["pypi-dependencies"]["rerun-sdk"]["extras"] == ["datafusion"]
    assert task["cmd"] == "python tools/apps/register_catalog_with_predictions.py"
    assert task["cwd"] == "packages/mv-api"
    assert environment["features"] == ["mv-api-catalog-register-mac"]
    assert environment["no-default-feature"] is True
    assert "cuda" not in environment["features"]
    assert "mv-api" not in environment["features"]


def test_mount_catalog_with_predictions_uses_simplecv_blueprints_and_registers_prediction_rrds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module: Any = _load_script_module()
    rrd_root: Path = tmp_path / "catalog"
    prediction_root: Path = tmp_path / "predictions"
    prediction_rrd: Path = prediction_root / "assembly101" / "all" / "seq_a" / "mvapi_coco133_upper_body_v1_full_red.rrd"
    prediction_rrd.parent.mkdir(parents=True, exist_ok=True)
    prediction_rrd.write_bytes(b"rrd")
    captured_mount_kwargs: dict[str, Any] = {}

    class _FakeRegistrationHandle:
        def __init__(self) -> None:
            self.wait_called: bool = False

        def wait(self) -> None:
            self.wait_called = True

    class _FakeDatasetEntry:
        def __init__(self) -> None:
            self.registration_handle: _FakeRegistrationHandle = _FakeRegistrationHandle()
            self.register_calls: list[tuple[list[str], dict[str, Any]]] = []

        def register(self, recording_uri: list[str], **kwargs: Any) -> _FakeRegistrationHandle:
            self.register_calls.append((recording_uri, kwargs))
            return self.registration_handle

    class _FakeClient:
        def __init__(self) -> None:
            self.datasets: dict[str, _FakeDatasetEntry] = {"assembly101": _FakeDatasetEntry()}

        def get_dataset(self, dataset_name: str) -> _FakeDatasetEntry:
            return self.datasets[dataset_name]

    class _FakeServer:
        def __init__(self) -> None:
            self.client_value: _FakeClient = _FakeClient()

        def client(self) -> _FakeClient:
            return self.client_value

        def url(self) -> str:
            return "rerun+http://127.0.0.1:9991"

    def _fake_mount_catalog(*args: Any, **kwargs: Any) -> _FakeServer:
        captured_mount_kwargs["args"] = args
        captured_mount_kwargs["kwargs"] = kwargs
        return _FakeServer()

    simplecv_module: ModuleType = ModuleType("simplecv")
    simplecv_apis_module: ModuleType = ModuleType("simplecv.apis")
    catalog_module: ModuleType = ModuleType("simplecv.apis.exoego_forge_catalog")
    cast(Any, catalog_module).mount_catalog = _fake_mount_catalog
    monkeypatch.setitem(sys.modules, "simplecv", simplecv_module)
    monkeypatch.setitem(sys.modules, "simplecv.apis", simplecv_apis_module)
    monkeypatch.setitem(sys.modules, "simplecv.apis.exoego_forge_catalog", catalog_module)
    config: Any = module.CatalogRegistrationConfig(
        rrd_root=rrd_root,
        prediction_root=prediction_root,
        prediction_rrds=(prediction_rrd,),
        port=9991,
        optimize_datasets=("assembly101",),
    )

    result: Any = module.mount_catalog_with_predictions(config)

    dataset_entry: _FakeDatasetEntry = result.server.client().datasets["assembly101"]
    assert captured_mount_kwargs["args"] == (rrd_root.resolve(),)
    assert captured_mount_kwargs["kwargs"]["datasets"] == ()
    assert captured_mount_kwargs["kwargs"]["port"] == 9991
    assert captured_mount_kwargs["kwargs"]["optimize_for_catalog"] is False
    assert captured_mount_kwargs["kwargs"]["optimize_datasets"] == ("assembly101",)
    assert dataset_entry.register_calls == [
        ([prediction_rrd.resolve().as_uri()], {"layer_name": "mvapi_coco133_upper_body_v1_full_red"})
    ]
    assert dataset_entry.registration_handle.wait_called is True
    assert result.catalog_url == "rerun+http://127.0.0.1:9991"
    assert result.registered_predictions[0].dataset_name == "assembly101"
