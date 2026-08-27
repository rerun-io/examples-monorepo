from pathlib import Path

from dataforge.paths import output_root, raw_root


def test_output_root_defaults_to_package_local_data(monkeypatch) -> None:
    monkeypatch.delenv("DATAFORGE_OUTPUT_ROOT", raising=False)
    root: Path = output_root()
    assert root == Path("data/dataforge/rrd")


def test_output_root_env_override(monkeypatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", "/mnt/nas/datasets/dataforge/rrd")
    assert output_root() == Path("/mnt/nas/datasets/dataforge/rrd")


def test_raw_root_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("DATAFORGE_RAW_ROOT", raising=False)
    assert raw_root() == Path("data/raw")
    monkeypatch.setenv("DATAFORGE_RAW_ROOT", "/mnt/nas/datasets")
    assert raw_root() == Path("/mnt/nas/datasets")
