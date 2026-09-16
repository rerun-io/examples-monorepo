from pathlib import Path

from dataforge.identity import SequenceIdentity
from dataforge.paths import BASE_LAYER, GT_LAYER, SIDECAR_DIR, output_root, raw_root, rrd_path, sidecar_path


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


def test_rrd_paths_are_layer_major() -> None:
    identity: SequenceIdentity = SequenceIdentity(dataset="msd", parts=("MI_valid_01",))
    root: Path = Path("/out")
    assert rrd_path(root, layer=BASE_LAYER, identity=identity) == root / "base" / f"{identity.recording_id}.rrd"
    assert rrd_path(root, layer=GT_LAYER, identity=identity) == root / "gt" / f"{identity.recording_id}.rrd"


def test_sidecars_are_grouped_per_recording_and_are_not_a_layer() -> None:
    """A sidecar is an *input* a derived layer rebuilds from, so it sits outside the layers.

    Keyed by recording id rather than by layer, because one sequence's sidecars
    serve every derived layer of it; ``register`` walks ``LAYERS`` and so never
    sees this directory.
    """
    identity: SequenceIdentity = SequenceIdentity(dataset="msd-index", parts=("MIO_others", "MIO09"))
    root: Path = Path("/out")

    assert sidecar_path(root, identity, "gt.csv") == root / "sidecars" / identity.recording_id / "gt.csv"
    assert SIDECAR_DIR not in (BASE_LAYER, GT_LAYER)
