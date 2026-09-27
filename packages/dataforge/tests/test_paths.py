import re
from pathlib import Path

import pytest

from dataforge import paths
from dataforge.identity import SequenceIdentity
from dataforge.paths import BASE_LAYER, GT_LAYER, SIDECAR_DIR, layer_targets, output_root, raw_root, require_outside, rrd_path, sidecar_path


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


def test_layer_targets_keep_previews_in_their_own_tree(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    identity: SequenceIdentity = SequenceIdentity(dataset="hocap", parts=("subject_5", "test"))
    full: dict[str, Path] = layer_targets(identity, (BASE_LAYER, GT_LAYER), frame_limit=None)
    limited: dict[str, Path] = layer_targets(identity, (BASE_LAYER, GT_LAYER), frame_limit=2)
    assert list(full) == list(limited) == [BASE_LAYER, GT_LAYER]
    assert full[BASE_LAYER] == rrd_path(tmp_path, layer=BASE_LAYER, identity=identity)
    assert limited[BASE_LAYER] == rrd_path(tmp_path / "preview-first2", layer=BASE_LAYER, identity=identity)
    with pytest.raises(ValueError, match="frame_limit must be positive, got 0"):
        layer_targets(identity, (BASE_LAYER,), frame_limit=0)


def test_require_outside_follows_symlinks_into_protected_roots(tmp_path: Path) -> None:
    raw: Path = tmp_path / "raw"
    raw.mkdir()
    (tmp_path / "out").symlink_to(raw, target_is_directory=True)
    require_outside([tmp_path / "elsewhere" / "base.rrd"], roots=[raw])
    with pytest.raises(ValueError, match="refusing to write beneath protected input"):
        require_outside([tmp_path / "elsewhere" / "base.rrd", tmp_path / "out" / "base.rrd"], roots=[raw])


def test_every_layer_name_is_distinct_and_the_nas_is_a_protected_root() -> None:
    layers = [value for name, value in vars(paths).items() if name.endswith("_LAYER")]
    assert {paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER, paths.ACTIONS_LAYER} <= set(layers)
    assert len(set(layers)) == len(layers)
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", layer) for layer in layers)
    with pytest.raises(ValueError, match="refusing to write beneath protected input"):
        require_outside([paths.NAS_ROOT / "dataforge" / "rrd"], roots=[paths.NAS_ROOT])
