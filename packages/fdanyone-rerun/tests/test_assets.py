"""The GVHMR run root: a copy of the installed tree beside the weights."""

from pathlib import Path

import pytest

from fdanyone_rerun.assets import GVHMR_TREE_ENTRIES, gvhmr_root


def _tree(root: Path) -> Path:
    (root / "hmr4d").mkdir(parents=True)
    (root / "hmr4d" / "__init__.py").touch()
    (root / "tools").mkdir()
    (root / "REVISION").write_text("6ec3ca39336c50492c0fae65fba2fb831fc7d866\n")
    (root / "LICENSE").touch()
    return root


def test_copies_the_tree_next_to_the_weights(tmp_path: Path) -> None:
    tree: Path = _tree(tmp_path / "prefix-a")
    root: Path = gvhmr_root(tmp_path / "models", tree)
    assert root == tmp_path / "models" / "gvhmr-root"
    assert all((root / name).exists() and not (root / name).is_symlink() for name in GVHMR_TREE_ENTRIES)
    assert (root / "hmr4d" / "__init__.py").resolve().is_relative_to(root)
    # A second call keeps the copy and upstream's checkpoint links beside it.
    (root / "inputs").mkdir()
    (root / "hmr4d" / "marker").touch()
    assert gvhmr_root(tmp_path / "models", tree) == root
    assert (root / "inputs").is_dir() and (root / "hmr4d" / "marker").is_file()


def test_refreshes_the_copy_for_another_revision(tmp_path: Path) -> None:
    root: Path = gvhmr_root(tmp_path / "models", _tree(tmp_path / "prefix-a"))
    (root / "inputs").mkdir()
    other: Path = _tree(tmp_path / "prefix-b")
    (other / "REVISION").write_text("0" * 40 + "\n")
    (other / "hmr4d" / "new.py").touch()
    gvhmr_root(tmp_path / "models", other)
    assert (root / "hmr4d" / "new.py").is_file()
    assert (root / "REVISION").read_text() == "0" * 40 + "\n"
    assert (root / "inputs").is_dir()


def test_names_a_missing_tree(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gvhmr-src"):
        gvhmr_root(tmp_path / "models", tmp_path / "missing")
