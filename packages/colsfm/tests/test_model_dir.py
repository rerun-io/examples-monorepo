"""Where the uncommitted model graphs and engines live: one directory, resolved in one place."""

from __future__ import annotations

from pathlib import Path

from colsfm import MODEL_DIR, MODEL_DIR_ENV, resolve_model_dir
from colsfm.matching import FLEXGLUE_MODEL_DIR
from colsfm.model_assets import RACO_DYNAMIC_ONNX_PATH, RACO_LIGHTGLUE_ONNX_PATH, RACO_ONNX_PATH, missing_raco_graphs_message


def test_the_environment_variable_wins_over_a_worktree_that_holds_graphs(tmp_path: Path) -> None:
    """`$COLSFM_MODEL_DIR` is taken as given, even when the worktree has its own graphs."""
    worktree: Path = tmp_path / "data" / "models"
    worktree.mkdir(parents=True)
    (worktree / "raco-aliked-b1-16.onnx").touch()
    chosen: Path = tmp_path / "shared"
    assert resolve_model_dir({MODEL_DIR_ENV: str(chosen)}, worktree, tmp_path / "cache") == chosen


def test_a_worktree_that_holds_graphs_keeps_them(tmp_path: Path) -> None:
    """A checkout that already exported its graphs keeps using them, nested ones included (today's layout)."""
    worktree: Path = tmp_path / "data" / "models"
    (worktree / "flexglue").mkdir(parents=True)
    (worktree / "flexglue" / "flexglue_descriptor.onnx").touch()
    assert resolve_model_dir({}, worktree, tmp_path / "cache") == worktree


def test_a_fresh_worktree_falls_back_to_the_host_cache(tmp_path: Path) -> None:
    """No graphs in the worktree (missing or empty directory): the host-local cache, shared by every worktree."""
    cache: Path = tmp_path / "cache"
    assert resolve_model_dir({}, tmp_path / "missing", cache) == cache
    empty: Path = tmp_path / "empty"
    empty.mkdir()
    assert resolve_model_dir({}, empty, cache) == cache
    assert resolve_model_dir({MODEL_DIR_ENV: ""}, empty, cache) == cache


def test_every_graph_path_derives_from_the_one_model_dir() -> None:
    """RaCo, LightGlue+ and FlexGlue all live under `MODEL_DIR`, so moving it moves all of them."""
    for path in (RACO_ONNX_PATH, RACO_DYNAMIC_ONNX_PATH, RACO_LIGHTGLUE_ONNX_PATH, FLEXGLUE_MODEL_DIR):
        assert path.is_relative_to(MODEL_DIR)


def test_a_missing_graph_names_the_model_dir_and_the_tasks_that_create_it() -> None:
    """The message says where the graphs were looked for, how to point elsewhere, and which pixi tasks make them."""
    message: str = missing_raco_graphs_message((RACO_ONNX_PATH,))
    assert str(MODEL_DIR) in message
    assert MODEL_DIR_ENV in message
    for task in ("colsfm-raco-export", "colsfm-download-flexglue", "colsfm-flexglue-fp8"):
        assert f"pixi run -e colsfm {task}" in message
