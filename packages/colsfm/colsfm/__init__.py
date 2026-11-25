"""colsfm — an open Python + pycolmap rebuild of NVIDIA cuSFM.

The package mirrors cuSFM's eight stages (keyframe selection, feature extraction and matching, loop closure, pose-graph
optimisation, triangulation and bundle adjustment, extrinsic refinement, export) and starts, as cuSFM does, from an
initial trajectory. Input comes from a Rerun catalog segment (`colsfm.catalog`) or a cuSFM `frames_meta.json` folder;
the run is written as a COLMAP model plus a Rerun layer (`colsfm.rerun_layer`). The reverse-engineered stage specs
this code follows live in the source fork, pablovela5620/pyCuSFM (branch feat/pycolmap-open-pipeline, `docs/spec/`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
"""`packages/colsfm`, so data paths resolve from any working directory."""

RESOURCES_DIR: Final[Path] = Path(__file__).resolve().parent / "resources"
"""Committed text resources: cuSFM's protobuf-text config profiles and the vendored descriptor set."""

DATA_DIR: Final[Path] = PACKAGE_DIR / "data"
"""Gitignored: downloaded datasets, run outputs, and the older home of the model graphs (`MODEL_DIR`)."""

MODEL_DIR_ENV: Final[str] = "COLSFM_MODEL_DIR"
"""Environment variable that names the model directory outright."""


def resolve_model_dir(environ: Mapping[str, str], worktree_models: Path, host_cache: Path) -> Path:
    """Pick the one directory the RaCo / FlexGlue ONNX graphs and their TensorRT engines live in.

    The graphs are exported or downloaded, not committed, and an engine is specific to one TensorRT version and one
    GPU, so the default is a host-local cache that every worktree on the machine shares (never the NAS). A worktree that
    already holds graphs keeps them, so an existing checkout runs as before.

    Args:
        environ: The process environment; a non-empty `COLSFM_MODEL_DIR` wins.
        worktree_models: This checkout's gitignored `data/models`.
        host_cache: The host-local cache to use otherwise.

    Returns:
        `$COLSFM_MODEL_DIR` when set, else `worktree_models` when it holds an ONNX graph, else `host_cache`.
    """
    if chosen := environ.get(MODEL_DIR_ENV):
        return Path(chosen).expanduser()
    if worktree_models.is_dir() and next(worktree_models.rglob("*.onnx"), None) is not None:
        return worktree_models
    return host_cache


MODEL_DIR: Final[Path] = resolve_model_dir(os.environ, DATA_DIR / "models", Path.home() / ".cache" / "colsfm" / "models")
"""Where every model graph and engine is read and written: `model_assets`, `matching` and the export / download /
FP8 tasks all derive their paths from it."""
