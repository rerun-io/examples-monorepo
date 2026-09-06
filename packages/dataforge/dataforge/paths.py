"""On-disk layout: layer-major rrd tree plus raw-download tree. Stdlib-only.

Layout (relative to the overridable roots; defaults are package-local because
pixi tasks run with ``cwd = packages/dataforge``):

    data/raw/<dataset>/...            upstream layout, untouched
    data/dataforge/rrd/<layer>/<recording_id>.rrd   layer in {base, gt}
    data/dataforge/rrd/blueprints/<name>[-table].rbl
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dataforge.identity import SequenceIdentity

BASE_LAYER: str = "base"
"""Sensor layer: the recording every dataset emits; the first path component under ``output_root()``."""

GT_LAYER: str = "gt"
"""Ground-truth trajectory layer; sibling of base (same recording ids, own directory)."""

LAYERS: tuple[str, ...] = (BASE_LAYER, GT_LAYER)
"""Every layer dataforge writes, base first.

The one list ``register`` and ``view`` walk, so a new derived layer is registered
and opened everywhere by adding it here. Layers share a recording id, so a viewer
handed several files of one sequence merges them onto the same entities.
"""


def output_root() -> Path:
    """Root of the converted rrd tree; override with ``DATAFORGE_OUTPUT_ROOT``."""
    return Path(os.environ.get("DATAFORGE_OUTPUT_ROOT", "data/dataforge/rrd"))


def raw_root() -> Path:
    """Root of raw dataset downloads; override with ``DATAFORGE_RAW_ROOT``."""
    return Path(os.environ.get("DATAFORGE_RAW_ROOT", "data/raw"))


def rrd_path(root: Path, *, layer: str, identity: SequenceIdentity) -> Path:
    """Layer-major rrd location: ``<root>/<layer>/<recording_id>.rrd``."""
    return root / layer / f"{identity.recording_id}.rrd"


def blueprint_path(root: Path, name: str, *, segment_table: bool = False) -> Path:
    """Blueprint location: ``<root>/blueprints/<name>[-table].rbl``."""
    return root / "blueprints" / f"{name}{'-table' if segment_table else ''}.rbl"


def remove_tree(root: Path) -> None:
    """Delete a scratch directory tree, reporting what it could not delete.

    ``shutil.rmtree(..., ignore_errors=True)`` is the wrong default for a raw
    tree under a size budget: a directory that quietly fails to go away is
    exactly how the next sequence's fetch finds the disk full and blames its own
    leftovers. A tree that was never there is not a failure, so only real errors
    are printed — and they are printed rather than raised, because every caller
    is already cleaning up after something else.

    Args:
        root: Directory to remove, along with everything under it.
    """
    if not root.exists():
        return
    shutil.rmtree(root, onexc=lambda _function, failed, error: print(f"  warning: could not remove {failed}: {type(error).__name__}: {error}"))
