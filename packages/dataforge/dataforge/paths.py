"""On-disk layout: layer-major rrd tree plus raw-download tree. Stdlib-only.

Layout (relative to the overridable roots; defaults are package-local because
pixi tasks run with ``cwd = packages/dataforge``):

    data/raw/<dataset>/...            upstream layout, untouched
    data/dataforge/rrd/<layer>/<recording_id>.rrd   layer in {base, gt}
    data/dataforge/rrd/blueprints/<name>[-table].rbl
"""

from __future__ import annotations

import os
from pathlib import Path

from dataforge.identity import SequenceIdentity

BASE_LAYER: str = "base"
"""Sensor layer: the recording every dataset emits; the first path component under ``output_root()``."""

GT_LAYER: str = "gt"
"""Ground-truth trajectory layer; sibling of base (same recording ids, own directory)."""


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
