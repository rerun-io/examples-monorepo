"""On-disk layout: layer-major rrd tree plus raw-download tree. Layout only, stdlib-only.

Layout (relative to the overridable roots; defaults are package-local because
pixi tasks run with ``cwd = packages/dataforge``):

    data/raw/<dataset>/...            upstream layout, untouched
    data/dataforge/rrd/<layer>/<recording_id>.rrd   layer declared by the dataset
    data/dataforge/rrd/blueprints/<name>[-table]-<stamp>.rbl   rewritten by every register run
    data/dataforge/rrd/sidecars/<recording_id>/<name>   small inputs a derived layer rebuilds from
"""

from __future__ import annotations

import os
from pathlib import Path

from dataforge.identity import SequenceIdentity

BASE_LAYER: str = "base"
"""Sensor layer: the recording every dataset emits; the first path component under ``output_root()``."""

GT_LAYER: str = "gt"
"""Ground-truth trajectory layer; sibling of base (same recording ids, own directory)."""

SENSOR_METADATA_LAYER: str = "sensor_metadata"
"""Static calibration backfill; sibling of base, separate from original measurements."""

HAND_POSE_LAYER: str = "hand_pose"
"""Measured hand annotations."""
CAPTIONS_LAYER: str = "captions"
"""Scene instructions and captions."""

OBJECT_POSE_LAYER: str = "object_pose"
"""Tracked rigid object poses and geometric census."""
OBJECT_MESH_LAYER: str = "object_mesh"
"""Static object-frame HOT3D geometry."""
HAND_MESH_LAYER: str = "hand_mesh"
"""Derived skinned world-frame hand geometry."""

PROJECTIONS_LAYER: str = "projections"
"""Derived 2D from 3D keypoints through each camera's lens model."""

LAYERS: tuple[str, ...] = (BASE_LAYER, GT_LAYER, SENSOR_METADATA_LAYER)
"""Common layers; register and view use each dataset's own layer declaration."""


def output_root() -> Path:
    """Root of the converted rrd tree; override with ``DATAFORGE_OUTPUT_ROOT``."""
    return Path(os.environ.get("DATAFORGE_OUTPUT_ROOT", "data/dataforge/rrd"))


def raw_root() -> Path:
    """Root of raw dataset downloads; override with ``DATAFORGE_RAW_ROOT``."""
    return Path(os.environ.get("DATAFORGE_RAW_ROOT", "data/raw"))


def rrd_path(root: Path, *, layer: str, identity: SequenceIdentity) -> Path:
    """Layer-major rrd location: ``<root>/<layer>/<recording_id>.rrd``."""
    return root / layer / f"{identity.recording_id}.rrd"


def blueprint_path(root: Path, name: str, *, stamp: str, segment_table: bool = False) -> Path:
    """Blueprint location: ``<root>/blueprints/<name>[-table]-<stamp>.rbl``; every register run stamps new files."""
    return root / "blueprints" / f"{name}{'-table' if segment_table else ''}-{stamp}.rbl"


SIDECAR_DIR: str = "sidecars"
"""Directory holding the small per-sequence inputs a derived layer rebuilds from.

Not a layer: nothing here is registered, and a viewer never opens it. It is what
lets the layer rule hold — the bulk source is deleted once base is published, so
whatever a derived layer still needs (a ground-truth csv, a calibration) has to
be small enough to keep and has to survive beside the rrds rather than in a
scratch directory the next run may wipe.
"""


def sidecar_path(root: Path, identity: SequenceIdentity, name: str) -> Path:
    """Sidecar location: ``<root>/sidecars/<recording_id>/<name>``.

    Keyed by recording id, not layer, because one sequence's sidecars serve every
    derived layer of it; a per-sequence directory then keeps a corpus's worth of
    them out of one flat listing.
    """
    return root / SIDECAR_DIR / identity.recording_id / name
