"""colsfm — an open Python + pycolmap rebuild of NVIDIA cuSFM.

The package mirrors cuSFM's eight stages (keyframe selection, feature extraction and matching, loop closure, pose-graph
optimisation, triangulation and bundle adjustment, extrinsic refinement, export) and starts, as cuSFM does, from an
initial trajectory. Input comes from a Rerun catalog segment (`colsfm.catalog`) or a cuSFM `frames_meta.json` folder;
the run is written as a COLMAP model plus a Rerun layer (`colsfm.rerun_layer`). The reverse-engineered stage specs
this code follows live in the source fork, pablovela5620/pyCuSFM (branch feat/pycolmap-open-pipeline, `docs/spec/`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
"""`packages/colsfm`, so data paths resolve from any working directory."""

RESOURCES_DIR: Final[Path] = Path(__file__).resolve().parent / "resources"
"""Committed text resources: cuSFM's protobuf-text config profiles and the vendored descriptor set."""

DATA_DIR: Final[Path] = PACKAGE_DIR / "data"
"""Gitignored: exported ONNX graphs, downloaded datasets, run outputs."""

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
