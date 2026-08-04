"""Canonical filesystem roots for datasets used by SimpleCV tools."""

import os
from pathlib import Path

DATASET_ROOT: Path = Path(os.environ.get("SIMPLECV_DATA_ROOT", "/mnt/nas/datasets")).expanduser()
"""Shared dataset mount; override with ``SIMPLECV_DATA_ROOT`` before importing SimpleCV."""

ARIA_GEN2_PILOT_ROOT: Path = DATASET_ROOT / "aria-gen2-pilot"
EGO_DEX_ROOT: Path = DATASET_ROOT / "ego-dex"
HOT3D_ROOT: Path = DATASET_ROOT / "hot3d"
MAMMA_SOURCE_ROOT: Path = DATASET_ROOT / "mamma"
MAMMA_AV1_1080_ROOT: Path = DATASET_ROOT / "mamma-av1-1080"
ROBOCAP_ROOT: Path = DATASET_ROOT / "robocap"
UMETRACK_SPLIT_ROOT: Path = DATASET_ROOT / "umetrack-split"
