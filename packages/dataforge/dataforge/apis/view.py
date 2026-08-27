"""``dataforge-view``: open one converted recording (and its embedded blueprint)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from dataforge import paths
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig, dataset_key
from dataforge.datasets.base import DataforgeDatasetConfig

LAYER: str = "base"
"""Layer whose recordings the viewer loads."""


@dataclass
class Config:
    """View one converted base-layer recording."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/save/connect behaviour; pass ``--rr-config.headless`` without a display."""
    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset to view; its registry key prefixes the recording filenames."""
    sequence: str | None = None
    """Recording id or sequence key to open; defaults to the first one found."""


def main(config: Config) -> None:
    """Load the selected rrd into the recording stream configured by ``rr_config``."""
    dataset_config: DataforgeDatasetConfig = config.dataset
    name: str = dataset_key(dataset_config)
    candidates: list[Path] = sorted((paths.output_root() / LAYER).glob(f"{name}__*.rrd"))
    if config.sequence is not None:
        wanted: str = config.sequence.replace("/", "__")
        candidates = [path for path in candidates if wanted in path.stem]
    if not candidates:
        raise FileNotFoundError(f"no {LAYER}-layer rrd for {name} (sequence={config.sequence}) under {paths.output_root() / LAYER}")
    rrd_path: Path = candidates[0]
    print(f"viewing {rrd_path}")
    rr.log_file_from_path(rrd_path)
