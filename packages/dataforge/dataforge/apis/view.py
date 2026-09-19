"""``dataforge-view``: open one converted recording — every layer of it — from disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from dataforge import paths
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig


@dataclass
class Config:
    """View one converted recording: its base layer and every derived layer on disk."""

    rr_config: RerunTyroConfig = field(default_factory=RerunTyroConfig)
    """Viewer/save/connect behaviour; pass ``--rr-config.headless`` without a display."""
    dataset: AnnotatedDatasetUnion = field(default_factory=RobocapConfig)
    """Dataset to view; its registry key prefixes the recording filenames."""
    sequence: str | None = None
    """Recording id or sequence key to open; defaults to the first one found."""


def main(config: Config) -> None:
    """Load the selected recording's layers into the stream configured by ``rr_config``.

    The base layer is what a sequence is selected by, but every layer shares its
    recording id, so handing the viewer the siblings too merges them onto the
    same entities — which is the only way an msd recording shows its ground truth.
    """
    dataset_config: DataforgeDatasetConfig = config.dataset
    dataset: DataforgeDataset = dataset_config.setup()
    name: str = dataset_config.name
    output_root: Path = paths.output_root()
    base_root: Path = output_root / paths.BASE_LAYER
    candidates: list[Path] = sorted(base_root.glob(f"{name}__*.rrd"))
    if config.sequence is not None:
        wanted: str = config.sequence.replace("/", "__")
        candidates = [path for path in candidates if wanted in path.stem]
    if not candidates:
        raise FileNotFoundError(f"no {paths.BASE_LAYER}-layer rrd for {name} (sequence={config.sequence}) under {base_root}")

    # The file name is the recording id, and the sibling layers are that same
    # name under another layer directory.
    selected: Path = candidates[0]
    found: list[Path] = [path for path in (output_root / layer / selected.name for layer in dataset.layers) if path.is_file()]
    print(f"viewing {', '.join(str(path) for path in found)}")
    for layer_path in found:
        rr.log_file_from_path(layer_path)
