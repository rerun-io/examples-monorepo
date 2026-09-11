"""``dataforge-view``: open one converted recording — every layer of it — from disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from dataforge import paths
from dataforge.datasets import AnnotatedDatasetUnion, RobocapConfig
from dataforge.datasets.base import DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity


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
    name: str = dataset_config.name
    output_root: Path = paths.output_root()
    base_root: Path = output_root / paths.BASE_LAYER
    candidates: list[Path] = sorted(base_root.glob(f"{name}__*.rrd"))
    if config.sequence is not None:
        wanted: str = config.sequence.replace("/", "__")
        candidates = [path for path in candidates if wanted in path.stem]
    if not candidates:
        raise FileNotFoundError(f"no {paths.BASE_LAYER}-layer rrd for {name} (sequence={config.sequence}) under {base_root}")

    # The stem is the recording id, and the recording id is the identity: the
    # sibling layers are the same id under another layer directory.
    identity: SequenceIdentity = SequenceIdentity(dataset=name, parts=tuple(candidates[0].stem.split("__")[1:]))
    found: list[Path] = []
    for layer in paths.LAYERS:
        layer_path: Path = paths.rrd_path(output_root, layer=layer, identity=identity)
        if layer_path.is_file():
            found.append(layer_path)
    print(f"viewing {', '.join(str(path) for path in found)}")
    for layer_path in found:
        rr.log_file_from_path(layer_path)
