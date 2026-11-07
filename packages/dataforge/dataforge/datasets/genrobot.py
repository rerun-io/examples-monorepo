"""Gen-HumanWholeBody (``genrobot``): the DAS-Ego release with the SMPL-X body22 track; the converter is ``dataforge.das_ego``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from dataforge import paths
from dataforge.das_ego.dataset import DasEgoConfig, DasEgoDataset
from dataforge.das_ego.pose import POSE_LAYER


@dataclass
class GenrobotConfig(DasEgoConfig):
    """Gen-HumanWholeBody episodes, fetched from the Hub one at a time."""

    command: ClassVar[str] = "genrobot"
    """Registry key and catalog dataset."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "genrobot")
    """Local scratch: the manifest and the MCAPs being converted. Never the NAS."""
    _target: type = field(default_factory=lambda: GenrobotDataset)
    """Dataset instantiated by setup."""


class GenrobotDataset(DasEgoDataset[GenrobotConfig]):
    """Gen-HumanWholeBody: body22 skeleton, hands, MANO, actions."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, POSE_LAYER, paths.ACTIONS_LAYER, paths.PROJECTIONS_LAYER, paths.HAND_MESH_LAYER)
    """Publication and loading order."""
