"""Where things live: run directories (contract run-layout.md), datasets and published checkpoints."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from hexels_bench.hexel_ply import HexelVariant

Dataset: TypeAlias = Literal["mipnerf360", "dtu", "tnt"]

UPSTREAM_DATASET: dict[Dataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tanks_and_temples"}
"""Run-layout dataset name -> upstream (HF and configs) dataset name."""
DATA_SUBDIR: dict[Dataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tnt_nvs"}
"""Run-layout dataset name -> scene root under the data directory (upstream data/ layout)."""
PUBLISHED_APPEARANCE: dict[HexelVariant, str] = {"v1": "v1-spatial-rgb", "v2": "v2-linear-view", "v3": "v3-sh3"}


@dataclass(frozen=True, slots=True)
class Roots:
    """The three roots the tools read and write."""

    data: Path = Path.home() / "gsplat-modern-data"
    """Datasets (upstream layout) and `hexels-pretrained/` (HF jhu-ccvl/Hexels)."""
    runs: Path = Path.home() / "hexels-work" / "runs"
    """Run outputs, one directory per dataset/scene/method."""

    def scene_dir(self, dataset: Dataset, scene: str) -> Path:
        return self.data / DATA_SUBDIR[dataset] / scene

    def contract(self, dataset: Dataset, scene: str) -> Path:
        return self.data / "hexels-pretrained" / "cameras" / "nvs" / UPSTREAM_DATASET[dataset] / scene / "contract.json"

    def published_model(self, dataset: Dataset, scene: str, variant: HexelVariant) -> Path:
        name: str = f"hexels_{PUBLISHED_APPEARANCE[variant]}_fixed-sfm_seed0_step030000"
        return self.data / "hexels-pretrained" / "nvs" / UPSTREAM_DATASET[dataset] / scene / name

    def run_dir(self, dataset: Dataset, scene: str, method: str) -> Path:
        return self.runs / dataset / scene / method

    def gt_dir(self, dataset: Dataset, scene: str) -> Path:
        """Shared ground truth of a scene's held-out views: `gt/test/<name>.png`, 16-bit."""
        return self.runs / dataset / scene / "gt" / "test"

    def published_surface_model(self, dataset: Dataset, scene: str) -> Path:
        """The paper's surface checkpoint of a DTU or T&T scene (V3, fixed MVS population, normal + distortion losses, all views)."""
        name: str = "hexels_v3-sh3_fixed-mvs_normal-and-distortion_seed0_step030000"
        return self.data / "hexels-pretrained" / "surface" / UPSTREAM_DATASET[dataset] / scene / name

    def surface_dir(self, dataset: Dataset, scene: str, method: str) -> Path:
        """Surface outputs of one method: cameras.json, surface.rrd, fuse.ply, fuse_post.ply, fusion.json, surface_metrics.json."""
        return self.run_dir(dataset, scene, method) / "surface"
