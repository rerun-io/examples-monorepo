"""Where things live: run directories (contract run-layout.md), datasets, published checkpoints and the
protocol scenes; and the one loader for our JSON records."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from serde import SerdeError
from serde.json import from_json

from hexels_bench.hexel_ply import HexelVariant

Dataset: TypeAlias = Literal["mipnerf360", "dtu", "tnt"]

UPSTREAM_DATASET: dict[Dataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tanks_and_temples"}
"""Run-layout dataset name -> upstream (HF and configs) dataset name."""
DATA_SUBDIR: dict[Dataset, str] = {"mipnerf360": "mipnerf360", "dtu": "dtu", "tnt": "tnt_nvs"}
"""Run-layout dataset name -> scene root under the data directory (upstream data/ layout)."""
PUBLISHED_APPEARANCE: dict[HexelVariant, str] = {"v1": "v1-spatial-rgb", "v2": "v2-linear-view", "v3": "v3-sh3"}
PROTOCOL_SCENES: dict[Dataset, tuple[str, ...]] = {
    "mipnerf360": ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"),
    "dtu": ("scan24", "scan37", "scan40", "scan55", "scan63", "scan65", "scan69", "scan83", "scan97", "scan105", "scan106", "scan110", "scan114", "scan118", "scan122"),
    "tnt": ("Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck", "Train", "Francis", "Horse", "Lighthouse"),
}
"""The 34 NVS scenes of upstream configs/nvs.json, in upstream order."""
REPRESENTATIVE_SCENES: dict[Dataset, tuple[str, ...]] = {"dtu": ("scan24", "scan114"), "mipnerf360": ("garden", "counter"), "tnt": ("Barn", "Truck")}
"""Scope cut (contracts/scope.md): new compute runs on these scenes only, upstream order."""
WORKSPACE: Path = Path(__file__).resolve().parents[3]
"""Pixi workspace root (holds pixi.toml and packages/)."""
GSPLAT_RELEASE: Path = WORKSPACE / "packages" / "gsplat-rust-renderer" / "target" / "release"
"""Release `gsplat` and `gsplat-train` binaries (pixi run -e gsplat-rust-renderer gsplat-build)."""


def read_json[Record](cls: type[Record], path: Path) -> Record:
    """Decode a JSON record (camera contract, run record, report); malformed input names the file."""
    try:
        return from_json(cls, path.read_text())
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: {error}") from error


@dataclass(frozen=True, slots=True)
class Roots:
    """The three roots the tools read and write."""

    data: Path = Path.home() / "gsplat-modern-data"
    """Datasets (upstream layout) and `hexels-pretrained/` (HF jhu-ccvl/Hexels)."""
    runs: Path = Path.home() / "hexels-work" / "runs"
    """Run outputs, one directory per dataset/scene/method."""
    brush_prepared: Path = Path.home() / "hexels-work" / "brush-prepared"
    """Brush-ready protocol scenes (`<dataset>/<scene>/`), prepared once and reused."""

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

    def published_image_fit(self, stem: str, appearance: str) -> Path:
        """The paper's published 2D fit of one Image-GS image (20K -> 10K pruned, seed 123, 5,100 steps)."""
        return self.data / "hexels-pretrained" / "images_2d" / stem / f"hexels_{appearance}_pruned-20000-to-10000_seed123_step005100"

    def surface_dir(self, dataset: Dataset, scene: str, method: str) -> Path:
        """Surface outputs of one method: cameras.json, surface.rrd, fuse.ply, fuse_post.ply, fusion.json, surface_metrics.json."""
        return self.run_dir(dataset, scene, method) / "surface"
