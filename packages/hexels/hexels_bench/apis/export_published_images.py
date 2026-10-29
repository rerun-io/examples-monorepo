"""Stage the published 2D fits (HF jhu-ccvl/Hexels `images_2d/`, the paper's 20K -> 10K pruned recipe):
`runs/imagegs/<stem>/hexels-2d-pub-cuda/model.ply` (2D hexel PLY) and the shared ground truth
`runs/imagegs/<stem>/gt/test/<stem>.png`. No GPU needed."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from hexels_bench.apis.fit_image import stage_image_gt
from hexels_bench.image2d import ImageFit, load_image_fit, write_hexel_image_ply
from hexels_bench.layout import Roots
from hexels_bench.results2d import FITS, FitSpec

Appearance: TypeAlias = Literal["spatial-rgb", "flat-rgb"]


@dataclass
class Config:
    """Stage published 2D fits as 2D hexel PLYs."""

    stems: list[str] = field(default_factory=list)
    """Image stems; empty means every image under hexels-pretrained/images_2d."""
    appearance: Appearance = "spatial-rgb"
    """spatial-rgb (the paper's rich Hexels) or flat-rgb."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def stage_published_image(roots: Roots, stem: str, appearance: Appearance) -> Path:
    fit: Path = roots.published_image_fit(stem, appearance) / "hexels_raw-fp32.pt"
    spec: FitSpec = next(spec for spec in FITS.values() if spec.published and spec.appearance == appearance)
    out: Path = roots.runs / "imagegs" / stem / spec.cuda_method / "model.ply"
    loaded: ImageFit = load_image_fit(fit)
    write_hexel_image_ply(out, loaded.image, str(fit), loaded.step)
    stage_image_gt(roots, stem)
    return out


def main(config: Config) -> None:
    stems: list[str] = config.stems or sorted(
        path.name for path in (config.roots.data / "hexels-pretrained" / "images_2d").iterdir() if path.is_dir()
    )
    for stem in stems:
        print(f"staged {stage_published_image(config.roots, stem, config.appearance)}")
