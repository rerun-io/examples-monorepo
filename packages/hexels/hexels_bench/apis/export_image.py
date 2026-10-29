"""Export an upstream 2D Hexel fit (render_parameters_*.pt / hexels_raw-fp32.pt) to a 2D hexel PLY."""

from dataclasses import dataclass
from pathlib import Path

import torch

from hexels_bench.image2d import HexelImage, load_image_fit, write_hexel_image_ply


@dataclass
class Config:
    """Export one 2D fit."""

    fit: Path
    """Upstream raw-FP32 fit file."""
    out: Path
    """Destination 2D hexel PLY."""


def main(config: Config) -> None:
    image: HexelImage = load_image_fit(config.fit)
    step: int = int(torch.load(config.fit, map_location="cpu", weights_only=True).get("step", -1))
    write_hexel_image_ply(config.out, image, str(config.fit), step)
    print(f"wrote {config.out}: {image.centers_xy.shape[0]} hexels, {image.width}x{image.height}")
