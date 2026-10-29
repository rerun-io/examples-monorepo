"""Export an upstream 2D Hexel fit (render_parameters_*.pt / hexels_raw-fp32.pt) to a 2D hexel PLY."""

from dataclasses import dataclass
from pathlib import Path

from hexels_bench.image2d import ImageFit, load_image_fit, write_hexel_image_ply


@dataclass
class Config:
    """Export one 2D fit."""

    fit: Path
    """Upstream raw-FP32 fit file."""
    out: Path
    """Destination 2D hexel PLY."""


def main(config: Config) -> None:
    fit: ImageFit = load_image_fit(config.fit)
    write_hexel_image_ply(config.out, fit.image, str(config.fit), fit.step)
    print(f"wrote {config.out}: {fit.image.centers_xy.shape[0]} hexels, {fit.image.width}x{fit.image.height}")
