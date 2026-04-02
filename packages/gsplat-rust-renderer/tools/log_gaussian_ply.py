"""Load a Gaussian PLY in Python and log it to the external Rust viewer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.gaussians3d import Gaussians3D

VIEW_ROOT: str = "/"
DEFAULT_PLY: Path = Path(__file__).resolve().parents[1] / "examples" / "chair.ply"


@dataclass
class LogPlyConfig:
    """Log a Gaussian splat PLY file to the custom Rust viewer."""

    rr_config: RerunTyroConfig
    """Rerun connection/output configuration. Use --rr-config.connect to send to the Rust viewer."""
    ply_path: Path = DEFAULT_PLY
    """Path to the Gaussian splat .ply file."""
    entity_path: str = "world/splats"
    """Rerun entity path to log the splats under."""


def splat_blueprint(entity_path: str, gaussians: Gaussians3D) -> rrb.Blueprint:
    """Create the smallest stock blueprint that binds one entity to the custom visualizer.

    Args:
        entity_path: The entity path where the Gaussians were logged.
        gaussians: The Gaussian data (used to compute camera bounds).

    Returns:
        A Rerun blueprint with the custom visualizer bound.
    """
    bounds_min: np.ndarray = gaussians.centers.min(axis=0)
    bounds_max: np.ndarray = gaussians.centers.max(axis=0)
    center: np.ndarray = 0.5 * (bounds_min + bounds_max)
    extent: np.ndarray = bounds_max - bounds_min
    distance: float = max(float(np.linalg.norm(extent)), 1.0) * 1.5

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin=VIEW_ROOT,
            name="Scene",
            overrides={entity_path: rrb.Visualizer("GaussianSplats3D")},
            eye_controls=rrb.EyeControls3D(
                position=center + np.array([distance, distance * 0.5, distance], dtype=np.float32),
                look_target=center,
                eye_up=(0.0, 1.0, 0.0),
            ),
        )
    )


def main(config: LogPlyConfig) -> None:
    """Load a PLY file and log it to the Rerun viewer.

    Args:
        config: CLI configuration parsed by tyro.
    """
    gaussians: Gaussians3D = Gaussians3D.from_ply(config.ply_path)

    rr.send_blueprint(splat_blueprint(config.entity_path, gaussians))
    rr.log(config.entity_path, rr.Clear(recursive=True), static=True)
    rr.log(config.entity_path, gaussians, static=True)

    print(f"Logged {config.ply_path} as {config.entity_path}")


if __name__ == "__main__":
    main(tyro.cli(LogPlyConfig))
