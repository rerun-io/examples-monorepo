"""Load a Gaussian PLY in Python and log it to the external Rust viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from gsplat_rust_renderer.gaussians3d import Gaussians3D

APP_ID: str = "gsplat-rust-renderer"
VIEWER_URL: str = "rerun+http://127.0.0.1:9876/proxy"
VIEW_ROOT: str = "/"
DEFAULT_ENTITY_PATH: str = "world/splats"
DEFAULT_PLY: Path = Path(__file__).resolve().parents[1] / "examples" / "chair.ply"


def args_from_argv() -> tuple[Path, str]:
    """Parse command-line arguments for PLY path and entity path.

    Returns:
        A tuple of ``(ply_path, entity_path)``.
    """
    args: list[str] = sys.argv[1:]
    if len(args) > 2:
        raise SystemExit(
            "usage: python tools/log_gaussian_ply.py [scene.ply] [entity/path]"
        )
    ply_path: Path = Path(args[0]) if args else DEFAULT_PLY
    entity_path: str = args[1] if len(args) == 2 else DEFAULT_ENTITY_PATH
    return ply_path, entity_path


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


def main() -> None:
    """Entry point: load PLY and log to the running Rust viewer."""
    ply_path: Path
    entity_path: str
    ply_path, entity_path = args_from_argv()
    gaussians: Gaussians3D = Gaussians3D.from_ply(ply_path)

    rr.init(APP_ID, spawn=False)
    rr.connect_grpc(VIEWER_URL)
    rr.send_blueprint(splat_blueprint(entity_path, gaussians))
    rr.log(entity_path, rr.Clear(recursive=True), static=True)
    rr.log(entity_path, gaussians, static=True)
    rr.disconnect()

    print(f"Logged {ply_path} to {VIEWER_URL} as {entity_path}")


if __name__ == "__main__":
    main()
