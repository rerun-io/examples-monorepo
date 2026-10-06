"""Load a Gaussian PLY in Python and log it to the external Rust viewer.

Logs the splat under ``/world/splats`` at ``frame=0`` on a sequence timeline
named ``"frame"`` using the native ``GaussianSplats3D`` component schema, without a visualizer override. The custom viewer selects compute automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.gaussians3d import SPLATS_ENTITY, compute_visualizer, splats_from_ply
from gsplat_rust_renderer.nerfbaselines import DEFAULT_SCENE, scene_ply_path

VIEW_ROOT: str = "/"


@dataclass
class LogPlyConfig:
    """Log a Gaussian splat PLY file to the custom Rust viewer."""

    rr_config: RerunTyroConfig
    """Rerun connection/output configuration. Use --rr-config.connect to send to the Rust viewer."""
    compute: bool = False
    """Explicitly select the custom compute visualizer; normally each viewer selects its renderer."""
    render_mode: Literal["default", "mip"] | None = None
    """Per-entity compute rendering mode, stored in the blueprint."""
    scene: str = DEFAULT_SCENE
    """nerfbaselines scene whose pretrained PLY is loaded when --ply-path is omitted."""
    ply_path: Path | None = None
    """Explicit Gaussian splat .ply path; defaults to the pretrained --scene PLY."""


def splat_blueprint(gaussians: rr.GaussianSplats3D, compute: bool = False, render_mode: Literal["default", "mip"] | None = None) -> rrb.Blueprint:
    """Build a minimal blueprint, with an optional explicit compute selection.

    Args:
        gaussians: The loaded splat data (used to frame the initial camera).

    Returns:
        A single-3D-view blueprint; renderer selection defaults to the viewer.
    """
    if render_mode is not None and not compute:
        raise ValueError("--render-mode requires --compute")
    assert gaussians.centers is not None
    # Percentile bounds: trained 3DGS scenes carry far-flung outlier splats
    # (sky/background), so a raw min/max bbox frames the camera much too far
    # out and the subject looks tiny on load.
    centers: Float32[ndarray, "n 3"] = gaussians.centers.as_arrow_array().flatten().to_numpy().reshape(-1, 3)
    bounds: Float32[ndarray, "2 3"] = np.percentile(centers, [2.0, 98.0], axis=0).astype(np.float32)
    bounds_min, bounds_max = bounds
    center: Float32[ndarray, "3"] = 0.5 * (bounds_min + bounds_max)
    extent: Float32[ndarray, "3"] = bounds_max - bounds_min
    distance: float = max(float(np.linalg.norm(extent)), 1.0) * 1.4
    # 3/4 view for a Z-up world (blender / nerf-synthetic convention).
    offset_dir: Float32[ndarray, "3"] = np.array([1.0, -1.0, 0.6], dtype=np.float32)
    offset_dir /= np.linalg.norm(offset_dir)

    return rrb.Blueprint(
        rrb.Spatial3DView(
            origin=VIEW_ROOT,
            name="Scene",
            overrides={SPLATS_ENTITY: compute_visualizer(render_mode or "default")} if compute else {},
            eye_controls=rrb.EyeControls3D(
                position=center + offset_dir * distance,
                look_target=center,
                eye_up=(0.0, 0.0, 1.0),
            ),
        )
    )


def main(config: LogPlyConfig) -> None:
    """Load a PLY file and log it to the Rerun viewer at frame 0.

    Args:
        config: CLI configuration parsed by tyro.
    """
    ply_path: Path = config.ply_path if config.ply_path is not None else scene_ply_path(config.scene)
    gaussians: rr.GaussianSplats3D = splats_from_ply(ply_path)

    rr.send_blueprint(splat_blueprint(gaussians, config.compute, config.render_mode))
    rr.set_time("frame", sequence=0)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(SPLATS_ENTITY, rr.Clear(recursive=True))
    rr.log(SPLATS_ENTITY, gaussians)

    print(f"logged {ply_path} as {SPLATS_ENTITY} ({len(gaussians.centers.as_arrow_array()) if gaussians.centers is not None else 0} splats) at frame=0")
