"""Load a Gaussian PLY in Python and log it to the external Rust viewer.

Logs the splat under ``/world/splats`` as a static scene using the native ``GaussianSplats3D`` component schema, without a visualizer override. The custom viewer selects compute automatically.
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

from gsplat_rust_renderer.gaussians3d import SPLATS_ENTITY, compute_visualizer, log_ply
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


def splat_blueprint(bounds: Float32[ndarray, "2 3"], compute: bool = False, render_mode: Literal["default", "mip"] | None = None) -> rrb.Blueprint:
    """Build a minimal blueprint, with an optional explicit compute selection.

    Args:
        bounds: Native decoded center percentiles used to frame the initial camera.

    Returns:
        A single-3D-view blueprint; renderer selection defaults to the viewer.
    """
    if render_mode is not None and not compute:
        raise ValueError("--render-mode requires --compute")
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
    """Load a static PLY scene and frame it in the Rerun viewer.

    Args:
        config: CLI configuration parsed by tyro.
    """
    ply_path: Path = config.ply_path if config.ply_path is not None else scene_ply_path(config.scene)
    bounds: Float32[ndarray, "2 3"] = log_ply(ply_path)

    rr.send_blueprint(splat_blueprint(bounds, config.compute, config.render_mode))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    print(f"logged {ply_path} as static {SPLATS_ENTITY}")
