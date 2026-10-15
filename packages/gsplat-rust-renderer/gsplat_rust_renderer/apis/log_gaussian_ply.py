"""Log a static native GaussianSplats3D scene; the viewer selects its renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Float32
from numpy import ndarray
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.gaussians3d import SPLATS_ENTITY, SplatSource, log_ply, splat_overrides
from gsplat_rust_renderer.nerfbaselines import scene_ply_path

VIEW_ROOT: str = "/"


@dataclass
class LogPlyConfig:
    """Log a Gaussian splat PLY file to the custom Rust viewer."""

    rr_config: RerunTyroConfig
    """Rerun connection/output configuration. Use --rr-config.connect to send to the Rust viewer."""
    source: Annotated[SplatSource, tyro.conf.OmitArgPrefixes] = field(default_factory=SplatSource)
    """Splat source and rendering choices, exposed without a flag prefix."""



def splat_blueprint(bounds: Float32[ndarray, "2 3"], compute: bool = False, render_mode: Literal["default", "mip"] | None = None) -> rrb.Blueprint:
    """Frame decoded center percentiles, with optional explicit compute selection.

    Args:
        bounds: Float32[ndarray, "2 3"] lower and upper center percentiles.
    """
    overrides: dict[str, rrb.Visualizer] = splat_overrides(compute, render_mode)
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
            overrides={**overrides},
            eye_controls=rrb.EyeControls3D(
                position=center + offset_dir * distance,
                look_target=center,
                eye_up=(0.0, 0.0, 1.0),
            ),
        )
    )


def main(config: LogPlyConfig) -> None:
    """Load a static PLY scene and frame it in the Rerun viewer."""
    ply_path: Path = config.source.ply_path if config.source.ply_path is not None else scene_ply_path(config.source.scene)
    bounds: Float32[ndarray, "2 3"] = log_ply(ply_path)

    rr.send_blueprint(splat_blueprint(bounds, config.source.compute, config.source.render_mode))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    print(f"logged {ply_path} as static {SPLATS_ENTITY}")
