"""Log native splats and dataset cameras with composited GT image planes.

The blueprint pairs a 3D view with capped train/test image grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int, UInt8
from numpy import ndarray
from simplecv.camera_parameters import PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from gsplat_rust_renderer.gaussians3d import SPLATS_ENTITY, compute_visualizer, log_ply
from gsplat_rust_renderer.nerfbaselines import DEFAULT_SCENE, scene_data_dir, scene_ply_path
from gsplat_rust_renderer.scene_io import load_nerf_cameras, load_rgb_composited

SPLITS: tuple[Literal["train", "test"], ...] = ("train", "test")


@dataclass
class LogSceneConfig:
    """Log splats + NeRF-synthetic dataset cameras + GT images to a rerun viewer."""

    rr_config: RerunTyroConfig
    """Viewer wiring (spawn/connect/save/serve) — use --rr-config.connect for a running viewer."""
    compute: bool = False
    """Explicitly select the custom compute visualizer; normally each viewer selects its renderer."""
    render_mode: Literal["default", "mip"] | None = None
    """Per-entity compute rendering mode, stored in the blueprint."""
    scene: str = DEFAULT_SCENE
    """nerfbaselines scene; resolves --scene-dir and --ply-path defaults when they are omitted."""
    scene_dir: Path | None = None
    """NeRF-synthetic scene dir (transforms_{train,test}.json); defaults to the --scene data dir."""
    ply_path: Path | None = None
    """Trained 3DGS PLY; defaults to the pretrained --scene PLY."""
    image_plane_distance: float = 0.2
    """Frustum image-plane distance in world units."""
    max_image_views: int = 8
    """Cap on camera image views shown per split in the blueprint grids (all cameras are still logged)."""


def _even_subset(count: int, cap: int) -> Int[ndarray, "k"]:
    """Return up to *cap* evenly-spaced indices into ``range(count)``."""
    if count <= 0 or cap <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, cap).round().astype(np.int64))


def log_split_cameras(scene_dir: Path, split: Literal["train", "test"], image_plane_distance: float) -> list[str]:
    """Log each camera as a Pinhole with its GT image; return the entity paths."""
    cameras: list[tuple[PinholeParameters, Path]] = load_nerf_cameras(scene_dir, split)
    cam_paths: list[str] = []
    for index, (camera, image_path) in enumerate(cameras):
        cam_path: str = f"/world/cameras/{split}_{index:04d}"
        log_pinhole(camera, cam_log_path=Path(cam_path), image_plane_distance=image_plane_distance)
        rgb: UInt8[ndarray, "h w 3"] = load_rgb_composited(image_path, background=255.0)
        rr.log(f"{cam_path}/pinhole/image", rr.Image(rgb))
        cam_paths.append(cam_path)
    return cam_paths


def scene_blueprint(split_cam_paths: dict[str, list[str]], max_image_views: int, compute: bool = False, render_mode: Literal["default", "mip"] | None = None) -> rrb.Blueprint:
    """Pair a 3D view with per-split image grids capped by max_image_views."""
    if render_mode is not None and not compute:
        raise ValueError("--render-mode requires --compute")
    grids: list[rrb.Grid] = []
    for split, cam_paths in split_cam_paths.items():
        subset: Int[ndarray, "k"] = _even_subset(len(cam_paths), max_image_views)
        grids.append(
            rrb.Grid(
                *[rrb.Spatial2DView(origin=f"{cam_paths[i]}/pinhole", name=Path(cam_paths[i]).name) for i in subset],
                grid_columns=2,
                name=f"{split} ({len(subset)}/{len(cam_paths)})",
            )
        )

    view3d = rrb.Spatial3DView(
        origin="/",
        name="splats + dataset cameras",
        overrides={SPLATS_ENTITY: compute_visualizer(render_mode or "default")} if compute else {},
        background=rrb.Background(color=(255, 255, 255), kind=rrb.BackgroundKind.SolidColor),
    )
    return rrb.Blueprint(
        rrb.Horizontal(view3d, rrb.Tabs(*grids, name="Camera views"), column_shares=[5, 3]),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),
    )


def main(config: LogSceneConfig) -> None:
    """Log static splats and frame-0 cameras, then send the scene blueprint."""
    scene_dir: Path = config.scene_dir if config.scene_dir is not None else scene_data_dir(config.scene)
    ply_path: Path = config.ply_path if config.ply_path is not None else scene_ply_path(config.scene)

    rr.set_time("frame", sequence=0)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_ply(ply_path)

    split_cam_paths: dict[str, list[str]] = {}
    for split in SPLITS:
        if (scene_dir / f"transforms_{split}.json").exists():
            split_cam_paths[split] = log_split_cameras(scene_dir, split, config.image_plane_distance)

    rr.send_blueprint(scene_blueprint(split_cam_paths, config.max_image_views, config.compute, config.render_mode))

    rec: rr.RecordingStream | None = rr.get_global_data_recording()
    assert rec is not None
    rec.flush(timeout_sec=120.0)
    total: int = sum(len(paths) for paths in split_cam_paths.values())
    print(f"logged {ply_path.name} + {total} cameras across {list(split_cam_paths)} at frame=0")
