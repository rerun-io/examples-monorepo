"""Log a trained splat PLY together with its dataset cameras and GT images.

Demonstrates the custom ``GaussianSplats3D`` visualizer coexisting with stock
rerun archetypes: every dataset camera becomes an ``rr.Pinhole`` frustum with
its GT photo on the image plane, and the photos are also browsable in a paged
2x2 grid driven by a ``page`` sequence timeline (scrub or play to flip
through them — hidden tab views are NOT free in rerun, ~0.2 ms/frame each,
so tabs of hundreds of views would tank the frame rate).

Supports both dataset layouts:

- NeRF-synthetic: ``<scene_dir>/transforms_test.json`` + ``test/r_*.png``
  (OpenGL/RUB c2w matrices, used unmodified via simplecv's RUB conventions);
- COLMAP: ``<scene_dir>/sparse/0/{cameras,images}.bin`` + ``images/``
  (world-to-cam RDF poses, used unmodified; intrinsics rescaled when the
  shipped images are smaller than the calibrated resolution).

Examples:
    # Into the live desktop viewer (port 9876):
    python tools/log_splats_with_cameras.py --rr-config.connect \\
        --scene-dir data/nerf-synthetic/ship --ply-path data/trained/ship.ply

    # COLMAP scene, custom framing:
    python tools/log_splats_with_cameras.py --rr-config.connect \\
        --scene-dir data/tandt/train --ply-path /tmp/train.ply \\
        --eye 2.84 -4.09 -6.12 --look-target 0.43 -0.07 0.31 --eye-up 0 -1 0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8
from numpy import ndarray
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from gsplat_rust_renderer.gaussians3d import Gaussians3D
from gsplat_rust_renderer.scene_io import load_nerf_cameras, load_rgb_composited, qvec_to_rotmat, read_colmap_cameras_bin, read_colmap_images_bin


@dataclass
class LogSplatsWithCamerasConfig:
    """Log splats + dataset cameras + GT images to a rerun viewer."""

    rr_config: RerunTyroConfig
    """Viewer wiring (spawn/connect/save/serve) — use --rr-config.connect for a running viewer."""
    scene_dir: Path = Path("data/nerf-synthetic/ship")
    """Dataset directory: NeRF-synthetic (transforms_test.json) or COLMAP (sparse/0)."""
    ply_path: Path = Path("data/trained/ship.ply")
    """Trained 3DGS PLY to render through the GaussianSplats3D visualizer."""
    max_cameras: int = 0
    """Cap on the number of cameras logged (0 = all)."""
    browser: Literal["tabs", "pages"] = "tabs"
    """Image panel style.  'tabs': 2x2 grids of named camera views, four per
    tab — every camera individually inspectable, but each view in the
    blueprint costs ~0.2 ms of CPU per frame whether or not its tab is
    showing, so hundreds of cameras pull the frame rate down (200 ≈ 15-20
    FPS).  'pages': four fixed views
    paging through all images on the 'page' sequence timeline — stays at
    60 FPS at any camera count."""
    image_plane_distance: float = 0.1
    """Frustum image-plane distance in world units."""
    eye: tuple[float, float, float] | None = None
    """Initial 3D eye position; omit to let the viewer auto-frame the scene."""
    look_target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial look-at target (used only when --eye is given)."""
    eye_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Up vector for the initial eye pose (used only when --eye is given)."""


def load_colmap_cameras(scene_dir: Path) -> list[tuple[PinholeParameters, Path]]:
    """Read COLMAP sparse cameras as (pinhole, image path) pairs.

    COLMAP poses are world-to-cam in RDF — exactly what simplecv's
    ``Extrinsics(cam_R_world=..., cam_t_world=...)`` expects, no conversion.
    Intrinsics are rescaled when the shipped images are smaller than the
    calibrated resolution (T&T ships half-res images).
    """
    calibrations = read_colmap_cameras_bin(scene_dir / "sparse" / "0" / "cameras.bin")
    poses = sorted(read_colmap_images_bin(scene_dir / "sparse" / "0" / "images.bin"), key=lambda im: im["name"])
    cameras: list[tuple[PinholeParameters, Path]] = []
    for im in poses:
        calib = calibrations[im["camera_id"]]
        fx, fy, cx, cy = calib["params"]
        image_path: Path = scene_dir / "images" / im["name"]
        with Image.open(image_path) as probe:
            width, height = probe.size
        sx: float = width / calib["width"]
        sy: float = height / calib["height"]
        camera = PinholeParameters(
            name=Path(im["name"]).stem,
            extrinsics=Extrinsics(
                cam_R_world=qvec_to_rotmat(im["qvec"]),
                cam_t_world=np.asarray(im["tvec"], dtype=np.float64),
            ),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RDF",
                fl_x=fx * sx,
                fl_y=fy * sy,
                cx=cx * sx,
                cy=cy * sy,
                height=height,
                width=width,
            ),
        )
        cameras.append((camera, image_path))
    return cameras


def main(config: LogSplatsWithCamerasConfig) -> None:
    if (config.scene_dir / "transforms_test.json").exists():
        cameras: list[tuple[PinholeParameters, Path]] = load_nerf_cameras(config.scene_dir, "test")
    elif (config.scene_dir / "sparse" / "0" / "cameras.bin").exists():
        cameras = load_colmap_cameras(config.scene_dir)
    else:
        raise FileNotFoundError(f"{config.scene_dir} is neither a NeRF-synthetic nor a COLMAP scene")
    if config.max_cameras > 0:
        cameras = cameras[: config.max_cameras]

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/splats", Gaussians3D.from_ply(config.ply_path), static=True)

    for k, (camera, image_path) in enumerate(cameras):
        cam_path = f"world/cameras/{camera.name}"
        log_pinhole(camera, cam_log_path=Path(cam_path), image_plane_distance=config.image_plane_distance, static=True)
        rgb: UInt8[ndarray, "h w 3"] = load_rgb_composited(image_path, 255.0)
        rr.log(f"{cam_path}/pinhole/image", rr.Image(rgb), static=True)
        rr.set_time("page", sequence=k // 4)
        rr.log(f"browser/{k % 4}", rr.Image(rgb))
    rr.set_time("page", sequence=0)

    view3d = rrb.Spatial3DView(
        origin="/",
        name="splats + dataset cameras",
        overrides={"world/splats": rrb.Visualizer("GaussianSplats3D")},
        background=rrb.Background(color=(255, 255, 255), kind=rrb.BackgroundKind.SolidColor),
        line_grid=False,
        eye_controls=(
            rrb.EyeControls3D(position=config.eye, look_target=config.look_target, eye_up=config.eye_up) if config.eye is not None else None
        ),
    )
    if config.browser == "tabs":
        image_panel = rrb.Tabs(
            *[
                rrb.Grid(
                    *[rrb.Spatial2DView(origin=f"world/cameras/{camera.name}/pinhole", name=camera.name) for camera, _ in cameras[i : i + 4]],
                    grid_columns=2,
                    name=f"{cameras[i][0].name}–{cameras[min(i + 3, len(cameras) - 1)][0].name}",
                )
                for i in range(0, len(cameras), 4)
            ]
        )
    else:
        image_panel = rrb.Grid(
            *[rrb.Spatial2DView(origin=f"browser/{i}", name=f"slot {i}") for i in range(4)],
            grid_columns=2,
            name="image browser (scrub the page timeline)",
        )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(view3d, image_panel, column_shares=[5, 3]),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="expanded"),
        )
    )
    rec: rr.RecordingStream | None = rr.get_global_data_recording()
    assert rec is not None
    rec.flush(timeout_sec=120.0)
    panel_desc: str = f"{(len(cameras) + 3) // 4} tabs of 4" if config.browser == "tabs" else f"{(len(cameras) + 3) // 4} pages of 4"
    print(f"logged {len(cameras)} cameras ({panel_desc}) + {config.ply_path.name}")
