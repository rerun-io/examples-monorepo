"""Render real GPU splats inside brush's own training blueprint.

brush-cli (main branch, --rerun-enabled) already logs a rich rerun blueprint —
loss/lr/psnr/ssim/splats/refine/memory time-series, per-view (GT | Render) eval
pairs, dataset cameras, the world frame — but its rerun SDK only knows coarse
``Ellipsoids3D``, so its 3D scene is a fuzzy ellipsoid cloud.  This sidecar
overlays the real thing.

Two modes:

1. ``--brush-native`` (the good one).  Join brush's *own* recording and do
   exactly two things brush can't: overlay a ``GaussianSplats3D`` snapshot at
   ``world/splats`` per ``export_NNNNN.ply`` (on brush's ``iterations``
   timeline), and re-send brush's ``send_default_blueprint`` replica with one
   change — a visualizer override pinning ``world/splats`` to the custom
   GaussianSplats3D visualizer (otherwise the built-in Points3D wins the
   entity, since splat centers are ``Position3D``-typed).  Result: brush's exact
   blueprint with GPU splats in the Scene view.  Brush owns everything else.

   Workflow (no brush patch needed — share brush's auto-assigned recording id):
     a. Start the viewer headless on :9876 (``gsplat-rust-renderer --headless``).
     b. Start brush — it connects to that viewer and self-assigns a ``rec_*`` id:
          brush-cli DATA --rerun-enabled --export-every 200 \\
            --export-path RUN_DIR --eval-every 500
        (omit --rerun-log-splats-every so ``world/splat/points`` stays empty).
     c. Read brush's recording id from the viewer (its Sources panel, or the
        rerun MCP ``viewer_state``: the active "Brush" recording's id).
     d. Join it:
          python tools/visualize_brush_training.py --brush-native \\
            --rr-config.connect --rr-config.application-id Brush \\
            --rr-config.recording-id rec_... \\
            --scene-dir DATA --export-dir RUN_DIR --total-iters N

2. Legacy standalone (no ``--brush-native``).  Own recording, blueprint built
   from brush's *stdout* (psnr/ssim/splat-count only — no loss/lr/memory, since
   those live solely in brush's native logging), frusta with GT thumbnails.
   Useful for replaying a finished run with no live brush process to join.
"""

import dataclasses
import json
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Float64, UInt8
from numpy import ndarray
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole

from gsplat_rust_renderer.gaussians3d import Gaussians3D

EXPORT_RE: re.Pattern[str] = re.compile(r"export_(\d+)\.ply$")
EVAL_LINE_RE: re.Pattern[str] = re.compile(r"Eval iter (\d+): PSNR ([0-9.]+|nan|inf), ssim ([0-9.]+|nan|inf)")
REFINE_LINE_RE: re.Pattern[str] = re.compile(r"Refine iter (\d+), (\d+) splats\.")


@dataclass
class VisualizeBrushTrainingConfig:
    """Tail a brush-cli export dir and log the training run to the custom viewer."""

    rr_config: RerunTyroConfig
    """Viewer wiring (spawn/connect/save/serve) — use --rr-config.connect for the
    running viewer, and pin --rr-config.recording-id so restarts reattach.

    Brush-join mode (--brush-native): point this at brush's own recording with
    ``--rr-config.connect --rr-config.application-id Brush
    --rr-config.recording-id <the id passed to brush via BRUSH_RERUN_RECORDING_ID>``
    so our GPU splats land in the same recording as brush's native scalars/eval."""
    brush_native: bool = False
    """Join brush's own rerun recording instead of building a parallel one.

    Brush (--rerun-enabled) already logs the full rich blueprint: loss/lr/psnr/
    ssim/splats/refine/memory time-series, per-view eval pairs, dataset cameras,
    and the world coordinate frame.  The one thing it can't do is render real GPU
    splats (its rerun SDK only knows coarse Ellipsoids3D).  In this mode the
    sidecar joins brush's recording (share the recording id via
    BRUSH_RERUN_RECORDING_ID) and does exactly two things: overlay a
    GaussianSplats3D snapshot at ``world/splats`` per exported PLY (on brush's
    ``iterations`` timeline), and re-send brush's send_default_blueprint with one
    change — a visualizer override pinning ``world/splats`` to GaussianSplats3D so
    the Scene view renders our splats instead of the built-in Points3D.  Launch
    brush WITHOUT --rerun-log-splats-every so ``world/splat/points`` stays empty.

    When False, the legacy standalone path runs: own recording, stdout-parsed
    psnr/ssim/splat-count plots, frusta with GT thumbnails — useful for replaying
    a finished run with no live brush process to join."""
    scene_dir: Path = Path("data/nerf-synthetic/lego")
    """Scene dir.  NeRF-synthetic (transforms_train.json + transforms_val.json)
    or a real COLMAP/nerfstudio capture (auto-detected via colmap/sparse/0)."""
    eval_split_every: int = 0
    """COLMAP captures only: brush's --eval-split-every (every Nth sorted view is
    held out for eval).  Set it to match the brush flag so the GT panels track
    the same held-out views.  0 = no eval panels (still get the PSNR plot)."""
    export_dir: Path = Path("/tmp/brush-runs/lego")
    """brush-cli --export-path dir being watched for export_*.ply and eval_*/."""
    brush_log: Path | None = None
    """brush-cli stdout capture; parsed for PSNR/SSIM and refine splat counts."""
    total_iters: int = 30000
    """brush --total-train-iters; the checkpoint at this iter ends the watch."""
    poll_interval: float = 2.0
    """Seconds between export-dir scans."""
    eval_views_logged: int = 2
    """Number of val views logged as render-vs-GT image pairs each eval."""
    sh_mode: Literal["final", "all", "none"] = "final"
    """Which snapshots keep SH coefficients: 'final' = only the last checkpoint
    (intermediate snapshots are DC-only, ~4x lighter — color still comes from
    the DC term, so this is artifact-free), 'all', or 'none'.

    Note: every snapshot logs the FULL splat geometry (centers, rotations,
    scales, opacities, colors) like brush does. An earlier 'partial' mode that
    skipped rotations/scales and leaned on latest-at corrupted the in-between
    frames — splats rotate and rescale every training step, so a partial frame
    showed fresh centers with stale geometry (visible as smeared splats between
    keyframes). Dropped in favor of brush-matching full snapshots."""
    step_stride: int = 100
    """Everything is logged on two timelines: the true ``iteration`` sequence
    and a dense ``step`` sequence (= iteration / step_stride, i.e. 1, 2, 3, …
    when this matches brush's --export-every) so scrubbing through snapshots
    is one timeline tick per checkpoint."""
    snapshot_stride: int = 1
    """Log a full splat snapshot only every Nth checkpoint (the final one always
    logs).  Each snapshot uploads the COMPLETE splat geometry, so for big scenes
    (~1M splats) trained fast, logging every checkpoint floods the viewer (it
    re-uploads tens of MB per snapshot) and overruns its gRPC history buffer.
    A stride bounds the snapshot count/rate while every logged snapshot stays
    self-consistent.  PSNR/SSIM/splat-count scalars are cheap and logged every
    eval regardless of this."""
    spin_speed: float = 0.0
    """Continuous orbit speed (rad/s) for the 3D view's eye controls; 0 = off.
    Nonzero keeps the viewer repainting every frame — used for live FPS
    validation while training."""
    max_cameras: int = 0
    """Cap on train cameras logged (0 = all)."""
    image_plane_distance: float = 0.4
    """Frustum image-plane distance in world units."""
    plane_thumb_px: int = 160
    """Longest edge of the GT images textured onto the 3D frustum planes.
    100 full-res 800px planes overwhelm the GB10 at 2.5K viewport; thumbnails
    keep the look at a fraction of the texture cost. 0 = full resolution."""
    follow: bool = True
    """Keep polling until the total_iters checkpoint lands; False = log what
    exists now and exit (works on a finished run)."""
    stall_timeout: float = 1800.0
    """Abort if no new artifact appears for this many seconds while following."""


def load_nerf_cameras(scene_dir: Path, split: Literal["train", "val", "test"]) -> list[tuple[PinholeParameters, Path]]:
    """Read NeRF-synthetic cameras of one split as (pinhole, image path) pairs.

    The c2w matrices are OpenGL/RUB; simplecv carries the convention through
    to the Pinhole's ``camera_xyz``, so they are used unmodified.
    """
    transforms = json.loads((scene_dir / f"transforms_{split}.json").read_text())
    camera_angle_x: float = transforms["camera_angle_x"]
    cameras: list[tuple[PinholeParameters, Path]] = []
    for frame in transforms["frames"]:
        image_path: Path = (scene_dir / frame["file_path"]).with_suffix(".png")
        with Image.open(image_path) as probe:
            width, height = probe.size
        focal: float = 0.5 * width / np.tan(0.5 * camera_angle_x)
        c2w: Float64[ndarray, "4 4"] = np.array(frame["transform_matrix"], dtype=np.float64)
        camera = PinholeParameters(
            name=image_path.stem,
            extrinsics=Extrinsics(world_R_cam=c2w[:3, :3], world_t_cam=c2w[:3, 3]),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RUB",
                fl_x=focal,
                fl_y=focal,
                cx=width / 2.0,
                cy=height / 2.0,
                height=height,
                width=width,
            ),
        )
        cameras.append((camera, image_path))
    return cameras


def load_rgb_composited(image_path: Path, background: float) -> UInt8[ndarray, "h w 3"]:
    """Load an image as raw RGB with alpha composited onto a constant background.

    Raw ``rr.Image`` on purpose — ``rr.EncodedImage`` makes the viewer
    re-decode every visible image every frame.
    """
    with Image.open(image_path) as img:
        rgba: UInt8[ndarray, "h w c"] = np.asarray(img.convert("RGBA"))
    alpha: Float64[ndarray, "h w 1"] = rgba[..., 3:].astype(np.float64) / 255.0
    rgb: UInt8[ndarray, "h w 3"] = (rgba[..., :3].astype(np.float64) * alpha + background * (1.0 - alpha)).astype(np.uint8)
    return rgb


# ── Minimal COLMAP binary parsers (for real nerfstudio/COLMAP captures) ──────
# Mirrors tools/log_splats_with_cameras.py; kept local to avoid a shared import
# between two standalone CLI tools (both intentionally self-contained).  brush's
# colmap loader reads the same sparse model, so the cameras we draw match the
# poses brush trains on.


def _read_colmap_cameras_bin(path: Path) -> dict[int, dict]:
    models: dict[int, tuple[str, int]] = {
        0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    }
    cameras: dict[int, dict] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            _name, n_params = models[model_id]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {"width": w, "height": h, "params": params}
    return cameras


def _read_colmap_images_bin(path: Path) -> list[dict]:
    images: list[dict] = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)  # image id
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)  # skip 2D points
            images.append({"qvec": qvec, "tvec": tvec, "camera_id": cam_id, "name": name.decode()})
    return images


def _qvec_to_rotmat(q: tuple[float, float, float, float]) -> Float64[ndarray, "3 3"]:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def colmap_sparse_dir(scene_dir: Path) -> Path | None:
    """Return the COLMAP ``sparse/0`` dir under ``scene_dir`` (nerfstudio nests
    it under ``colmap/``), or None if this isn't a COLMAP capture."""
    for candidate in (scene_dir / "colmap" / "sparse" / "0", scene_dir / "sparse" / "0"):
        if (candidate / "cameras.bin").exists():
            return candidate
    return None


def thumbnail_image_path(scene_dir: Path, name: str) -> Path:
    """Prefer a downscaled copy (nerfstudio ships images_8/4/2) over the full-res
    original — frusta/GT only need small images and the originals can be 12MP."""
    for sub in ("images_8", "images_4", "images_2", "images"):
        p: Path = scene_dir / sub / name
        if p.exists():
            return p
    return scene_dir / "images" / name


def load_colmap_cameras(scene_dir: Path) -> list[tuple[PinholeParameters, Path]]:
    """Read a COLMAP sparse model as (pinhole, thumbnail-image-path) pairs, sorted
    by image name (the order brush registers them, so eval-split selection lines
    up).  COLMAP poses are world-to-cam in RDF — used unmodified.  Distortion is
    ignored for the frustum (pinhole approximation is fine for visualization)."""
    sparse: Path | None = colmap_sparse_dir(scene_dir)
    assert sparse is not None
    calibrations = _read_colmap_cameras_bin(sparse / "cameras.bin")
    poses = sorted(_read_colmap_images_bin(sparse / "images.bin"), key=lambda im: im["name"])
    cameras: list[tuple[PinholeParameters, Path]] = []
    for im in poses:
        calib = calibrations[im["camera_id"]]
        fx, fy, cx, cy = calib["params"][:4]
        thumb: Path = thumbnail_image_path(scene_dir, im["name"])
        with Image.open(thumb) as probe:
            width, height = probe.size
        sx: float = width / calib["width"]
        sy: float = height / calib["height"]
        camera = PinholeParameters(
            name=Path(im["name"]).stem,
            extrinsics=Extrinsics(cam_R_world=_qvec_to_rotmat(im["qvec"]), cam_t_world=np.asarray(im["tvec"], dtype=np.float64)),
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions="RDF", fl_x=fx * sx, fl_y=fy * sy, cx=cx * sx, cy=cy * sy, height=height, width=width
            ),
        )
        cameras.append((camera, thumb))
    return cameras


def estimate_up(cameras: list[tuple[PinholeParameters, Path]], cam_up_local: Float64[ndarray, "3"] | None = None) -> Float64[ndarray, "3"]:
    """Estimate world up from the cameras: each camera's image-up axis in world
    points roughly toward gravity-up, so the mean is a robust up estimate (the
    world frame is otherwise arbitrarily oriented).  ``cam_up_local`` is the
    image-up direction in camera-local coords — ``[0,-1,0]`` for RDF (COLMAP,
    the default) and ``[0,1,0]`` for RUB (NeRF-synthetic / OpenGL)."""
    if cam_up_local is None:
        cam_up_local = np.array([0.0, -1.0, 0.0])
    ups: Float64[ndarray, "n 3"] = np.array([cam.extrinsics.world_R_cam @ cam_up_local for cam, _ in cameras])
    up: Float64[ndarray, "3"] = ups.mean(axis=0)
    return up / np.linalg.norm(up)


def up_to_z_rotation(up: Float64[ndarray, "3"]) -> Float64[ndarray, "3 3"]:
    """Rotation R with ``R @ up == +Z`` (Rodrigues), so applying it to the world
    makes the estimated up axis vertical (the viewer assumes +Z up)."""
    z: Float64[ndarray, "3"] = np.array([0.0, 0.0, 1.0])
    v: Float64[ndarray, "3"] = np.cross(up, z)
    s: float = float(np.linalg.norm(v))
    c: float = float(np.dot(up, z))
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx: Float64[ndarray, "3 3"] = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def set_iteration_time(config: VisualizeBrushTrainingConfig, iteration: int) -> None:
    """Stamp subsequent logs with both the true iteration and the dense step index."""
    rr.set_time("iteration", sequence=iteration)
    rr.set_time("step", sequence=round(iteration / config.step_stride))


def awaiting_stable_size(path: Path, pending_sizes: dict[Path, int]) -> bool:
    """True while ``path``'s size is still changing across polls.

    brush writes each PLY/PNG in one non-atomic call, so a file can be observed
    half-written; wait for two equal, non-zero size scans before reading it.
    Records the current size in ``pending_sizes`` for the next comparison.
    """
    size: int = path.stat().st_size
    if pending_sizes.get(path) != size or size == 0:
        pending_sizes[path] = size
        return True
    return False


def eval_image_path(scene_dir: Path, name: str) -> Path:
    """A decent-resolution image for the eval GT panel (bigger than the frustum
    thumbnail) — nerfstudio ships images_4/images_2 downscales of the originals."""
    for sub in ("images_4", "images_2", "images"):
        p: Path = scene_dir / sub / name
        if p.exists():
            return p
    return scene_dir / "images" / name


def log_camera_frustum(
    cam_path: str, camera: PinholeParameters, image_path: Path, config: VisualizeBrushTrainingConfig, conventions: Literal["RDF", "RUB"]
) -> None:
    """Log a camera as a Pinhole frustum with its photo on the image plane.

    The plane image is a thumbnail; the Pinhole is rescaled to the thumbnail's
    resolution so the photo fills the whole plane (not just its top-left corner)
    while the frustum FOV stays identical to the full-res camera.
    """
    rgb: UInt8[ndarray, "h w 3"] = load_rgb_composited(image_path, background=255.0)
    if config.plane_thumb_px > 0:
        h, w = rgb.shape[:2]
        scale: float = config.plane_thumb_px / max(h, w)
        tw: int = max(1, round(w * scale))
        th: int = max(1, round(h * scale))
        plane_rgb: UInt8[ndarray, "th tw 3"] = np.asarray(Image.fromarray(rgb).resize((tw, th), Image.Resampling.BILINEAR))
        plane_camera = PinholeParameters(
            name=camera.name,
            extrinsics=camera.extrinsics,
            intrinsics=Intrinsics.from_focal_principal_point(
                camera_conventions=conventions,
                fl_x=camera.intrinsics.fl_x * (tw / w),
                fl_y=camera.intrinsics.fl_y * (th / h),
                cx=tw / 2.0,
                cy=th / 2.0,
                height=th,
                width=tw,
            ),
        )
    else:
        plane_rgb = rgb
        plane_camera = camera
    log_pinhole(plane_camera, cam_log_path=Path(cam_path), image_plane_distance=config.image_plane_distance, static=True)
    rr.log(f"{cam_path}/pinhole/image", rr.Image(plane_rgb), static=True)


def log_static_scene(config: VisualizeBrushTrainingConfig) -> list[tuple[PinholeParameters, Path]]:
    """Log coordinates, dataset-camera frusta with GT thumbnail planes, and the
    static GT halves of the eval comparison rows.

    Handles both NeRF-synthetic scenes and real COLMAP/nerfstudio captures
    (auto-detected); returns the eval cameras whose renders will be tracked.
    """
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if colmap_sparse_dir(config.scene_dir) is not None:
        conventions: Literal["RDF", "RUB"] = "RDF"
        all_cameras: list[tuple[PinholeParameters, Path]] = load_colmap_cameras(config.scene_dir)
        # COLMAP's world frame is arbitrarily oriented (its up is rarely +Z), so
        # the splats brush exports come out tilted under a +Z-up viewer.  Rotate
        # the whole world subtree (splats + cameras) so the camera-estimated up
        # is vertical.  The splats and cameras stay mutually consistent.
        r_up: Float64[ndarray, "3 3"] = up_to_z_rotation(estimate_up(all_cameras))
        rr.log("world", rr.Transform3D(mat3x3=r_up.tolist()), static=True)
        # brush holds out every eval_split_every-th sorted view for eval.
        eval_cameras: list[tuple[PinholeParameters, Path]] = (
            [
                (camera, eval_image_path(config.scene_dir, thumb.name))
                for i, (camera, thumb) in enumerate(all_cameras)
                if config.eval_split_every > 0 and i % config.eval_split_every == 0
            ][: config.eval_views_logged]
        )
    else:
        conventions = "RUB"
        all_cameras = load_nerf_cameras(config.scene_dir, "train")
        val_split: Literal["val", "test"] = "val" if (config.scene_dir / "transforms_val.json").exists() else "test"
        eval_cameras = load_nerf_cameras(config.scene_dir, val_split)[: config.eval_views_logged]

    if config.max_cameras > 0:
        all_cameras = all_cameras[: config.max_cameras]
    for camera, image_path in all_cameras:
        log_camera_frustum(f"world/cameras/{camera.name}", camera, image_path, config, conventions)

    for camera, image_path in eval_cameras:
        # Brush eval renders composite on black; match it for the GT half.
        gt: UInt8[ndarray, "h w 3"] = load_rgb_composited(image_path, background=0.0)
        rr.log(f"eval/{camera.name}/gt", rr.Image(gt), static=True)

    # Named series so the plot legends read psnr/ssim/num_splats.
    rr.log("plots/psnr", rr.SeriesLines(names="psnr (brush eval)"), static=True)
    rr.log("plots/ssim", rr.SeriesLines(names="ssim (brush eval)"), static=True)
    rr.log("plots/num_splats", rr.SeriesLines(names="splat count"), static=True)
    return eval_cameras


def send_blueprint(config: VisualizeBrushTrainingConfig, eval_cameras: list[tuple[PinholeParameters, Path]]) -> None:
    # Spin mode = continuous full-rate repaint for FPS validation; the eval-image
    # and time-series side views cost ~20 ms/frame under continuous repaint in the
    # viewer (measured), so the spinning layout is 3D-only and excludes
    # the camera frusta.  Everything stays logged either way — rerun without
    # --spin-speed (or switch blueprints) to inspect the panels.
    spinning: bool = config.spin_speed > 0.0
    view3d = rrb.Spatial3DView(
        origin="/",
        name="splats live (spin)" if spinning else "splats + train cameras",
        contents=["+ $origin/**", "- /eval/**", "- /plots/**"] + (["- world/cameras/**"] if spinning else []),
        overrides={"world/splats": rrb.Visualizer("GaussianSplats3D")},
        background=rrb.Background(color=(255, 255, 255), kind=rrb.BackgroundKind.SolidColor),
        line_grid=False,
        eye_controls=(
            rrb.EyeControls3D(
                kind="orbital",
                position=(3.4, -3.4, 2.0),
                look_target=(0.0, 0.0, 0.3),
                eye_up=(0.0, 0.0, 1.0),
                spin_speed=config.spin_speed,
            )
            if spinning
            else None
        ),
    )
    if spinning:
        rr.send_blueprint(
            rrb.Blueprint(
                view3d,
                rrb.BlueprintPanel(state="collapsed"),
                rrb.SelectionPanel(state="collapsed"),
                rrb.TimePanel(state="collapsed"),
            )
        )
        return

    # Layout follows brush's send_default_blueprint: a Vertical split with the
    # 3D scene + per-view (Ground truth | Render) eval pairs on top, and the
    # metric time-series in a graph row below.  Two intentional deviations from
    # brush: we keep the 3D scene prominent (brush gives it 1/4 width — ours is a
    # real GPU splat render, not coarse ellipsoids), and we surface only the
    # series we parse from brush's stdout (psnr/ssim/splat count).
    eval_panel = rrb.Grid(
        *[
            rrb.Horizontal(
                rrb.Spatial2DView(origin=f"eval/{camera.name}/gt", name="Ground truth"),
                rrb.Spatial2DView(origin=f"eval/{camera.name}/render", name="Render"),
                name=camera.name,
            )
            for camera, _ in eval_cameras
        ],
        grid_columns=1,
        name="Eval views",
    )
    main_row = rrb.Horizontal(view3d, eval_panel, column_shares=[2, 1])
    graphs = rrb.Horizontal(
        rrb.TimeSeriesView(origin="plots", contents=["+ plots/psnr", "+ plots/ssim"], name="Quality"),
        rrb.TimeSeriesView(origin="plots", contents=["+ plots/num_splats"], name="Splats"),
    )
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(main_row, graphs, row_shares=[3, 2]),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="expanded"),
            auto_layout=False,
            auto_views=False,
        )
    )


def log_checkpoint(ply_path: Path, with_sh: bool) -> int:
    """Log one checkpoint PLY as the full ``world/splats`` snapshot.

    The caller must stamp the timeline first.  Every geometry component (centers,
    rotations, scales, opacities, colors) is logged every time — like brush — so
    each timeline point renders a self-consistent splat cloud.  ``with_sh``
    controls only whether the (heavy, view-dependent) higher-order SH coefficients
    ride along; DC color is always present, so dropping them is artifact-free.
    Returns the splat count.
    """
    splats: Gaussians3D = Gaussians3D.from_ply(ply_path)
    if not with_sh and splats.sh_coefficients is not None:
        splats = dataclasses.replace(splats, sh_coefficients=None)
    num_splats: int = splats.centers.shape[0]
    rr.log("world/splats", splats)
    return num_splats


# ── Brush-join mode ──────────────────────────────────────────────────────────
# Overlay GPU splats on brush's own recording and replicate brush's blueprint.


def count_eval_views(config: VisualizeBrushTrainingConfig) -> int:
    """Number of eval views brush logs as ``eval/view_{i}`` — drives the eval-tab
    layout of the replicated blueprint.  Mirrors brush's own split logic: the
    full ``transforms_val``/``transforms_test`` set for NeRF-synthetic, or every
    ``eval_split_every``-th sorted view for a COLMAP capture with no val file.
    Reads only metadata (no image decodes), so it stays cheap for big sets."""
    if colmap_sparse_dir(config.scene_dir) is not None:
        if config.eval_split_every <= 0:
            return 0
        sparse: Path | None = colmap_sparse_dir(config.scene_dir)
        assert sparse is not None
        num_views: int = len(_read_colmap_images_bin(sparse / "images.bin"))
        return sum(1 for i in range(num_views) if i % config.eval_split_every == 0)
    val_split: Literal["val", "test"] = "val" if (config.scene_dir / "transforms_val.json").exists() else "test"
    transforms: dict = json.loads((config.scene_dir / f"transforms_{val_split}.json").read_text())
    return len(transforms["frames"])


def brush_blueprint(num_eval_views: int, splat_entity: str = "world/splats") -> rrb.Blueprint:
    """Python replica of brush's ``VisualizeTools::send_default_blueprint`` with a
    single change: a visualizer override pinning ``splat_entity`` to the custom
    GaussianSplats3D visualizer (brush's coarse Ellipsoids3D path is unused — we
    overlay real GPU splats).  Everything else — the Vertical(main_row, graphs)
    split, the Quality/Splats/Refine/Memory/Other graph tabs, the per-view
    (Ground truth | Render) eval cells grouped 4-per-tab — matches brush exactly
    (crates/brush-rerun/src/visualize_tools.rs).  Legend names, scalars, eval
    images and cameras all come from brush's own logging into this recording, so
    this only builds the layout tree."""
    # Scene: a plain Spatial3DView over world/**, with the one visualizer override.
    scene_view = rrb.Spatial3DView(
        name="Scene",
        origin="world",
        contents=["world/**"],
        overrides={splat_entity: rrb.Visualizer("GaussianSplats3D")},
    )

    # Each eval view = a Horizontal[Ground truth, Render] cell; groups of up to 4
    # become a 2-column Grid, and >4 views split those grids into switchable tabs.
    def eval_cell(i: int) -> rrb.Horizontal:
        return rrb.Horizontal(
            rrb.Spatial2DView(name="Ground truth", origin=f"eval/view_{i}/ground_truth", contents=["$origin/**"]),
            rrb.Spatial2DView(name="Render", origin=f"eval/view_{i}/render", contents=["$origin/**"]),
            name=f"view {i}",
        )

    def eval_group(start: int, end: int) -> rrb.Container:
        cells: list[rrb.Horizontal] = [eval_cell(i) for i in range(start, end)]
        if len(cells) == 1:
            return cells[0]
        return rrb.Grid(*cells, grid_columns=2, name=f"views {start}-{end - 1}")

    group_size: int = 4
    if num_eval_views == 0:
        main_row: rrb.Container = rrb.Horizontal(scene_view)
    else:
        if num_eval_views <= group_size:
            eval_panel: rrb.Container = eval_group(0, num_eval_views)
        else:
            num_groups: int = (num_eval_views + group_size - 1) // group_size
            groups: list[rrb.Container] = [
                eval_group(g * group_size, min(g * group_size + group_size, num_eval_views)) for g in range(num_groups)
            ]
            eval_panel = rrb.Tabs(*groups, name="Eval views")
        main_row = rrb.Horizontal(eval_panel, scene_view, column_shares=[3.0, 1.0])

    quality_tabs = rrb.Tabs(
        rrb.TimeSeriesView(name="PSNR", contents=["psnr/eval"]),
        rrb.TimeSeriesView(name="PSNR per view", contents=["psnr/per_view/**"]),
        rrb.TimeSeriesView(name="SSIM", contents=["ssim/eval"]),
        rrb.TimeSeriesView(name="SSIM per view", contents=["ssim/per_view/**"]),
        rrb.TimeSeriesView(name="Loss", contents=["loss/**"]),
        name="Quality",
    )
    splats_view = rrb.TimeSeriesView(name="Splats", contents=["splats/**"])
    refine_view = rrb.TimeSeriesView(
        name="Refine",
        contents=[
            "refine/num_split_oversized",
            "refine/num_split_high_grad",
            "refine/num_pruned",
            "refine/num_pruned_non_finite",
            "refine/effective_growth",
        ],
    )
    memory_view = rrb.TimeSeriesView(name="Memory", contents=["memory/**"])
    other_tabs = rrb.Tabs(
        rrb.TimeSeriesView(name="Throughput", contents=["train/step_ms", "refine/duration_ms"]),
        rrb.TimeSeriesView(name="Learning rates", contents=["lr/**"]),
        name="Other",
    )
    graphs = rrb.Horizontal(quality_tabs, splats_view, refine_view, memory_view, other_tabs)

    return rrb.Blueprint(
        rrb.Vertical(main_row, graphs, row_shares=[3.0, 2.0]),
        auto_layout=False,
        auto_views=False,
    )


def orient_brush_native(config: VisualizeBrushTrainingConfig) -> None:
    """Re-orient brush's world so the scene is +Z up in the viewer.

    Brush logs ``world`` as ``RIGHT_HAND_Y_DOWN`` for every dataset, and the
    Scene view's origin is ``world``, so that label drives the view's up axis.
    It's right for COLMAP (whose world really is Y-down) but wrong for the
    blender Z-up NeRF-synthetic world — the bulldozer renders on its side.
    Brush's ``opengl_c2w_to_pose`` only flips the *camera's* local axes, leaving
    the world frame intact, so the cameras we load from ``scene_dir`` live in the
    same frame as brush's splats: estimate gravity-up from them and rotate the
    whole ``world`` subtree so up is +Z, then relabel ``world`` (and root) Z-up,
    overriding brush's Y-down.  Mirrors the standalone path's orientation fix."""
    if colmap_sparse_dir(config.scene_dir) is not None:
        cameras: list[tuple[PinholeParameters, Path]] = load_colmap_cameras(config.scene_dir)
        up: Float64[ndarray, "3"] = estimate_up(cameras)  # RDF image-up = -Y
    else:
        cameras = load_nerf_cameras(config.scene_dir, "train")
        up = estimate_up(cameras, np.array([0.0, 1.0, 0.0]))  # RUB image-up = +Y
    r_up: Float64[ndarray, "3 3"] = up_to_z_rotation(up)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world", rr.Transform3D(mat3x3=r_up.tolist()), static=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def run_brush_native(config: VisualizeBrushTrainingConfig) -> None:
    """Join brush's recording, overlay GPU splats per exported PLY on brush's
    ``iterations`` timeline, and send brush's blueprint replica (with the splat
    visualizer override).  Brush owns the scalars, eval and cameras; we add the
    splats and re-orient the world to +Z up."""
    orient_brush_native(config)
    num_eval_views: int = count_eval_views(config)
    print(f"brush-join mode: overlaying world/splats on brush's recording; {num_eval_views} eval views, watching {config.export_dir}")

    done_plys: dict[int, int] = {}  # iteration -> splat count
    pending_sizes: dict[Path, int] = {}  # candidate files awaiting a stable size
    n_checkpoints_seen: int = 0  # for snapshot_stride
    blueprint_sent: bool = False
    last_progress: float = time.monotonic()

    while True:
        progressed: bool = False
        candidates: list[tuple[int, Path]] = sorted(
            (int(m.group(1)), p) for p in config.export_dir.glob("export_*.ply") if (m := EXPORT_RE.search(p.name)) and "lod" not in p.name
        )
        for iteration, ply_path in candidates:
            if iteration in done_plys:
                continue
            if config.follow and awaiting_stable_size(ply_path, pending_sizes):
                continue
            is_final: bool = iteration >= config.total_iters
            if n_checkpoints_seen % config.snapshot_stride == 0 or is_final:
                with_sh: bool = config.sh_mode == "all" or (config.sh_mode == "final" and is_final)
                rr.set_time("iterations", sequence=iteration)
                num_splats: int = log_checkpoint(ply_path, with_sh)
                done_plys[iteration] = num_splats
                # Send our blueprint only after the first real snapshot exists.  By
                # now brush has long since sent its own default blueprint, so this
                # send wins the activation and the Scene view picks up our splat
                # visualizer override (a single send avoids resetting the camera).
                if not blueprint_sent:
                    rr.send_blueprint(brush_blueprint(num_eval_views))
                    blueprint_sent = True
                print(f"iter {iteration:>6}: logged {num_splats} splats ({ply_path.name}, sh={'on' if with_sh else 'off'})")
            else:
                done_plys[iteration] = 0  # processed but skipped by snapshot_stride
            n_checkpoints_seen += 1
            progressed = True

        if progressed:
            last_progress = time.monotonic()
        finished: bool = any(it >= config.total_iters for it in done_plys)
        if finished or not config.follow:
            break
        if time.monotonic() - last_progress > config.stall_timeout:
            raise RuntimeError(f"no new brush artifacts in {config.export_dir} for {config.stall_timeout:.0f}s — is brush-cli still running?")
        time.sleep(config.poll_interval)

    rec: rr.RecordingStream | None = rr.get_global_data_recording()
    assert rec is not None
    rec.flush(timeout_sec=120.0)
    final_iter: int = max(done_plys) if done_plys else 0
    print(f"done: {len([v for v in done_plys.values() if v])} splat snapshots (final iter {final_iter}, {done_plys.get(final_iter, 0)} splats)")


def main(config: VisualizeBrushTrainingConfig) -> None:
    if config.brush_native:
        run_brush_native(config)
        return

    eval_cameras: list[tuple[PinholeParameters, Path]] = log_static_scene(config)
    send_blueprint(config, eval_cameras)
    print(f"static scene logged: train cameras + {len(eval_cameras)} eval views tracked from {config.export_dir}")

    done_plys: dict[int, int] = {}  # iteration -> splat count
    pending_sizes: dict[Path, int] = {}  # candidate files awaiting a stable size
    done_eval_imgs: set[Path] = set()
    done_eval_dirs: set[int] = set()  # eval dirs whose tracked renders are all logged
    done_eval_iters: set[int] = set()  # brush prints each eval line twice under RUST_LOG=info
    n_checkpoints_seen: int = 0  # for snapshot_stride
    log_offset: int = 0
    last_progress: float = time.monotonic()

    while True:
        progressed: bool = False

        # 1. Eval renders for the tracked views (brush saves eval_<iter>/<name>.png).
        #    Logged before the checkpoints: on finished-run replays this is the
        #    cheap part, so the eval panels and metric plots fill in seconds
        #    instead of after the multi-minute PLY grind.  Fully-logged eval dirs
        #    are remembered so later polls don't re-glob/-stat them.
        for eval_dir in sorted(config.export_dir.glob("eval_*")):
            name: str = eval_dir.name.removeprefix("eval_")
            if not name.isdigit() or int(name) in done_eval_dirs or not eval_dir.is_dir():
                continue
            iteration = int(name)
            pending_here: bool = False
            for camera, _ in eval_cameras:
                # brush names eval renders "<img_filename>.png", so the original
                # extension survives: NeRF-synthetic r_98.png -> r_98.png.png,
                # COLMAP frame_00001.jpg -> frame_00001.jpg.png.  Match the stem
                # plus any middle extension(s) ending in png.
                matches: list[Path] = sorted(eval_dir.rglob(f"{camera.name}.*png*"))
                if not matches:
                    pending_here = True
                    continue
                render_path: Path = matches[0]
                if render_path in done_eval_imgs:
                    continue
                if config.follow and awaiting_stable_size(render_path, pending_sizes):
                    pending_here = True
                    continue
                with Image.open(render_path) as img:
                    render: UInt8[ndarray, "h w 3"] = np.asarray(img.convert("RGB"))
                set_iteration_time(config, iteration)
                rr.log(f"eval/{camera.name}/render", rr.Image(render))
                done_eval_imgs.add(render_path)
                progressed = True
            if not pending_here:
                done_eval_dirs.add(iteration)

        # 2. Brush stdout: PSNR/SSIM eval lines + refine splat counts.  Read only
        #    the bytes appended since the last poll — the log grows to many MB
        #    over a run, so re-reading the whole file each poll is wasteful.
        if config.brush_log is not None and config.brush_log.exists():
            with open(config.brush_log, "rb") as log_file:
                log_file.seek(log_offset)
                chunk: bytes = log_file.read()
            log_offset += len(chunk)
            new_text: str = chunk.decode(errors="replace")
            for m in EVAL_LINE_RE.finditer(new_text):
                if int(m.group(1)) in done_eval_iters:
                    continue
                done_eval_iters.add(int(m.group(1)))
                set_iteration_time(config, int(m.group(1)))
                rr.log("plots/psnr", rr.Scalars(float(m.group(2))))
                rr.log("plots/ssim", rr.Scalars(float(m.group(3))))
                progressed = True
                print(f"iter {int(m.group(1)):>6}: psnr={m.group(2)} ssim={m.group(3)}")
            for m in REFINE_LINE_RE.finditer(new_text):
                set_iteration_time(config, int(m.group(1)))
                rr.log("plots/num_splats", rr.Scalars(float(m.group(2))))

        # 3. Checkpoint PLYs: process once the size is stable across two scans
        #    (brush writes the file in one async call, but not atomically).
        candidates: list[tuple[int, Path]] = sorted(
            (int(m.group(1)), p) for p in config.export_dir.glob("export_*.ply") if (m := EXPORT_RE.search(p.name)) and "lod" not in p.name
        )
        for iteration, ply_path in candidates:
            if iteration in done_plys:
                continue
            # Live mode only: wait for brush's non-atomic write to settle (a
            # finished run's files are all complete).
            if config.follow and awaiting_stable_size(ply_path, pending_sizes):
                continue
            is_final: bool = iteration >= config.total_iters
            if n_checkpoints_seen % config.snapshot_stride == 0 or is_final:
                with_sh: bool = config.sh_mode == "all" or (config.sh_mode == "final" and is_final)
                set_iteration_time(config, iteration)
                num_splats: int = log_checkpoint(ply_path, with_sh)
                rr.log("plots/num_splats", rr.Scalars(float(num_splats)))
                done_plys[iteration] = num_splats
                print(f"iter {iteration:>6}: logged {num_splats} splats ({ply_path.name}, sh={'on' if with_sh else 'off'})")
            else:
                done_plys[iteration] = 0  # processed but skipped by snapshot_stride
            n_checkpoints_seen += 1
            progressed = True

        if progressed:
            last_progress = time.monotonic()

        finished: bool = any(it >= config.total_iters for it in done_plys)
        if finished or not config.follow:
            break
        if time.monotonic() - last_progress > config.stall_timeout:
            raise RuntimeError(f"no new brush artifacts in {config.export_dir} for {config.stall_timeout:.0f}s — is brush-cli still running?")
        time.sleep(config.poll_interval)

    rec: rr.RecordingStream | None = rr.get_global_data_recording()
    assert rec is not None
    rec.flush(timeout_sec=120.0)
    final_iter: int = max(done_plys) if done_plys else 0
    print(f"done: {len(done_plys)} checkpoints (final iter {final_iter}, {done_plys.get(final_iter, 0)} splats), {len(done_eval_imgs)} eval renders")


if __name__ == "__main__":
    main(tyro.cli(VisualizeBrushTrainingConfig))
