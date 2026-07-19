"""Re-log regression check for the gsplat viewer (requires a GPU + built viewer binary).

Logs a splat cloud, screenshots it, then RE-LOGS the same entity twice —
(a) same splat count with different colors/positions, (b) different splat
count — and asserts the rendered output follows the new data each time.

Exercises the bug where the viewer's per-entity GPU caches (CloudSignature /
batch_cache attribute buffers) served stale data after a re-log.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from rerun.experimental import ViewerClient

from gsplat_rust_renderer.gaussians3d import SPLATS_ENTITY, SPLATS_VISUALIZER, Gaussians3D

BINARY: str = str(Path(__file__).resolve().parents[2] / "target" / "release" / "gsplat-rust-renderer")
"""The custom viewer binary built from this package (cargo build --release). This
file lives at ``<pkg>/gsplat_rust_renderer/apis/``, so the package root (where
``target/`` sits) is ``parents[2]``."""


def make_cloud(n_side: int, rgb: tuple[float, float, float], spread: float) -> Gaussians3D:
    """A flat n_side x n_side grid of fat splats in the z=0 plane, one color."""
    xs = np.linspace(-spread, spread, n_side, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs)
    centers = np.stack([gx.ravel(), gy.ravel(), np.zeros(n_side * n_side, np.float32)], axis=1)
    n = centers.shape[0]
    quaternions = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
    scales = np.full((n, 3), 0.18, np.float32)
    # opacity lives in the color alpha channel now (0.95 -> 242/255).
    rgba = np.array([*(round(c * 255.0) for c in rgb), 242], np.uint8)
    colors_rgba = np.tile(rgba, (n, 1))
    return Gaussians3D(centers=centers, quaternions_xyzw=quaternions, scales=scales, colors_rgba=colors_rgba)


def screenshot(viewer: ViewerClient, rec, path: Path) -> np.ndarray:
    # Full-frame capture (panels are collapsed, so the single view fills the
    # frame): per-view `view_id` captures can hang without a diagnostic on
    # 0.34 when the view can't be resolved to a rendered frame.
    rec.flush(timeout_sec=15.0)
    time.sleep(2.5)  # ingest + render settle
    viewer.save_screenshot(str(path))
    # The write is async on the viewer side; a fixed settle then tolerant
    # retries has proven far more reliable than tight stat-polling here.
    img = None
    for _ in range(10):
        time.sleep(2.0)
        try:
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            break
        except (FileNotFoundError, OSError):
            continue
    if img is None:
        raise RuntimeError(f"screenshot {path} never completed")
    h, w = img.shape[:2]
    crop = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return crop.reshape(-1, 3).mean(axis=0)


def dominant(mean_rgb: np.ndarray) -> str:
    return "rgb"[int(np.argmax(mean_rgb))]


def main() -> int:
    # Fresh directory per run: reusing fixed paths lets an async
    # save_screenshot race read a stale PNG from a PREVIOUS run and falsely
    # pass the gate (flagged by adversarial review).
    out = Path(tempfile.mkdtemp(prefix="relog-check-"))
    print(f"screenshots -> {out}")
    # Detached: an attached (child-process) viewer shares its lifetime and
    # output pipes with this process, which has proven racy around re-log +
    # screenshot (truncated PNGs when the SDK's background write thread
    # aborts). Detach and kill explicitly in `finally` instead.
    viewer = ViewerClient.spawn(
        headless=True,
        port=9951,
        executable_path=BINARY,
        hide_welcome_screen=True,
        detach_process=True,
    )
    failures: list[str] = []
    try:
        rr.init("relog_check")
        rr.connect_grpc(viewer.url)
        rec = rr.get_global_data_recording()
        assert rec is not None

        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(SPLATS_ENTITY, make_cloud(8, (1.0, 0.05, 0.05), 1.2), static=True)

        view = rrb.Spatial3DView(
            origin="/",
            name="relog",
            overrides={SPLATS_ENTITY: rrb.Visualizer(SPLATS_VISUALIZER)},
            background=rrb.Background(color=(0, 0, 0), kind=rrb.BackgroundKind.SolidColor),
            line_grid=False,
            eye_controls=rrb.EyeControls3D(
                position=(0.0, 0.0, 5.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0)
            ),
        )
        rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
        m1 = screenshot(viewer, rec, out / "s1_red.png")
        print(f"S1 (red, 64 splats):   mean RGB {np.round(m1, 3)} -> {dominant(m1)}")
        if dominant(m1) != "r" or m1[0] < 0.05:
            failures.append(f"S1 should render the initial RED cloud, got mean {m1}")

        # (a) Re-log: SAME count, different color (green) and shifted positions.
        rr.log(SPLATS_ENTITY, make_cloud(8, (0.05, 1.0, 0.05), 1.2), static=True)
        m2 = screenshot(viewer, rec, out / "s2_green.png")
        print(f"S2 (re-log same count): mean RGB {np.round(m2, 3)} -> {dominant(m2)}")
        if dominant(m2) != "g":
            failures.append(
                f"S2 should be GREEN after re-logging same-count cloud, got mean {m2} (stale render)"
            )

        # (b) Re-log: DIFFERENT count (4x splats), blue, wider spread.
        rr.log(SPLATS_ENTITY, make_cloud(16, (0.05, 0.05, 1.0), 1.8), static=True)
        m3 = screenshot(viewer, rec, out / "s3_blue.png")
        print(f"S3 (re-log 4x count):   mean RGB {np.round(m3, 3)} -> {dominant(m3)}")
        if dominant(m3) != "b":
            failures.append(
                f"S3 should be BLUE after re-logging 4x-count cloud, got mean {m3} (stale render)"
            )
    finally:
        viewer.close()

    if failures:
        print("RELOG CHECK: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("RELOG CHECK: PASS")
    return 0
