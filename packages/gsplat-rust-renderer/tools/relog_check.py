"""Re-log regression check for the gsplat viewer (requires a GPU + built viewer binary).

Logs a splat cloud, screenshots it, then RE-LOGS the same entity twice —
(a) same splat count with different colors/positions, (b) different splat
count — and asserts the rendered output follows the new data each time.

Exercises the bug where the viewer's per-entity GPU caches (CloudSignature /
batch_cache attribute buffers) served stale data after a re-log.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from rerun.experimental import ViewerClient

from gsplat_rust_renderer.gaussians3d import Gaussians3D

BINARY: str = str(Path(__file__).resolve().parents[1] / "target" / "release" / "gsplat-rust-renderer")
"""The custom viewer binary built from this package (cargo build --release)."""
OUT: Path = Path("/tmp/relog-check")
ENTITY: str = "world/splats"


def make_cloud(n_side: int, rgb: tuple[float, float, float], spread: float) -> Gaussians3D:
    """A flat n_side x n_side grid of fat splats in the z=0 plane, one color."""
    xs = np.linspace(-spread, spread, n_side, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs)
    centers = np.stack([gx.ravel(), gy.ravel(), np.zeros(n_side * n_side, np.float32)], axis=1)
    n = centers.shape[0]
    quaternions = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
    scales = np.full((n, 3), 0.18, np.float32)
    opacities = np.full((n,), 0.95, np.float32)
    colors = np.tile(np.array(rgb, np.float32), (n, 1))
    return Gaussians3D(
        centers=centers, quaternions_xyzw=quaternions, scales=scales,
        opacities=opacities, colors_dc=colors,
    )


def screenshot(viewer: ViewerClient, rec, view_id: str, path: Path) -> np.ndarray:
    rec.flush(timeout_sec=15.0)
    time.sleep(2.5)  # ingest + render settle
    viewer.save_screenshot(str(path), view_id=view_id)
    deadline = time.time() + 10.0
    while time.time() < deadline and (not path.exists() or path.stat().st_size == 0):
        time.sleep(0.25)
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    crop = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return crop.reshape(-1, 3).mean(axis=0)


def dominant(mean_rgb: np.ndarray) -> str:
    return "rgb"[int(np.argmax(mean_rgb))]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    viewer = ViewerClient.spawn(
        headless=True, port=9931, executable_path=BINARY, hide_welcome_screen=True
    )
    failures: list[str] = []
    try:
        rr.init("relog_check")
        rr.connect_grpc(viewer.url)
        rec = rr.get_global_data_recording()
        assert rec is not None

        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(ENTITY, make_cloud(8, (1.0, 0.05, 0.05), 1.2), static=True)

        view = rrb.Spatial3DView(
            origin="/",
            name="relog",
            overrides={ENTITY: rrb.Visualizer("GaussianSplats3D")},
            background=rrb.Background(color=(0, 0, 0), kind=rrb.BackgroundKind.SolidColor),
            line_grid=False,
            eye_controls=rrb.EyeControls3D(
                position=(0.0, 0.0, 5.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0)
            ),
        )
        rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
        view_id = str(view.id)

        m1 = screenshot(viewer, rec, view_id, OUT / "s1_red.png")
        print(f"S1 (red, 64 splats):   mean RGB {np.round(m1, 3)} -> {dominant(m1)}")
        if dominant(m1) != "r" or m1[0] < 0.05:
            failures.append(f"S1 should render the initial RED cloud, got mean {m1}")

        # (a) Re-log: SAME count, different color (green) and shifted positions.
        rr.log(ENTITY, make_cloud(8, (0.05, 1.0, 0.05), 1.2), static=True)
        m2 = screenshot(viewer, rec, view_id, OUT / "s2_green.png")
        print(f"S2 (re-log same count): mean RGB {np.round(m2, 3)} -> {dominant(m2)}")
        if dominant(m2) != "g":
            failures.append(
                f"S2 should be GREEN after re-logging same-count cloud, got mean {m2} (stale render)"
            )

        # (b) Re-log: DIFFERENT count (4x splats), blue, wider spread.
        rr.log(ENTITY, make_cloud(16, (0.05, 0.05, 1.0), 1.8), static=True)
        m3 = screenshot(viewer, rec, view_id, OUT / "s3_blue.png")
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


if __name__ == "__main__":
    sys.exit(main())
