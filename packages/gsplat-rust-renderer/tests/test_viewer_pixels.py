"""Public viewer and standalone pixel contracts; each capture has a process timeout."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from gsplat_rust_renderer.nerfbaselines import scene_ply_path

ROOT: Path = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def binaries() -> Path:
    subprocess.run(
        ["cargo", "build", "--locked", "-p", "gsplat-viewer", "-p", "gsplat-cli", "--features", "gsplat-cli/probe"],
        cwd=ROOT, env=os.environ, check=True, timeout=1200,
    )
    return ROOT / "target/debug"


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["relog", "views", "portable", "depth"])
def test_viewer_pixels(binaries: Path, tmp_path: Path, mode: str) -> None:
    """Check upload replacement, per-view modes, stock compatibility and opaque depth."""
    run_capture(binaries, tmp_path, mode)


@pytest.mark.golden
@pytest.mark.parametrize("mode", ["pair", "pair-white"])
def test_viewer_pair_pixels(binaries: Path, tmp_path: Path, mode: str) -> None:
    """Compare black and white composition with the standalone fixed-eye render."""
    run_capture(binaries, tmp_path, mode)


def run_capture(binaries: Path, out: Path, mode: str) -> None:
    ply: Path = Path(os.environ.get("GSPLAT_TEST_PLY", str(scene_ply_path("lego"))))
    if not ply.exists():
        pytest.skip(f"Required Lego native viewer asset missing: {ply}")
    environment: dict[str, str] = {**os.environ, "RUST_LOG": "info,gsplat_viewer::cache=debug"}
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("viewer_check.py")), "--binaries", str(binaries), "--ply", str(ply), "--out", str(out), "--mode", mode],
        check=False, timeout=180, capture_output=True, text=True, env=environment,
    )
    print(result.stdout, result.stderr, sep="\n")
    assert result.returncode == 0
    names: list[str] = {
        "depth": ["framing.png", "depth.png", "bad-entity.png"],
        "portable": [f"{choice}-{viewer}.png" for choice in ("portable", "explicit-native", "explicit-compute") for viewer in ("compute", "stock")],
        "relog": ["lego.png", "relog-r-64.png", "relog-g-64.png", "relog-b-256.png"],
        "views": ["lego.png", "mode-0.png", "mode-1.png"],
    }.get(mode, ["lego.png"])
    if mode == "portable":
        assert result.stderr.count("Uploaded ") == 2, "Only automatic and explicit compute choices may upload"
        assert (out / "portable.rrd").stat().st_size > 0
    for name in names:
        with Image.open(out / name) as screenshot:
            screenshot.verify()
    if mode == "views":
        assert result.stderr.count("Uploaded 1 Gaussian splats") == 1, "Two views must share the one-splat upload"


@pytest.mark.integration
@pytest.mark.parametrize(("limit", "scale", "moving", "automatic"), [(1024, 0.18, False, False), (65536, 10.0, True, False), (1024, 0.18, False, True)])
def test_storage_capacity_uses_visible_native_fallback(binaries: Path, tmp_path: Path, limit: int, scale: float, moving: bool, automatic: bool) -> None:
    """Upload and intersection capacity failures retain visible native splats, including while moving."""
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    recording_path: Path = tmp_path / "fallback.rrd"
    rr.init("capacity-fallback", strict=True)
    rr.save(recording_path)
    view = rrb.Spatial3DView(origin="/", contents=["world/splats"], line_grid=False,
        overrides={"world/splats": rrb.Visualizer("ComputeGaussianSplats3D")},
        background=rrb.Background(color=(0, 0, 0), kind="SolidColor"),
        eye_controls=rrb.EyeControls3D(position=(0.0, 0.0, 5.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0), spin_speed=1.2566370614359172 if moving else 0.0) if not automatic else None)
    rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    axis = np.linspace(-1.2, 1.2, 8, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    centers = np.stack([gx.ravel(), gy.ravel(), np.zeros(64, dtype=np.float32)], axis=1)
    if automatic:
        centers += np.array([100.0, 200.0, 300.0], dtype=np.float32)
    rr.log("world/splats", rr.GaussianSplats3D(centers=centers, scales=[scale] * 3, colors=[255, 0, 0, 255], spherical_harmonics_degree=0), static=True)
    recording = rr.get_global_data_recording()
    assert recording is not None
    recording.flush(timeout_sec=30.0)
    rr.disconnect()
    report: Path = tmp_path / "fallback.json"
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [str(binaries / "gsplat"), "probe", str(recording_path), "--out", str(report), "--window-size", "800x600", "--storage-binding-limit", str(limit), *(["--require-motion"] if moving else [])],
        text=True, capture_output=True, check=False, timeout=180,
    )
    print(result.stdout, result.stderr, sep="\n")
    assert result.returncode == 0
    with Image.open(report.with_suffix(".png")) as screenshot:
        pixels = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
    red = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 60) & (pixels[:, :, 2] < 60)
    assert float(red.mean()) > (0.005 if automatic else 0.05), "Capacity failure left the view blank instead of drawing native quads"
    assert "Gaussian compute fallback" in result.stderr
