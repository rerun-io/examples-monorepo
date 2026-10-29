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


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["hexels", "hexels-tracked", "hexels-portable"])
def test_hexel_viewer_pixels(binaries: Path, tmp_path: Path, mode: str) -> None:
    """Synthetic V3 Hexels3D: compute pixels match gsplat render (explicit and tracked eye); stock Rerun draws the fallback."""
    result: subprocess.CompletedProcess[str] = run_check(binaries, tmp_path, mode, Path("/dev/null"))
    assert result.stderr.count("Uploaded 25 hexels") == 1, "One upload per viewer process; the Points3D fallback must not draw"
    assert "Hexel compute fallback" not in result.stderr
    names: list[str] = {"hexels": ["hexels.png"], "hexels-tracked": ["tracked-0.png", "tracked-1.png"],
                        "hexels-portable": ["hexels-portable-compute.png", "hexels-portable-stock.png"]}[mode]
    for name in names:
        with Image.open(tmp_path / name) as screenshot:
            screenshot.verify()


def run_check(binaries: Path, out: Path, mode: str, ply: Path) -> subprocess.CompletedProcess[str]:
    environment: dict[str, str] = {**os.environ, "RUST_LOG": "info,gsplat_viewer::cache=debug"}
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("viewer_check.py")), "--binaries", str(binaries), "--ply", str(ply), "--out", str(out), "--mode", mode],
        check=False, timeout=180, capture_output=True, text=True, env=environment,
    )
    print(result.stdout, result.stderr, sep="\n")
    assert result.returncode == 0
    return result


def run_capture(binaries: Path, out: Path, mode: str) -> None:
    ply: Path = Path(os.environ.get("GSPLAT_TEST_PLY", str(scene_ply_path("lego"))))
    if not ply.exists():
        pytest.skip(f"Required Lego native viewer asset missing: {ply}")
    result: subprocess.CompletedProcess[str] = run_check(binaries, out, mode, ply)
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


@pytest.mark.integration
def test_hexel_capacity_failure_draws_the_points_fallback(binaries: Path, tmp_path: Path) -> None:
    """25 V3 hexels need 6,500 storage bytes; a 1,024-byte device limit leaves the Points3D-style fallback visible."""
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb

    from gsplat_rust_renderer.hexels3d import HEXELS_ENTITY, Hexels3D, log_hexels

    axis: np.ndarray = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    red: np.ndarray = np.zeros((25, 9), dtype=np.float32)
    red[:, 0] = 1.0
    hexels: Hexels3D = Hexels3D(
        centers=np.stack([gx.ravel(), gy.ravel(), np.zeros(25, dtype=np.float32)], axis=1),
        quaternions_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (25, 1)),
        scales=np.full((25, 2), 0.22, dtype=np.float32),
        opacities=np.ones(25, dtype=np.float32),
        sharpness=np.full(25, 20.0, dtype=np.float32),
        color_field=red,
        sh_coefficients=np.zeros((25, 45), dtype=np.float32),
    )
    recording_path: Path = tmp_path / "hexel-fallback.rrd"
    rr.init("hexel-capacity-fallback", strict=True)
    rr.save(recording_path)
    view = rrb.Spatial3DView(origin="/", contents=[HEXELS_ENTITY], line_grid=False, background=rrb.Background(color=(0, 0, 0), kind="SolidColor"),
        eye_controls=rrb.EyeControls3D(position=(0.0, 0.0, 4.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0)))
    rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
    log_hexels(hexels)
    recording = rr.get_global_data_recording()
    assert recording is not None
    recording.flush(timeout_sec=30.0)
    rr.disconnect()
    report: Path = tmp_path / "fallback.json"
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [str(binaries / "gsplat"), "probe", str(recording_path), "--out", str(report), "--window-size", "800x600", "--storage-binding-limit", "1024"],
        text=True, capture_output=True, check=False, timeout=180,
    )
    print(result.stdout, result.stderr, sep="\n")
    assert result.returncode == 0
    with Image.open(report.with_suffix(".png")) as screenshot:
        pixels = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
    reds = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 60) & (pixels[:, :, 2] < 60)
    assert int(reds.sum()) > 25 * 4, "Capacity failure left the view blank instead of drawing the hexel centres"
    assert "Hexel compute fallback" in result.stderr
