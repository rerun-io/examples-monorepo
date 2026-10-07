"""Public viewer and standalone pixel contracts; each capture has a process timeout."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from gsplat_rust_renderer.apis.calibration_scene import CheckConfig, GenerateConfig, check, generate

ROOT: Path = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def binaries() -> Path:
    environment: dict[str, str] = dict(os.environ)
    environment.update(CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0", CARGO_INCREMENTAL="0")
    subprocess.run(
        ["cargo", "build", "--locked", "--all-features", "-p", "gsplat-viewer", "-p", "gsplat-cli"],
        cwd=ROOT, env=environment, check=True, timeout=1200,
    )
    return ROOT / "target/debug"


@pytest.mark.integration
def test_native_lego_and_same_entity_relog(binaries: Path, tmp_path: Path) -> None:
    """Native Lego is visible and 64 red -> 64 green -> 256 blue replaces the upload."""
    run_capture(binaries, tmp_path, "relog")


@pytest.mark.integration
def test_two_views_share_upload_and_override_mode(binaries: Path, tmp_path: Path) -> None:
    """One data generation serves two views with different blueprint render modes."""
    run_capture(binaries, tmp_path, "views")


@pytest.mark.integration
def test_recording_without_override_in_compute_and_stock_viewers(binaries: Path, tmp_path: Path) -> None:
    """The same saved native recording works in both viewers without an override."""
    run_capture(binaries, tmp_path, "portable")


@pytest.mark.golden
def test_viewer_matches_standalone_at_fixed_eye(binaries: Path, tmp_path: Path) -> None:
    """Compare the same camera and crop, with the specified 39 dB compositing allowance."""
    run_capture(binaries, tmp_path, "pair")


@pytest.mark.golden
def test_viewer_white_background_color(binaries: Path, tmp_path: Path) -> None:
    """White catches linearization of premultiplied display color at soft edges."""
    run_capture(binaries, tmp_path, "pair-white")


def run_capture(binaries: Path, out: Path, mode: str) -> None:
    ply: Path = Path(os.environ.get("GSPLAT_TEST_PLY", str(ROOT / "data/nerfbaselines/pretrained/lego/checkpoint/point_cloud/iteration_30000/point_cloud.ply")))
    if not ply.exists():
        pytest.skip(f"Required Lego native viewer asset missing: {ply}")
    environment: dict[str, str] = dict(os.environ)
    environment["RUST_LOG"] = "info,gsplat_viewer::cache=debug"
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-m", "gsplat_rust_renderer.apis.viewer_check", "--binaries", str(binaries), "--ply", str(ply), "--out", str(out), "--mode", mode],
        check=False, timeout=180, capture_output=True, text=True, env=environment,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    names: list[str] = ["lego.png"]
    if mode == "depth":
        names = ["framing.png", "depth.png", "bad-entity.png"]
    if mode == "portable":
        names = [f"{choice}-{viewer}.png" for choice in ("portable", "explicit-native", "explicit-compute") for viewer in ("compute", "stock")]
        assert result.stderr.count("Uploaded ") == 2, "Only automatic and explicit compute choices may upload"
        assert (out / "portable.rrd").stat().st_size > 0
    if mode == "relog":
        names += ["relog-r-64.png", "relog-g-64.png", "relog-b-256.png"]
    elif mode == "views":
        names += ["mode-0.png", "mode-1.png"]
    for name in names:
        with Image.open(out / name) as screenshot:
            screenshot.verify()
    if mode == "views":
        assert result.stderr.count("Uploaded 1 Gaussian splats") == 1, "Two views must share the one-splat upload"


@pytest.mark.golden
def test_standalone_calibration_geometry_and_occlusion(binaries: Path, tmp_path: Path) -> None:
    """Analytic marker positions check handedness, image Y, focal length, and occlusion."""
    generate(GenerateConfig(out_dir=tmp_path))
    image: Path = tmp_path / "calibration.png"
    # This background lies between the existing dark/light gray marker bands.
    subprocess.run(
        [str(binaries / "gsplat"), "render", "--ply", str(tmp_path / "calibration.ply"), "--camera", str(tmp_path / "transforms_test.json"), "--output", str(image), "--background", "0.575,0.575,0.575"],
        check=True, timeout=120,
    )
    with pytest.raises(SystemExit) as result:
        check(CheckConfig(image=image, scene_dir=tmp_path, renderer="gsplat-core", report_json=tmp_path / "calibration-report.json"))
    assert result.value.code == 0

@pytest.mark.integration
def test_opaque_depth_and_automatic_framing(binaries: Path, tmp_path: Path) -> None:
    """Front opaque content occludes splats; rear content cannot punch through them."""
    run_capture(binaries, tmp_path, "depth")


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
    view = rrb.Spatial3DView(
        origin="/", contents=["world/splats"], line_grid=False,
        overrides={"world/splats": rrb.Visualizer("ComputeGaussianSplats3D")},
        background=rrb.Background(color=(0, 0, 0), kind="SolidColor"),
        eye_controls=rrb.EyeControls3D(position=(0.0, 0.0, 5.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0), spin_speed=1.2566370614359172 if moving else 0.0) if not automatic else None,
    )
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
        [str(binaries / "gsplat-viewer-probe"), str(recording_path), "--out", str(report), "--window-size", "800x600", "--storage-binding-limit", str(limit), *(["--require-motion"] if moving else [])],
        text=True, capture_output=True, check=False, timeout=180,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0
    with Image.open(report.with_suffix(".png")) as screenshot:
        pixels = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
    red = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 60) & (pixels[:, :, 2] < 60)
    assert float(red.mean()) > (0.005 if automatic else 0.05), "Capacity failure left the view blank instead of drawing native quads"
    (tmp_path / "probe.log").write_text(result.stdout + result.stderr)
    assert "Gaussian compute fallback" in result.stderr
