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
        ["cargo", "build", "--locked", "-p", "gsplat-viewer", "-p", "gsplat-render"],
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
    environment["GSPLAT_UPLOAD_PROBE"] = "1"
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
        assert result.stderr.count("GSPLAT_UPLOAD count=") == 2, "Only automatic and explicit compute choices may upload"
        assert (out / "portable.rrd").stat().st_size > 0
    if mode == "relog":
        names += ["relog-r-64.png", "relog-g-64.png", "relog-b-256.png"]
    elif mode == "views":
        names += ["mode-0.png", "mode-1.png"]
    for name in names:
        with Image.open(out / name) as screenshot:
            screenshot.verify()
    if mode == "views":
        assert result.stderr.count("GSPLAT_UPLOAD count=1\n") == 1, "Two views must share the one-splat upload"


@pytest.mark.golden
def test_standalone_calibration_geometry_and_occlusion(binaries: Path, tmp_path: Path) -> None:
    """Analytic marker positions check handedness, image Y, focal length, and occlusion."""
    generate(GenerateConfig(out_dir=tmp_path))
    image: Path = tmp_path / "calibration.png"
    # This background lies between the existing dark/light gray marker bands.
    subprocess.run(
        [str(binaries / "gsplat-render"), "--ply", str(tmp_path / "calibration.ply"), "--camera", str(tmp_path / "transforms_test.json"), "--output", str(image), "--background", "0.575,0.575,0.575"],
        check=True, timeout=120,
    )
    with pytest.raises(SystemExit) as result:
        check(CheckConfig(image=image, scene_dir=tmp_path, renderer="gsplat-core", report_json=tmp_path / "calibration-report.json"))
    assert result.value.code == 0

@pytest.mark.integration
def test_opaque_depth_and_automatic_framing(binaries: Path, tmp_path: Path) -> None:
    """Front opaque content occludes splats; rear content cannot punch through them."""
    run_capture(binaries, tmp_path, "depth")
