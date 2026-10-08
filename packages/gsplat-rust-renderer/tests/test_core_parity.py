"""Golden contract: the shared benchmark gates unclipped RGB, alpha, and white RGB."""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from serde import serde
from serde.json import from_json

from gsplat_rust_renderer.evaluation import GSPLAT_BIN
from gsplat_rust_renderer.nerfbaselines import scene_data_dir, scene_ply_path


@serde
@dataclass(frozen=True, slots=True)
class Metrics:
    """Typed projection of the Rust-owned metric schema."""

    psnr: float


@serde
@dataclass(frozen=True, slots=True)
class Parity:
    """The three independent channel means required by the parity gate."""

    mean: Metrics
    mean_alpha_psnr: float
    mean_white_psnr: float


@pytest.mark.golden
@pytest.mark.parametrize("case", [
    "lego", "lego-mip", "lego-indirect", "cactus", "cactus-mip-floor",
    "garden-pinhole", "garden-kb4", "garden-rt8", "garden-thin-prism",
])
def test_core_matches_brush_float_channels(case: str, tmp_path: Path) -> None:
    """Use gsplat parity for every render, camera conversion, and metric calculation."""
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = Path(os.environ.get("GSPLAT_MODERN_DATA", str(Path.home() / "gsplat-modern-data")))
    scene: str = case.split("-", 1)[0]
    ply: Path = scene_ply_path("lego") if scene == "lego" else (data / "cactus/cactus.ply" if scene == "cactus" else
        data / "nerfbaselines-pretrained/mipnerf360/garden/checkpoint/point_cloud/iteration_30000/point_cloud.ply")
    cameras: Path = scene_data_dir("lego") / "transforms_test.json"
    arguments: list[str] = ["--path", str(cameras), "--res", "native"]
    assets: list[Path] = [ply]
    if scene == "lego":
        assets.append(cameras)
    elif scene == "cactus":
        arguments = ["--path", "orbit:60", "--res", "1280x720"]
    else:
        cameras = root / "tests/fixtures/core-parity" / f"{case}.json"
        assets.append(cameras)
        arguments = ["--path", str(cameras), "--res", "native"]
    for asset in assets:
        if not asset.exists():
            pytest.skip(f"Required core parity asset missing: {asset}")
    if "mip" in case:
        arguments.extend(["--render-mode", "mip"])
    if case == "cactus-mip-floor":
        arguments.extend(["--min-scale", "0.002", "--splat-scale", "1.3"])
    if case == "lego-indirect":
        arguments.extend(["--initial-capacity", "1"])
    assert GSPLAT_BIN.is_file(), "Build gsplat with tests-golden, or set GSPLAT_BIN to a built benchmark"
    output: Path = tmp_path / f"{case}.json"
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [str(GSPLAT_BIN), "parity", "--impl", "ours", "--oracle", "brush", "--ply", str(ply), *arguments, "--out", str(output)],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report: Parity = from_json(Parity, output.read_text())
    assert min(report.mean.psnr, report.mean_alpha_psnr, report.mean_white_psnr) >= 40.0
