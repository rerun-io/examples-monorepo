"""Golden contract: the shared benchmark gates unclipped RGB, alpha, and white RGB."""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from serde import serde
from serde.json import from_json


@serde
@dataclass(frozen=True, slots=True)
class Metrics:
    """Typed projection of the Rust-owned metric schema."""

    psnr: float
    """Mean PSNR in dB over the scored RGB images."""


@serde
@dataclass(frozen=True, slots=True)
class Parity:
    """The three independent channel means required by the parity gate."""

    mean: Metrics
    """RGB rendered over black."""
    mean_alpha_psnr: float
    """Mean alpha-channel PSNR in dB."""
    mean_white_psnr: float
    """Mean RGB PSNR over white in dB."""


@pytest.mark.golden
@pytest.mark.parametrize("case", [
    "lego", "lego-mip", "lego-indirect", "cactus", "cactus-mip-floor",
    "garden-pinhole", "garden-kb4", "garden-rt8", "garden-thin-prism",
])
def test_core_matches_brush_float_channels(case: str, tmp_path: Path) -> None:
    """Use gsplat-bench for every render, camera conversion, and metric calculation."""
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = Path(os.environ.get("GSPLAT_MODERN_DATA", str(Path.home() / "gsplat-modern-data")))
    scene: str = case.split("-", 1)[0]
    ply: Path = data / "cactus/cactus.ply" if scene == "cactus" else (
        data / "nerfbaselines-pretrained" / ("blender" if scene == "lego" else "mipnerf360")
        / scene / "checkpoint/point_cloud/iteration_30000/point_cloud.ply"
    )
    cameras: Path = data / "nerf-synthetic/lego/transforms_test.json"
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
    binary: Path = root / "target/release/gsplat-bench"
    assert binary.is_file(), "Build the release benchmark with the tests-golden Pixi task"
    output: Path = tmp_path / f"{case}.json"
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [str(binary), "parity", "--impl", "ours", "--oracle", "brush", "--ply", str(ply), *arguments, "--out", str(output)],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report: Parity = from_json(Parity, output.read_text())
    assert min(report.mean.psnr, report.mean_alpha_psnr, report.mean_white_psnr) >= 40.0
