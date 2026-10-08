"""Run Rust GPU contracts through the package integration lane."""
import os
import re
import subprocess
from pathlib import Path

import pytest

from gsplat_rust_renderer.nerfbaselines import scene_data_dir, scene_ply_path


def run_rust_contract(package: str, suite: str, assets: tuple[str, ...]) -> None:
    """Missing external assets are pytest skips; Rust failures remain failures."""
    root: Path = Path(__file__).resolve().parents[1]
    environment: dict[str, str] = dict(os.environ)
    defaults: dict[str, Path] = {
        "GSPLAT_LEGO": Path(environment.get("GSPLAT_NERF_INIT_ROOT", str(root / "data/nerf-synthetic-init"))) / "lego",
        "GSPLAT_TEST_PLY": scene_ply_path("lego"),
        "GSPLAT_TEST_CAMERAS": scene_data_dir("lego") / "transforms_test.json",
        "GSPLAT_TEST_GT": scene_data_dir("lego") / "test/r_0.png",
        "GSPLAT_TEST_COLMAP": root / "data/mipnerf360/garden/sparse/0",
    }
    for key in assets:
        path: Path = Path(environment.get(key, str(defaults[key])))
        if not path.exists():
            pytest.skip(f"Required Rust integration asset {key} missing: {path}")
        if key == "GSPLAT_LEGO" and not (path / "points3d.ply").is_file():
            pytest.skip(f"Required initialized Lego cloud missing: {path / 'points3d.ply'}")
        environment[key] = str(path)
    selection: list[str] = ["--lib"] if suite == "lib" else ["--test", suite]
    if suite.startswith("bin:"):
        selection = ["--bin", suite.removeprefix("bin:")]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["cargo", "test", "--locked", "--package", package, *selection, "--", "--ignored", "--nocapture", "--test-threads=1"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    print(result.stdout, result.stderr, sep="\n")
    if result.returncode == 0 and "SKIP:" in result.stderr:
        pytest.skip(result.stderr.split("SKIP:", 1)[1].splitlines()[0].strip())
    assert result.returncode == 0, f"Rust integration contract failed: {package}/{suite}"
    assert re.search(r"test result: ok\. [1-9][0-9]* passed", result.stdout), f"Rust test selection was empty: {package}/{suite}"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("package", "suite", "assets"),
    [
        ("gsplat-cli", "evaluation", ("GSPLAT_TEST_PLY", "GSPLAT_TEST_CAMERAS", "GSPLAT_TEST_GT")),
        ("gsplat-cli", "renderers", ("GSPLAT_TEST_PLY", "GSPLAT_TEST_CAMERAS", "GSPLAT_TEST_COLMAP")),
        ("gsplat-cli", "lib", ()),
        ("gsplat-core", "lib", ()),
        ("gsplat-core", "render", ()),
        ("gsplat-core", "views", ()),
        ("gsplat-core", "indirect", ()),
    ],
)
def test_rust_gpu_contract(package: str, suite: str, assets: tuple[str, ...]) -> None:
    """Every ignored test in a selected target runs, including newly added tests."""
    run_rust_contract(package, suite, assets)
