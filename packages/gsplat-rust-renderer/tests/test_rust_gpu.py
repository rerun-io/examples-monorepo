"""Run Rust GPU contracts through the package integration lane."""
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    ("suite", "name", "assets"),
    [
        ("evaluation", "brush_quantization_premultiplication_and_identity", ()),
        ("evaluation", "evaluator_lpips_matches_reference", ()),
        ("evaluation", "float_parity_preserves_highlights_and_detects_alpha", ()),
        ("evaluation", "float_ssim_agrees_with_brush_on_byte_exact_inputs", ()),
        ("evaluation", "lego_float_evaluation_matches_brush_eval_stats", ("GSPLAT_TEST_PLY", "GSPLAT_TEST_CAMERAS", "GSPLAT_TEST_GT")),
        ("renderers", "all_renderers_nonblack_and_brush_identity", ("GSPLAT_TEST_PLY", "GSPLAT_TEST_CAMERAS")),
        ("renderers", "independent_brush_renders_have_exact_identity_on_one_splat", ()),
        ("renderers", "old_core_renders_at_4k", ()),
        ("renderers", "garden_colmap_projects_observed_points", ("GSPLAT_TEST_COLMAP",)),
    ],
)
def test_rust_gpu_contract(suite: str, name: str, assets: tuple[str, ...]) -> None:
    """Missing external assets are pytest skips; Rust failures remain failures."""
    root: Path = Path(__file__).resolve().parents[1]
    environment: dict[str, str] = dict(os.environ)
    defaults: dict[str, Path] = {
        "GSPLAT_TEST_PLY": root / "data/nerfbaselines/pretrained/lego/checkpoint/point_cloud/iteration_30000/point_cloud.ply",
        "GSPLAT_TEST_CAMERAS": root / "data/nerfbaselines/data/lego/transforms_test.json",
        "GSPLAT_TEST_GT": root / "data/nerfbaselines/data/lego/test/r_0.png",
        "GSPLAT_TEST_COLMAP": root / "data/mipnerf360/garden/sparse/0",
    }
    for key in assets:
        path: Path = Path(environment.get(key, str(defaults[key])))
        if not path.exists():
            pytest.skip(f"Required Rust integration asset {key} missing: {path}")
        environment[key] = str(path)
    environment["CARGO_PROFILE_DEV_DEBUG"] = "0"
    environment["CARGO_PROFILE_TEST_DEBUG"] = "0"
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["cargo", "test", "--locked", "--workspace", "--test", suite, name, "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Rust integration contract failed: {suite}::{name}"
    assert "1 passed" in result.stdout, f"Rust test selection was empty: {suite}::{name}"
