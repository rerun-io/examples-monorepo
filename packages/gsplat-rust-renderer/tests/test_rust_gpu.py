"""Run Rust GPU contracts through the package integration lane."""
import os
import subprocess
from pathlib import Path

import pytest


def run_rust_contract(package: str, suite: str | None, name: str, assets: tuple[str, ...]) -> None:
    """Missing external assets are pytest skips; Rust failures remain failures."""
    root: Path = Path(__file__).resolve().parents[1]
    environment: dict[str, str] = dict(os.environ)
    defaults: dict[str, Path] = {
        "GSPLAT_LEGO": Path(environment.get("GSPLAT_NERF_INIT_ROOT", str(root / "data/nerf-synthetic-init"))) / "lego",
        "GSPLAT_TEST_PLY": root / "data/nerfbaselines/pretrained/lego/checkpoint/point_cloud/iteration_30000/point_cloud.ply",
        "GSPLAT_TEST_CAMERAS": root / "data/nerfbaselines/data/lego/transforms_test.json",
        "GSPLAT_TEST_GT": root / "data/nerfbaselines/data/lego/test/r_0.png",
        "GSPLAT_TEST_COLMAP": root / "data/mipnerf360/garden/sparse/0",
    }
    for key in assets:
        path: Path = Path(environment.get(key, str(defaults[key])))
        if not path.exists():
            pytest.skip(f"Required Rust integration asset {key} missing: {path}")
        if key == "GSPLAT_LEGO" and not (path / "points3d.ply").is_file():
            pytest.skip(f"Required initialized Lego cloud missing: {path / 'points3d.ply'}")
        environment[key] = str(path)
    environment["CARGO_PROFILE_DEV_DEBUG"] = "0"
    environment["CARGO_PROFILE_TEST_DEBUG"] = "0"
    environment["CARGO_INCREMENTAL"] = "0"
    selection: list[str] = ["--lib"] if suite is None else ["--test", suite]
    if suite == "bin:gsplat-bench":
        selection = ["--bin", "gsplat-bench"]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        ["cargo", "test", "--locked", "--package", package, *selection, name, "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"],
        cwd=root, env=environment, text=True, capture_output=True, check=False,
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode == 0 and "SKIP:" in result.stderr:
        pytest.skip(result.stderr.split("SKIP:", 1)[1].splitlines()[0].strip())
    assert result.returncode == 0, f"Rust integration contract failed: {suite}::{name}"
    assert "1 passed" in result.stdout, f"Rust test selection was empty: {suite}::{name}"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("package", "suite", "name", "assets"),
    [
        ("gsplat-train", "recording", "lego_training_recording_contains_snapshots_cameras_curves_and_eval_pairs", ("GSPLAT_LEGO",)),
        ("gsplat-eval", "evaluation", "brush_quantization_premultiplication_and_identity", ()),
        ("gsplat-eval", "evaluation", "evaluator_lpips_matches_reference", ()),
        ("gsplat-eval", "evaluation", "float_parity_preserves_highlights_and_detects_alpha", ()),
        ("gsplat-eval", "evaluation", "float_ssim_agrees_with_brush_on_byte_exact_inputs", ()),
        ("gsplat-eval", "evaluation", "lego_float_evaluation_matches_brush_eval_stats", ("GSPLAT_TEST_PLY", "GSPLAT_TEST_CAMERAS", "GSPLAT_TEST_GT")),
        ("gsplat-bench", "renderers", "all_renderers_nonblack_and_brush_identity", ("GSPLAT_TEST_PLY", "GSPLAT_TEST_CAMERAS")),
        ("gsplat-bench", "renderers", "independent_brush_renders_have_exact_identity_on_one_splat", ()),
        ("gsplat-bench", None, "renderers::coverage::sh_1_2_4_and_varying_scale_floors_match_brush", ()),
        ("gsplat-bench", None, "renderers::coverage::fisheye_keeps_visible_splats_behind_the_camera", ()),
        ("gsplat-bench", None, "renderers::coverage::eight_k_sorts_five_digits_and_dispatches_beyond_65535_tiles", ()),
        ("gsplat-bench", "renderers", "garden_colmap_projects_observed_points", ("GSPLAT_TEST_COLMAP",)),
        ("gsplat-core", None, "primitive_tests::gpu_counts_cross_recursive_boundaries_and_reuse_scratch", ()),
        ("gsplat-core", None, "primitive_tests::inclusive_scan_crosses_recursive_block_boundaries", ()),
        ("gsplat-core", None, "primitive_tests::radix_sort_is_stable_for_duplicates_and_partial_blocks", ()),
        ("gsplat-core", None, "primitive_tests::radix_sort_crosses_the_70m_reduced_histogram_boundary", ()),
        ("gsplat-core", "render", "instance_affine_preserves_projection_and_covariance", ()),
        ("gsplat-core", "render", "tiny_invertible_instances_are_valid", ()),
        ("gsplat-core", "render", "optional_depth_is_alpha_weighted_and_normal_color_is_preserved", ()) ,
        ("gsplat-core", "render", "centered_gaussian_has_analytic_color_alpha_and_background", ()),
        ("gsplat-core", "views", "one_scene_renders_two_views_in_one_submit", ()),
        ("gsplat-core", None, "view::tests::feedback_recovers_after_capacity_error_and_bounds_pending_frames", ()),
        ("gsplat-core", "indirect", "overflow_preserves_target_then_grows_and_rerenders_exactly", ()),
        ("gsplat-core", None, "gpu::tests::wrapped_intersection_count_cannot_enable_raster", ()),
    ],
)
def test_rust_gpu_contract(package: str, suite: str | None, name: str, assets: tuple[str, ...]) -> None:
    """Execute GPU behavior contracts through the integration task."""
    run_rust_contract(package, suite, name, assets)


@pytest.mark.golden
def test_training_conversion_matches_rerun_loader() -> None:
    """Compare every native component against the pretrained Lego PLY reference."""
    run_rust_contract("gsplat-train", "recording", "pretrained_lego_conversion_matches_rerun_loader", ("GSPLAT_TEST_PLY",))
