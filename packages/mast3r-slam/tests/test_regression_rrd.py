"""Regression tests: compare current inference output against baseline .rrd.

The baseline is generated once via:
    pixi run -e mast3r-slam --frozen _generate-test-baseline

These tests run inference on the same 100 frames with the deterministic
config (single_thread=True) and compare key metrics against the baseline.
"""

from pathlib import Path

import pytest
import rerun as rr

from conftest import BASELINE_RRD, PACKAGE_DIR


def _get_entity_paths(rrd_path: Path) -> list[str]:
    """Load an .rrd and return all entity paths."""
    with rr.server.Server(datasets={"ds": [str(rrd_path.resolve())]}) as server:
        client = server.client()
        ds = client.get_dataset("ds")
        schema = ds.schema()
        return schema.entity_paths()


def _count_keyframes(rrd_path: Path) -> int:
    """Count unique keyframe indices in the .rrd."""
    entity_paths: list[str] = _get_entity_paths(rrd_path)
    kf_indices: set[str] = set()
    for path in entity_paths:
        parts: list[str] = path.split("/")
        for part in parts:
            if part.startswith("keyframe-"):
                kf_indices.add(part)
    return len(kf_indices)


@pytest.fixture(scope="module")
def current_rrd_path(tmp_path_factory) -> Path:
    """Run inference and produce a fresh .rrd for comparison."""
    import os

    os.chdir(PACKAGE_DIR)

    from mast3r_slam.api.inference import InferenceConfig, mast3r_slam_inference
    from simplecv.rerun_log_utils import RerunTyroConfig

    output_rrd: Path = tmp_path_factory.mktemp("rrd") / "current_100frames.rrd"
    inf_config: InferenceConfig = InferenceConfig(
        rr_config=RerunTyroConfig(save=output_rrd),
        dataset="data/normal-apt-tour.MOV",
        config="config/test.yaml",
        img_size=224,
        max_frames=100,
        no_viz=True,
    )
    mast3r_slam_inference(inf_config)
    return output_rrd


def test_baseline_rrd_exists(baseline_rrd_path: Path) -> None:
    """Verify baseline .rrd file exists and is non-empty."""
    assert baseline_rrd_path.exists()
    assert baseline_rrd_path.stat().st_size > 0


def test_rrd_has_keyframe_entities(baseline_rrd_path: Path) -> None:
    """Verify the baseline .rrd contains keyframe entities."""
    n_keyframes: int = _count_keyframes(baseline_rrd_path)
    assert n_keyframes > 0, "No keyframe entities found in baseline RRD"


def test_camera_path_logged(baseline_rrd_path: Path) -> None:
    """Verify camera path entity exists in the baseline."""
    paths: list[str] = _get_entity_paths(baseline_rrd_path)
    assert "/world/path" in paths, f"No /world/path entity found. Paths: {paths[:20]}"


def test_current_camera_logged(baseline_rrd_path: Path) -> None:
    """Verify current camera entity exists in the baseline."""
    paths: list[str] = _get_entity_paths(baseline_rrd_path)
    assert "/world/current_camera" in paths, "No /world/current_camera entity found"


def test_keyframe_count_regression(baseline_rrd_path: Path, current_rrd_path: Path) -> None:
    """Number of keyframes should match between baseline and current within tolerance."""
    baseline_kf: int = _count_keyframes(baseline_rrd_path)
    current_kf: int = _count_keyframes(current_rrd_path)

    # Allow 10% variance due to CUDA non-determinism
    tolerance: float = 0.1
    lower: int = int(baseline_kf * (1 - tolerance))
    upper: int = int(baseline_kf * (1 + tolerance))
    assert lower <= current_kf <= upper, f"Keyframe count {current_kf} outside [{lower}, {upper}] (baseline={baseline_kf})"
