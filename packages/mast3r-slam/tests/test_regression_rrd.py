"""Regression tests: validate baseline .rrd structure and content.

The baseline is generated once via:
    pixi run -e mast3r-slam --frozen _generate-test-baseline

These tests only read the pre-generated baseline .rrd — they do NOT
run inference. To compare two runs, generate a fresh RRD and diff
manually or via a separate pixi task.
"""

from pathlib import Path

import pytest
import rerun as rr


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


def test_edges_logged(baseline_rrd_path: Path) -> None:
    """Verify factor graph edges entity exists in the baseline."""
    paths: list[str] = _get_entity_paths(baseline_rrd_path)
    assert "/world/edges" in paths, "No /world/edges entity found"


def test_keyframe_count_reasonable(baseline_rrd_path: Path) -> None:
    """Verify the number of keyframes is in a reasonable range for 100 frames."""
    n_keyframes: int = _count_keyframes(baseline_rrd_path)
    # With subsample=5, 100 input frames -> ~20 actual frames -> 5-20 keyframes typical
    assert 3 <= n_keyframes <= 50, f"Unexpected keyframe count: {n_keyframes}"
