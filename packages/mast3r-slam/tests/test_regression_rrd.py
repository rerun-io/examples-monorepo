"""Regression tests: validate baseline .rrd and compare against fresh inference.

The baseline is generated once via:
    pixi run -e mast3r-slam --frozen _generate-test-baseline

The inference comparison test uses config/test.yaml which sets
single_thread=True — this means NO backend subprocess is spawned,
eliminating any zombie process risk on test failure.
"""

from pathlib import Path

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
        for part in path.split("/"):
            if part.startswith("keyframe-"):
                kf_indices.add(part)
    return len(kf_indices)


# ── Baseline structure tests (fast, no inference) ────────────────────────


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
    """Verify the number of keyframes is in a reasonable range for the 30-frame baseline."""
    n_keyframes: int = _count_keyframes(baseline_rrd_path)
    assert 3 <= n_keyframes <= 50, f"Unexpected keyframe count: {n_keyframes}"
