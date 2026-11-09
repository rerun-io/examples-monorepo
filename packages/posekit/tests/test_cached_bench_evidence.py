"""Saved benchmark evidence must contain the requested intermediates."""

import os
from pathlib import Path

import pytest
from rerun.chunk import RrdReader

pytestmark = pytest.mark.integration


def test_cached_benchmark_contains_full_evidence() -> None:
    location: str | None = os.environ.get("EGOHAND_BENCH_EVIDENCE")
    if location is None or not Path(location).is_file():
        pytest.skip("Set EGOHAND_BENCH_EVIDENCE to a completed cached benchmark RRD.")
    entities: set[str] = set(RrdReader(Path(location)).stream().collect().schema().entity_paths())
    for required in (
        "/camera/image",
        "/camera/image/detection",
        "/camera/mesh/left",
        "/crops/left",
        "/error/left/pixels",
        "/parameters/left/scales",
        "/timing/total",
    ):
        assert required in entities, f"Missing benchmark evidence: {required}"
