"""The cached bench fails a run unless every dataset passes all nine accuracy bands."""

from dataclasses import replace

import pytest

pytest.importorskip("egohand_inference", reason="the cached bench needs the posekit-hands Pixi environment")

from posekit.apis.hands_cached_bench import Result, failed_bands


def result(dataset: str) -> Result:
    return Result(
        model="default",
        gpu="test",
        dataset=dataset,
        frames=10,
        hands_per_frame=1.0,
        gt_hands=10,
        matched_hands=9,
        recall=0.9,
        mean_iou=0.8,
        side_accuracy=1.0,
        epe_2d_median_px=10.0,
        epe_2d_p95_px=30.0,
        absolute_3d_median_mm=20.0,
        absolute_3d_p95_mm=50.0,
        relative_3d_median_mm=15.0,
        relative_3d_p95_mm=40.0,
        mean_ms=None,
        median_ms=None,
        p95_ms=None,
    )


def test_bands_pass_within_five_percent() -> None:
    baseline: list[Result] = [result("a"), result("b")]
    candidate: list[Result] = [replace(result("a"), recall=0.9 * 0.96, epe_2d_p95_px=30.0 * 1.04), result("b")]
    assert failed_bands(baseline, candidate) == []


def test_bands_name_each_failure_and_missing_dataset() -> None:
    baseline: list[Result] = [result("a"), result("b")]
    candidate: list[Result] = [replace(result("a"), mean_iou=0.7, absolute_3d_median_mm=None)]
    assert failed_bands(baseline, candidate) == ["a: mean_iou", "a: absolute_3d_median_mm", "b: missing"]
