"""Behavior tests for the shared keypoint logging helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray

from posekit.rerun_logging import log_person_points2d


def _batches_by_component(archetype: Any) -> dict[str, Any]:
    return {batch.component_descriptor().component: batch for batch in archetype.as_component_batches()}


def _unpack_rgba(packed: int) -> tuple[int, int, int]:
    return (packed >> 24) & 0xFF, (packed >> 16) & 0xFF, (packed >> 8) & 0xFF


def test_log_person_points2d_colors_by_confidence_and_keeps_raw_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, Any]] = []
    monkeypatch.setattr(rr, "log", lambda entity_path, archetype, **kwargs: logged.append((entity_path, archetype)))

    xy: Float32[ndarray, "3 2"] = np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
    confidence: Float32[ndarray, "3"] = np.asarray([0.0, 1.0, 0.1], dtype=np.float32)
    log_person_points2d(
        "image/person_0/keypoints",
        xy,
        confidence,
        0.3,
        keypoint_ids=np.arange(3, dtype=np.uint16),
        class_ids=0,
    )

    assert len(logged) == 1
    entity_path, archetype = logged[0]
    assert entity_path == "image/person_0/keypoints"
    batches: dict[str, Any] = _batches_by_component(archetype)

    positions: Float32[ndarray, "3 2"] = np.asarray(batches["Points2D:positions"].as_arrow_array().to_pylist(), dtype=np.float32)
    np.testing.assert_allclose(positions[1], xy[1])
    assert np.isnan(positions[0]).all(), "below-threshold positions must be masked"
    assert np.isnan(positions[2]).all(), "below-threshold positions must be masked"

    packed_colors: list[int] = batches["Points2D:colors"].as_arrow_array().to_pylist()
    assert _unpack_rgba(packed_colors[0]) == (255, 0, 0), "confidence 0.0 must render red"
    assert _unpack_rgba(packed_colors[1]) == (0, 255, 0), "confidence 1.0 must render green"

    raw_scores: Float32[ndarray, "3"] = np.asarray(
        batches["simplecv.KeypointConfidence2D:confidences"].as_arrow_array().to_pylist(), dtype=np.float32
    )
    np.testing.assert_allclose(raw_scores, confidence, err_msg="raw confidence values must ride along unmasked")
