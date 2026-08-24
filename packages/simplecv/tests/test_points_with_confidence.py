"""Behavior tests for the confidence-carrying point archetypes."""

from __future__ import annotations

from typing import Any

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence


def _batches_by_component(archetype: Any) -> dict[str, Any]:
    return {batch.component_descriptor().component: batch for batch in archetype.as_component_batches()}


def _unpack_rgba(packed: int) -> tuple[int, int, int]:
    return (packed >> 24) & 0xFF, (packed >> 16) & 0xFF, (packed >> 8) & 0xFF


def test_points2d_with_confidence_derives_gradient_colors_when_none() -> None:
    positions: Float32[ndarray, "3 2"] = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    confidences: Float32[ndarray, "3"] = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)

    archetype: Points2DWithConfidence = Points2DWithConfidence(positions=positions, confidences=confidences)

    batches: dict[str, Any] = _batches_by_component(archetype)
    packed_colors: list[int] = batches["Points2D:colors"].as_arrow_array().to_pylist()
    assert _unpack_rgba(packed_colors[0]) == (255, 0, 0), "confidence 0.0 must derive red"
    assert _unpack_rgba(packed_colors[1]) == (255, 255, 0), "confidence 0.5 must derive yellow"
    assert _unpack_rgba(packed_colors[2]) == (0, 255, 0), "confidence 1.0 must derive green"


def test_points2d_columns_derive_no_default_colors() -> None:
    positions: Float32[ndarray, "2 2"] = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    confidences: Float32[ndarray, "2"] = np.asarray([0.0, 1.0], dtype=np.float32)

    columns: Any = Points2DWithConfidence.columns(positions=positions, confidences=confidences)

    components: set[str] = {column.component_descriptor().component for column in columns}
    assert "Points2D:positions" in components
    assert "simplecv.KeypointConfidence2D:confidences" in components
    # The columnar path must NOT invent colors: exoego ingest relies on
    # annotation-context class colors winning when none are passed.
    assert "Points2D:colors" not in components


def test_points3d_with_confidence_derives_gradient_colors_when_none() -> None:
    positions: Float32[ndarray, "2 3"] = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    confidences: Float32[ndarray, "2"] = np.asarray([0.0, 1.0], dtype=np.float32)

    archetype: Any = Points3DWithConfidence(positions=positions, confidences=confidences)

    batches: dict[str, Any] = _batches_by_component(archetype)
    packed_colors: list[int] = batches["Points3D:colors"].as_arrow_array().to_pylist()
    assert _unpack_rgba(packed_colors[0]) == (255, 0, 0), "confidence 0.0 must derive red"
    assert _unpack_rgba(packed_colors[1]) == (0, 255, 0), "confidence 1.0 must derive green"


def test_points2d_with_confidence_keeps_explicit_colors() -> None:
    positions: Float32[ndarray, "2 2"] = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    confidences: Float32[ndarray, "2"] = np.asarray([0.0, 1.0], dtype=np.float32)
    explicit: np.ndarray = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    archetype: Points2DWithConfidence = Points2DWithConfidence(positions=positions, confidences=confidences, colors=explicit)

    batches: dict[str, Any] = _batches_by_component(archetype)
    packed_colors: list[int] = batches["Points2D:colors"].as_arrow_array().to_pylist()
    assert _unpack_rgba(packed_colors[0]) == (1, 2, 3)
    assert _unpack_rgba(packed_colors[1]) == (4, 5, 6)
