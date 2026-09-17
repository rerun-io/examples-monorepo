"""Blueprint layout, depth encoding, and summary formatting of the full-segment demo."""

import numpy as np
import pytest
from jaxtyping import Float32, UInt16
from numpy import ndarray

from zipdepth.apis.eval_catalog import MetricCatalogDepthMetrics

pytest.importorskip("arkitscenes_download", reason="ARKitScenes catalog dependencies live in the zipdepth catalog lane")

from zipdepth.apis.infer_segment_rerun import (  # noqa: E402
    METRICS_ROOT,
    ULTRAWIDE_ROOT,
    WIDE_ROOT,
    CameraSummary,
    demo_blueprint,
    depth_mm,
    summary_table,
)


def view_origins(node: object) -> list[str]:
    """Collect every view origin below one blueprint container."""
    origin: object = getattr(node, "origin", None)
    if origin is not None:
        return [str(origin)]
    return [found for child in getattr(node, "contents", []) or [] for found in view_origins(child)]


def test_demo_blueprint_holds_both_camera_rows_and_the_metric_series() -> None:
    """Give every logged image its own view, keep the panels open, and plot the scalars."""
    blueprint = demo_blueprint()

    origins: list[str] = view_origins(blueprint.root_container)

    assert origins == [
        f"{ULTRAWIDE_ROOT}/rgb",
        f"{ULTRAWIDE_ROOT}/prompt_footprint",
        f"{ULTRAWIDE_ROOT}/depth_pred",
        f"{ULTRAWIDE_ROOT}/depth_target",
        f"{ULTRAWIDE_ROOT}/abs_error",
        f"{WIDE_ROOT}/rgb",
        f"{WIDE_ROOT}/depth_pred",
        f"{WIDE_ROOT}/depth_target",
        f"{WIDE_ROOT}/abs_error",
        METRICS_ROOT,
    ]
    assert blueprint.collapse_panels is False


def test_depth_millimetres_quantize_and_clamp_the_metric_range() -> None:
    """Convert metres to the layers' uint16 millimetre encoding without overflow."""
    depth_m_hw: Float32[ndarray, "1 3"] = np.array([[0.0, 1.234, 100.0]], dtype=np.float32)

    quantized_hw: UInt16[ndarray, "1 3"] = depth_mm(depth_m_hw)

    assert quantized_hw.dtype == np.uint16
    assert quantized_hw.tolist() == [[0, 1234, 65535]]


def test_summary_table_reports_every_camera_and_region() -> None:
    """Print one row per camera and region, with the ultrawide footprint split."""
    metrics: MetricCatalogDepthMetrics = MetricCatalogDepthMetrics(abs_rel=0.125, delta1=0.875, mae=0.25)
    summaries: list[CameraSummary] = [
        CameraSummary(camera="wide", frame_count=3, regions={"whole": metrics}),
        CameraSummary(camera="ultrawide", frame_count=2, regions={"whole": metrics, "outside": metrics}),
    ]

    lines: list[str] = summary_table(summaries).splitlines()

    assert len(lines) == 2 + 3
    assert lines[2].split() == ["wide", "whole", "3", "0.1250", "0.8750", "0.2500"]
    assert lines[4].split() == ["ultrawide", "outside", "2", "0.1250", "0.8750", "0.2500"]
