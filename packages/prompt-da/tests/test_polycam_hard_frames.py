"""CPU tests for Polycam hard-frame metrics."""

import pytest
import torch
from jaxtyping import Float32
from torch import Tensor

from rerun_prompt_da.apis.polycam_hard_frames import FrameMetrics, compute_frame_metrics


def test_compute_frame_metrics_splits_teacher_edges_from_flat_regions() -> None:
    """Report student and bilinear errors over one known edge partition."""
    teacher_hw: Float32[Tensor, "2 2"] = torch.tensor([[0.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    student_hw: Float32[Tensor, "2 2"] = torch.tensor([[1.0, 2.0], [3.0, 6.0]], dtype=torch.float32)
    baseline_hw: Float32[Tensor, "2 2"] = torch.tensor([[4.0, 3.0], [2.0, 3.0]], dtype=torch.float32)

    metrics: FrameMetrics = compute_frame_metrics(
        frame_index=7,
        teacher_hw=teacher_hw,
        student_hw=student_hw,
        baseline_hw=baseline_hw,
        edge_quantile=0.75,
    )

    assert metrics.frame_index == 7
    assert metrics.student_overall_dev_m == pytest.approx(2.5)
    assert metrics.student_edge_dev_m == pytest.approx(4.0)
    assert metrics.student_flat_dev_m == pytest.approx(2.0)
    assert metrics.baseline_overall_dev_m == pytest.approx(2.5)
    assert metrics.baseline_edge_dev_m == pytest.approx(1.0)
    assert metrics.baseline_flat_dev_m == pytest.approx(3.0)
    assert metrics.baseline_to_student_overall_ratio == pytest.approx(1.0)
    assert metrics.baseline_to_student_edge_ratio == pytest.approx(0.25)
    assert metrics.baseline_to_student_flat_ratio == pytest.approx(1.5)
