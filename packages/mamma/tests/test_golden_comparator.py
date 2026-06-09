"""Comparator self-consistency: golden vs golden must be exactly zero."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mamma.eval.golden import fit3d_comparison, landmarks_diagnostics

_GOLDEN_ROOT: Path = Path(__file__).parent.parent / "data" / "golden"
_MA3D: Path = _GOLDEN_ROOT / "ma_3d/baseline/indoors/crossing_arms/verts_joints_body_id-00.npz"
_MA2D: Path = _GOLDEN_ROOT / "ma_2d/baseline/indoors/crossing_arms"

needs_golden = pytest.mark.skipif(not _MA3D.exists(), reason="golden artifact not downloaded")


@needs_golden
def test_fit3d_golden_vs_golden_is_zero() -> None:
    golden = np.load(_MA3D, allow_pickle=True)
    cmp = fit3d_comparison(_MA3D, golden["pred_joints"], golden["pred_vertices"])
    assert cmp.mpjpe_mm == 0.0
    assert cmp.pve_mm == 0.0
    assert cmp.frames_compared == 30


@needs_golden
def test_landmarks_golden_vs_golden_is_zero() -> None:
    cams: list[str] = ["A001", "B001"]
    predicted = {}
    predicted_vis = {}
    for cam in cams:
        golden = np.load(_MA2D / f"{cam}.npz", allow_pickle=True)
        predicted[cam] = golden["landmarks"][:, 0]
        predicted_vis[cam] = golden["visibilities"][:, 0]
    diags = landmarks_diagnostics(_MA2D, cams, predicted, predicted_vis, pred_scale_to_golden=1.0)
    for d in diags:
        assert d.px_p50 == 0.0
        assert d.px_p99 == 0.0
        assert d.vis_p99 == 0.0


@needs_golden
def test_fit3d_detects_offset() -> None:
    golden = np.load(_MA3D, allow_pickle=True)
    shifted = golden["pred_joints"] + 0.05  # 5 cm in every axis
    cmp = fit3d_comparison(_MA3D, shifted, None)
    assert abs(cmp.mpjpe_mm - 50.0 * np.sqrt(3)) < 1.0
