"""Hand metrics on worked examples (EgoHand paper Sec. 6.1 definitions)."""

import numpy as np
import pytest

from posekit.eval.hand_metrics import box_iou, hand_errors, summarize


def _hand() -> np.ndarray:
    """A 21-joint hand: wrist at the origin, joint i at (i, 2i, 3i) mm along a bent chain."""
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[1:] = np.stack([np.arange(1, 21), (np.arange(1, 21) % 4) * 2.0, np.arange(1, 21) * 0.5], axis=1)
    return joints / 1000.0


def test_identical_hands_have_zero_error() -> None:
    gt = _hand()
    errors = hand_errors(gt, gt, gt[:, :2] * 1000, gt[:, :2] * 1000, np.ones(21, dtype=bool))
    assert errors.mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.pa_mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.rte_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.epe_2d_px == pytest.approx(0.0, abs=1e-9)


def test_a_translated_hand_only_has_root_error() -> None:
    gt = _hand()
    pred = gt + np.array([0.03, -0.04, 0.0])
    errors = hand_errors(pred, gt, np.zeros((21, 2)), np.zeros((21, 2)), np.ones(21, dtype=bool))
    assert errors.mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.pa_mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.rte_mm == pytest.approx(50.0)


def test_a_hand_scaled_about_its_wrist_is_wrong_until_similarity_aligned() -> None:
    gt = _hand()
    pred = gt * 2.0
    errors = hand_errors(pred, gt, np.zeros((21, 2)), np.zeros((21, 2)), np.ones(21, dtype=bool))
    # Wrist-relative error of joint i is |G_i|; the wrist contributes 0.
    expected = float(np.linalg.norm(gt, axis=1).mean() * 1000.0)
    assert errors.mpjpe_mm == pytest.approx(expected)
    assert errors.pa_mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.rte_mm == pytest.approx(0.0, abs=1e-9)


def test_a_rotated_hand_is_removed_by_procrustes() -> None:
    gt = _hand()
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    errors = hand_errors(gt @ rotation.T, gt, np.zeros((21, 2)), np.zeros((21, 2)), np.ones(21, dtype=bool))
    assert errors.pa_mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.mpjpe_mm > 1.0


def test_masked_joints_do_not_count() -> None:
    gt = _hand()
    pred = gt.copy()
    pred[7] += np.array([1.0, 0.0, 0.0])
    gt_2d = np.zeros((21, 2))
    pred_2d = np.zeros((21, 2))
    pred_2d[7] = [100.0, 0.0]
    mask = np.ones(21, dtype=bool)
    mask[7] = False
    errors = hand_errors(pred, gt, pred_2d, gt_2d, mask)
    assert errors.mpjpe_mm == pytest.approx(0.0, abs=1e-9)
    assert errors.epe_2d_px == pytest.approx(0.0, abs=1e-9)


def test_epe_is_the_mean_joint_pixel_distance() -> None:
    gt = _hand()
    pred_2d = np.zeros((21, 2))
    pred_2d[:, 0] = 3.0
    pred_2d[:, 1] = 4.0
    errors = hand_errors(gt, gt, pred_2d, np.zeros((21, 2)), np.ones(21, dtype=bool))
    assert errors.epe_2d_px == pytest.approx(5.0)


def test_summary_reports_mean_and_p90() -> None:
    summary = summarize(np.arange(1.0, 11.0))
    assert summary.mean == pytest.approx(5.5)
    assert summary.p90 == pytest.approx(9.1)
    assert summary.count == 10
    assert summarize(np.array([])).mean is None


def test_box_iou() -> None:
    assert box_iou(np.array([0.0, 0, 10, 10]), np.array([5.0, 0, 15, 10])) == pytest.approx(50 / 150)
    assert box_iou(np.array([0.0, 0, 10, 10]), np.array([20.0, 20, 30, 30])) == 0.0
