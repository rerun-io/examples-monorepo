"""Hand-computed evaluation cases."""
import pytest

pytest.importorskip('torch', reason='requires handtrack environment')
import torch

from handtrack.eval.metrics import detection_metrics, keynet_metrics, pipeline_metrics, tracking_metrics


def test_detection_rule() -> None:
    points = torch.full((4, 21, 2), 100.0)
    points[:, 0, 0] = 90.0
    points[:, 1, 0] = 110.0
    front = torch.ones(4, 21, dtype=torch.bool)
    boxes = torch.tensor([[90., 90., 110., 110.], [87.5, 87.5, 112.5, 112.5], [91., 90., 111., 110.], [90., 90., 110., 110.]])
    front[3, 16:] = False
    result = detection_metrics(boxes, torch.ones(4), points, front, torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]))
    assert result.total.true_positive == 1
    assert result.total.ground_truth == 3
    assert result.total.predicted == 4
    assert result.total.precision == 0.25
    assert result.total.recall == pytest.approx(1 / 3)


def test_keypoint_pipeline_and_tracking() -> None:
    xy = torch.zeros(2, 21, 2)
    pred = xy.clone()
    pred[0, :, 0] = 3.0
    affine = torch.eye(3).repeat(2, 1, 1)
    affine[:, 0, 0] = 0.5
    result = keynet_metrics(pred, xy, affine, torch.ones(2, 21), torch.zeros(2, 21), torch.tensor([0.8, 0.9]), torch.ones(2), torch.tensor([True, False]), torch.tensor([True, False]))
    assert result.error_px == 6.0
    assert result.distance_mm == 1.0
    assert result.presence.precision == 1.0
    trajectory = torch.arange(5).float()[:, None, None, None].expand(5, 1, 21, 3) * 0.01
    metrics = pipeline_metrics(trajectory + 0.001, trajectory, torch.ones(5, 1, dtype=torch.bool))
    assert metrics.mkpe_mm == pytest.approx(3 ** 0.5, abs=1e-5)
    assert metrics.mka_mm == pytest.approx(0.0, abs=1e-5)
    assert metrics.mka_gt_mm == pytest.approx(0.0, abs=1e-5)
    tracking = tracking_metrics(torch.tensor([False, True, True, True, False, False, True, True]), torch.tensor([False, False, True, True, True, False, False, False]))
    assert tracking.visible_tracked_fraction == 0.4
    assert tracking.acquire_frames == (1, None)
    assert tracking.drop_frames == (1,)
    assert tracking.tracked_without_hand == 1


def test_presence_threshold_and_behind_camera_points() -> None:
    points = torch.full((2, 21, 2), 100.0)
    points[:, 0, 0] = 90.0
    points[:, 1, 0] = 110.0
    points[:, -1] = 999.0
    front = torch.ones(2, 21, dtype=torch.bool)
    front[:, -1] = False
    result = detection_metrics(torch.tensor([[90., 90., 110., 110.]]).repeat(2, 1), torch.tensor([0.5, 0.51]), points, front, torch.zeros(2, dtype=torch.int64), torch.tensor([0, 1]))
    assert result.total.predicted == result.total.true_positive == 1
    assert result.total.recall == 0.5
    assert result.by_camera_hand[(0, 0)].precision is None
    assert result.by_camera_hand[(0, 1)].precision == 1.0


def test_tracking_censored_drop_and_empty_metrics() -> None:
    result = tracking_metrics(torch.tensor([True, False, False]), torch.tensor([True, True, True]))
    assert result.acquire_frames == (0,)
    assert result.drop_frames == (None,)
    assert result.tracked_without_hand == 2
    empty = pipeline_metrics(torch.zeros(1, 1, 21, 3), torch.zeros(1, 1, 21, 3), torch.zeros(1, 1, dtype=torch.bool))
    assert empty.mkpe_mm is None and empty.mka_mm is None and empty.mka_gt_mm is None


def test_acceleration_and_mirrored_crop() -> None:
    trajectory = torch.zeros(3, 1, 21, 3)
    trajectory[2, :, :, 0] = 0.002
    result = pipeline_metrics(trajectory, trajectory, torch.ones(3, 1, dtype=torch.bool))
    assert result.mkpe_mm == 0.0
    assert result.mka_mm == pytest.approx(2.0)
    assert result.mka_gt_mm == pytest.approx(2.0)
    affine = torch.tensor([[[-0.5, 0., 95.], [0., 0.5, 0.], [0., 0., 1.]]])
    key = keynet_metrics(torch.ones(1, 21, 2), torch.zeros(1, 21, 2), affine, torch.zeros(1, 21), torch.zeros(1, 21), torch.tensor([0.5]), torch.ones(1), torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    assert key.error_px == pytest.approx(8 ** 0.5)
    assert key.presence.recall == key.presence.precision == 1.0
