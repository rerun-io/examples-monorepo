"""CPU contract checks against an optional UmeTrack checkout (no catalog or GPU)."""
from pathlib import Path

import numpy as np
import pytest
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch

from handtrack.hand.pose import generic_hand_model
from handtrack.reference.catalog import CameraSpec, ReferenceLabels
from handtrack.reference.geometry import FisheyeRays, circle_crop
from handtrack.reference.upstream import PoseStage, load_umetrack

pytestmark = pytest.mark.integration


def test_upstream_camera_skinning_and_circle_crop_contract():
    root = Path("/home/pablo/handtrack-data/umetrack_baseline/UmeTrack")
    if not (root / "lib/tracker/tracker.py").is_file():
        pytest.skip(f"UmeTrack checkout missing: {root}")
    api = load_umetrack(root)
    model: HandModelTorch = generic_hand_model()
    wrists = np.tile(np.eye(4), (1, 2, 1, 1))
    wrists[:, :, 2, 3] = 500.
    labels = ReferenceLabels([CameraSpec(640, 480, (240., 240.), (320., 240.), [.01, -.002, .001, 0., .001, -.001, .00001, -.000001], 90.)] * 4,
                             np.tile(np.eye(4), (1, 4, 1, 1)), wrists, np.zeros((1, 2, 22)), np.ones((1, 2)),
                             np.ones(1, dtype=bool), np.zeros(1, dtype=np.int64), model)
    stage = PoseStage(api, labels, None)
    stage.set_frame(0)
    gt = stage.ground_truth(0)
    circle, visible = stage.gt_circles(gt)
    assert visible.min() == 21
    crops = stage.circle_crops(circle, visible, 1.0, gt=True)
    upstream = stage.pose_crops(gt)
    assert list(upstream[0]) == [0, 1]
    for hand in (0, 1):
        points = stage.landmarks(gt[hand], hand)
        assert points.shape == (21, 3)
        for index, crop in crops[hand].items():
            assert isinstance(crop, api.camera.PinholePlaneCameraModel)
            pixels = crop.eye_to_window(crop.world_to_eye(points))
            assert ((pixels >= 0) & (pixels <= 95)).all()
            source = stage.cameras[index]
            rays = FisheyeRays(source)
            native = np.array([[320., 240.], [100., 80.], [500., 400.]])
            np.testing.assert_allclose(source.eye_to_window(rays.window_to_eye(native)), native, atol=1e-5)
            params = circle_crop(rays, circle[index, hand], 90.0, hand, 1.0)
            np.testing.assert_allclose(params.camera_to_world[:3, 2], rays.window_to_eye(circle[index, hand, None, :2])[0], atol=1e-6)
    ratios, angles = stage.crop_diagnostics(gt, circle, visible)
    assert len(ratios) == len(angles) == 4
    assert np.isfinite(ratios).all()


def test_upstream_clears_only_ended_hand_memory(monkeypatch):
    import torch

    root = Path("/home/pablo/handtrack-data/umetrack_baseline/UmeTrack")
    if not (root / "lib/tracker/tracker.py").is_file():
        pytest.skip(f"UmeTrack checkout missing: {root}")
    api = load_umetrack(root)
    # No GPU probe or weights: exercise the actual upstream tracker with a fake CPU regressor.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    class FakeModel:
        def __init__(self):
            self.memory = []

        def to(self, device):
            return self

        def getInputImageSizes(self):
            return (96, 96)

        def regress_pose_use_skeleton(self, data, desc, skeleton):
            assert skeleton is not None
            self.memory.append(desc.use_memory.tolist())
            count = len(desc.hand_idx)
            wrist = torch.eye(4).repeat(count, 1, 1)
            wrist[:, 2, 3] = .5
            return api.tracker.RegressorOutput(torch.zeros(count, 22), wrist)

    model = FakeModel()
    tracker = api.tracker.HandTracker(model, api.tracker.HandTrackerOpts())
    camera = api.camera.PinholePlaneCameraModel(96, 96, (100., 100.), (47.5, 47.5), [], camera_to_world_xf=np.eye(4))
    sample = api.tracker.InputFrame([api.tracker.ViewData(np.zeros((96, 96), dtype=np.uint8), camera, 0.)])
    from dataclasses import fields
    profile = generic_hand_model()
    hand_model = api.hand.HandModel(**{field.name: getattr(profile, field.name).float() for field in fields(profile)})
    tracker.track_frame(sample, hand_model, {0: {0: camera}, 1: {0: camera}})
    tracker.track_frame(sample, hand_model, {1: {0: camera}})
    tracker.track_frame(sample, hand_model, {0: {0: camera}, 1: {0: camera}})
    tracker.track_frame(sample, hand_model, {})
    tracker.track_frame(sample, hand_model, {0: {0: camera}, 1: {0: camera}})
    assert model.memory == [[False, False], [True], [False, True], [False, False]]
