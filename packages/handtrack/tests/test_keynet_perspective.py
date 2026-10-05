import pytest
import torch

import handtrack.keynet_perspective as stage
from handtrack.geometry.camera import CameraRig, project, world_to_cameras
from handtrack.geometry.letterbox import letterbox_for
from handtrack.hand.pose import HandPose, Side, generic_hand_model, landmarks
from handtrack.keynet_perspective import PerspectiveKeyNetEstimator
from handtrack.labels.perspective import CROP_CENTRE, CROP_MARGIN, to_crop, unproject
from handtrack.models.keynet import KeyNetF
from handtrack.tracker import CropRequest


def _rig() -> CameraRig:
    coefficients = torch.tensor([[0.30, -0.10, 0.05, -0.01, 0.002, 0.0, 1e-4, -2e-4]] * 2)
    cam_from_rig = torch.eye(4).repeat(2, 1, 1)
    cam_from_rig[1, 0, 3] = -0.05
    return CameraRig(names=("/a", "/b"), image_size=torch.tensor([[640.0, 480.0]] * 2), cam_from_rig=cam_from_rig,
                     focal=torch.tensor([[240.0, 240.0]] * 2), principal=torch.tensor([[320.5, 238.0]] * 2), fisheye62=coefficients)


def _pose(x: float) -> HandPose:
    return HandPose(rotation=torch.eye(3), translation=torch.tensor([x, 0.0, 0.35]), joint_angles=torch.zeros(22))


def test_decoded_crop_points_come_back_as_the_true_net_keypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    rig, model = _rig(), generic_hand_model()
    letterboxes = (letterbox_for(640, 480),) * 2
    world_from_rig = torch.eye(4)
    poses = (_pose(-0.04), _pose(0.05))
    truth_net = {}
    for side in Side:
        points_cam = world_to_cameras(rig, world_from_rig, landmarks(model, poses[side], side))
        truth_net[side] = letterboxes[0].to_net(project(rig, points_cam))  # both letterboxes are the identity map
    # Views: the left hand tracked in camera 0 and the right hand tracked in camera 1.
    request = CropRequest(camera=torch.tensor([0, 1]), side=torch.tensor([0, 1]), crop_from_net=torch.eye(3).repeat(2, 1, 1),
                          keypoint_input=torch.zeros(2, 63), poses=poses, world_from_rig=world_from_rig, circles=torch.zeros(2, 3),
                          native_images=(torch.zeros(480, 640, dtype=torch.uint8),) * 2)
    estimator = PerspectiveKeyNetEstimator(KeyNetF(), rig, letterboxes, (0.0, 30.0), model, phi=1.0)
    planned = [estimator._cameras(request, int(request.camera[i]), int(request.side[i]), i) for i in range(2)]
    exact = torch.stack([to_crop(cameras, world_to_cameras(rig, world_from_rig, landmarks(model, poses[s], Side(s)))[c][None])[0][0]
                         for (cameras, _), c, s in zip(planned, (0, 1), (0, 1), strict=True)])
    monkeypatch.setattr(stage, "decode_heatmaps", lambda heatmaps: (exact.to(heatmaps.device), torch.ones(2, 21)))
    estimate = estimator(torch.zeros(2, 480, 640, dtype=torch.uint8), 0, request)
    torch.testing.assert_close(estimate.points_net[0], truth_net[Side.LEFT][0], atol=0.05, rtol=0)
    torch.testing.assert_close(estimate.points_net[1], truth_net[Side.RIGHT][1], atol=0.05, rtol=0)
    assert estimate.presence.shape == (2,) and bool(torch.isfinite(estimate.d_rel_mm).all()) and float(planned[0][1].abs().sum()) > 0
    # An acquisition (no pose for that hand): the crop is aimed through its DetNet circle, and its keypoint input is zeros.
    circle = torch.cat([truth_net[Side.RIGHT][0].mean(0), torch.tensor([60.0])])[None]
    acquire = CropRequest(camera=torch.tensor([0]), side=torch.tensor([1]), crop_from_net=torch.eye(3)[None], keypoint_input=torch.zeros(1, 63),
                          poses=(poses[0], None), world_from_rig=world_from_rig, circles=circle, native_images=request.native_images)
    cameras, features = estimator._cameras(acquire, 0, 1, 0)
    assert float(features.abs().sum()) == 0.0 and bool(cameras.mirror.all())
    monkeypatch.setattr(stage, "decode_heatmaps", lambda heatmaps: (torch.full((1, 21, 2), CROP_CENTRE, device=heatmaps.device), torch.ones(1, 21)))
    centre = estimator(torch.zeros(2, 480, 640, dtype=torch.uint8), 0, acquire).points_net[0]
    torch.testing.assert_close(centre, circle[0, :2].expand(21, 2), atol=0.05, rtol=0)  # the crop centre maps back onto the circle centre
    # The circle's radius spans CROP_CENTRE / CROP_MARGIN crop pixels, as a training crop's farthest landmark does.
    edge_ray = unproject(rig.select([0]), (circle[0, :2] + torch.tensor([60.0, 0.0]))[None])
    uv, _ = to_crop(cameras, edge_ray[None])
    assert float((uv[0, 0] - CROP_CENTRE).norm()) == pytest.approx(CROP_CENTRE / CROP_MARGIN, rel=0.02)


def test_a_view_without_a_usable_crop_camera_reports_absent_not_nan() -> None:
    rig, model = _rig(), generic_hand_model()
    behind = HandPose(rotation=torch.eye(3), translation=torch.tensor([0.0, 0.0, -0.4]), joint_angles=torch.zeros(22))
    request = CropRequest(camera=torch.tensor([0, 1]), side=torch.tensor([0, 1]), crop_from_net=torch.eye(3).repeat(2, 1, 1),
                          keypoint_input=torch.zeros(2, 63), poses=(behind, _pose(0.05)), world_from_rig=torch.eye(4),
                          circles=torch.tensor([[float("nan")] * 3, [320.0, 240.0, 50.0]]), native_images=(torch.zeros(480, 640, dtype=torch.uint8),) * 2)
    estimator = PerspectiveKeyNetEstimator(KeyNetF(), rig, (letterbox_for(640, 480),) * 2, (0.0, 0.0), model, phi=1.0)
    estimate = estimator(torch.zeros(2, 480, 640, dtype=torch.uint8), 0, request)
    assert bool(torch.isfinite(estimate.points_net).all()) and bool(torch.isfinite(estimate.presence).all()) and bool(torch.isfinite(estimate.d_rel_mm).all())
    assert float(estimate.presence[0]) == 0.0 and float(estimate.confidence[0].abs().sum()) == 0.0
