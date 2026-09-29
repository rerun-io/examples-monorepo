"""CPU contract for the installed upstream checkout and pretrained weights; no catalog."""

from pathlib import Path

import numpy as np
import pytest
import torch
from test_tracker import _scene

from handtrack.geometry.camera import CameraRig, project, world_to_cameras
from handtrack.geometry.letterbox import letterbox_for
from handtrack.hand.pose import HandPose, Side, landmarks
from handtrack.labels.circles import enclosing_circles
from handtrack.labels.crops import crop_boxes, crop_from_net
from handtrack.reference.upstream import PoseStage, load_umetrack
from handtrack.tracker import CropRequest
from handtrack.umetrack import RigPoseNetwork, UmeTrackEstimator

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("show3d", [False, True])
def test_pretrained_cpu_inference_supports_our_rigs_and_both_crop_sources(show3d: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path("/home/pablo/handtrack-data/umetrack_baseline/UmeTrack")
    weights = root / "pretrained_models/pretrained_weights.torch"
    if not weights.is_file() or not (root / "lib/tracker/tracker.py").is_file():
        pytest.skip(f"UmeTrack source/weights missing: {root}")
    # No device probing or GPU access, including the upstream constructor.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    torch.set_num_threads(1)
    scene = _scene()
    width, height, count = (1024, 1280, 2) if show3d else (640, 480, 4)
    rig = CameraRig(tuple(f"/world/rig_0{int(show3d)}/cam_0{i}" for i in range(count)),
        torch.tensor([[float(width), float(height)]] * count), torch.eye(4).repeat(count, 1, 1),
        torch.tensor([[300.0, 300.0]] * count), torch.tensor([[(width - 1) / 2, (height - 1) / 2]] * count),
        None if show3d else torch.zeros(count, 8))
    letters = tuple(letterbox_for(width, height) for _ in range(count))
    world_from_rig = torch.eye(4)
    world_from_rig[:3, 3] = torch.tensor([0.7, -0.2, 0.1])
    poses = tuple(HandPose(torch.eye(3), torch.tensor([0.05 * (-1 if side == Side.LEFT else 1), 0.0, 0.5]) + world_from_rig[:3, 3], torch.zeros(22)) for side in Side)
    pixels = torch.stack([project(rig, world_to_cameras(rig, world_from_rig, landmarks(scene.model, poses[side], side))) for side in Side], dim=1)
    net = torch.stack([letters[camera].to_net(pixels[camera]) for camera in range(count)])
    circles = torch.from_numpy(enclosing_circles(net.numpy(), np.ones((count, 2, 21), dtype=bool)))
    api = load_umetrack(root)
    stage = PoseStage(api, None, weights, hand_model=scene.model, device=torch.device("cpu"))
    backend = RigPoseNetwork(stage, rig, (-90.0 if show3d else 0.0,) * count)
    adapter = UmeTrackEstimator(backend, rig, letters, 1.0, 0.8733532444680852)
    images = tuple(torch.full((height, width), 127, dtype=torch.uint8) for _ in range(count))
    # Both hands, two requested views per hand, deliberately interleaved order.
    cameras = torch.tensor([0, 1, 1, 0])
    sides = torch.tensor([0, 1, 0, 1])
    selected = circles[cameras, sides]
    for guess in ((None, None), poses):
        request = CropRequest(cameras, sides, crop_from_net(crop_boxes(selected), sides == 1), torch.zeros((4, 63)),
                              (guess[0], guess[1]), world_from_rig, selected, images)
        output = adapter(torch.stack([letters[camera].apply(image) for camera, image in enumerate(images)]), 0, request)
        assert output.uses_detnet_presence
        assert output.presence.tolist() == [1.0] * 4
        assert torch.isfinite(output.points_net).all() and torch.isfinite(output.d_rel_mm).all()
        assert torch.all(output.confidence == 1.0)
        # Coincident source cameras must receive identical projections, with no second right-hand mirror.
        torch.testing.assert_close(output.points_net[0], output.points_net[2])
        torch.testing.assert_close(output.points_net[1], output.points_net[3])
