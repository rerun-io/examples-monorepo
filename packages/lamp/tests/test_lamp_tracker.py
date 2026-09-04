"""Public streaming LAMP tracker tests with learned-stage doubles."""

import numpy as np
import torch
from jaxtyping import UInt8
from posekit.models.base import PersonDetector, TopDownPose2d
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.skeletons import COCO_17
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from torch import Tensor, nn

from lamptrack.cameras import RigCamera
from lamptrack.models.lamp import Frameset, LampTracker
from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings


class _Detector(PersonDetector):
    """Return one person box in every camera."""

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        device = frames_rgb.device
        batch = int(frames_rgb.shape[0])
        return BoxDetections(
            xyxy=torch.tensor([[20.0, 20.0, 80.0, 100.0]], device=device).repeat(batch, 1),
            scores=torch.full((batch,), 0.9, device=device),
            frame_indices=torch.arange(batch, device=device),
        )


class _Pose(TopDownPose2d):
    """Return row-aligned COCO-17 keypoints."""

    skeleton = COCO_17

    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Keypoints2d:
        count = detections.num_detections
        one_pose = torch.stack(
            (
                torch.linspace(30.0, 70.0, 17, device=frames_rgb.device),
                torch.linspace(35.0, 85.0, 17, device=frames_rgb.device),
            ),
            dim=1,
        )
        xy = one_pose[None].repeat(count, 1, 1)
        scores = torch.full((count, 17), 0.9, device=frames_rgb.device)
        scores[:, 0] = 0.4
        return Keypoints2d(xy, scores, detections.frame_indices, self.skeleton)


class _RecordingLampNet(nn.Module):
    """Return a valid upright skeleton and retain the binary model input."""

    def __init__(self) -> None:
        super().__init__()
        self.keypoints: list[Tensor] = []

    def forward(
        self,
        keypoints: list[Tensor],
        camera_params: list[Tensor],
        camera_poses: list[Tensor],
        ground_planes: Tensor,
    ) -> dict[str, Tensor]:
        del camera_params, camera_poses, ground_planes
        self.keypoints = [view.detach().cpu().clone() for view in keypoints]
        batch, steps = keypoints[0].shape[:2]
        joints = torch.zeros((batch, steps, 24, 3), dtype=torch.float32)
        joints[..., 2] = 2.0
        joints[..., 4, 2] = 1.4
        joints[..., 5, 2] = 1.4
        joints[..., 7, 2] = 0.8
        joints[..., 8, 2] = 0.8
        rotations = torch.eye(3).expand(batch, steps, 24, 3, 3).clone()
        translation = joints[:, :, 0]
        return {
            "skel_w": joints,
            "transl": translation,
            "global_orient_rotmat": rotations[:, :, :1],
            "body_pose_rotmat": rotations[:, :, 1:],
            "betas": torch.zeros((batch, 10), dtype=torch.float32),
        }


def _camera(name: str) -> RigCamera:
    """Build an identity pinhole camera for association tests."""
    intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions="RDF", fl_x=100.0, fl_y=100.0, cx=64.0, cy=64.0, width=128, height=128
    )
    extrinsics = Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.zeros(3))
    return RigCamera(PinholeParameters(name=name, extrinsics=extrinsics, intrinsics=intrinsics))


def test_step_returns_smoothed_person_and_binary_keypoints() -> None:
    """Two frames fill the window, preserve one ID, and lift binary COCO input."""
    model = _RecordingLampNet()
    lifter = Lifter(model, torch.device("cpu"), LifterSettings(snippet_length=2))
    tracker = LampTracker(
        detector=_Detector(), pose=_Pose(), lifter=lifter, device=torch.device("cpu"), window=2, keypoint_conf_min=0.5
    )
    tracker.configure_cameras(tuple(_camera(f"cam_{index}") for index in range(4)))
    images = np.zeros((4, 128, 128, 3), dtype=np.uint8)
    world_T_rig = np.eye(4, dtype=np.float64)

    first = tracker.step(Frameset(1_000_000_000, images, world_T_rig))
    second = tracker.step(Frameset(1_100_000_000, images, world_T_rig))

    assert first.people == {}
    assert set(second.people) == {1}
    state = second.people[1]
    assert state.timestamps_ns.tolist() == [1_000_000_000, 1_100_000_000]
    assert state.joints_world.shape == (2, 24, 3)
    assert state.betas.shape == (10,)
    assert state.root_T.shape == (2, 4, 4)
    assert state.rotations.shape == (2, 24, 3, 3)
    assert all(output.track_ids is not None and output.track_ids.tolist() == [1] for output in second.boxes_by_camera.values())
    assert model.keypoints[0][0, -1, 0].tolist() == [0.0, 0.0, 0.0]
    assert model.keypoints[0][0, -1, 1, 2].item() == 1.0
