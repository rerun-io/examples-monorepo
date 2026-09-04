"""Fast shape and finiteness tests for the LAMP lifter contract."""

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings, SnippetData


class _ShapeLampNet(nn.Module):
    """Small learned-model boundary double with the public output contract."""

    def forward(
        self,
        keypoints: list[Tensor],
        camera_params: list[Tensor],
        camera_poses: list[Tensor],
        ground_planes: Tensor,
    ) -> dict[str, Tensor]:
        """Derive finite, SMPL-shaped outputs from every input family."""
        batch, steps = keypoints[0].shape[:2]
        signal = torch.stack([view[..., :2].mean(dim=(2, 3)) for view in keypoints]).mean(dim=0)
        camera_signal = torch.stack([params[..., :4].mean(dim=2) for params in camera_params]).mean(dim=0)
        pose_signal = torch.stack([poses[..., :3, 3].mean(dim=2) for poses in camera_poses]).mean(dim=0)
        floor_signal = torch.nan_to_num(ground_planes, nan=0.0).mean(dim=(1, 2))[:, None]
        scalar = signal + camera_signal * 1e-5 + pose_signal * 1e-4 + floor_signal * 1e-4
        translation = torch.stack((scalar, scalar * 0.5, scalar * 0.25), dim=-1)
        rotations = torch.eye(3, dtype=torch.float32).expand(batch, steps, 24, 3, 3).clone()
        return {
            "skel_w": translation[:, :, None].expand(batch, steps, 24, 3).clone(),
            "transl": translation,
            "global_orient_rotmat": rotations[:, :, :1],
            "body_pose_rotmat": rotations[:, :, 1:],
            "betas": scalar.mean(dim=1, keepdim=True).expand(batch, 10).clone(),
        }


def _snippet(steps: int, camera_width: int) -> SnippetData:
    """Create a deterministic random four-view snippet."""
    generator = np.random.default_rng(steps * 100 + camera_width)
    keypoints = []
    camera_params = []
    camera_poses = []
    for view_idx in range(4):
        view_keypoints = generator.uniform(0.0, 512.0, size=(steps, 17, 3)).astype(np.float32)
        view_keypoints[..., 2] = 1.0
        keypoints.append(view_keypoints)
        params = generator.normal(size=(steps, camera_width)).astype(np.float32)
        params[:, :4] += np.array([500.0, 500.0, 256.0, 256.0], dtype=np.float32)
        camera_params.append(params)
        poses = np.tile(np.eye(4, dtype=np.float32), (steps, 1, 1))
        poses[:, 0, 3] = view_idx * 0.1
        camera_poses.append(poses)
    return SnippetData(
        person_id=1,
        snippet_timestamps_ns=list(range(steps)),
        view_cam_indices=[0, 1, 2, 3],
        kp2ds_per_view=keypoints,
        Ts_gw_cam_per_view=camera_poses,
        cam_params_per_view=camera_params,
        T_gravityWorld_world=np.eye(4, dtype=np.float32),
    )


@pytest.mark.parametrize("steps", [5, 20])
@pytest.mark.parametrize("camera_width", [4, 16])
@pytest.mark.parametrize("floor_z", [None, 0.0])
def test_lifter_random_snippet_shapes_are_finite(steps: int, camera_width: int, floor_z: float | None) -> None:
    """Both camera models and floor modes produce the public SMPL shapes."""
    lifter = Lifter(_ShapeLampNet(), torch.device("cpu"), LifterSettings(snippet_length=steps))
    lifter.set_floor_plane(floor_z)

    outputs = lifter.lift_all_steps_batched({1: _snippet(steps, camera_width)})[1]

    assert len(outputs) == steps
    for _, skeleton in outputs:
        assert skeleton.kp_world.shape == (24, 3)
        assert skeleton.shape.shape == (10,)
        assert skeleton.T_world_pelvis.shape == (4, 4)
        assert skeleton.joints_rot_mat.shape == (24, 3, 3)
        assert np.isfinite(skeleton.kp_world).all()
        assert np.isfinite(skeleton.T_world_pelvis).all()
        assert np.isfinite(skeleton.joints_rot_mat).all()
