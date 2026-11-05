"""Virtual-camera translation retains the released solver's masks and fallback."""

import pytest
import torch

pytest.importorskip("warp", reason="EgoHand requires posekit-hands")

from egohand.utils.cam_utils import depth_parameter_to_legacy_camera
from egohand.utils.fixed_depth_translation import camera_scale_to_translation_z, solve_lateral_translation_full_perspective
from posekit.models.egohand_head import solve_virtual_translation


def test_virtual_translation_matches_general_solver_with_invalid_points() -> None:
    generator = torch.Generator().manual_seed(95)
    joints = torch.randn((6, 25, 3), generator=generator) * 0.03
    direct = torch.randn((6, 25, 2), generator=generator) * 0.2
    presence = torch.rand((6, 25), generator=generator)
    presence[1] = 0.0
    joints[2, 4] = float("nan")
    direct[3, 5] = float("inf")
    focal = torch.tensor([90.0, 180.0, 250.0, 350.0, 500.0, 800.0])
    raw = torch.randn((6, 3), generator=generator) * 0.1
    boxes = torch.tensor([[127.5, 127.5, 256.0]]).expand(6, -1)
    intrinsic = torch.eye(3).expand(6, -1, -1).clone()
    intrinsic[:, 0, 0] = intrinsic[:, 1, 1] = focal
    intrinsic[:, :2, 2] = 127.5
    camera = depth_parameter_to_legacy_camera(raw, boxes, intrinsic, "log_lambda", log_value_bounds=(-3.5, 3.0))
    depth = camera_scale_to_translation_z(camera[:, 0], boxes, intrinsic)
    pixels = direct * 256.0 + 127.5
    mask = (presence >= 0.5) & (direct >= -0.5).all(-1) & (direct < 0.5).all(-1) & (pixels >= 0.0).all(-1) & (pixels < 256.0).all(-1)
    solved = solve_lateral_translation_full_perspective(joints, pixels, depth, intrinsic, point_mask=mask)
    expected_xy = torch.where(solved.valid[:, None], solved.translation_xy, raw[:, 1:])
    expected = torch.cat((expected_xy, depth[:, None]), dim=-1)
    torch.testing.assert_close(solve_virtual_translation(joints, raw, direct, presence, focal), expected, atol=1e-6, rtol=1e-5)
