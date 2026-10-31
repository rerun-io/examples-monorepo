"""Pose gradient selection preserves hand-only and frozen-leg policies."""

import numpy as np
import pytest
import torch

pytest.importorskip("soma", reason="requires the selfcap-soma environment")
from selfcap_soma.fitting import pose_gradient_mask


@pytest.mark.parametrize("hands_only", [False, True])
@pytest.mark.parametrize("freeze_legs", [False, True])
def test_pose_gradient_mask_composes_hand_and_leg_policies(hands_only: bool, freeze_legs: bool) -> None:
    names = ("Hips", "LeftLeg", "LeftForeArm", "LeftHand", "RightHand", "Neck")
    mask = pose_gradient_mask(names, np.array([1], dtype=np.int64), hands_only, freeze_legs, torch.device("cpu"))
    expected = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0] if hands_only else [1.0] * 6
    if freeze_legs:
        expected[1] = 0.0
    torch.testing.assert_close(mask, torch.tensor(expected).reshape(1, 6, 1), rtol=0.0, atol=0.0)
