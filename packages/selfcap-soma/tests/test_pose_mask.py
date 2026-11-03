"""Pose gradient selection freezes only the leg chain."""

import numpy as np
import pytest
import torch

pytest.importorskip("soma", reason="requires the selfcap-soma environment")
from selfcap_soma.fitting import pose_gradient_mask


@pytest.mark.parametrize("freeze_legs", [False, True])
def test_pose_gradient_mask_freezes_only_the_legs(freeze_legs: bool) -> None:
    names = ("Hips", "LeftLeg", "LeftForeArm", "LeftHand", "RightHand", "Neck")
    mask = pose_gradient_mask(names, np.array([1], dtype=np.int64), freeze_legs, torch.device("cpu"))
    expected = [1.0] * 6
    if freeze_legs:
        expected[1] = 0.0
    torch.testing.assert_close(mask, torch.tensor(expected).reshape(1, 6, 1), rtol=0.0, atol=0.0)
