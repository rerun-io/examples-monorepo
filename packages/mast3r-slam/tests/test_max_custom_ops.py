from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from mast3r_slam import _backends as cuda_be
from mast3r_slam import gn_backends
from mast3r_slam import max_ops


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_load_gn_custom_ops() -> None:
    ops = max_ops.load_gn_custom_ops()
    assert hasattr(ops, "pose_retr")


def test_pose_retr_matches_cuda() -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    poses = torch.randn(6, 8, device="cuda", generator=gen, dtype=torch.float32)
    poses[:, 6] = 1.0
    poses[:, 7] = 1.0
    dx = torch.randn(5, 7, device="cuda", generator=gen, dtype=torch.float32) * 1e-3
    num_fix = 1

    expected = poses.clone()
    actual = poses.clone()
    cuda_be.pose_retr(expected, dx, num_fix)
    max_ops.pose_retr(actual, dx, num_fix)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_gn_pose_retr_prefers_max_when_available() -> None:
    gen = torch.Generator(device="cuda").manual_seed(1)
    poses = torch.randn(6, 8, device="cuda", generator=gen, dtype=torch.float32)
    poses[:, 6] = 1.0
    poses[:, 7] = 1.0
    dx = torch.randn(5, 7, device="cuda", generator=gen, dtype=torch.float32) * 1e-3

    expected = poses.clone()
    actual = poses.clone()
    cuda_be.pose_retr(expected, dx, 1)
    gn_backends.pose_retr(actual, dx, 1)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
