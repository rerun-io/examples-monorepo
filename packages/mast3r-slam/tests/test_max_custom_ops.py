from __future__ import annotations
# ruff: noqa: I001

import importlib.util

import pytest
import torch

import mast3r_slam._backends as cuda_be  # pyrefly: ignore[missing-import]
import mast3r_slam.gn_backends as gn_backends
import mast3r_slam.max_ops as max_ops

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(importlib.util.find_spec("max") is None, reason="MAX Python package not installed in this environment"),
]


def test_load_gn_custom_ops() -> None:
    ops = max_ops.load_gn_custom_ops()
    assert hasattr(ops, "pose_retr")
    assert hasattr(ops, "pose_retr_launch")
    assert hasattr(ops, "pose_copy_launch")


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


def test_pose_copy_launch_smoke() -> None:
    ops = max_ops.load_gn_custom_ops()
    gen = torch.Generator(device="cuda").manual_seed(2)
    src = torch.randn(6, 8, device="cuda", generator=gen, dtype=torch.float32)
    dst = torch.empty_like(src)
    ops.pose_copy_launch(dst, src)
    torch.testing.assert_close(dst, src, atol=0.0, rtol=0.0)


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
