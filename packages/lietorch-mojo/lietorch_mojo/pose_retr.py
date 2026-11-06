"""MASt3R-SLAM's Sim3 pose retraction on the Mojo backend.

`pose_retr(poses, dx, num_fix)` is the update of mast3r-slam's
`pose_retr_py` (backend/mojo/gn.mojo): in place, for every pose k >= num_fix,
T_new[k] = Exp(dx[k - num_fix]) * T_old[k]. It equals
`Sim3(poses[num_fix:]).retr(dx)` row for row. The Mojo side lives in
`lietorch_mojo/mojo/pose_retr.mojo`, which other Mojo builds import with
`-I lietorch_mojo/mojo`.
"""

import torch
from torch import Tensor

from lietorch_mojo import _lietorch_mojo_backends as _ext
from lietorch_mojo import backends

POSE_WIDTH: int = 8
TANGENT_WIDTH: int = 7


def pose_retr(poses: Tensor, dx: Tensor, num_fix: int) -> None:
    """Retract `poses` (N, 8) in place by `dx` (N - num_fix, 7); the first `num_fix` poses stay fixed.

    Both tensors must be contiguous, of one dtype (float32 or float64) and on
    one device (CPU or cuda:0). On CUDA the kernel runs on PyTorch's current
    stream, asynchronously.
    """
    if num_fix < 0:
        raise ValueError(f"num_fix must be >= 0, got {num_fix}")
    if poses.dim() != 2 or poses.shape[1] != POSE_WIDTH:
        raise ValueError(f"poses must have shape (N, {POSE_WIDTH}), got {tuple(poses.shape)}")
    num_poses: int = poses.shape[0]
    num_vars: int = max(num_poses - num_fix, 0)
    if tuple(dx.shape) != (num_vars, TANGENT_WIDTH):
        raise ValueError(f"dx must have shape ({num_vars}, {TANGENT_WIDTH}), got {tuple(dx.shape)}")
    if not poses.is_contiguous() or not dx.is_contiguous():
        raise ValueError("poses and dx must be contiguous (poses is updated in place)")
    if poses.dtype != dx.dtype or poses.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"poses and dx must share dtype float32 or float64, got {poses.dtype} and {dx.dtype}")
    if poses.device != dx.device:
        raise ValueError(f"poses and dx must be on one device, got {poses.device} and {dx.device}")
    if num_vars == 0:
        return
    is_f64: bool = poses.dtype == torch.float64
    if poses.is_cuda:
        if poses.get_device() != 0:
            raise RuntimeError(f"lietorch_mojo runs on cuda:0 only, got {poses.device}")
        stream: int = torch._C._cuda_getCurrentRawStream(0)
        _ext.pose_retr_cuda(backends._runtime(True), stream, poses.data_ptr(), dx.data_ptr(), num_poses, num_fix, is_f64)
    elif poses.is_cpu:
        _ext.pose_retr_cpu(backends._runtime(False), poses.data_ptr(), dx.data_ptr(), num_poses, num_fix, is_f64)
    else:
        raise RuntimeError(f"lietorch_mojo supports CPU and CUDA tensors, got {poses.device}")
