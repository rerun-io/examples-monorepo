from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor


def _ops_root() -> Path:
    return Path(__file__).resolve().parent / "backend" / "max_ops" / "gn_ops"


@lru_cache(maxsize=1)
def load_gn_custom_ops():
    from max.experimental.torch import CustomOpLibrary

    return CustomOpLibrary(_ops_root())


@lru_cache(maxsize=32)
def _num_fix_tensor(device_index: int, num_fix: int) -> Tensor:
    return torch.tensor([num_fix], device=torch.device("cuda", device_index), dtype=torch.int32)


def pose_retr(poses: Tensor, dx: Tensor, num_fix: int) -> Tensor:
    ops = load_gn_custom_ops()
    device_index = poses.device.index
    if device_index is None:
        raise ValueError("pose_retr requires a CUDA tensor")
    num_fix_tensor = _num_fix_tensor(device_index, num_fix)
    if hasattr(ops, "pose_retr_launch"):
        ops.pose_retr_launch(poses, poses, dx.contiguous(), num_fix_tensor)
    else:
        ops.pose_retr(poses, poses, dx.contiguous(), num_fix_tensor)
    return poses


def has_pose_retr() -> bool:
    try:
        _ = load_gn_custom_ops()
    except Exception:
        return False
    return True
