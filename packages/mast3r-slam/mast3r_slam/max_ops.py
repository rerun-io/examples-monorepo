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


def pose_retr(poses: Tensor, dx: Tensor, num_fix: int) -> Tensor:
    ops = load_gn_custom_ops()
    out = poses.new_empty(poses.shape)
    num_fix_tensor = torch.tensor([num_fix], device=poses.device, dtype=torch.int32)
    ops.pose_retr(out, poses, dx.contiguous(), num_fix_tensor)
    poses.copy_(out)
    return poses


def has_pose_retr() -> bool:
    try:
        _ = load_gn_custom_ops()
    except Exception:
        return False
    return True
