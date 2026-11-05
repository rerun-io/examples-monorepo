# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Any

import torch


def recursive_to(x: Any, target: torch.device | str):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        if target == "numpy":
            return x.numpy()
        else:
            return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x

