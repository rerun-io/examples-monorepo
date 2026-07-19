# This module must stay importable without heavy deps: stdlib-light CLIs
# (e.g. the arkitscenes downloader subprocess) import format_bytes at startup.
from collections.abc import Callable
from typing import Any


def format_bytes(num_bytes: int) -> str:
    """Format a byte count with binary units."""
    value: float = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def debug_numpy(tensor: Any) -> Callable:
    """
    Convert a tensor to a numpy array if it is not already one.
    """
    import numpy as np
    from lovely_numpy import lo

    if isinstance(tensor, np.ndarray):
        return lo(tensor)
    tensor = tensor.clone().detach().cpu().numpy()
    return lo(tensor)
