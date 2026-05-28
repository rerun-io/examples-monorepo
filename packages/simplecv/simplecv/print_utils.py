from collections.abc import Callable
from typing import Any

import numpy as np
from lovely_numpy import lo


def debug_numpy(tensor: np.ndarray | Any) -> Callable:
    """
    Convert a tensor to a numpy array if it is not already one.
    """
    if isinstance(tensor, np.ndarray):
        return lo(tensor)
    tensor = tensor.clone().detach().cpu().numpy()
    return lo(tensor)
