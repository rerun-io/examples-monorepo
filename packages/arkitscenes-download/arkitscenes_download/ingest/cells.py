"""Decoding of catalog reader cells into plain values."""

from typing import Any

import numpy as np


def component_instance(value: Any, column_name: str) -> Any:
    """Unwrap the single component instance from one reader cell.

    Component cells arrive as a one-element list around the instance (itself a
    list of scalars or bytes); scalar-shaped cells arrive as the value directly.
    """
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"column {column_name!r} has an empty cell")
        if len(value) == 1 and isinstance(value[0], (list, np.ndarray, bytes)):
            return value[0]
    return value
