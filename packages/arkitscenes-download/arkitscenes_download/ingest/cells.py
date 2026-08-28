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


def landscape_quarter_turns(row: dict[str, Any]) -> int:
    """Return counter-clockwise quarter turns that bring a stored segment to landscape; 0 for landscape segments.

    Args:
        row: Catalog segment row with scalar- or list-shaped orientation properties.

    Returns:
        Counter-clockwise quarter turns in ``[0, 3]``.

    Raises:
        KeyError: If a required portrait property is absent.
        ValueError: If a property is empty or invalid.
    """
    orientation_name: str = "property:capture:orientation"
    if orientation_name not in row or row[orientation_name] is None:
        raise KeyError(f"segment {row.get('rerun_segment_id', '<unknown>')!r} is missing {orientation_name!r}")
    orientation: Any = component_instance(row[orientation_name], orientation_name)
    if isinstance(orientation, (list, np.ndarray)):
        if len(orientation) != 1:
            raise ValueError(f"column {orientation_name!r} must contain one value")
        orientation = orientation[0]
    if orientation not in ("portrait", "landscape"):
        raise ValueError(f"column {orientation_name!r} has invalid value {orientation!r}")
    if orientation == "landscape":
        return 0

    turns_name: str = "property:capture:orientation_quarter_turns_ccw"
    if turns_name not in row or row[turns_name] is None:
        raise KeyError(f"segment {row.get('rerun_segment_id', '<unknown>')!r} is missing {turns_name!r}")
    turns: Any = component_instance(row[turns_name], turns_name)
    if isinstance(turns, (list, np.ndarray)):
        if len(turns) != 1:
            raise ValueError(f"column {turns_name!r} must contain one value")
        turns = turns[0]
    return (-int(turns)) % 4
