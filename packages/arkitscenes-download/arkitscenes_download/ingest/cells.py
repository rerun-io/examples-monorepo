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


def segment_property(row: dict[str, Any], name: str) -> Any:
    """Return one segment property as a scalar.

    Segment rows come from the catalog reader's ``to_pylist()`` as untyped cells whose shape
    depends on the column: a scalar, a one-element list, a one-element array, or a nested
    component instance. This is the one place that narrows that shape; callers validate the value.

    Raises:
        KeyError: If the property is absent or null.
        ValueError: If the property cell does not hold exactly one value.
    """
    if row.get(name) is None:
        raise KeyError(f"segment {row.get('rerun_segment_id', '<unknown>')!r} is missing {name!r}")
    value: Any = component_instance(row[name], name)
    if isinstance(value, (list, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f"column {name!r} must contain one value")
        value = value[0]
    return value


def portrait_from_segment_row(row: dict[str, Any]) -> bool:
    """Return whether the segment was captured in portrait, from its stored orientation property."""
    orientation: Any = segment_property(row, "property:capture:orientation")
    if orientation not in ("portrait", "landscape"):
        raise ValueError(f"column 'property:capture:orientation' has invalid value {orientation!r}")
    return orientation == "portrait"


def landscape_quarter_turns(row: dict[str, Any]) -> int:
    """Return counter-clockwise quarter turns that bring a stored segment to landscape; 0 for landscape segments."""
    if not portrait_from_segment_row(row):
        return 0
    return (-int(segment_property(row, "property:capture:orientation_quarter_turns_ccw"))) % 4
