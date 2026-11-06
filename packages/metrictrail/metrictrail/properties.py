"""Recording properties: every non-null entry becomes one ``AnyValues`` field."""

from typing import TypeAlias

import rerun as rr

PropertyValue: TypeAlias = str | int | float | list[float] | list[int]
"""A value a property field may hold."""


def any_values(values: dict[str, PropertyValue | None]) -> rr.AnyValues:
    """An ``AnyValues`` bundle of the non-null entries of ``values``."""
    bundle: rr.AnyValues = rr.AnyValues()
    for name, value in values.items():
        if value is not None:
            bundle = bundle.with_component_from_data(name, value)
    return bundle
