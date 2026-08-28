"""Behavior checks for catalog cell parsing."""

import numpy as np
import pytest

from arkitscenes_download.ingest.cells import landscape_quarter_turns


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            {
                "property:capture:orientation": ["portrait"],
                "property:capture:orientation_quarter_turns_ccw": np.array([1]),
            },
            3,
        ),
        ({"property:capture:orientation": "landscape"}, 0),
    ],
)
def test_landscape_quarter_turns_parses_list_and_scalar_properties(row: dict[str, object], expected: int) -> None:
    """Return the counter-clockwise rotation that restores stored frames to landscape."""
    assert landscape_quarter_turns(row) == expected
