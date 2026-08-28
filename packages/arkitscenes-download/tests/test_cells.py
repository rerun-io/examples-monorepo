"""Behavior checks for catalog cell parsing."""

import numpy as np
import pytest

from arkitscenes_download.ingest.cells import landscape_quarter_turns, portrait_from_segment_row, segment_property


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("portrait", "portrait"), (["portrait"], "portrait"), (np.array([3]), 3), ([[7]], 7)],
)
def test_segment_property_narrows_every_reader_cell_shape(cell: object, expected: object) -> None:
    """Return the one scalar behind a scalar, list, array, or nested component cell."""
    assert segment_property({"property:x": cell}, "property:x") == expected


def test_segment_property_rejects_missing_and_multi_valued_cells() -> None:
    """Name the segment when a property is absent; refuse cells with several values."""
    with pytest.raises(KeyError, match="'seg-1' is missing 'property:x'"):
        segment_property({"rerun_segment_id": "seg-1", "property:x": None}, "property:x")
    with pytest.raises(ValueError, match="exactly one value|must contain one value"):
        segment_property({"property:x": ["a", "b"]}, "property:x")


def test_portrait_from_segment_row_validates_the_orientation_value() -> None:
    """Map the stored orientation to a portrait flag and reject anything else."""
    assert portrait_from_segment_row({"property:capture:orientation": ["portrait"]}) is True
    assert portrait_from_segment_row({"property:capture:orientation": "landscape"}) is False
    with pytest.raises(ValueError, match="invalid value 'upside-down'"):
        portrait_from_segment_row({"property:capture:orientation": "upside-down"})
    with pytest.raises(KeyError, match="orientation"):
        portrait_from_segment_row({})


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
        ({"property:capture:orientation": ["portrait"], "property:capture:orientation_quarter_turns_ccw": [3]}, 1),
        ({"property:capture:orientation": "landscape"}, 0),
    ],
)
def test_landscape_quarter_turns_undoes_the_stored_rotation(row: dict[str, object], expected: int) -> None:
    """Return the counter-clockwise rotation that restores stored frames to landscape; landscape needs no turns property."""
    assert landscape_quarter_turns(row) == expected
