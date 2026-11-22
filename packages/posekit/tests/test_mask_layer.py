"""The mask layer contract's leaf names: consumers on other packages match these strings."""

from posekit.mask_layer import MASK_NAME, MATTE_NAME


def test_leaf_names_are_pinned() -> None:
    assert MASK_NAME == "people_mask"
    assert MATTE_NAME == "people_matte"
