"""Instance masks keep sparse uint8 IDs without sorting every image."""

import io

import numpy as np
import pytest
from PIL import Image

from dataforge.datasets.mamma_syn import instance_mask_pngs


def test_sparse_instance_ids_use_bounded_id_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """IDs 1 and 255 remain distinct; zero stays transparent background."""
    source: io.BytesIO = io.BytesIO()
    Image.fromarray(np.array([[0, 1, 255], [255, 0, 1]], dtype=np.uint8)).save(source, format="PNG")

    def reject_sort(*_args, **_kwargs) -> None:
        raise AssertionError("sorting all mask pixels")

    monkeypatch.setattr(np, "unique", reject_sort)
    masks: dict[int, bytes] = instance_mask_pngs(source.getvalue(), (3, 2), 0)
    assert list(masks) == [0, 254]
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(masks[0]))), [[0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(masks[254]))), [[0, 0, 1], [1, 0, 0]])
