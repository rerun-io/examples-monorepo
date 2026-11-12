"""A SOMA-X stream window whose rows all lack a source fit is written, every row untrusted (smplx cannot pose an empty batch)."""

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

pytest.importorskip("soma", reason="SOMA-X is installed only in the dataforge-soma-x environments")
torch = pytest.importorskip("torch")

from dataforge import soma_x  # noqa: E402
from dataforge.apis import soma_x as api  # noqa: E402
from dataforge.soma_x_fit import HandFitter  # noqa: E402

pytestmark = pytest.mark.integration


def test_a_window_without_a_present_row_is_written_untrusted() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA on the host")
    rows: int = 5
    temporal: dict[str, Float32[ndarray, "t c"]] = {"betas": np.full((rows, 10), np.nan, dtype=np.float32)}
    window: api.SourceWindow = api.empty_window(
        np.arange(rows, dtype=np.int64), np.arange(rows, dtype=np.int64), np.zeros(rows, dtype=bool), 778, temporal, slice(0, rows)
    )
    fit: soma_x.SomaXFit = api.fit_window(HandFitter("right"), window)
    assert not fit.trusted.any()
    assert fit.vertices.shape[0] == 0
    assert np.isnan(fit.fit_error_mm).all()
