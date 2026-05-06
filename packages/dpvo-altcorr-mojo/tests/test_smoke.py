from __future__ import annotations

import pytest
import torch

pytest.importorskip("max", reason="MAX is required for Mojo custom ops")
backend = pytest.importorskip("dpvo_altcorr_mojo.backend", reason="dpvo_altcorr_mojo is not importable")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Mojo custom ops")


def test_smoke_scale_custom_op() -> None:
    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = backend.smoke_scale(x)
    torch.cuda.synchronize()
    assert torch.equal(y, x * 2.0)
