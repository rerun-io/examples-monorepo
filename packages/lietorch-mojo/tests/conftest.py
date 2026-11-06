"""Shared pytest setup: device and dtype params, and the module-level skip for an unbuilt Mojo extension.

pytest cannot skip from a conftest import, so each test module that imports
`lietorch_mojo` calls `require_mojo_extension()` at its top instead.
"""

import pytest
import torch


def require_mojo_extension() -> None:
    """Skip the calling test module when the Mojo extension is not built (a fresh checkout: the *.so is gitignored)."""
    pytest.importorskip(
        "lietorch_mojo._lietorch_mojo_backends",
        reason="build the Mojo extension first (see packages/lietorch-mojo/README.md)",
    )


# CPU runs in the unit lane; CUDA needs a GPU, so it is an integration test.
DEVICES: list = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")],
    ),
]
DTYPES: list = [torch.float32, torch.float64]
