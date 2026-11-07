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


# CPU runs in the unit lane; CUDA and MPS need a GPU, so they are integration tests.
DEVICES: list = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")],
    ),
    pytest.param(
        "mps",
        marks=[pytest.mark.integration, pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")],
    ),
]
DTYPES: list = [torch.float32, torch.float64]
# The lietorch build whose frozen outputs a device compares against: lietorch has no MPS build, and the Mojo
# Metal kernels evaluate the GPU formulas, so MPS compares against lietorch's CUDA build.
LIETORCH_BUILD: dict[str, str] = {"cpu": "cpu", "cuda": "cuda", "mps": "cuda"}


# Tests in float64 only (gradcheck, exact-Jacobian identities, float64 gradient references): nothing to run on MPS.
FLOAT64_MODULES: frozenset[str] = frozenset({"test_gradcheck", "test_rxso3_sim3_gradcheck", "test_groups_generic"})
FLOAT64_TESTS: frozenset[str] = frozenset(
    {"test_broadcast_gradients_match", "test_lie_group_parameter_gradients_match", "test_lie_group_parameter_sgd_trajectory_matches"}
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """MPS has no float64 (neither PyTorch's MPS backend nor Metal): skip float64 cases and float64-only modules there."""
    for item in items:
        params: dict = getattr(getattr(item, "callspec", None), "params", {})
        if params.get("device") != "mps":
            continue
        if params.get("dtype") is torch.float64 or item.module.__name__ in FLOAT64_MODULES or item.originalname in FLOAT64_TESTS:
            item.add_marker(pytest.mark.skip(reason="MPS has no float64"))
        elif item.originalname == "test_autograd_gradients_match" and params.get("name") == "Sim3" and params["suite"].name == "unit":
            # Open (2026-11-07, Apple M4): Sim3 float32 composite gradients at unit-scale tangents differ from
            # lietorch by up to 2.2e-4 on MPS (1e-5 bound); every backend op passes 1e-5 on its own. Not yet
            # attributed (Metal float32 exp/log, or MPS's own ops in the composite). DPVO uses SE3 only.
            item.add_marker(pytest.mark.xfail(strict=True, reason="Sim3 float32 composite gradients on MPS: open"))
