"""The compiled kernels (vipe-src's vipe_ext, packages/dpvo's DPVO kernels through dpvo.kernels) load and compute (GPU)."""

import importlib
from types import ModuleType

import pytest

torch = pytest.importorskip("torch", reason="torch comes from the metrictrail env's cuda feature")
pytest.importorskip("vipe_ext", reason="vipe_ext is compiled by recipes/vipe-src (metrictrail env)")

from metrictrail import kernels  # noqa: E402
from metrictrail.kernels import KernelCheck  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cuda() -> "torch.device":
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device (asset: a GPU)")
    return torch.device("cuda")


def test_vipe_ext_has_exactly_the_built_submodules() -> None:
    vipe_ext: ModuleType = importlib.import_module("vipe_ext")  # compiled, no stubs for pyrefly
    assert kernels.missing_vipe_ext_submodules() == []
    assert [name for name in kernels.VIPE_EXT_DROPPED if hasattr(vipe_ext, name)] == []


def test_vipe_dpvo_kernels_are_not_built() -> None:
    """MetricTrail's DPVO kernels are packages/dpvo's (dpvo.kernels); vipe-src builds no DPVO extension package."""
    with pytest.raises(ImportError):
        importlib.import_module("vipe.metrictrail.dpvo_ext")


@pytest.mark.parametrize("check", [kernels.check_scatter_max, kernels.check_patchify, kernels.check_reproject])
def test_kernel_matches_torch_reference(check, cuda) -> None:
    result: KernelCheck = check(cuda)
    assert result.max_abs_error < 1e-4, result
