"""ONNX Runtime's CUDA provider is checked by running a kernel, not by listing providers.

conda-forge's CUDA 13 build lists ``CUDAExecutionProvider`` on the GB10 (sm_121) and creates sessions there, but has no
kernel image for that GPU: the first inference fails (2026-11-27 smoke run, ``cudaErrorNoKernelImageForDevice``).
"""

import sys
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from trtkit.onnx_cuda import require_cuda_kernels


@pytest.fixture(autouse=True)
def _fresh_probe() -> Iterator[None]:
    """The probe caches a passing device per process; every case here starts uncached."""
    require_cuda_kernels.cache_clear()
    yield
    require_cuda_kernels.cache_clear()


class _Session:
    def __init__(self, model: bytes, providers: list[tuple[str, dict[str, int]]], taken_by: str, fail_with: str | None) -> None:
        assert model and providers == [("CUDAExecutionProvider", {"device_id": 0})]
        self._taken_by: str = taken_by
        self._fail_with: str | None = fail_with

    def get_providers(self) -> list[str]:
        return [self._taken_by]

    def run(self, output_names: object, feeds: dict[str, object]) -> list[object]:
        if self._fail_with is not None:
            raise RuntimeError(self._fail_with)
        return [feeds["x"]]


def _fake_ort(monkeypatch: pytest.MonkeyPatch, fail_with: str | None = None, taken_by: str = "CUDAExecutionProvider", listed: bool = True) -> None:
    def session(model: bytes, sess_options: SimpleNamespace, providers: list[tuple[str, dict[str, int]]]) -> _Session:
        assert sess_options.log_severity_level == 3, "the probe runs quiet, as the runtime's own session does"
        return _Session(model, providers, taken_by, fail_with)

    available: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"] if listed else ["CPUExecutionProvider"]
    fake: SimpleNamespace = SimpleNamespace(InferenceSession=session, SessionOptions=SimpleNamespace, get_available_providers=lambda: available)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)


def test_a_provider_without_kernels_for_this_gpu_points_at_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ort(monkeypatch, fail_with="CUDA error cudaErrorNoKernelImageForDevice:no kernel image is available for execution on the device")
    with pytest.raises(RuntimeError, match="TensorRtBackendConfig") as raised:
        require_cuda_kernels(device_id=0)
    assert "no kernel image" in str(raised.value)


def test_a_session_that_fell_back_to_the_cpu_proves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ort(monkeypatch, taken_by="CPUExecutionProvider")
    with pytest.raises(RuntimeError, match="fell back"):
        require_cuda_kernels(device_id=0)


def test_an_unlisted_provider_says_so_before_any_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ort(monkeypatch, listed=False)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        require_cuda_kernels(device_id=0)


def test_a_provider_that_runs_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ort(monkeypatch)
    require_cuda_kernels(device_id=0)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_installed_provider_runs_here_except_on_the_gb10() -> None:
    if torch.cuda.get_device_capability(0) == (12, 1):
        with pytest.raises(RuntimeError, match="TensorRtBackendConfig"):
            require_cuda_kernels(device_id=0)
    else:
        require_cuda_kernels(device_id=0)
