"""The Mojo scalar math (sin, cos, atan, sqrt) against float64 references, on CPU and GPU.

The math is reached through a test-only extension, `tests/mojo/math_probe.mojo`,
built here on first use (about 4 s) and cached under `tests/mojo/.build/` by
source hash, so the production extension carries no test hooks.
"""

import hashlib
import importlib.util
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import torch
from conftest import DEVICES, DTYPES
from torch import Tensor

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
MOJO_DIR: Path = PACKAGE_DIR / "lietorch_mojo" / "mojo"
PROBE_SOURCE: Path = PACKAGE_DIR / "tests" / "mojo" / "math_probe.mojo"


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    mojo: str | None = shutil.which("mojo")
    if mojo is None:
        pytest.skip("needs the `mojo` compiler on PATH to build tests/mojo/math_probe.mojo (e.g. the mast3r-slam-dev env)")
    digest = hashlib.sha256()
    for source in [PROBE_SOURCE, *sorted(MOJO_DIR.glob("*.mojo"))]:
        digest.update(source.read_bytes())
    built: Path = PACKAGE_DIR / "tests" / "mojo" / ".build" / digest.hexdigest()[:16] / "math_probe.so"
    if not built.exists():
        built.parent.mkdir(parents=True, exist_ok=True)
        partial: Path = built.with_name(f"math_probe.{os.getpid()}.so")
        cmd: list[str] = [mojo, "build", "--emit", "shared-lib", "-I", str(MOJO_DIR), "-o", str(partial), str(PROBE_SOURCE)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"building the math probe failed:\n{result.stdout}\n{result.stderr}"
        partial.replace(built)
    spec = importlib.util.spec_from_file_location("math_probe", built)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _math_probe(probe: ModuleType, x: Tensor) -> Tensor:
    """x (B,) -> (B, 4) of [sin, cos, atan, sqrt(|x|)] from lie_math.mojo, on x's device."""
    xc: Tensor = x.contiguous().view(-1, 1)
    out: Tensor = xc.new_empty(xc.shape[0], 4)
    is_f64: bool = xc.dtype == torch.float64
    if xc.is_cuda:
        torch.cuda.synchronize()  # the probe runs on its own stream
        probe.probe_cuda(is_f64, xc.shape[0], xc.data_ptr(), out.data_ptr())
    else:
        probe.probe_cpu(is_f64, xc.shape[0], xc.data_ptr(), out.data_ptr())
    return out


def _probe_inputs() -> Tensor:
    xs: list[Tensor] = [
        torch.linspace(-20.0, 20.0, 40001, dtype=torch.float64),
        torch.logspace(-12, 0, 2000, dtype=torch.float64),
        -torch.logspace(-12, 0, 2000, dtype=torch.float64),
        torch.tensor([0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi, 0.4375, 0.6875, 1.1875, 2.4375, 1e3, 4e3, 1e6, 1e10], dtype=torch.float64),
    ]
    return torch.cat(xs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_math_probe_within_one_ulp(probe: ModuleType, device: str, dtype: torch.dtype) -> None:
    x: Tensor = _probe_inputs().to(dtype)
    # Reduction ranges documented in lie_math.mojo: |x| < 8192 (float32), < 2^20 * pi/2 (float64).
    limit: float = 8192.0 if dtype == torch.float32 else 1.6e6
    trig_mask: Tensor = x.abs() < limit
    out: Tensor = _math_probe(probe, x.to(device)).cpu().double()
    xd: Tensor = x.double()
    ref: Tensor = torch.stack([torch.sin(xd), torch.cos(xd), torch.atan(xd), xd.abs().sqrt()], dim=-1)
    eps: float = torch.finfo(dtype).eps
    # Error in units of eps * max(|ref|, tiny): sin/cos measured relative to max(|ref|, eps) near their zeros.
    scale: Tensor = ref.abs().clamp_min(torch.finfo(dtype).tiny)
    scale[:, :2] = ref[:, :2].abs().clamp_min(eps)
    err_ulps: Tensor = (out - ref).abs() / (scale * eps)
    assert err_ulps[trig_mask][:, :2].max().item() <= 1.0
    assert err_ulps[:, 2].max().item() <= 1.0
    assert err_ulps[:, 3].max().item() <= 0.5


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_math_cpu_and_gpu_agree(probe: ModuleType, dtype: torch.dtype) -> None:
    """Same Mojo source on both devices: sin, cos and sqrt are bitwise equal; atan within one ulp.

    (The CPU and NVPTX backends contract a few multiply-adds in atan differently.)
    """
    x: Tensor = _probe_inputs().to(dtype)
    cpu: Tensor = _math_probe(probe, x)
    gpu: Tensor = _math_probe(probe, x.cuda()).cpu()
    assert torch.equal(cpu[:, [0, 1, 3]], gpu[:, [0, 1, 3]])
    ulp: Tensor = torch.finfo(dtype).eps * cpu[:, 2].abs().clamp_min(torch.finfo(dtype).tiny)
    assert ((cpu[:, 2] - gpu[:, 2]).abs() <= ulp).all()


def test_production_extension_has_no_test_hooks() -> None:
    """The math probe lives only in the test extension (verify-idiomatic-mojo r2, §12)."""
    sources: str = "\n".join(p.read_text() for p in MOJO_DIR.glob("*.mojo"))
    assert "math_probe" not in sources and "MATH_PROBE" not in sources
