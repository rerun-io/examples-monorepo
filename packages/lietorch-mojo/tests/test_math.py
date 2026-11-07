"""The scalar math the group code uses (sincos_, sqrt_ from lie_math.mojo; std.math.atan) against exact results.

The math is reached through a test-only extension, `tests/mojo/math_probe.mojo`,
built here on first use (about 4 s) and cached under `tests/mojo/.build/` by
source hash, so the production extension carries no test hooks. Errors are
measured in ulps of the exact result: float32 against float64 torch, float64
against mpmath at 50 digits.
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

from lietorch_mojo.metal import gpu_address

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
    """x (B,) -> (B, 4) of [sin, cos, atan, sqrt(|x|)] on x's device."""
    xc: Tensor = x.contiguous().view(-1, 1)
    out: Tensor = xc.new_empty(xc.shape[0], 4)
    is_f64: bool = xc.dtype == torch.float64
    if xc.is_cuda:
        torch.cuda.synchronize()  # the probe runs on its own stream
        probe.probe_cuda(is_f64, xc.shape[0], xc.data_ptr(), out.data_ptr())
    elif xc.is_mps:
        torch.mps.synchronize()
        probe.probe_cuda(is_f64, xc.shape[0], gpu_address(xc), gpu_address(out))
    else:
        probe.probe_cpu(is_f64, xc.shape[0], xc.data_ptr(), out.data_ptr())
    return out


def _probe_inputs(dtype: torch.dtype) -> Tensor:
    """Small, medium, near-multiple-of-pi/2 and huge arguments, plus the hardest reduction cases."""
    g = torch.Generator().manual_seed(0)
    top: float = 300.0 if dtype == torch.float64 else 38.0
    xs: list[Tensor] = [
        torch.linspace(-20.0, 20.0, 4001, dtype=torch.float64),
        torch.logspace(-12, 0, 500, dtype=torch.float64),
        -torch.logspace(-12, 0, 500, dtype=torch.float64),
        torch.randint(1, 2**30, (1000,), generator=g).double() * (math.pi / 2),
        10.0 ** (torch.rand(2000, generator=g, dtype=torch.float64) * (top - 3.0) + 3.0),
        torch.tensor(
            [
                0.0,
                math.pi / 4,
                math.pi / 2,
                math.pi,
                0.4375,
                0.6875,
                1.1875,
                2.4375,
                1024.0,
                1e6,
                1647099.3291652855,
                1e18,
                1e22,
                # Closest float32 values to multiples of pi/2 below 1024, and the hardest float64 cases.
                252.89820861816406,
                505.7964172363281,
                1011.5928344726562,
                6381956970095103.0 * 2.0**797,
                5261692873635770.0 * 2.0**499,
            ],
            dtype=torch.float64,
        ),
    ]
    x = torch.cat(xs).to(dtype)
    x = torch.cat([x, -x])
    return x[torch.isfinite(x)]


def _exact(x: Tensor) -> Tensor:
    """(B, 4) exact [sin, cos, atan, sqrt(|x|)] as float64 (rounded once from the exact value)."""
    if x.dtype == torch.float32:
        xd: Tensor = x.double()
        return torch.stack([torch.sin(xd), torch.cos(xd), torch.atan(xd), xd.abs().sqrt()], dim=-1)
    mpmath = pytest.importorskip("mpmath", reason="float64 references need mpmath")
    mpmath.mp.dps = 50
    rows: list[list[float]] = []
    for v in x.tolist():
        m = mpmath.mpf(v)
        rows.append([float(mpmath.sin(m)), float(mpmath.cos(m)), float(mpmath.atan(m)), float(mpmath.sqrt(abs(m)))])
    return torch.tensor(rows, dtype=torch.float64)


def _ulps(out: Tensor, ref: Tensor, dtype: torch.dtype) -> Tensor:
    """|out - ref| in ulps of `dtype` at |ref| (with the subnormal floor)."""
    digits, min_exp = (24, -126) if dtype == torch.float32 else (53, -1022)
    _, exp = torch.frexp(ref)
    ulp: Tensor = torch.ldexp(torch.ones_like(ref), (exp - 1).clamp_min(min_exp) - (digits - 1))
    return (out.double() - ref).abs() / ulp


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_math_within_one_ulp(probe: ModuleType, device: str, dtype: torch.dtype) -> None:
    x: Tensor = _probe_inputs(dtype)
    if device == "mps":
        # No float64 on Metal: sincos_ above |x| = 1024 is std.math's float32 sin/cos there (lie_math.mojo).
        x = x[x.abs() <= 1024.0]
    out: Tensor = _math_probe(probe, x.to(device)).cpu()
    err: Tensor = _ulps(out, _exact(x), dtype)
    assert torch.isfinite(out).all()
    assert err[:, :3].max().item() <= 1.0, f"max ulp {err[:, :3].amax(dim=0).tolist()} at x={x[err[:, :3].amax(dim=1).argmax()].item()}"
    assert err[:, 3].max().item() <= 0.5


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_math_cpu_and_gpu_agree(probe: ModuleType, dtype: torch.dtype) -> None:
    """Same Mojo source on both devices: sqrt is bitwise equal; sin, cos and atan within one ulp of each other.

    (The CPU and NVPTX backends contract polynomial multiply-adds differently.)
    """
    x: Tensor = _probe_inputs(dtype)
    cpu: Tensor = _math_probe(probe, x)
    gpu: Tensor = _math_probe(probe, x.cuda()).cpu()
    assert torch.equal(cpu[:, 3], gpu[:, 3])
    assert (_ulps(gpu[:, :3], cpu[:, :3].double(), dtype) <= 1.0).all()


def test_production_extension_has_no_test_hooks() -> None:
    """The math probe lives only in the test extension (verify-idiomatic-mojo r2, §12)."""
    sources: str = "\n".join(p.read_text() for p in MOJO_DIR.glob("*.mojo"))
    assert "math_probe" not in sources and "MATH_PROBE" not in sources
