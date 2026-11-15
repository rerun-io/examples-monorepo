"""Raw backend ops: lietorch_mojo.backends vs frozen upstream lietorch_backends outputs, all four groups.

Every forward op and every backward op (fed the same upstream gradients) on
the input sets of `cases.py`, float32 and float64, CPU and CUDA. The inputs
and lietorch's outputs come from `fixtures/lietorch_references.rrd`
(`freeze_references.py`), so no lietorch is needed; CPU runs in the unit
lane, CUDA is integration.

Error metric: `parity.row_error`. Tolerances: 1e-5 (float32), 1e-10 (float64).

Reference build: SE3 and SO3 compare against lietorch on the device under
test. RxSO3 and Sim3 in float32 compare against lietorch's CPU build on every
device, and Mojo must also be no farther than lietorch (same device) from a
float64 truth on the same float32 inputs (`parity.F32_CPU_REFERENCE_GROUPS`).

Float32 rotation angles between lietorch's EPS (1e-6) and ~1e-2 are
ill-conditioned for the SE3 Jacobians (see se3.mojo calcQ); lietorch's own
float32 error there reaches 1e-4 (CUDA) to 1e-1 (CPU). That zone is tested
against a float64 reference instead: Mojo's float32 error must not exceed
lietorch's float32 error on the same device (plus 1e-5).
"""

import pytest
import torch
from conftest import DEVICES, DTYPES, require_mojo_extension

require_mojo_extension()

from parity import (  # noqa: E402
    ALL_OPS,
    DTYPE_NAMES,
    F32_CPU_REFERENCE_GROUPS,
    TOLERANCE,
    OpInputs,
    call_op,
    reference_device,
    row_error,
)
from references import Record, entities, record  # noqa: E402
from torch import Tensor  # noqa: E402

import lietorch_mojo  # noqa: E402
from lietorch_mojo import backends  # noqa: E402

GROUP_NAMES: tuple[str, ...] = ("SO3", "RxSO3", "SE3", "Sim3")
CASE_PARAMS: list = [pytest.param(e, id=e.removeprefix("/ops/").replace("/", "-")) for g in GROUP_NAMES for e in entities(f"/ops/{g}")]


def _inputs(rec: Record, dtype: torch.dtype, device: str) -> OpInputs:
    prefix: str = f"in/{DTYPE_NAMES[dtype]}/"
    return OpInputs.from_named({k.removeprefix(prefix): v for k, v in rec.tensors.items() if k.startswith(prefix)}, device)


def _on(tensors: list[Tensor], device: str) -> list[Tensor]:
    return [t.to(device) for t in tensors]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", ALL_OPS)
@pytest.mark.parametrize("entity", CASE_PARAMS)
def test_op_matches_lietorch(entity: str, op: str, dtype: torch.dtype, device: str) -> None:
    group: str = entity.split("/")[2]
    G = getattr(lietorch_mojo, group)
    rec: Record = record(entity)
    inputs: OpInputs = _inputs(rec, dtype, device)
    ours: list[Tensor] = call_op(backends, G, op, inputs)
    assert all(bool(torch.isfinite(o).all()) for o in ours)
    ref_device: str = reference_device(dtype, device, group, F32_CPU_REFERENCE_GROUPS)  # module docstring
    ref: list[Tensor] = _on(rec.outputs(f"out/{ref_device}/{DTYPE_NAMES[dtype]}/{op}"), ref_device)
    err: float = row_error(_on(ours, ref_device), ref)
    assert err <= TOLERANCE[dtype], f"{entity} {op} {dtype} {device}: row error {err:.3e}"
    if dtype == torch.float32 and group in F32_CPU_REFERENCE_GROUPS:
        truth: list[Tensor] = _on([t.float() for t in rec.outputs(f"truth/{device}/{op}")], device)
        ours_err: float = row_error(ours, truth)
        lietorch_err: float = row_error(_on(rec.outputs(f"out/{device}/float32/{op}"), device), truth)
        assert ours_err <= lietorch_err + TOLERANCE[dtype], f"{entity} {op} {device}: Mojo {ours_err:.3e} vs lietorch {lietorch_err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ALL_OPS)
@pytest.mark.parametrize("group", ("SE3", "SO3"))
def test_small_angle_float32_no_less_accurate_than_lietorch(group: str, op: str, device: str) -> None:
    G = getattr(lietorch_mojo, group)
    rec: Record = record(f"/ops_small_angle/{group}")
    ours: list[Tensor] = call_op(backends, G, op, _inputs(rec, torch.float32, device))
    # float64 truth evaluated on exactly the float32-rounded inputs.
    truth32: list[Tensor] = _on([t.float() for t in rec.outputs(f"truth/{device}/{op}")], device)
    ref: list[Tensor] = _on(rec.outputs(f"out/{device}/float32/{op}"), device)
    ours_err: float = row_error(ours, truth32)
    lietorch_err: float = row_error(ref, truth32)
    assert ours_err <= lietorch_err + 1e-5, f"{op} {device}: Mojo {ours_err:.3e} vs lietorch {lietorch_err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group", GROUP_NAMES)
def test_backward_tail_slots_are_zero(group: str, dtype: torch.dtype, device: str) -> None:
    """Gradients w.r.t. group elements live in the first K of N slots; the rest are zero (as lietorch)."""
    G = getattr(lietorch_mojo, group)
    N, K = G.embedded_dim, G.manifold_dim
    inputs: OpInputs = _inputs(record(f"/ops/{group}/random"), dtype, device)
    for op in ("logm_backward", "inv_backward", "mul_backward", "adj_backward", "adjT_backward", "act_backward", "act4_backward"):
        outs: list[Tensor] = call_op(backends, G, op, inputs)
        # dX comes first; mul_backward's dY is a group gradient too (act4's dp is 4 wide, like an SO3 row, but is not).
        group_grads: list[Tensor] = outs[:2] if op == "mul_backward" else outs[:1]
        for g in group_grads:
            assert g.shape[-1] == N, op
            assert torch.all(g[:, K:] == 0), op
