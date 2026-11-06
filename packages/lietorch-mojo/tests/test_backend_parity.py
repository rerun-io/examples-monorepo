"""Raw backend ops: lietorch_mojo.backends vs lietorch's compiled lietorch_backends, SE3 and SO3.

Every forward op and every backward op (fed the same upstream gradients) on
the input sets of `cases.py`, float32 and float64, CPU and CUDA.

Error metric (`row_error`): per element, max |ours - lietorch| over its output
row, divided by max(|lietorch row|, 1); the test reports the worst element.
Tolerances: 1e-5 (float32), 1e-10 (float64).

Float32 rotation angles between lietorch's EPS (1e-6) and ~1e-2 are
ill-conditioned for the SE3 Jacobians (see se3.mojo calcQ); lietorch's own
float32 error there reaches 1e-4 (CUDA) to 1e-1 (CPU). That zone is tested
against a float64 reference instead: Mojo's float32 error must not exceed
lietorch's float32 error on the same device (plus 1e-5).
"""

from dataclasses import replace

import pytest
import torch

pytest.importorskip("lietorch_backends", reason="needs the compiled lietorch (ai-demos build) for parity")
from conftest import require_mojo_extension  # noqa: E402

require_mojo_extension()

import lietorch_backends  # noqa: E402
from cases import Se3Case, se3_cases, se3_ill_conditioned_f32_case, so3_case, so3_cases  # noqa: E402
from conftest import DEVICES, DTYPES  # noqa: E402
from parity import BACKWARD_OPS, FORWARD_OPS, GROUP_DIMS, SE3_ID, SO3_ID, OpInputs, call_op, make_inputs  # noqa: E402
from torch import Tensor  # noqa: E402

from lietorch_mojo import backends  # noqa: E402

pytestmark = pytest.mark.integration

TOLERANCE: dict[torch.dtype, float] = {torch.float32: 1e-5, torch.float64: 1e-10}
CASES: list[Se3Case] = se3_cases()
GROUP_IDS: list[int] = [SE3_ID, SO3_ID]
GROUP_ID_NAMES: list[str] = ["SE3", "SO3"]
# (group id, case) for every group with a Mojo backend.
GROUP_CASES: list[tuple[int, Se3Case]] = [(SE3_ID, c) for c in CASES] + [(SO3_ID, c) for c in so3_cases()]
GROUP_CASE_IDS: list[str] = [f"{'SE3' if g == SE3_ID else 'SO3'}-{c.name}" for g, c in GROUP_CASES]
ILL_CONDITIONED: dict[int, Se3Case] = {SE3_ID: se3_ill_conditioned_f32_case(), SO3_ID: so3_case(se3_ill_conditioned_f32_case())}


def row_error(ours: list[Tensor], ref: list[Tensor]) -> float:
    worst: float = 0.0
    for o, r in zip(ours, ref, strict=True):
        assert o.shape == r.shape and o.dtype == r.dtype and o.device == r.device
        od: Tensor = o.double().reshape(o.shape[0], -1)
        rd: Tensor = r.double().reshape(r.shape[0], -1)
        if od.numel() == 0:
            continue
        err: Tensor = (od - rd).abs().amax(dim=1) / rd.abs().amax(dim=1).clamp_min(1.0)
        worst = max(worst, err.max().item())
    return worst


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op", FORWARD_OPS + BACKWARD_OPS)
@pytest.mark.parametrize("group_case", GROUP_CASES, ids=GROUP_CASE_IDS)
def test_op_matches_lietorch(group_case: tuple[int, Se3Case], op: str, dtype: torch.dtype, device: str) -> None:
    group_id, case = group_case
    inputs: OpInputs = make_inputs(case, dtype, device, group_id=group_id)
    ref: list[Tensor] = call_op(lietorch_backends, op, inputs, group_id)
    ours: list[Tensor] = call_op(backends, op, inputs, group_id)
    assert all(bool(torch.isfinite(o).all()) for o in ours)
    err: float = row_error(ours, ref)
    assert err <= TOLERANCE[dtype], f"{op} {case.name} {dtype} {device}: row error {err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", FORWARD_OPS + BACKWARD_OPS)
@pytest.mark.parametrize("group_id", GROUP_IDS, ids=GROUP_ID_NAMES)
def test_small_angle_float32_no_less_accurate_than_lietorch(group_id: int, op: str, device: str) -> None:
    case: Se3Case = ILL_CONDITIONED[group_id]
    f32: OpInputs = make_inputs(case, torch.float32, device, group_id=group_id)
    # float64 truth evaluated on exactly the float32-rounded inputs.
    f64: OpInputs = replace(
        f32,
        X=f32.X.double(),
        Y=f32.Y.double(),
        a=f32.a.double(),
        p3=f32.p3.double(),
        p4=f32.p4.double(),
        grads={k: v.double() for k, v in f32.grads.items()},
    )
    truth: list[Tensor] = call_op(lietorch_backends, op, f64, group_id)
    ref: list[Tensor] = [t.float() for t in call_op(lietorch_backends, op, f32, group_id)]
    ours: list[Tensor] = call_op(backends, op, f32, group_id)
    truth32: list[Tensor] = [t.float() for t in truth]
    ours_err: float = row_error(ours, truth32)
    lietorch_err: float = row_error(ref, truth32)
    assert ours_err <= lietorch_err + 1e-5, f"{op} {device}: Mojo {ours_err:.3e} vs lietorch {lietorch_err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_id", GROUP_IDS, ids=GROUP_ID_NAMES)
def test_backward_tail_slots_are_zero(group_id: int, dtype: torch.dtype, device: str) -> None:
    """Gradients w.r.t. group elements live in the first K of N slots; the rest are zero (as lietorch)."""
    N, K = GROUP_DIMS[group_id]
    case: Se3Case = CASES[0] if group_id == SE3_ID else so3_case(CASES[0])
    inputs: OpInputs = make_inputs(case, dtype, device, group_id=group_id)
    for op in ("logm_backward", "inv_backward", "mul_backward", "adj_backward", "adjT_backward", "act_backward", "act4_backward"):
        outs: list[Tensor] = call_op(backends, op, inputs, group_id)
        # dX comes first; mul_backward's dY is a group gradient too (act4's dp is 4 wide, like an SO3 row, but is not).
        group_grads: list[Tensor] = outs[:2] if op == "mul_backward" else outs[:1]
        for g in group_grads:
            assert g.shape[-1] == N, op
            assert torch.all(g[:, K:] == 0), op
