"""The public API on all four groups against frozen upstream lietorch outputs: forward, gradients, LieGroupParameter.

- Forward: every public method through the API, including broadcasting shapes
  (`parity.api_forward_outputs`; `scale` is SE3-only).
- Backward: autograd gradients of composite functions (`parity.parity_composites`)
  in two input suites (`parity.COMPOSITE_SUITES`), gradients of broadcast
  operands, LieGroupParameter gradients and the examples/pgo SGD trajectory.
- No kernels: repr, setitem and shapes, and the group conversions.

Inputs and lietorch's outputs come from `fixtures/lietorch_references.rrd`
(`freeze_references.py`); CPU runs in the unit lane, CUDA is integration.
Error metric: `parity.max_error`. Tolerances: 1e-5 for float32, 1e-10 for float64.
"""

import pytest
import torch
from conftest import DEVICES, DTYPES, LIETORCH_BUILD, require_mojo_extension

require_mojo_extension()

from groups_common import GROUP_NAMES  # noqa: E402
from parity import (  # noqa: E402
    BROADCAST_SHAPES,
    COMPOSITE_SUITES,
    CONVERSIONS,
    DTYPE_NAMES,
    TOLERANCE,
    CompositeSuite,
    api_forward_outputs,
    broadcast_gradients,
    composite_gradients,
    lie_group_parameter_gradient,
    max_error,
    parity_composites,
    reference_device,
    sgd_trajectory,
    shapes_id,
)
from references import Record, record  # noqa: E402
from torch import Tensor  # noqa: E402

import lietorch_mojo  # noqa: E402

F64: torch.dtype = torch.float64


def _prefixed(rec: Record, prefix: str, device: str) -> dict[str, Tensor]:
    """The record's tensors under `prefix/`, prefix stripped, on `device`."""
    return {k.removeprefix(prefix + "/"): v.to(device) for k, v in rec.tensors.items() if k.startswith(prefix + "/")}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", GROUP_NAMES)
@pytest.mark.parametrize("shapes", BROADCAST_SHAPES, ids=[shapes_id(s) for s in BROADCAST_SHAPES])
def test_api_forward_matches(shapes: tuple[tuple[int, ...], tuple[int, ...]], name: str, dtype: torch.dtype, device: str) -> None:
    rec: Record = record(f"/api/forward/{name}/{shapes_id(shapes)}")
    ours: dict[str, Tensor] = api_forward_outputs(lietorch_mojo, name, _prefixed(rec, f"in/{DTYPE_NAMES[dtype]}", device))
    ref: dict[str, Tensor] = _prefixed(rec, f"out/{LIETORCH_BUILD[device]}/{DTYPE_NAMES[dtype]}", "cpu")
    assert ours.keys() == ref.keys()
    for k, r in ref.items():
        err: float = max_error(ours[k].cpu(), r)
        assert err <= TOLERANCE[dtype], f"{k}: {err:.3e}"


COMPOSITE_PARAMS: list = [
    pytest.param(suite, name, composite, id=f"{suite.name}-{name}-{composite}")
    for suite in COMPOSITE_SUITES
    for name in GROUP_NAMES
    for composite in parity_composites(name)
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("suite", "name", "composite"), COMPOSITE_PARAMS)
def test_autograd_gradients_match(suite: CompositeSuite, name: str, composite: str, dtype: torch.dtype, device: str) -> None:
    rec: Record = record(f"/api/composite/{suite.name}/{name}/{composite}")
    inputs: dict[str, Tensor] = _prefixed(rec, f"in/{DTYPE_NAMES[dtype]}", device)
    weight: Tensor = inputs.pop("weight")

    def stored_weight(shape: torch.Size) -> Tensor:
        assert shape == weight.shape
        return weight

    ours: list[Tensor] = composite_gradients(lietorch_mojo, name, composite, inputs, stored_weight)
    ref_device: str = reference_device(dtype, LIETORCH_BUILD[device], name, suite.f32_cpu_reference_groups)  # parity.COMPOSITE_SUITES
    ref: list[Tensor] = rec.outputs(f"grad/{ref_device}/{DTYPE_NAMES[dtype]}")
    for i, (o, r) in enumerate(zip(ours, ref, strict=True)):
        err: float = max_error(o.cpu(), r)
        assert err <= TOLERANCE[dtype], f"{composite} input {i}: {err:.3e}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUP_NAMES)
def test_broadcast_gradients_match(name: str, device: str) -> None:
    """Gradients of operands that broadcast ((4, 1) against (1, 6)) are reduced the same way as upstream."""
    rec: Record = record(f"/api/broadcast_gradients/{name}")
    ours: list[Tensor] = broadcast_gradients(lietorch_mojo, name, _prefixed(rec, "in", device))
    for i, (o, r) in enumerate(zip(ours, rec.outputs(f"grad/{device}"), strict=True)):
        assert max_error(o.cpu(), r) <= TOLERANCE[F64], f"input {i}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUP_NAMES)
def test_lie_group_parameter_gradients_match(name: str, device: str) -> None:
    rec: Record = record(f"/api/lie_group_parameter/{name}")
    ours: Tensor = lie_group_parameter_gradient(lietorch_mojo, name, _prefixed(rec, "in", device))
    assert max_error(ours.cpu(), rec.tensor(f"grad/{device}")) <= TOLERANCE[F64]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", GROUP_NAMES)
def test_lie_group_parameter_sgd_trajectory_matches(name: str, device: str) -> None:
    """The upstream examples/pgo optimisation (SGD + momentum on a LieGroupParameter) follows the same path."""
    rec: Record = record(f"/api/sgd/{name}")
    grad0, final = sgd_trajectory(lietorch_mojo, name, _prefixed(rec, "in", device))
    assert max_error(grad0.cpu(), rec.tensor(f"grad0/{device}")) <= TOLERANCE[F64]
    assert max_error(final.detach().cpu(), rec.tensor(f"final/{device}")) <= 1e-8


@pytest.mark.parametrize("name", GROUP_NAMES)
def test_repr_setitem_and_shapes_match(name: str) -> None:
    rec: Record = record(f"/api/repr/{name}")
    data: Tensor = rec.tensor("in/x")
    G = getattr(lietorch_mojo, name)
    assert repr(G(data)) == rec.texts["repr"]
    X = G(data.clone())
    X[1:3] = G(data[:2])
    assert torch.equal(X.data[1:3], data[:2])
    assert X.shape == torch.Size([4])
    assert X.tangent_shape == torch.Size([4, G.manifold_dim])
    assert (X.embedded_dim, X.manifold_dim) == (G.embedded_dim, G.manifold_dim)


def test_group_conversions_match() -> None:
    """Constructors that take another group (no kernels involved), Sim3 = SE3 at unit scale, and SE3.scale."""
    rec: Record = record("/api/conversions")
    inputs: dict[str, Tensor] = _prefixed(rec, "in", "cpu")
    for k, build in CONVERSIONS.items():
        assert torch.equal(build(lietorch_mojo, inputs), rec.tensor(f"out/{k}")), k
