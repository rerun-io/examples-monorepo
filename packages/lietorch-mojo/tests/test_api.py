"""Public API behaviour that needs no upstream lietorch (CPU, unit lane)."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from conftest import require_mojo_extension
from torch import Tensor

require_mojo_extension()

import lietorch_mojo  # noqa: E402
from lietorch_mojo import SE3, SO3, Sim3, backends  # noqa: E402


def test_identity_accepts_ints_tuples_and_lists() -> None:
    for X in (SE3.Identity(2, 3), SE3.Identity((2, 3)), SE3.Identity([2, 3])):
        assert X.shape == torch.Size([2, 3])
        assert torch.equal(X.data[1, 2], SE3.id_elem)
    X = SE3.Identity(4, dtype=torch.float64)
    assert X.dtype == torch.float64
    assert SE3.IdentityLike(X).shape == X.shape


def test_random_and_conversions() -> None:
    X = SE3.Random(3, 4, sigma=0.1, dtype=torch.float64)
    assert X.shape == torch.Size([3, 4]) and X.dtype == torch.float64
    q: Tensor = X.quaternion()
    assert torch.allclose(q.norm(dim=-1), torch.ones(3, 4, dtype=torch.float64))
    assert X.float(None).dtype == torch.float32
    assert X.double(None).dtype == torch.float64
    assert X.to(torch.float32).dtype == torch.float32
    assert X.cpu().device.type == "cpu"
    assert len(X.unbind(0)) == 3
    assert X.view((12,)).shape == torch.Size([12])
    assert X[1].shape == torch.Size([4])


def test_mul_dispatches_on_operand_type() -> None:
    X = SE3.Random(5, dtype=torch.float64)
    p: Tensor = torch.randn(5, 3, dtype=torch.float64)
    assert torch.allclose(X * p, X.act(p))
    assert torch.allclose((X * X).data, X.mul(X).data)


def test_conversions_between_groups() -> None:
    X = SE3.Random(2, dtype=torch.float64)
    assert torch.equal(SO3(X).data, X.data[..., 3:7])
    assert torch.equal(SE3(SO3(X)).data[..., 3:], X.data[..., 3:])
    assert torch.equal(Sim3(X).data[..., :7], X.data)
    assert torch.equal(Sim3(SO3(X)).data[..., 3:7], X.data[..., 3:7])


def test_lie_group_parameter_retracts_and_steps() -> None:
    X = SE3.Random(4, dtype=torch.float64)
    P = lietorch_mojo.LieGroupParameter(X)
    assert P.shape == torch.Size([4, 6]) and P.requires_grad
    assert torch.allclose(P.retr().data, X.data)
    P.add_(torch.zeros(4, 6, dtype=torch.float64), alpha=1.0)
    assert torch.allclose(P.group.data, X.data)


def test_backend_input_errors() -> None:
    X: Tensor = SE3.Identity(4, dtype=torch.float64).data
    with pytest.raises(RuntimeError, match="must be contiguous"):
        backends.logm(3, X.t().contiguous().t())
    with pytest.raises(TypeError):
        backends.logm(3, X.half())
    with pytest.raises(ValueError):
        backends.mul(3, X, X.float())
    with pytest.raises(ValueError, match="unknown lietorch group id"):
        backends.logm(7, torch.zeros(4, 5, dtype=torch.float64))
    assert backends.logm(3, X[:0]).shape == (0, 6)
    assert backends.logm(1, SO3.Identity(4, dtype=torch.float64).data[:0]).shape == (0, 3)


# ── inputs upstream accepts (regression: beartype rejected them in the dev env) ──


def test_setitem_accepts_a_plain_tensor() -> None:
    """dpvo projective_ops.py writes `Gij[..., 3:] = torch.as_tensor([0, 0, 0, 1])`; upstream indexes `data` and takes `item.data`."""
    G = SE3.Random(2, 3, dtype=torch.float64)
    t: Tensor = G.data[..., :3].clone()
    G[..., 3:] = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    assert torch.equal(G.data[..., :3], t)
    assert torch.equal(G.data[..., 3:], SE3.Identity(2, 3, dtype=torch.float64).data[..., 3:])
    G[0] = SE3.Identity(3, dtype=torch.float64)
    assert torch.equal(G.data[0], SE3.Identity(3, dtype=torch.float64).data)


def test_constructors_accept_what_upstream_accepts() -> None:
    assert SE3.Identity(np.int64(3)).shape == torch.Size([3])
    assert SE3.Identity((np.int64(2), 3)).shape == torch.Size([2, 3])
    assert SE3.Random(2, sigma=1).shape == torch.Size([2])
    assert SE3.Random(2, requires_grad=True).data.requires_grad
    gen: torch.Generator = torch.Generator().manual_seed(0)
    assert torch.equal(SE3.Random(2, generator=gen).data, SE3.Random(2, generator=torch.Generator().manual_seed(0)).data)


def test_cat_and_stack_accept_tuples() -> None:
    X = SE3.Random(2, 3, dtype=torch.float64)
    assert lietorch_mojo.cat((X, X), 0).shape == torch.Size([4, 3])
    assert lietorch_mojo.stack((X, X), 0).shape == torch.Size([2, 2, 3])


def test_mul_by_a_number_returns_none_like_upstream() -> None:
    X = SE3.Random(2, dtype=torch.float64)
    assert X.__mul__(2.0) is None


def test_lie_group_parameter_add_accepts_int_alpha() -> None:
    X = SE3.Random(2, dtype=torch.float64)
    P = lietorch_mojo.LieGroupParameter(X)
    P.add_(torch.zeros(2, 6, dtype=torch.float64), alpha=1)
    assert torch.allclose(P.group.data, X.data)


def test_upstream_module_names_exist() -> None:
    from lietorch_mojo.gradcheck import get_analytical_jacobian, get_numerical_jacobian, gradcheck, gradgradcheck
    from lietorch_mojo.group_ops import ToMatrix
    from lietorch_mojo.groups import ToMatrix as GroupsToMatrix

    assert GroupsToMatrix is ToMatrix
    assert all(callable(f) for f in (gradcheck, gradgradcheck, get_analytical_jacobian, get_numerical_jacobian))


def test_names_vipe_uses_exist() -> None:
    """The lietorch names ViPE's vendored copies (vipe/ext/lietorch, metrictrail dpvo) call, on every group.

    ViPE's `.rotation` is a field of its own `ScaledTransform` dataclass (vipe/utils/geometry.py), not a lietorch name.
    """
    from lietorch_mojo import RxSO3, group_ops

    methods: tuple[str, ...] = (
        "adj", "adjT", "act", "vec", "translation", "matrix", "retr", "Identity", "IdentityLike", "InitFromVec", "Random",
        "embedded_dim", "manifold_dim", "Jinv", "quaternion", "exp", "log", "inv", "mul", "detach", "view", "unbind",
    )  # fmt: skip
    for group in (SO3, RxSO3, SE3, Sim3):
        missing: list[str] = [name for name in methods if not hasattr(group, name)]
        assert not missing, f"{group.group_name}: {missing}"
    assert callable(SE3.scale)
    assert all(hasattr(group_ops, name) for name in ("Adj", "AdjT", "Act3", "Act4", "Jinv", "ToMatrix", "ToVec", "FromVec"))
    assert all(callable(getattr(backends, name)) for name in ("act4", "act4_backward", "Jinv", "as_matrix", "projector"))
    assert all(callable(getattr(lietorch_mojo, name)) for name in ("cat", "stack", "LieGroupParameter"))


def test_gradcheck_module_returns_both_jacobians_like_upstream() -> None:
    """Upstream's gradcheck returns (analytical, numerical) for the first output; run_tests.py compares them."""
    from lietorch_mojo.gradcheck import gradcheck

    a: Tensor = (0.3 * torch.randn(2, 6, dtype=torch.float64)).requires_grad_()
    result = gradcheck(lambda t: SE3.exp(t).log(), [a], eps=1e-4)
    assert isinstance(result, tuple)
    analytical: tuple[Tensor, ...] = result[0]
    numerical: tuple[Tensor, ...] = result[1]
    assert analytical[0].shape == numerical[0].shape == (12, 12)
    assert torch.allclose(analytical[0], numerical[0], atol=1e-6)
    assert torch.allclose(analytical[0], torch.eye(12, dtype=torch.float64), atol=1e-8)


# ── the Mojo boundary checks its inputs (regression: wrong widths read or wrote out of bounds) ──


@pytest.mark.parametrize(
    "call",
    [
        lambda: backends.expm(3, torch.zeros(8, 3)),
        lambda: backends.adj(3, SE3.Identity(4).data, torch.zeros(4, 3)),
        lambda: backends.act(3, SE3.Identity(4).data, torch.zeros(4, 4)),
        lambda: backends.mul(3, SE3.Identity(4).data, SE3.Identity(5).data),
        lambda: backends.logm(3, SE3.Identity(4).data.reshape(2, 2, 7)),
        lambda: backends.logm_backward(3, torch.zeros(4, 7), SE3.Identity(4).data),
    ],
    ids=["expm_short_rows", "adj_short_tangent", "act_wrong_point_width", "mul_batch_mismatch", "logm_3d", "logm_backward_wide_grad"],
)
def test_backend_rejects_wrong_shapes(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        call()


def test_extension_rejects_a_wrong_tensor_count() -> None:
    from lietorch_mojo import _lietorch_mojo_backends as ext

    X: Tensor = SE3.Identity(4).data
    for ptrs in ((X.data_ptr(),), (X.data_ptr(),) * 3, (X.data_ptr(),) * 66):
        with pytest.raises(Exception, match="takes 2 tensors"):
            ext.launch_cpu(backends._runtime(False), 4 | (3 << 8), 4, ptrs)  # OP_INV
    with pytest.raises(Exception, match="unknown op code"):
        ext.launch_cpu(backends._runtime(False), 99 | (3 << 8), 4, (X.data_ptr(),) * 2)
    with pytest.raises(Exception, match="not implemented"):
        ext.launch_cpu(backends._runtime(False), 4 | (7 << 8), 4, (X.data_ptr(),) * 2)  # no group 7
    with pytest.raises(Exception, match="expected Mojo .launch.GPURuntime. type value"):  # the downcast is type checked
        ext.launch_cuda(backends._runtime(False), 0, 4 | (3 << 8), 4, (X.data_ptr(),) * 2)


def test_cpu_rows_are_independent_of_the_batch_size() -> None:
    """Regression: CPU closures captured locals whose stack slots the compiler reused, so inputs read the outputs."""
    a: Tensor = torch.randn(5000, 6, dtype=torch.float64)
    full: Tensor = backends.expm(3, a)
    for n in (1, 2, 5, 2048, 2049):
        assert torch.equal(backends.expm(3, a[:n].contiguous()), full[:n])


def test_numpy_scalars_and_out_are_accepted_as_upstream_does() -> None:
    """Regression (verify-api r2): with beartype on (PIXI_DEV_MODE=1), these upstream-legal calls raised."""
    X = SE3.Identity(2)
    assert lietorch_mojo.cat([X, X], np.int64(0)).shape == torch.Size([4])
    assert lietorch_mojo.stack([X, X], np.int64(1)).shape == torch.Size([2, 2])
    Y = SE3.Identity(2, 3)
    assert [Z.shape for Z in Y.unbind(np.int64(1))] == [torch.Size([2])] * 3
    assert SE3.Random(2, sigma=np.float32(0.5)).shape == torch.Size([2])
    out: Tensor = torch.empty(2, 6)
    assert SE3.Random(2, out=out).shape == torch.Size([2])
