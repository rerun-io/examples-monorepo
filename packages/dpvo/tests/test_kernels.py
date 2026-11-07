"""Unit tests of the kernel-op dispatch (no GPU: fake backends on the ``meta`` device)."""

from collections.abc import Sequence
from typing import NamedTuple

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")

from torch import Tensor  # noqa: E402

from dpvo import altcorr, fastba, kernels  # noqa: E402
from dpvo.fastba import BADepthPrior  # noqa: E402

META = torch.device("meta")


class FakeCorr:
    """Records every call; returns fixed tensors."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def forward(self, fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, radius: int) -> Sequence[Tensor]:
        self.calls.append(("forward", (fmap1, fmap2, coords, ii, jj, radius)))
        return (torch.empty(1, ii.shape[0], 2 * radius + 1, 2 * radius + 1, 3, 3, device=META),)

    def backward(self, fmap1: Tensor, fmap2: Tensor, coords: Tensor, ii: Tensor, jj: Tensor, grad: Tensor, radius: int) -> Sequence[Tensor]:
        self.calls.append(("backward", (fmap1, fmap2, coords, ii, jj, grad, radius)))
        return torch.empty_like(fmap1), torch.empty_like(fmap2)

    def patchify_forward(self, net: Tensor, coords: Tensor, radius: int) -> Sequence[Tensor]:
        self.calls.append(("patchify_forward", (net, coords, radius)))
        return (torch.empty(coords.shape[0], coords.shape[1], net.shape[1], 2 * radius + 2, 2 * radius + 2, device=META),)

    def patchify_backward(self, net: Tensor, coords: Tensor, grad: Tensor, radius: int) -> Sequence[Tensor]:
        self.calls.append(("patchify_backward", (net, coords, grad, radius)))
        return (torch.empty_like(net),)


class FakeBA:
    """Records every call; returns fixed tensors."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.linked: tuple[Tensor, Tensor] = (torch.empty(4, dtype=torch.int64, device=META), torch.empty(4, dtype=torch.int64, device=META))

    def forward(
        self,
        poses: Tensor,
        patches: Tensor,
        intrinsics: Tensor,
        target: Tensor,
        weight: Tensor,
        lmbda: Tensor,
        depth_prior: Tensor,
        depth_prior_weight: Tensor,
        depth_prior_lambda: float,
        depth_prior_huber: float,
        ii: Tensor,
        jj: Tensor,
        kk: Tensor,
        M: int,
        t0: int,
        t1: int,
        iterations: int,
        eff_impl: bool,
    ) -> Sequence[Tensor]:
        args = (poses, patches, intrinsics, target, weight, lmbda, depth_prior, depth_prior_weight, depth_prior_lambda, depth_prior_huber)
        self.calls.append(("forward", (*args, ii, jj, kk, M, t0, t1, iterations, eff_impl)))
        return []

    def neighbors(self, ii: Tensor, jj: Tensor) -> Sequence[Tensor]:
        self.calls.append(("neighbors", (ii, jj)))
        return list(self.linked)

    def reproject(self, poses: Tensor, patches: Tensor, intrinsics: Tensor, ii: Tensor, jj: Tensor, kk: Tensor) -> Tensor:
        self.calls.append(("reproject", (poses, patches, intrinsics, ii, jj, kk)))
        return torch.empty(1, ii.shape[0], 2, 3, 3, device=META)

    def solve_system(
        self, J_Ginv_i: Tensor, J_Ginv_j: Tensor, ii: Tensor, jj: Tensor, res: Tensor, ep: float, lm: float, freen: int
    ) -> Sequence[Tensor]:
        self.calls.append(("solve_system", (J_Ginv_i, J_Ginv_j, ii, jj, res, ep, lm, freen)))
        return [torch.empty(3, 7, device=META)]


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeCorr, FakeBA]:
    """Register fake kernels for ``meta`` in an empty, test-local registry."""
    monkeypatch.setattr(kernels, "_BACKENDS", {})
    corr, ba = FakeCorr(), FakeBA()
    kernels.register("meta", corr=corr, ba=ba)
    return corr, ba


class BAInputs(NamedTuple):
    """The nine tensor arguments of ``fastba.BA``, in order."""

    poses: Tensor
    patches: Tensor
    intrinsics: Tensor
    target: Tensor
    weight: Tensor
    lmbda: Tensor
    ii: Tensor
    jj: Tensor
    kk: Tensor


def _ba_problem(n_frames: int = 3, n_patches: int = 6, n_edges: int = 4) -> BAInputs:
    edge: Tensor = torch.empty(n_edges, dtype=torch.int64, device=META)
    return BAInputs(
        poses=torch.empty(1, n_frames, 7, device=META),
        patches=torch.empty(1, n_patches, 3, 3, 3, device=META),
        intrinsics=torch.empty(1, n_frames, 4, device=META),
        target=torch.empty(1, n_edges, 2, device=META),
        weight=torch.empty(1, n_edges, 2, device=META),
        lmbda=torch.empty(1, device=META),
        ii=edge,
        jj=edge.clone(),
        kk=edge.clone(),
    )


def test_lookup_returns_the_registered_backend(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    corr, ba = fake_backend

    assert kernels.corr_kernels(META) is corr
    assert kernels.ba_kernels(META) is ba


def test_lookup_miss_raises_lookup_error_naming_the_device_and_register(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    with pytest.raises(LookupError, match=r"'xla'.*dpvo\.kernels\.register"):
        kernels.ba_kernels(torch.device("xla"))
    with pytest.raises(LookupError, match="'xla'"):
        kernels.corr_kernels(torch.device("xla"))


def test_register_replaces_an_earlier_backend(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    corr, ba = FakeCorr(), FakeBA()
    kernels.register("meta", corr=corr, ba=ba)

    assert kernels.corr_kernels(META) is corr
    assert kernels.ba_kernels(META) is ba


def test_ba_without_prior_passes_tensors_through_and_disables_the_prior(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    _, ba = fake_backend
    problem = _ba_problem()

    fastba.BA(*problem, 1, 3, M=2, iterations=4, eff_impl=True)

    ((name, args),) = ba.calls
    assert name == "forward"
    poses_data = args[0]
    assert isinstance(poses_data, Tensor) and poses_data.shape == problem.poses.shape  # poses.data: new Python object
    for passed, given in zip(args[1:6], problem[1:6], strict=True):
        assert passed is given
    prior, prior_weight, lam, huber = args[6:10]
    assert isinstance(prior, Tensor) and isinstance(prior_weight, Tensor)
    assert prior.shape == (0,) and prior.dtype == torch.float32 and prior.device == META
    assert prior_weight.shape == (0,) and prior_weight.dtype == torch.float32
    assert (lam, huber) == (0.0, 0.3)
    assert args[10] is problem.ii and args[11] is problem.jj and args[12] is problem.kk
    assert args[13:] == (2, 1, 3, 4, True)


def test_ba_with_prior_passes_the_prior_tensors_without_copies(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    _, ba = fake_backend
    problem = _ba_problem(n_patches=6)
    prior = BADepthPrior(disps=torch.empty(6, device=META), weights=torch.empty(6, device=META), lam=1.0, huber=0.5)

    fastba.BA(*problem, 1, 3, depth_prior=prior)

    ((_, args),) = ba.calls
    assert args[6] is prior.disps and args[7] is prior.weights
    assert args[8:10] == (1.0, 0.5)
    assert args[13:] == (-1, 1, 3, 2, False)


def test_ba_rejects_a_prior_that_does_not_cover_every_patch_slot(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    problem = _ba_problem(n_patches=6)
    prior = BADepthPrior(disps=torch.empty(5, device=META), weights=torch.empty(5, device=META), lam=1.0)

    with pytest.raises(ValueError, match="5 values"):
        fastba.BA(*problem, 1, 3, depth_prior=prior)


@pytest.mark.parametrize(
    ("n_weights", "lam", "huber", "message"),
    [(5, 1.0, 0.3, "shape"), (6, -1.0, 0.3, "lam"), (6, float("nan"), 0.3, "lam"), (6, 1.0, 0.0, "huber")],
)
def test_depth_prior_validates_its_fields(n_weights: int, lam: float, huber: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BADepthPrior(disps=torch.empty(6, device=META), weights=torch.empty(n_weights, device=META), lam=lam, huber=huber)


def test_neighbors_reproject_and_solve_system_dispatch_on_the_input_device(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    _, ba = fake_backend
    problem = _ba_problem()

    ix, jx = fastba.neighbors(problem.kk, problem.jj)
    coords = fastba.reproject(problem.poses, problem.patches, problem.intrinsics, problem.ii, problem.jj, problem.kk)
    jac = torch.empty(2, 7, 7, device=META)
    edge = torch.empty(2, dtype=torch.int64, device=META)
    delta = fastba.solve_system(jac, jac, edge, edge, torch.empty(2, 7, device=META), 0.0, 1e-6, -1)

    assert ix is ba.linked[0] and jx is ba.linked[1]
    assert coords.shape == (1, 4, 2, 3, 3)
    assert delta.shape == (3, 7)
    assert [name for name, _ in ba.calls] == ["neighbors", "reproject", "solve_system"]


def test_corr_and_patchify_dispatch_on_the_input_device(fake_backend: tuple[FakeCorr, FakeBA]) -> None:
    corr_backend, _ = fake_backend
    fmap1 = torch.empty(1, 2, 8, 12, 16, device=META)
    fmap2 = torch.empty(1, 2, 8, 6, 8, device=META)
    coords = torch.empty(1, 4, 2, 3, 3, device=META)
    edge = torch.empty(4, dtype=torch.int64, device=META)

    volume = altcorr.corr(fmap1, fmap2, coords, edge, edge, radius=3)
    patches = altcorr.patchify(torch.empty(1, 8, 12, 16, device=META), torch.empty(1, 5, 2, device=META), 0, mode="nearest")

    assert volume.shape == (1, 4, 7, 7, 3, 3)
    assert patches.shape == (1, 5, 8, 2, 2)
    assert [name for name, _ in corr_backend.calls] == ["forward", "patchify_forward"]
    assert corr_backend.calls[0][1][:5] == (fmap1, fmap2, coords, edge, edge)
