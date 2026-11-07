"""GPU tests of the shared fastba BA with the depth-prior kernel (map/kernels.md section 6.4)."""

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")

from jaxtyping import Float32, Int64  # noqa: E402
from torch import Tensor  # noqa: E402

from dpvo import fastba  # noqa: E402
from dpvo.fastba import BADepthPrior  # noqa: E402

pytestmark = pytest.mark.integration

PATCHES_PER_FRAME = 16
N_FRAMES = 2
INTRINSICS = (50.0, 50.0, 32.0, 24.0)


@pytest.fixture
def cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    return torch.device("cuda")


def _patches(disps: Float32[Tensor, "n"], device: torch.device) -> Float32[Tensor, "1 n 3 3 3"]:
    """Patches at fixed pixel centres with a uniform inverse depth per patch."""
    n: int = disps.shape[0]
    gen = torch.Generator().manual_seed(1)
    centres: Float32[Tensor, "n 2"] = torch.rand(n, 2, generator=gen) * torch.tensor([48.0, 32.0]) + torch.tensor([8.0, 8.0])
    offsets: Float32[Tensor, "3"] = torch.tensor([-1.0, 0.0, 1.0])
    patches: Float32[Tensor, "n 3 3 3"] = torch.empty(n, 3, 3, 3)
    patches[:, 0] = centres[:, 0, None, None] + offsets[None, None, :]
    patches[:, 1] = centres[:, 1, None, None] + offsets[None, :, None]
    patches[:, 2] = disps.cpu()[:, None, None]
    return patches[None].to(device)


def _pose_problem(device: torch.device) -> dict[str, Tensor]:
    """Two frames, 16 patches each, every patch observed once in the other frame; perturbed start.

    Each patch has exactly one edge, so in the depth-only solve (t0 == t1) every C[k]/u[k] slot
    receives its atomicAdds from one thread and the result is bit-reproducible. The pose solve sums
    many edges into the same B/v slots, so its float atomicAdd order (and the last bits) vary per run.
    """
    n_patches: int = N_FRAMES * PATCHES_PER_FRAME
    gen = torch.Generator().manual_seed(0)
    true_disps: Float32[Tensor, "n"] = 0.3 + 0.7 * torch.rand(n_patches, generator=gen)
    true_poses: Float32[Tensor, "1 f 7"] = torch.zeros(1, N_FRAMES, 7)
    true_poses[0, :, 6] = 1.0
    true_poses[0, 1, :3] = torch.tensor([0.15, -0.05, 0.02])
    intrinsics: Float32[Tensor, "1 f 4"] = torch.tensor(INTRINSICS).repeat(1, N_FRAMES, 1)

    kk: Int64[Tensor, "e"] = torch.arange(n_patches)
    ii: Int64[Tensor, "e"] = kk // PATCHES_PER_FRAME
    jj: Int64[Tensor, "e"] = 1 - ii

    true_patches: Float32[Tensor, "1 n 3 3 3"] = _patches(true_disps, device)
    edges = (ii.to(device), jj.to(device), kk.to(device))
    coords: Float32[Tensor, "1 e 2 3 3"] = fastba.reproject(true_poses.to(device), true_patches, intrinsics.to(device), *edges)
    target: Float32[Tensor, "1 e 2"] = coords[..., 1, 1] + 0.3 * torch.randn(1, kk.shape[0], 2, generator=gen).to(device)

    start_poses: Float32[Tensor, "1 f 7"] = true_poses.clone()
    start_poses[0, 1, :3] += torch.tensor([0.02, 0.01, -0.01])
    start_disps: Float32[Tensor, "n"] = true_disps * (1.0 + 0.2 * (torch.rand(n_patches, generator=gen) - 0.5))
    return {
        "poses": start_poses.to(device),
        "patches": _patches(start_disps, device),
        "intrinsics": intrinsics.to(device),
        "target": target,
        "weight": torch.ones(1, kk.shape[0], 2, device=device),
        "lmbda": torch.tensor([1e-4], device=device),
        "ii": edges[0],
        "jj": edges[1],
        "kk": edges[2],
    }


def _run_ba(problem: dict[str, Tensor], t0: int, eff_impl: bool, depth_prior: BADepthPrior | None) -> tuple[Tensor, Tensor]:
    poses: Tensor = problem["poses"].clone()
    patches: Tensor = problem["patches"].clone()
    fastba.BA(
        poses, patches, problem["intrinsics"], problem["target"], problem["weight"], problem["lmbda"],
        problem["ii"], problem["jj"], problem["kk"], t0, N_FRAMES,
        M=PATCHES_PER_FRAME, iterations=2, eff_impl=eff_impl, depth_prior=depth_prior,
    )  # fmt: skip
    torch.cuda.synchronize()
    return poses, patches


def _priors(n_patches: int, device: torch.device) -> tuple[BADepthPrior, BADepthPrior]:
    """The same non-empty prior with lam = 0 (disabled) and lam = 1 (enabled)."""
    gen = torch.Generator().manual_seed(2)
    disps: Float32[Tensor, "n"] = (0.3 + torch.rand(n_patches, generator=gen)).to(device)
    weights: Float32[Tensor, "n"] = torch.ones(n_patches, device=device)
    return BADepthPrior(disps=disps, weights=weights, lam=0.0), BADepthPrior(disps=disps, weights=weights, lam=1.0)


def test_zero_lambda_prior_is_bit_identical_to_no_prior_in_the_depth_only_solve(cuda: torch.device) -> None:
    problem = _pose_problem(cuda)
    disabled, enabled = _priors(problem["patches"].shape[1], cuda)

    poses_none, patches_none = _run_ba(problem, N_FRAMES, False, None)
    _, patches_again = _run_ba(problem, N_FRAMES, False, None)
    poses_zero, patches_zero = _run_ba(problem, N_FRAMES, False, disabled)
    _, patches_on = _run_ba(problem, N_FRAMES, False, enabled)

    assert not torch.equal(patches_none, problem["patches"])  # the BA moved the depths
    assert torch.equal(patches_again, patches_none)  # this problem is bit-reproducible
    assert torch.equal(patches_zero, patches_none) and torch.equal(poses_zero, poses_none)
    assert (patches_on - patches_none).abs().max() > 1e-3  # lam > 0 reaches the solve


@pytest.mark.parametrize("eff_impl", [False, True])
def test_zero_lambda_prior_matches_no_prior_in_the_pose_solve(cuda: torch.device, eff_impl: bool) -> None:
    """Pose + depth solve: equal up to the run-to-run atomicAdd noise of the BA kernels.

    Measured on the 5090 (20 runs): dense path 0, block-E path up to 2.5e-4 between two runs
    without a prior; lam = 1 moves poses by 2.3e-2 and depths by 2.0e-1.
    """
    problem = _pose_problem(cuda)
    disabled, enabled = _priors(problem["patches"].shape[1], cuda)

    poses_none, patches_none = _run_ba(problem, 1, eff_impl, None)
    poses_zero, patches_zero = _run_ba(problem, 1, eff_impl, disabled)
    poses_on, patches_on = _run_ba(problem, 1, eff_impl, enabled)

    assert (poses_none - problem["poses"]).abs().max() > 1e-3  # the BA moved the poses
    torch.testing.assert_close(poses_zero, poses_none, rtol=0.0, atol=1e-3)
    torch.testing.assert_close(patches_zero, patches_none, rtol=0.0, atol=1e-3)
    assert (patches_on - patches_none).abs().max() > 1e-2  # lam > 0 reaches the solve, far above the noise
    assert (poses_on - poses_none).abs().max() > 1e-2


def test_depth_only_prior_matches_the_closed_form_update(cuda: torch.device) -> None:
    """With t0 == t1 (both frames fixed) and zero-weight reprojection edges, dZ = u / (C + lmbda) from the prior alone."""
    current: Float32[Tensor, "6"] = torch.tensor([0.5, 0.5, 0.8, 0.4, 0.6, 0.7])
    prior_disps: Float32[Tensor, "6"] = torch.tensor([0.55, 1.5, 0.8, 0.3, -1.0, 9.0])  # inside / beyond huber, r=0, ..., skip, unused
    prior_weights: Float32[Tensor, "6"] = torch.tensor([1.0, 2.0, 1.0, 0.0, 1.0, 1.0])
    lam, huber, lmbda = 1.5, 0.3, 1e-4
    patches: Float32[Tensor, "1 6 3 3 3"] = _patches(current, cuda)
    intrinsics: Float32[Tensor, "1 2 4"] = torch.tensor(INTRINSICS, device=cuda).repeat(1, 2, 1)
    poses: Float32[Tensor, "1 2 7"] = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1], [0.1, 0, 0, 0, 0, 0, 1]], device=cuda)[None]
    kk: Int64[Tensor, "5"] = torch.arange(5, device=cuda)  # patch 5 is not in the BA
    ii: Int64[Tensor, "5"] = torch.zeros(5, dtype=torch.int64, device=cuda)
    jj: Int64[Tensor, "5"] = torch.ones(5, dtype=torch.int64, device=cuda)
    zeros: Float32[Tensor, "1 5 2"] = torch.zeros(1, 5, 2, device=cuda)
    prior = BADepthPrior(disps=prior_disps.to(cuda), weights=prior_weights.to(cuda), lam=lam, huber=huber)

    fastba.BA(poses, patches, intrinsics, zeros, zeros, torch.tensor([lmbda], device=cuda), ii, jj, kk, 2, 2, iterations=1, depth_prior=prior)

    residual: Float32[Tensor, "6"] = torch.log(prior_disps.clamp_min(1e-30)) - torch.log(current)
    robust: Float32[Tensor, "6"] = torch.where(residual.abs() <= huber, 1.0, huber / residual.abs().clamp_min(1e-6))
    active: Float32[Tensor, "6"] = ((prior_weights > 0) & (prior_disps > 0)).float()
    w: Float32[Tensor, "6"] = active * lam * prior_weights * robust
    jac: Float32[Tensor, "6"] = 1.0 / current.clamp_min(1e-4)
    dz: Float32[Tensor, "6"] = (w * residual * jac) / (w * jac * jac + lmbda)
    dz[5] = 0.0
    expected: Float32[Tensor, "6"] = (current + dz).clamp_min(1e-4)

    assert residual[1].abs() > huber >= residual[0].abs()  # both Huber branches are covered
    actual: Float32[Tensor, "6 3 3"] = patches[0, :, 2].cpu()
    torch.testing.assert_close(actual, expected[:, None, None].expand(6, 3, 3), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(patches[0, :, :2].cpu(), _patches(current, cuda)[0, :, :2].cpu(), rtol=0.0, atol=0.0)
