"""Patchifier centroid draws: unchanged without a mask, ViPE's rejection sampler with one."""

import math

import pytest

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch_mojo", reason="DPVO requires lietorch_mojo")

from beartype.roar import BeartypeException  # noqa: E402
from jaxtyping import Bool, Int64  # noqa: E402
from torch import Tensor  # noqa: E402

from dpvo.net import Patchifier, sample_valid_centroids, valid_interior_cells  # noqa: E402

# ── frozen copy of ViPE's sampler (vipe/metrictrail/dynamic.py:172-248, vipe-src build 4) ────────
VIPE_CANDIDATE_FACTOR = 4
VIPE_MAX_REJECTION_ROUNDS = 64


def vipe_valid_interior_cells(valid):
    height, width = valid.shape[-2:]
    return int(valid[..., 1 : height - 1, 1 : width - 1].sum().item())


def vipe_sample_valid_centroids(valid, frames, count, *, max_rounds=VIPE_MAX_REJECTION_ROUNDS):
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError(f"centroid validity must be a (h,w) bool grid, got {tuple(valid.shape)} {valid.dtype}")
    height, width = int(valid.shape[0]), int(valid.shape[1])
    available = vipe_valid_interior_cells(valid)
    if available == 0:
        raise ValueError("no valid centroid position")
    accept_rate = available / float((height - 2) * (width - 2))
    xs = []
    ys = []
    for _ in range(int(frames)):
        kept_x = []
        kept_y = []
        found = 0
        for _round in range(int(max_rounds)):
            draws = max(VIPE_CANDIDATE_FACTOR * count, math.ceil(2.0 * (count - found) / accept_rate))
            x = torch.randint(1, width - 1, size=[draws], device=valid.device)
            y = torch.randint(1, height - 1, size=[draws], device=valid.device)
            keep = valid[y, x]
            kept_x.append(x[keep])
            kept_y.append(y[keep])
            found += int(keep.sum().item())
            if found >= count:
                break
        else:
            raise RuntimeError(f"rejection sampling found {found}/{count} valid centroids in {max_rounds} rounds")
        xs.append(torch.cat(kept_x)[:count])
        ys.append(torch.cat(kept_y)[:count])
    return torch.stack(xs), torch.stack(ys)


# ────────────────────────────────────────────────────────────────────────


def sparse_mask(height: int, width: int, fraction: float, seed: int) -> Bool[Tensor, "h w"]:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    return torch.rand(height, width, generator=generator) < fraction


@pytest.mark.parametrize(("fraction", "frames", "count"), [(0.9, 1, 96), (0.05, 2, 80), (0.002, 1, 3)])
def test_sampler_matches_vipe_draw_order(fraction: float, frames: int, count: int) -> None:
    valid: Bool[Tensor, "h w"] = sparse_mask(60, 80, fraction, seed=3)
    valid[30, 40] = True  # at least one interior cell
    torch.manual_seed(11)
    expected = vipe_sample_valid_centroids(valid, frames, count)
    after_vipe: Tensor = torch.rand(1)
    torch.manual_seed(11)
    actual: tuple[Int64[Tensor, "f c"], Int64[Tensor, "f c"]] = sample_valid_centroids(valid, frames, count)
    after_ours: Tensor = torch.rand(1)

    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
    assert torch.equal(after_ours, after_vipe)  # same number of draws
    assert bool(valid[actual[1], actual[0]].all())
    assert valid_interior_cells(valid) == vipe_valid_interior_cells(valid)


def test_sampler_errors_match_vipe() -> None:
    border_only: Bool[Tensor, "h w"] = torch.zeros(8, 8, dtype=torch.bool)
    border_only[0] = True
    with pytest.raises(ValueError, match="no valid centroid position"):
        sample_valid_centroids(border_only, 1, 4)
    with pytest.raises((ValueError, BeartypeException)):  # beartype rejects the rank first in the dev env
        sample_valid_centroids(torch.ones(1, 8, 8, dtype=torch.bool), 1, 4)  # pyrefly: ignore[bad-argument-type]
    one_cell: Bool[Tensor, "h w"] = torch.zeros(8, 8, dtype=torch.bool)
    one_cell[4, 4] = True
    with pytest.raises(RuntimeError, match="rejection sampling found 0/4 valid centroids in 0 rounds"):
        sample_valid_centroids(one_cell, 1, 4, max_rounds=0)


# ── the patchifier itself (CUDA patch-extraction kernel) ────────────────


@pytest.fixture(scope="module")
def cuda_patchifier() -> Patchifier:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for the patchify kernel")
    torch.manual_seed(0)
    return Patchifier(3).cuda().eval()


def image(seed: int) -> Tensor:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 1, 3, 64, 96, generator=generator) - 0.5).cuda()


@pytest.mark.integration
@pytest.mark.parametrize("gradient_bias", [False, True])
def test_unmasked_draws_unchanged(cuda_patchifier: Patchifier, gradient_bias: bool) -> None:
    with torch.no_grad():
        torch.manual_seed(5)
        _, _, _, patches, _ = cuda_patchifier(image(0), patches_per_image=16, gradient_bias=gradient_bias)
        after: Tensor = torch.rand(1, device="cuda")

        # The pre-change draws (P net.py:306-321 at main 347c62081): x then y on the default CUDA generator.
        torch.manual_seed(5)
        width: int = 3 * 16 if gradient_bias else 16
        x: Tensor = torch.randint(1, 96 // 4 - 1, size=[1, width], device="cuda")
        y: Tensor = torch.randint(1, 64 // 4 - 1, size=[1, width], device="cuda")
        expected_after: Tensor = torch.rand(1, device="cuda")

    assert torch.equal(after, expected_after)
    centres: Tensor = patches[0, :, :2, 1, 1]
    if gradient_bias:
        candidates: set[tuple[float, float]] = set(zip(x[0].float().tolist(), y[0].float().tolist(), strict=True))
        assert all(tuple(c) in candidates for c in centres.tolist())
    else:
        assert torch.equal(centres, torch.stack([x[0], y[0]], dim=-1).float())


@pytest.mark.integration
def test_masked_draws_use_sampler(cuda_patchifier: Patchifier) -> None:
    valid: Bool[Tensor, "h4 w4"] = torch.zeros(16, 24, dtype=torch.bool, device="cuda")
    valid[4:9, 10:20] = True
    with torch.no_grad():
        torch.manual_seed(9)
        _, _, _, patches, _ = cuda_patchifier(image(1), patches_per_image=32, centroid_valid=valid)
        torch.manual_seed(9)
        x, y = sample_valid_centroids(valid, 1, 32)

    assert torch.equal(patches[0, :, :2, 1, 1], torch.stack([x[0], y[0]], dim=-1).float())
    with pytest.raises(ValueError, match="gradient_bias"):
        cuda_patchifier(image(1), patches_per_image=32, gradient_bias=True, centroid_valid=valid)
    with pytest.raises(ValueError, match="does not match"):
        cuda_patchifier(image(1), patches_per_image=32, centroid_valid=valid[:-1])
