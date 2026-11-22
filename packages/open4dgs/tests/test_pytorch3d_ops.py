"""The torch stand-ins for PyTorch3D's knn_points and sample_farthest_points against brute-force references."""

import itertools

import pytest
import torch

from open4dgs.pytorch3d_ops import knn_points, sample_farthest_points


def test_knn_points_matches_brute_force_across_chunks() -> None:
    gen = torch.Generator().manual_seed(0)
    p1 = torch.rand((2, 37, 3), generator=gen)
    p2 = torch.rand((2, 23, 3), generator=gen)
    result = knn_points(p1, p2, K=4, return_nn=True, chunk_size=8)
    full = ((p1[:, :, None, :] - p2[:, None, :, :]) ** 2).sum(-1)
    want_d, want_i = full.sort(dim=-1)
    torch.testing.assert_close(result.dists, want_d[..., :4])
    assert torch.equal(result.idx, want_i[..., :4])
    assert result.knn is not None
    torch.testing.assert_close(result.knn[0, 5, 2], p2[0, want_i[0, 5, 2]])


def test_knn_points_self_query_puts_each_point_first() -> None:
    pts = torch.rand((1, 50, 3), generator=torch.Generator().manual_seed(1))
    result = knn_points(pts, pts, K=3)
    assert torch.equal(result.idx[0, :, 0], torch.arange(50))
    assert result.knn is None


def test_knn_points_respects_lengths2_and_pads_missing_neighbours() -> None:
    p1 = torch.zeros((1, 2, 3))
    p2 = torch.tensor([[[0.0, 0, 1], [0, 0, 2], [0, 0, 0.5]]])
    result = knn_points(p1, p2, lengths2=torch.tensor([2]), K=3)
    assert result.idx[0, 0].tolist() == [0, 1, -1]
    torch.testing.assert_close(result.dists[0, 0], torch.tensor([1.0, 4.0, 0.0]))


def test_sample_farthest_points_matches_exhaustive_greedy() -> None:
    pts = torch.rand((1, 12, 3), generator=torch.Generator().manual_seed(2))
    _, idx = sample_farthest_points(pts, K=4)
    picks = [0]
    for _ in range(3):
        rest = [i for i in range(12) if i not in picks]
        picks.append(max(rest, key=lambda i: min(float(((pts[0, i] - pts[0, j]) ** 2).sum()) for j in picks)))
    assert idx[0].tolist() == picks


def test_sample_farthest_points_per_item_k_and_lengths_pad_with_minus_one() -> None:
    pts = torch.rand((2, 10, 3), generator=torch.Generator().manual_seed(3))
    sampled, idx = sample_farthest_points(pts, lengths=torch.tensor([10, 4]), K=torch.tensor([5, 6]))
    assert idx.shape == (2, 5)
    assert (idx[1, 4:] == -1).all() and (idx[1, :4] >= 0).all() and (idx[1, :4] < 4).all()
    assert len(set(idx[0].tolist())) == 5
    for b, k in itertools.product(range(2), range(5)):
        if idx[b, k] >= 0:
            torch.testing.assert_close(sampled[b, k], pts[b, idx[b, k]])
        else:
            assert (sampled[b, k] == 0).all()


def test_knn_points_adaptive_chunks_equal_fixed_chunks() -> None:
    gen = torch.Generator().manual_seed(4)
    p1 = torch.rand((1, 3000, 3), generator=gen)
    p2 = torch.rand((1, 2500, 3), generator=gen)
    fixed = knn_points(p1, p2, K=5, chunk_size=4096)  # one chunk: the pre-2026-11-21 behaviour at this size
    tiny_budget = knn_points(p1, p2, K=5, memory_budget=64 * 2500 * 4)  # 64 rows per chunk
    torch.testing.assert_close(tiny_budget.dists, fixed.dists, rtol=0, atol=0)
    assert torch.equal(tiny_budget.idx, fixed.idx)
    padded = knn_points(p1, p2, lengths2=torch.tensor([2000]), K=5, memory_budget=100 * 2500 * 4)
    assert int(padded.idx.max()) < 2000


def _assert_valid_self_knn(result_d: torch.Tensor, result_i: torch.Tensor, pts: torch.Tensor, want_d: torch.Tensor) -> None:
    """``result`` is a K-NN of ``pts`` with ``want_d``'s sorted distances: same distances, and each pick at its distance."""
    torch.testing.assert_close(result_d, want_d, rtol=1e-5, atol=1e-6)
    picked = (pts[result_i.clamp(min=0)] - pts[:, None]).square().sum(-1)
    torch.testing.assert_close(picked, want_d, rtol=1e-5, atol=1e-6)
    assert all(len(set(row)) == len(row) for row in result_i.tolist())


@pytest.mark.integration
@pytest.mark.parametrize("k", [1, 4, 33, 200])
def test_knn_points_binned_self_query_equals_torch_path(k: int) -> None:
    """The FastGraphCompute path (CUDA self-query) gives the torch path's neighbours, with exact duplicates included."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    pytest.importorskip("fastgraphcompute", reason="the FastGraphCompute path needs recipes/fastgraphcompute-src")
    from open4dgs import pytorch3d_ops

    gen = torch.Generator().manual_seed(5)
    pts = torch.rand((5000, 3), generator=gen)
    pts[100:110] = pts[7]  # 11 coincident points: ties at distance 0
    pts[200] = pts[201]
    pts = pts.cuda()
    assert pytorch3d_ops._binned_select_knn() is not None
    binned = knn_points(pts[None], pts[None], K=k)  # self-query: FastGraphCompute
    reference = knn_points(pts[None], pts[None].clone(), K=k)  # p2 another tensor: the torch path
    assert binned.idx.shape == reference.idx.shape == (1, 5000, k) and binned.idx.dtype == torch.int64
    assert torch.equal(binned.idx[0, :, 0], torch.arange(5000, device="cuda"))  # the point itself first, duplicates too
    exact = (pts[:, None, :] - pts[None, :, :]).square().sum(-1).sort(dim=1).values  # direct differences, no matmul
    _assert_valid_self_knn(binned.dists[0], binned.idx[0], pts, exact[:, :k])
    # Where the K-th and (K+1)-th distances differ the neighbour set is unique: both paths must return it.
    unique_set = exact[:, k] > exact[:, k - 1] + 1e-5  # beyond the torch path's cdist rounding (matmul form)
    same = (binned.idx[0].sort(dim=1).values == reference.idx[0].sort(dim=1).values).all(1)
    assert bool(same[unique_set].all())


@pytest.mark.integration
def test_knn_points_binned_k_above_point_count_pads_like_torch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    pytest.importorskip("fastgraphcompute", reason="the FastGraphCompute path needs recipes/fastgraphcompute-src")
    pts = torch.rand((1, 6, 3), generator=torch.Generator().manual_seed(6)).cuda()
    binned = knn_points(pts, pts, K=9, return_nn=True)
    reference = knn_points(pts, pts.clone(), K=9, return_nn=True)
    assert torch.equal(binned.idx, reference.idx)  # 6 distinct points, no ties: identical including the -1 padding
    torch.testing.assert_close(binned.dists, reference.dists, rtol=1e-5, atol=1e-6)
    assert binned.knn is not None and reference.knn is not None
    torch.testing.assert_close(binned.knn, reference.knn)
