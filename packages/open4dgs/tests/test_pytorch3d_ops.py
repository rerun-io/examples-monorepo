"""The torch stand-ins for PyTorch3D's knn_points and sample_farthest_points against brute-force references."""

import itertools

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
