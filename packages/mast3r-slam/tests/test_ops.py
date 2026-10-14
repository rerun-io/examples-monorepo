"""Op tier (decision 18): Sim3 group identities, the pose-graph normal equations, frame intrinsics, ATE under a Sim3."""

import lietorch
import numpy as np
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.spatial.transform import Rotation

from mast3r_slam.ops.image import frame_intrinsics, resize_img_with_transform
from mast3r_slam.ops.optim import POSE_DIM, solve_pose_blocks
from mast3r_slam.verify import _ate

TANGENTS = arrays(np.float64, (2, 7), elements=st.floats(-1.0, 1.0))


@given(tangents=TANGENTS)
@settings(max_examples=40, deadline=None)
def test_sim3_compose_with_inverse_is_identity(tangents: np.ndarray) -> None:
    a, b = (lietorch.Sim3.exp(torch.from_numpy(t)[None]) for t in tangents)
    loop, ab, b_inv_a_inv = a * a.inv(), a * b, b.inv() * a.inv()
    assert isinstance(loop, lietorch.Sim3) and isinstance(ab, lietorch.Sim3) and isinstance(b_inv_a_inv, lietorch.Sim3)
    np.testing.assert_allclose(loop.log().numpy(), np.zeros((1, 7)), atol=1e-9)
    np.testing.assert_allclose(ab.inv().matrix().numpy(), b_inv_a_inv.matrix().numpy(), rtol=1e-7, atol=1e-9)


@given(n_free=st.integers(1, 4), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=40, deadline=None)
def test_solve_pose_blocks_solves_the_assembled_system(n_free: int, seed: int) -> None:
    """Chain graph over poses 0..n_free (pose 0 pinned): the step solves H dx = -b of the assembled blocks."""
    rng = np.random.default_rng(seed)
    ii = torch.arange(n_free, dtype=torch.long) - 1  # free-pose index of each edge's first pose (-1: pinned)
    jj = torch.arange(n_free, dtype=torch.long)
    jac = rng.normal(size=(n_free, 2, 9, POSE_DIM))  # per edge: residual Jacobian w.r.t. both poses
    hs = torch.from_numpy(np.stack([jac[:, a].transpose(0, 2, 1) @ jac[:, b] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]))
    gs = torch.from_numpy(rng.normal(size=(2, n_free, POSE_DIM)))
    dense = np.zeros((n_free * POSE_DIM, n_free * POSE_DIM))
    rhs = np.zeros(n_free * POSE_DIM)
    for e in range(n_free):
        blocks = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        for (a, b), h in blocks.items():
            ia, ib = (int(ii[e]), int(jj[e]))[a], (int(ii[e]), int(jj[e]))[b]
            if ia >= 0 and ib >= 0:
                dense[ia * 7 : ia * 7 + 7, ib * 7 : ib * 7 + 7] += hs[h, e].numpy()
        for a, idx in enumerate((int(ii[e]), int(jj[e]))):
            if idx >= 0:
                rhs[idx * 7 : idx * 7 + 7] += gs[a, e].numpy()
    assume(np.linalg.cond(dense) < 1e8)
    dx = solve_pose_blocks(hs.float(), gs.float(), ii, jj, n_free)
    assert dx.shape == (n_free, POSE_DIM) and dx.dtype == torch.float64
    np.testing.assert_allclose(dense @ dx.numpy().reshape(-1), -rhs, rtol=1e-3, atol=1e-3)


def test_solve_pose_blocks_gives_a_zero_step_when_the_system_is_singular() -> None:
    hs = torch.zeros(4, 1, POSE_DIM, POSE_DIM) - torch.eye(POSE_DIM)  # negative definite: Cholesky fails
    dx = solve_pose_blocks(hs, torch.ones(2, 1, POSE_DIM), torch.tensor([-1]), torch.tensor([0]), 1)
    assert torch.count_nonzero(dx) == 0


@given(width=st.integers(64, 1600), aspect=st.floats(0.5, 2.0), size=st.sampled_from([224, 512]), data=st.data())
@settings(max_examples=40, deadline=None)
def test_frame_intrinsics_map_projections_through_resize_and_crop(width: int, aspect: float, size: int, data: st.DataObject) -> None:
    height: int = max(32, round(width * aspect))
    fx, fy = data.draw(st.floats(50.0, 2000.0)), data.draw(st.floats(50.0, 2000.0))
    K = np.array([[fx, 0.0, width / 2 + 3.0], [0.0, fy, height / 2 - 2.0], [0.0, 0.0, 1.0]])
    point = np.array([data.draw(st.floats(-1.0, 1.0)), data.draw(st.floats(-1.0, 1.0)), data.draw(st.floats(0.5, 5.0))])
    _, crop = resize_img_with_transform(np.zeros((height, width, 3), dtype=np.float32), size)  # type: ignore[arg-type]
    native = K @ point / point[2]
    expected = np.array([native[0] / crop.scale_w - crop.half_crop_w, native[1] / crop.scale_h - crop.half_crop_h])
    framed = frame_intrinsics(K, height, width, size) @ point / point[2]  # type: ignore[arg-type]
    np.testing.assert_allclose(framed[:2], expected, rtol=1e-4, atol=1e-2)


@given(scale=st.floats(0.05, 20.0), rotvec=arrays(np.float64, 3, elements=st.floats(-3.0, 3.0)), shift=arrays(np.float64, 3, elements=st.floats(-10.0, 10.0)))
@settings(max_examples=40, deadline=None)
def test_ate_is_invariant_to_a_sim3_of_the_estimate(scale: float, rotvec: np.ndarray, shift: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    gt_t = np.arange(0, 200_000_000, 1_000_000, dtype=np.int64)
    gt_xyz = np.cumsum(rng.normal(size=(gt_t.size, 3)), axis=0)
    est_t = gt_t[::7] + 1_000_000 // 2
    est_xyz = gt_xyz[::7] + rng.normal(scale=0.3, size=(est_t.size, 3))
    moved = scale * Rotation.from_rotvec(rotvec).apply(est_xyz) + shift
    base, moved_ate = _ate(est_t, est_xyz, gt_t, gt_xyz), _ate(est_t, moved, gt_t, gt_xyz)
    assert base[2] == moved_ate[2] == est_t.size
    np.testing.assert_allclose(moved_ate[0], base[0], rtol=1e-6)


def test_ate_associates_only_within_the_gap_it_is_given() -> None:
    gt_t = np.arange(0, 64_000_000, 16_000_000, dtype=np.int64)  # GT 16 ms apart: every estimate below is 8 ms from both neighbours
    gt_xyz = np.random.default_rng(1).normal(size=(gt_t.size, 3))
    est_t = gt_t[:-1] + 8_000_000
    assert _ate(est_t, gt_xyz[:-1], gt_t, gt_xyz, 5_000_000)[2] == 0
    assert _ate(est_t, gt_xyz[:-1], gt_t, gt_xyz, 10_000_000)[2] == est_t.size
