"""Optimization utilities for loop closure.

Contains two groups of functions:
1. **Edge reduction** (``reduce_edges``, ``umeyama_alignment``, ``ransac_umeyama``):
   Only need numba + numpy.  Used by both GPU and classical loop closure.
2. **PGO** (``residual``, ``perform_updates``, ``run_DPVO_PGO``, etc.):
   Additionally need pypose, scipy, einops.  Only used by classical loop closure.

The heavy dependencies (pypose, scipy) are imported lazily so that GPU
loop closure works without them.
"""

import numba as nb
import numpy as np
import torch

# NOTE: pypose, scipy, einops are imported lazily inside functions that need them,
# so that the GPU loop closure path (reduce_edges only) works without these deps.


def make_pypose_Sim3(rot: np.ndarray, t: np.ndarray, s: float) -> object:
    """Create a PyPose Sim(3) element from rotation matrix, translation, and scale."""
    import pypose as pp
    from scipy.spatial.transform import Rotation as R

    q = R.from_matrix(rot).as_quat()
    data = np.concatenate([t, q, np.array(s).reshape((1,))])
    return pp.Sim3(data)


def SE3_to_Sim3(x: object) -> object:
    """Convert SE3 to Sim3 with unit scale."""
    import pypose as pp

    out = torch.cat((x.data, torch.ones_like(x.data[..., :1])), dim=-1)
    return pp.Sim3(out)


@nb.njit(cache=True)
def _format(es):
    return np.asarray(es, dtype=np.int64).reshape((-1, 2))[1:]


@nb.njit(cache=True)
def reduce_edges(flow_mag, ii, jj, max_num_edges, nms):
    """NMS filtering of loop closure edge candidates sorted by flow magnitude."""
    es = [(-1, -1)]

    if ii.size == 0:
        return _format(es)

    Ni, Nj = (ii.max() + 1), (jj.max() + 1)
    ignore_lookup = np.zeros((Ni, Nj), dtype=nb.bool_)

    idxs = np.argsort(flow_mag)
    for idx in idxs:
        if len(es) > max_num_edges:
            break

        i = ii[idx]
        j = jj[idx]
        mag = flow_mag[idx]

        if (j - i) < 30:
            continue

        if mag >= 1000:
            continue

        if ignore_lookup[i, j]:
            continue

        es.append((i, j))

        for di in range(-nms, nms + 1):
            i1 = i + di
            if 0 <= i1 < Ni:
                ignore_lookup[i1, j] = True

    return _format(es)


@nb.njit(cache=True)
def umeyama_alignment(x, y):
    """Umeyama alignment: least-squares Sim(m) between two point sets.

    See Umeyama (1991), "Least-squares estimation of transformation
    parameters between two point patterns".
    """
    m, n = x.shape

    mean_x = x.sum(axis=1) / n
    mean_y = y.sum(axis=1) / n

    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        return None, None, None

    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1

    r = u.dot(s).dot(v)
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s))
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


@nb.njit(cache=True)
def ransac_umeyama(src_points, dst_points, iterations=1, threshold=0.1):
    """RANSAC wrapper around Umeyama alignment for robust Sim(3) estimation."""
    best_inliers = 0
    best_R = None
    best_t = None
    best_s = None
    for _ in range(iterations):
        indices = np.random.choice(src_points.shape[0], 3, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        rot, t, s = umeyama_alignment(src_sample.T, dst_sample.T)
        if t is None:
            continue

        transformed = (src_points @ (rot * s).T) + t
        distances = np.sum((transformed - dst_points) ** 2, axis=1) ** 0.5
        inlier_mask = distances < threshold
        inliers = np.sum(inlier_mask)

        if inliers > best_inliers:
            best_R, best_t, best_s = umeyama_alignment(src_points[inlier_mask].T, dst_points[inlier_mask].T)
            best_inliers = inliers

        if inliers > 100:
            break

    return best_R, best_t, best_s, best_inliers


def batch_jacobian(func, x):
    from einops import rearrange

    def _func_sum(*x):
        return func(*x).sum(dim=0)
    _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
    return rearrange(torch.stack((b, c)), "N O B I -> N B O I", N=2)


def _residual(C, Gi, Gj):
    import pypose as pp
    from einops import parse_shape

    assert parse_shape(C, "N _") == parse_shape(Gi, "N _") == parse_shape(Gj, "N _")
    out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
    return out.Log().tensor()


def residual(Ginv, input_poses, dSloop, ii, jj, jacobian=False):
    from einops import parse_shape

    device = Ginv.device
    assert parse_shape(input_poses, "_ d") == dict(d=7)
    pred_inv_poses = SE3_to_Sim3(input_poses).Inv()

    n, _ = pred_inv_poses.shape
    kk = torch.arange(1, n, device=device)
    ll = kk - 1

    Ti = pred_inv_poses[kk]
    Tj = pred_inv_poses[ll]
    dSij = Tj @ Ti.Inv()

    constants = torch.cat((dSij, dSloop), dim=0)
    iii = torch.cat((kk, ii))
    jjj = torch.cat((ll, jj))
    resid = _residual(constants, Ginv[iii], Ginv[jjj])

    if not jacobian:
        return resid

    J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
    return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)


def run_DPVO_PGO(pred_poses, loop_poses, loop_ii, loop_jj, queue):
    """Wraps PGO in a multiprocessing-friendly function."""
    final_est = perform_updates(pred_poses, loop_poses, loop_ii, loop_jj, iters=30)

    safe_i = loop_ii.max().item() + 1
    aa = SE3_to_Sim3(pred_poses.cpu())
    final_est = (aa[[safe_i]] * final_est[[safe_i]].Inv()) * final_est
    output = final_est[:safe_i]
    queue.put(output)


def perform_updates(input_poses, dSloop, ii_loop, jj_loop, iters, ep=0.0, lmbda=1e-6, fix_opt_window=False):
    """Run the Levenberg-Marquardt algorithm for pose-graph optimization."""
    import pypose as pp

    from .. import _cuda_ba

    input_poses = input_poses.clone()

    freen = torch.cat((ii_loop, jj_loop)).max().item() + 1 if fix_opt_window else -1

    Ginv = SE3_to_Sim3(input_poses).Inv().Log()

    residual_history: list[float] = []

    for itr in range(iters):
        resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = residual(Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)
        residual_history.append(resid.square().mean().item())
        (delta_pose,) = _cuda_ba.solve_system(J_Ginv_i, J_Ginv_j, iii, jjj, resid, ep, lmbda, freen)
        assert Ginv.shape == delta_pose.shape
        Ginv_tmp = Ginv + delta_pose

        new_resid = residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
        if new_resid.square().mean() < residual_history[-1]:
            Ginv = Ginv_tmp
            lmbda /= 2
        else:
            lmbda *= 2

        if (residual_history[-1] < 1e-5) and (itr >= 4) and ((residual_history[-5] / residual_history[-1]) < 1.5):
            break

    return pp.Exp(Ginv).Inv()
