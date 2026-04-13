"""Gauss-Newton bundle adjustment with Schur complement for DPVO.

Implements a single iteration of the weighted Gauss-Newton optimisation
that jointly refines SE3 camera poses and per-patch inverse depths.
The key algorithmic idea is the **Schur complement** trick: because each
depth variable only connects to at most two poses, the depth block of the
Hessian is diagonal and can be eliminated analytically, reducing the
system from ``(6N + M)`` unknowns to a dense ``6N × 6N`` system (where
``N`` is the number of active poses and ``M`` the number of patches).
See Sec. 3.3 of Teed et al. (2022).

The normal equations are assembled from analytical Jacobians provided by
:func:`~dpvo.projective_ops.transform`:

    [ B   E ] [δx] = [v]
    [ Eᵀ  C ] [δz]   [w]

where ``B`` is the pose-pose Hessian block (6×6 per pose pair), ``E`` is
the pose-depth cross term, ``C`` is the depth-depth diagonal, and ``v``,
``w`` are the corresponding gradient vectors.

After eliminating depths via the Schur complement
``S = B - E · C⁻¹ · Eᵀ``, the reduced system ``S · dX = v - E · C⁻¹ · w``
is solved with :class:`CholeskySolver` (Cholesky factorisation).  Depths
are then back-substituted: ``dZ = C⁻¹ · (w - Eᵀ · dX)``.

Updates are applied via retraction on SE3 (exponential map) for poses
and additive update for inverse depths.
"""

import torch
from jaxtyping import Bool, Float, Int
from lietorch import SE3
from torch import Tensor

from . import projective_ops as pops
from .scatter_utils import scatter_sum


class CholeskySolver(torch.autograd.Function):
    """Differentiable Cholesky linear solver with graceful failure.

    Solves ``H x = b`` where ``H`` is symmetric positive-definite, using
    Cholesky decomposition.  If the decomposition fails (``H`` is not SPD,
    which can happen with degenerate geometry), the forward pass returns
    zeros and the backward pass returns ``None`` gradients instead of
    crashing the training loop.

    This is essential for training stability: some edge configurations
    can produce rank-deficient Hessians, and silently returning zero
    updates is preferable to a hard failure.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, H: Float[Tensor, "batch n n"], b: Float[Tensor, "batch n m"]) -> Float[Tensor, "batch n m"]:
        """Solve H x = b via Cholesky decomposition.

        Args:
            ctx: Autograd context for saving tensors.
            H: Symmetric positive-definite matrix (the Hessian / Schur
                complement), shape ``(batch, n, n)``.
            b: Right-hand side, shape ``(batch, n, m)``.

        Returns:
            Solution ``x = H⁻¹ b``, or zeros if Cholesky fails.
        """
        # cholesky_ex returns info=0 on success, nonzero on failure
        U: Float[Tensor, "batch n n"]
        info: Int[Tensor, "batch"]
        U, info = torch.linalg.cholesky_ex(H)

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs: Float[Tensor, "batch n m"] = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_x: Float[Tensor, "batch n m"]) -> tuple[Float[Tensor, "batch n n"] | None, Float[Tensor, "batch n m"] | None]:
        """Compute gradients through the linear solve.

        Uses the implicit function theorem:
        ``dL/dH = -x · (H⁻¹ dL/dx)ᵀ`` and ``dL/db = H⁻¹ dL/dx``.

        Args:
            ctx: Autograd context with saved tensors.
            grad_x: Upstream gradient w.r.t. the solution ``x``.

        Returns:
            Gradients ``(dL/dH, dL/db)``, or ``(None, None)`` if the
            forward Cholesky failed.
        """
        if ctx.failed:
            return None, None

        U: Float[Tensor, "batch n n"]
        xs: Float[Tensor, "batch n m"]
        U, xs = ctx.saved_tensors
        dz: Float[Tensor, "batch n m"] = torch.cholesky_solve(grad_x, U)
        dH: Float[Tensor, "batch n n"] = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz


# ---------------------------------------------------------------------------
# Scatter utilities for assembling the normal equations
# ---------------------------------------------------------------------------

def safe_scatter_add_mat(
    A: Float[Tensor, "batch edges *rest"],
    ii: Int[Tensor, "edges"],
    jj: Int[Tensor, "edges"],
    n: int,
    m: int,
) -> Float[Tensor, "batch n_times_m *rest"]:
    """Scatter-add per-edge matrices into a flattened (n x m) block matrix.

    Edges with out-of-bounds indices are silently skipped (important
    because the first ``fixedp`` poses are shifted to negative indices
    to exclude them from optimisation).

    Args:
        A: Per-edge block matrices, shape ``(batch, edges, ...)``.
        ii: Row block indices for each edge.
        jj: Column block indices for each edge.
        n: Number of row blocks.
        m: Number of column blocks.

    Returns:
        Accumulated block matrix of shape ``(batch, n*m, ...)``, which
        the caller reshapes to ``(batch, n, m, ...)``.
    """
    v: Bool[Tensor, "edges"] = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)


def safe_scatter_add_vec(
    b: Float[Tensor, "batch edges *rest"],
    ii: Int[Tensor, "edges"],
    n: int,
) -> Float[Tensor, "batch n *rest"]:
    """Scatter-add per-edge vectors into a length-n accumulator.

    Edges with out-of-bounds indices are silently skipped.

    Args:
        b: Per-edge vectors, shape ``(batch, edges, ...)``.
        ii: Destination indices for each edge.
        n: Output dimension size.

    Returns:
        Accumulated vector of shape ``(batch, n, ...)``.
    """
    v: Bool[Tensor, "edges"] = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)


# ---------------------------------------------------------------------------
# Retraction operators
# ---------------------------------------------------------------------------

def disp_retr(
    disps: Float[Tensor, "batch n_patches ps ps"],
    dz: Float[Tensor, "batch edges 1 1"],
    ii: Int[Tensor, "edges"],
) -> Float[Tensor, "batch n_patches ps ps"]:
    """Apply additive retraction to inverse-depth maps.

    Accumulates per-patch depth updates ``dz`` (which may be indexed by
    a compressed index from ``torch.unique``) back into the full disparity
    tensor using scatter-add.

    Args:
        disps: Current inverse-depth maps, shape ``(batch, n_patches, ps, ps)``.
        dz: Depth updates per unique patch, shape ``(batch, m, 1, 1)``.
        ii: Mapping from compressed patch indices back to full indices.

    Returns:
        Updated inverse-depth tensor (same shape as ``disps``).
    """
    ii: Int[Tensor, "edges"] = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])


def pose_retr(
    poses: SE3,
    dx: Float[Tensor, "batch edges 6"],
    ii: Int[Tensor, "edges"],
) -> SE3:
    """Apply retraction on the SE3 manifold to update poses.

    Accumulates per-pose Lie algebra updates ``dx`` via scatter-add (in
    case multiple edges contribute to the same pose), then applies the
    retraction ``pose_new = pose · exp(accumulated_dx)`` via lietorch's
    ``.retr()`` method.

    Args:
        poses: Current SE3 poses.
        dx: Lie algebra updates (6-DoF: translation + rotation), one
            per optimised pose.
        ii: Pose indices for each update.

    Returns:
        Updated SE3 poses.
    """
    ii: Int[Tensor, "edges"] = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


# ---------------------------------------------------------------------------
# Block matrix operations for the 6×6 pose blocks
# ---------------------------------------------------------------------------

def block_matmul(
    A: Float[Tensor, "b n1 m1 p1 q1"],
    B: Float[Tensor, "b n2 m2 p2 q2"],
) -> Float[Tensor, "b n1 m2 p1 q2"]:
    """Block matrix multiplication.

    Treats the 5-D tensors as block matrices where each ``(p, q)`` sub-block
    is an element.  Reshapes to standard 2-D matrices, performs matmul,
    then reshapes back.

    For DPVO, this multiplies ``(n, m, 6, 6)`` blocks -- e.g. computing
    the Schur complement ``E · Q · Eᵀ`` or the back-substitution
    ``Eᵀ · dX``.

    Args:
        A: Left block matrix, shape ``(b, n1, m1, p1, q1)``.
        B: Right block matrix, shape ``(b, n2, m2, p2, q2)``.
            ``m1`` must equal ``n2`` and ``q1`` must equal ``p2``.

    Returns:
        Product block matrix, shape ``(b, n1, m2, p1, q2)``.
    """
    b: int
    n1: int
    m1: int
    p1: int
    q1: int
    b, n1, m1, p1, q1 = A.shape
    _b: int
    n2: int
    m2: int
    p2: int
    q2: int
    _b, n2, m2, p2, q2 = B.shape
    # Interleave block and sub-block dims to get standard matrix layout
    A: Float[Tensor, "b n1_p1 m1_q1"] = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B: Float[Tensor, "b n2_p2 m2_q2"] = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_solve(
    A: Float[Tensor, "b n1 m1 p1 q1"],
    B: Float[Tensor, "b n2 m2 p2 q2"],
    ep: float = 1.0,
    lm: float = 1e-4,
) -> Float[Tensor, "b n1 m2 p1 q2"]:
    """Solve a block linear system ``A x = B`` with Levenberg-Marquardt damping.

    Reshapes to standard matrix form, adds Levenberg-Marquardt-style
    regularisation ``(ep + lm · diag(A)) · I`` to the diagonal for
    numerical stability, then solves via :class:`CholeskySolver`.

    Args:
        A: Block Hessian matrix (the Schur complement S).
        B: Block right-hand side.
        ep: Constant damping added to diagonal (default 1.0).
        lm: Multiplicative damping factor for ``diag(A)`` (default 1e-4).

    Returns:
        Solution ``x`` as a block matrix.
    """
    b: int
    n1: int
    m1: int
    p1: int
    q1: int
    b, n1, m1, p1, q1 = A.shape
    _b: int
    n2: int
    m2: int
    p2: int
    q2: int
    _b, n2, m2, p2, q2 = B.shape
    A: Float[Tensor, "b n1_p1 m1_q1"] = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B: Float[Tensor, "b n2_p2 m2_q2"] = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)

    # Levenberg-Marquardt damping: A' = A + (ep + lm * A) * I
    # This ensures A' is positive-definite even when A is near-singular.
    A: Float[Tensor, "b n1_p1 m1_q1"] = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)

    X: Float[Tensor, "b n1_p1 m2_q2"] = CholeskySolver.apply(A, B)
    return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A: Float[Tensor, "b n1 m1 p1 q1"]) -> None:
    """Visualize a block matrix as a heatmap (debugging utility).

    Args:
        A: Block matrix to visualize.
    """
    import matplotlib.pyplot as plt
    b: int
    n1: int
    m1: int
    p1: int
    q1: int
    b, n1, m1, p1, q1 = A.shape
    A: Float[Tensor, "b n1_p1 m1_q1"] = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()


# ---------------------------------------------------------------------------
# Main bundle adjustment function
# ---------------------------------------------------------------------------

def BA(
    poses: SE3,
    patches: Float[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float[Tensor, "1 n_frames 4"],
    targets: Float[Tensor, "1 n_edges 2"],
    weights: Float[Tensor, "1 n_edges 2"],
    lmbda: float | Float[Tensor, "..."],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    bounds: tuple[float, float, float, float],
    ep: float = 100.0,
    PRINT: bool = False,
    fixedp: int = 1,
    structure_only: bool = False,
) -> tuple[SE3, Float[Tensor, "1 n_patches 3 ps ps"]]:
    """One iteration of Gauss-Newton bundle adjustment with Schur complement.

    Jointly optimises SE3 camera poses and per-patch inverse depths by
    minimising the weighted reprojection error between predicted and
    target 2-D coordinates.

    The algorithm proceeds as follows:

    1. **Reproject** all patches and compute analytical Jacobians
       ``(Ji, Jj, Jz)`` via :func:`~dpvo.projective_ops.transform`.
    2. **Filter** edges: discard points behind the camera (``Z < 0.2``),
       with large residuals (> 250 px), or outside ``bounds``.
    3. **Assemble** the weighted normal equations:
       - ``B``: 6×6 pose-pose Hessian blocks (scatter-accumulated per pose pair).
       - ``E``: 6×1 pose-depth cross terms.
       - ``C``: 1×1 depth-depth diagonal (scalar per patch).
       - ``v``, ``w``: gradient vectors.
    4. **Schur complement**: eliminate depths to get
       ``S = B - E · C⁻¹ · Eᵀ`` and solve ``S · dX = v - E · C⁻¹ · w``
       for pose updates.
    5. **Back-substitute** to recover depth updates:
       ``dZ = C⁻¹ · (w - Eᵀ · dX)``.
    6. **Apply retractions**: SE3 exponential map for poses, additive
       update (clamped to [1e-3, 10]) for inverse depths.

    The first ``fixedp`` poses are held fixed (gauge freedom). By default
    ``fixedp=1``, so the first pose is the reference.

    See Sec. 3.3 of Teed et al. (2022) for the full derivation.

    Args:
        poses: SE3 camera poses, shape ``(1, N, 7)``.
        patches: Patch coordinates ``(x, y, inv_depth)``.
        intrinsics: Camera intrinsics ``(fx, fy, cx, cy)`` per frame.
        targets: Target 2-D pixel coordinates from the GRU update
            (reprojected center pixel + learned delta).
        weights: Per-edge confidence weights (sigmoid output of the GRU).
        lmbda: Damping parameter for the depth-depth diagonal ``C``.
        ii: Source frame index per edge.
        jj: Target frame index per edge.
        kk: Patch index per edge.
        bounds: ``(x_min, y_min, x_max, y_max)`` image bounds for
            filtering out-of-frame reprojections.
        ep: Levenberg-Marquardt constant damping for the Schur system.
        PRINT: If True, print mean residual magnitude.
        fixedp: Number of leading poses to hold fixed (default 1).
        structure_only: If True, only update depths (skip pose updates).

    Returns:
        A 2-tuple of ``(updated_poses, updated_patches)``.
    """
    b: int = 1
    n: int = max(ii.max().item(), jj.max().item()) + 1

    # ----- Step 1: Reproject all edges and compute Jacobians -----
    coords: Float[Tensor, "1 n_edges ps ps 2"]
    v: Float[Tensor, "1 n_edges ps ps"]
    Ji: Float[Tensor, "1 n_edges 2 6"]
    Jj: Float[Tensor, "1 n_edges 2 6"]
    Jz: Float[Tensor, "1 n_edges 2 1"]
    coords, v, (Ji, Jj, Jz) = \
        pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    # ----- Step 2: Compute residuals and validity mask -----
    p: int = coords.shape[3]
    # Reprojection residual: target pixel - current projection (center pixel)
    r: Float[Tensor, "1 n_edges 2"] = targets - coords[...,p//2,p//2,:]

    # Discard edges with points behind camera (v from transform is Z > 0.2)
    # and large residuals (likely outliers)
    v: Float[Tensor, "1 n_edges"] = v * (r.norm(dim=-1) < 250).float()

    # Discard edges whose reprojection falls outside image bounds
    in_bounds: Bool[Tensor, "1 n_edges"] = \
        (coords[...,p//2,p//2,0] > bounds[0]) & \
        (coords[...,p//2,p//2,1] > bounds[1]) & \
        (coords[...,p//2,p//2,0] < bounds[2]) & \
        (coords[...,p//2,p//2,1] < bounds[3])

    v: Float[Tensor, "1 n_edges"] = v * in_bounds.float()

    if PRINT:
        print((r * v[...,None]).norm(dim=-1).mean().item())

    # Apply validity mask to residuals and weights
    r: Float[Tensor, "1 n_edges 2 1"] = (v[...,None] * r).unsqueeze(dim=-1)
    weights: Float[Tensor, "1 n_edges 2 1"] = (v[...,None] * weights).unsqueeze(dim=-1)

    # ----- Step 3: Assemble the normal equations -----
    # Weighted Jacobian transposes: w · Jᵀ
    wJiT: Float[Tensor, "1 n_edges 6 2"] = (weights * Ji).transpose(2,3)
    wJjT: Float[Tensor, "1 n_edges 6 2"] = (weights * Jj).transpose(2,3)
    wJzT: Float[Tensor, "1 n_edges 1 2"] = (weights * Jz).transpose(2,3)

    # Pose-pose Hessian blocks: B = Jᵀ W J (accumulated for all 4 (i, j) combinations)
    Bii: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJiT, Ji)
    Bij: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJiT, Jj)
    Bji: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJjT, Ji)
    Bjj: Float[Tensor, "1 n_edges 6 6"] = torch.matmul(wJjT, Jj)

    # Pose-depth cross terms: E = J_poseᵀ W J_depth
    Eik: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJiT, Jz)
    Ejk: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJjT, Jz)

    # Gradient vectors: v = J_poseᵀ W r,  w = J_depthᵀ W r
    vi: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJiT, r)
    vj: Float[Tensor, "1 n_edges 6 1"] = torch.matmul(wJjT, r)

    # Fix first `fixedp` poses by shifting indices so they become negative
    # (and are filtered out by safe_scatter_add_*)
    ii: Int[Tensor, "n_edges"] = ii.clone()
    jj: Int[Tensor, "n_edges"] = jj.clone()

    n: int = n - fixedp
    ii: Int[Tensor, "n_edges"] = ii - fixedp
    jj: Int[Tensor, "n_edges"] = jj - fixedp

    # Compress patch indices to contiguous range [0, m)
    kx: Int[Tensor, "m"]
    kk: Int[Tensor, "n_edges"]
    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m: int = len(kx)

    # Scatter per-edge blocks into the global Hessian
    B: Float[Tensor, "1 n n 6 6"] = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)

    E: Float[Tensor, "1 n m 6 1"] = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1)

    # C is diagonal (scalar per patch): J_zᵀ W J_z
    C: Float[Tensor, "1 m 1 1"] = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)

    v: Float[Tensor, "1 n 1 6 1"] = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)

    w: Float[Tensor, "1 m 1 1"] = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk, m)

    if isinstance(lmbda, Tensor):
        lmbda: Float[Tensor, "..."] = lmbda.reshape(*C.shape)

    # Q = (C + λ)⁻¹: inverse of the damped depth diagonal
    Q: Float[Tensor, "1 m 1 1"] = 1.0 / (C + lmbda)

    # ----- Step 4 & 5: Schur complement solve -----
    # EQ = E · Q (pre-multiply for efficiency)
    EQ: Float[Tensor, "1 n m 6 1"] = E * Q[:,None]

    if structure_only or n == 0:
        # Structure-only: just update depths, no pose optimisation
        dZ: Float[Tensor, "1 m 1 1"] = (Q * w).view(b, -1, 1, 1)

    else:
        # Schur complement: S = B - E · Q · Eᵀ
        S: Float[Tensor, "1 n n 6 6"] = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        # Reduced RHS: y = v - E · Q · w
        y: Float[Tensor, "1 n 1 6 1"] = v - block_matmul(EQ, w.unsqueeze(dim=2))
        # Solve for pose updates: S · dX = y
        dX: Float[Tensor, "1 n 1 6 1"] = block_solve(S, y, ep=ep, lm=1e-4)

        # Back-substitute for depth updates: dZ = Q · (w - Eᵀ · dX)
        dZ: Float[Tensor, "1 m 1 1"] = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        dX: Float[Tensor, "1 n_poses 6"] = dX.view(b, -1, 6)
        dZ: Float[Tensor, "1 m 1 1"] = dZ.view(b, -1, 1, 1)

    # ----- Step 6: Apply retractions -----
    x: Float[Tensor, "1 n_patches ps ps"]
    y_coord: Float[Tensor, "1 n_patches ps ps"]
    disps: Float[Tensor, "1 n_patches ps ps"]
    x, y_coord, disps = patches.unbind(dim=2)
    # Additive retraction for inverse depths, clamped to valid range
    disps: Float[Tensor, "1 n_patches ps ps"] = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches: Float[Tensor, "1 n_patches 3 ps ps"] = torch.stack([x, y_coord, disps], dim=2)

    if not structure_only and n > 0:
        # SE3 retraction: pose_new = pose · exp(dX) for each optimised pose
        poses: SE3 = pose_retr(poses, dX, fixedp + torch.arange(n))

    return poses, patches
