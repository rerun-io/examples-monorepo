"""Loop closure's pose-graph normal equations, matrix-free, for the preconditioned conjugate gradient in
`dpvo.fastba.solve_system`.

Edge e links poses ii[e] and jj[e] through 7x7 Jacobian blocks Ji[e] and Jj[e] (Sim(3) tangent) and a
7-vector residual. The normal matrix A = J^T J is never formed: every node gathers over its incident edges
(CSR lists: `offsets` (n + 1,), `incident` (2E,) holding 2 e + side, side 0 for ii and 1 for jj), so the
sums are deterministic and need no atomics. Everything is float64.

- `pgo_blocks_element`: node k's diagonal block sum J_s^T J_s (the block-Jacobi preconditioner) and
  gradient sum J_s^T r.
- `pgo_apply_element`: (A x)[k] + damp[k] * x[k], with A x = J^T (J x) per incident edge.
- `pgo_chain_factor` / `pgo_chain_solve`: the preconditioner, an exact block Cholesky of the odometry chain
  (the block-tridiagonal part of the damped A: diagonal blocks `diag` (n, 7, 7) and the (k + 1, k) blocks
  `sub` (n - 1, 7, 7)). A long chain has a condition number that grows like n^2; with the chain solved exactly,
  the loop edges are a low-rank correction and the conjugate gradient needs few iterations. Both are one
  sequential sweep over the poses (one thread).
"""

from max.gpu import global_idx
from std.math import sqrt

from layout import ImmTileTensor, MutTileTensor

from ba import L2
from corr import L1, L3

comptime F64x8 = SIMD[DType.float64, 8]
"""A 7-vector of the Sim(3) tangent in one SIMD register (lane 7 unused)."""


@always_inline
def _block(
    Ji: ImmTileTensor[DType.float64, L3, _], Jj: ImmTileTensor[DType.float64, L3, _], e: Int, side: Int, row: Int, col: Int
) -> Float64:
    comptime assert Ji.flat_rank == 3 and Jj.flat_rank == 3
    return Ji[e, row, col][0] if side == 0 else Jj[e, row, col][0]


@always_inline
def _row(x: ImmTileTensor[DType.float64, L2, _], k: Int) -> F64x8:
    comptime assert x.flat_rank == 2
    var v = F64x8(0)
    comptime for c in range(7):
        v[c] = x[k, c][0]
    return v


def pgo_blocks_element(
    k: Int,
    Ji: ImmTileTensor[DType.float64, L3, _],
    Jj: ImmTileTensor[DType.float64, L3, _],
    res: ImmTileTensor[DType.float64, L2, _],
    offsets: ImmTileTensor[DType.int64, L1, _],
    incident: ImmTileTensor[DType.int64, L1, _],
    blocks: MutTileTensor[DType.float64, L3, _],
    grad: MutTileTensor[DType.float64, L2, _],
):
    """blocks[k] = sum of J_s^T J_s and grad[k] = sum of J_s^T r over the edges incident to node k."""
    comptime assert res.flat_rank == 2 and offsets.flat_rank == 1 and incident.flat_rank == 1
    comptime assert blocks.flat_rank == 3 and grad.flat_rank == 2
    var g = F64x8(0)
    var b = Array[F64x8, 7](fill=F64x8(0))
    for a in range(Int(offsets[k][0]), Int(offsets[k + 1][0])):
        var code = Int(incident[a][0])
        var e = code >> 1
        var side = code & 1
        comptime for r in range(7):
            var j_row = F64x8(0)
            comptime for c in range(7):
                j_row[c] = _block(Ji, Jj, e, side, r, c)
            g += j_row * res[e, r][0]
            comptime for c in range(7):
                b[c] += j_row * j_row[c]
    comptime for c1 in range(7):
        grad[k, c1] = g[c1]
        comptime for c2 in range(7):
            blocks[k, c1, c2] = b[c1][c2]


def pgo_apply_element(
    k: Int,
    Ji: ImmTileTensor[DType.float64, L3, _],
    Jj: ImmTileTensor[DType.float64, L3, _],
    ii: ImmTileTensor[DType.int64, L1, _],
    jj: ImmTileTensor[DType.int64, L1, _],
    offsets: ImmTileTensor[DType.int64, L1, _],
    incident: ImmTileTensor[DType.int64, L1, _],
    damp: ImmTileTensor[DType.float64, L2, _],
    x: ImmTileTensor[DType.float64, L2, _],
    y: MutTileTensor[DType.float64, L2, _],
):
    """y[k] = (J^T J x)[k] + damp[k] * x[k]."""
    comptime assert ii.flat_rank == 1 and jj.flat_rank == 1 and offsets.flat_rank == 1 and incident.flat_rank == 1
    comptime assert damp.flat_rank == 2 and y.flat_rank == 2
    var acc = _row(damp, k) * _row(x, k)
    for a in range(Int(offsets[k][0]), Int(offsets[k + 1][0])):
        var code = Int(incident[a][0])
        var e = code >> 1
        var side = code & 1
        var xi = _row(x, Int(ii[e][0]))
        var xj = _row(x, Int(jj[e][0]))
        comptime for r in range(7):
            var ji_row = F64x8(0)
            var jj_row = F64x8(0)
            var own_row = F64x8(0)
            comptime for c in range(7):
                ji_row[c] = Ji[e, r, c][0]
                jj_row[c] = Jj[e, r, c][0]
            own_row = ji_row if side == 0 else jj_row
            var w = (ji_row * xi).reduce_add() + (jj_row * xj).reduce_add()
            acc += own_row * w
    comptime for c in range(7):
        y[k, c] = acc[c]


def pgo_blocks_kernel(
    Ji: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    Jj: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    res: ImmTileTensor[DType.float64, L2, ImmUntrackedOrigin],
    offsets: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    incident: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    blocks: MutTileTensor[DType.float64, L3, MutUntrackedOrigin],
    grad: MutTileTensor[DType.float64, L2, MutUntrackedOrigin],
):
    var k = global_idx.x
    if k < Int(grad.dim[0]()):
        pgo_blocks_element(k, Ji, Jj, res, offsets, incident, blocks, grad)


def pgo_apply_kernel(
    Ji: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    Jj: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    ii: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    jj: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    offsets: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    incident: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    damp: ImmTileTensor[DType.float64, L2, ImmUntrackedOrigin],
    x: ImmTileTensor[DType.float64, L2, ImmUntrackedOrigin],
    y: MutTileTensor[DType.float64, L2, MutUntrackedOrigin],
):
    var k = global_idx.x
    if k < Int(y.dim[0]()):
        pgo_apply_element(k, Ji, Jj, ii, jj, offsets, incident, damp, x, y)


comptime Block = Array[F64x8, 7]
"""A 7x7 matrix as seven SIMD rows (lane 7 unused)."""


@always_inline
def _load_block(t: ImmTileTensor[DType.float64, L3, _], k: Int) -> Block:
    comptime assert t.flat_rank == 3
    var m = Block(fill=F64x8(0))
    comptime for r in range(7):
        comptime for c in range(7):
            m[r][c] = t[k, r, c][0]
    return m^


@always_inline
def _store_block(t: MutTileTensor[DType.float64, L3, _], k: Int, m: Block):
    comptime assert t.flat_rank == 3
    comptime for r in range(7):
        comptime for c in range(7):
            t[k, r, c] = m[r][c]


@always_inline
def _cholesky(a: Block) -> Block:
    """Lower-triangular L with L L^T = a."""
    var l = Block(fill=F64x8(0))
    comptime for j in range(7):
        var d = a[j][j] - (l[j] * l[j]).reduce_add()
        var root = sqrt(d)
        l[j][j] = root
        comptime for i in range(j + 1, 7):
            l[i][j] = (a[i][j] - (l[i] * l[j]).reduce_add()) / root
    return l^


@always_inline
def _forward(l: Block, b: F64x8) -> F64x8:
    """y with L y = b (L lower-triangular)."""
    var y = F64x8(0)
    comptime for i in range(7):
        y[i] = (b[i] - (l[i] * y).reduce_add()) / l[i][i]
    return y


@always_inline
def _backward(l: Block, b: F64x8) -> F64x8:
    """x with L^T x = b (L lower-triangular)."""
    var x = F64x8(0)
    comptime for step in range(7):
        comptime i = 6 - step
        var s = b[i]
        comptime for j in range(i + 1, 7):
            s -= l[j][i] * x[j]
        x[i] = s / l[i][i]
    return x


@always_inline
def _mat_vec(m: Block, v: F64x8) -> F64x8:
    var y = F64x8(0)
    comptime for i in range(7):
        y[i] = (m[i] * v).reduce_add()
    return y


@always_inline
def _mat_t_vec(m: Block, v: F64x8) -> F64x8:
    var y = F64x8(0)
    comptime for i in range(7):
        y += m[i] * v[i]
    return y


def pgo_chain_factor(
    diag: ImmTileTensor[DType.float64, L3, _],
    sub: ImmTileTensor[DType.float64, L3, _],
    l_diag: MutTileTensor[DType.float64, L3, _],
    l_sub: MutTileTensor[DType.float64, L3, _],
):
    """Block Cholesky of the block-tridiagonal matrix (diag, sub): l_diag[k] lower-triangular, l_sub[k] = sub[k] l_diag[k]^-T."""
    var n = Int(diag.dim[0]())
    var below = Block(fill=F64x8(0))
    for k in range(n):
        var s = _load_block(diag, k)
        if k > 0:
            comptime for i in range(7):
                comptime for j in range(7):
                    s[i][j] -= (below[i] * below[j]).reduce_add()
        var l = _cholesky(s)
        _store_block(l_diag, k, l)
        if k + 1 < n:
            var b = _load_block(sub, k)
            comptime for i in range(7):
                below[i] = _forward(l, b[i])
            _store_block(l_sub, k, below)


def pgo_chain_solve(
    l_diag: ImmTileTensor[DType.float64, L3, _],
    l_sub: ImmTileTensor[DType.float64, L3, _],
    rhs: ImmTileTensor[DType.float64, L2, _],
    solution: MutTileTensor[DType.float64, L2, _],
):
    """solution = (L L^T)^-1 rhs for the block Cholesky factor of `pgo_chain_factor`."""
    comptime assert solution.flat_rank == 2
    var n = Int(l_diag.dim[0]())
    var y = F64x8(0)
    for k in range(n):
        var b = _row(rhs, k)
        if k > 0:
            b -= _mat_vec(_load_block(l_sub, k - 1), y)
        y = _forward(_load_block(l_diag, k), b)
        comptime for c in range(7):
            solution[k, c] = y[c]
    var x = F64x8(0)
    for step in range(n):
        var k = n - 1 - step
        var b = F64x8(0)
        comptime for c in range(7):
            b[c] = solution[k, c][0]
        if k + 1 < n:
            b -= _mat_t_vec(_load_block(l_sub, k), x)
        x = _backward(_load_block(l_diag, k), b)
        comptime for c in range(7):
            solution[k, c] = x[c]


def pgo_chain_factor_kernel(
    diag: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    sub: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    l_diag: MutTileTensor[DType.float64, L3, MutUntrackedOrigin],
    l_sub: MutTileTensor[DType.float64, L3, MutUntrackedOrigin],
):
    if global_idx.x == 0:
        pgo_chain_factor(diag, sub, l_diag, l_sub)


def pgo_chain_solve_kernel(
    l_diag: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    l_sub: ImmTileTensor[DType.float64, L3, ImmUntrackedOrigin],
    rhs: ImmTileTensor[DType.float64, L2, ImmUntrackedOrigin],
    solution: MutTileTensor[DType.float64, L2, MutUntrackedOrigin],
):
    if global_idx.x == 0:
        pgo_chain_solve(l_diag, l_sub, rhs, solution)
