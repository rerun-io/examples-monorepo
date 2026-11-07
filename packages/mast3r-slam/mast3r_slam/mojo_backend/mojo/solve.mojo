"""The pose step of one Gauss-Newton iteration on the CPU, on one thread (the CUDA extension's Eigen step:
`SparseBlock` + `SimplicialLLT`, then dx = -x).

`gn_solve` assembles H and g in float64 from the per-edge sums of `gn.mojo` (the lower triangle of each 14 × 14
[T_i, T_j] block, then the gradients of T_i and T_j) and solves H x = g by a Cholesky factorisation that keeps
the envelope of H: row r stores columns first[r] .. r, where first[r] is the first column of the earliest pose
that shares an edge with r's pose. L has the same envelope, so a graph whose edges reach back a few keyframes
costs about n w² operations, not n³ / 3. No pivoting, as Eigen's LLT with its fill-reducing ordering.
"""

from std.math import sqrt

comptime POSE_DIM = 7
comptime H_DIM = 14 * (14 + 1) // 2
comptime SUMS = H_DIM + 14

comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime F64Ptr = MutPointer[Float64, MutUntrackedOrigin]
comptime I64Ptr = MutPointer[Int64, MutUntrackedOrigin]


@always_inline
def _dot(p: ImmPointer[Float64, _], q: ImmPointer[Float64, _], start: Int, end: Int) -> Float64:
    """Sum of p[k] * q[k] over k in [start, end)."""
    var acc = SIMD[DType.float64, 4](0)
    var k = start
    while k + 4 <= end:
        acc += p.unsafe_load[width=4](k) * q.unsafe_load[width=4](k)
        k += 4
    var s = acc.reduce_add()
    while k < end:
        s += p[unsafe_offset=k] * q[unsafe_offset=k]
        k += 1
    return s


def gn_solve(num_poses: Int, num_edges: Int, sums: F32Ptr, ii: I64Ptr, jj: I64Ptr, dx: F32Ptr) -> Bool:
    """dx (num_poses, 7) = -H⁻¹ g, or zeros when H is not positive definite. ii / jj (num_edges,) are the edges'
    pose indices among the optimised poses; a negative index is a fixed pose and its blocks are dropped."""
    var dim = POSE_DIM * num_poses
    var first_pose = List[Int](capacity=num_poses)
    for p in range(num_poses):
        first_pose.append(p)
    for e in range(num_edges):
        var a = Int(ii[unsafe_offset=e])
        var b = Int(jj[unsafe_offset=e])
        if a >= 0 and b >= 0:
            first_pose[max(a, b)] = min(first_pose[max(a, b)], min(a, b))
    # Row r of the envelope is env[base[r] + c] for c in first[r] .. r.
    var first = List[Int](capacity=dim)
    var base = List[Int](capacity=dim)
    var size = 0
    for r in range(dim):
        var f = POSE_DIM * first_pose[r // POSE_DIM]
        first.append(f)
        base.append(size - f)
        size += r - f + 1
    var env = List[Float64](length=size, fill=0.0)
    var x = List[Float64](length=dim, fill=0.0)
    var h = env.unsafe_ptr()

    for e in range(num_edges):
        var pose = (Int(ii[unsafe_offset=e]), Int(jj[unsafe_offset=e]))
        var row = sums.unsafe_offset(e * SUMS)
        for n in range(14):
            var pn = pose[0] if n < POSE_DIM else pose[1]
            if pn < 0:
                continue
            var rn = POSE_DIM * pn + n % POSE_DIM
            x[rn] += Float64(row[unsafe_offset=H_DIM + n])
            for m in range(n + 1):
                var pm = pose[0] if m < POSE_DIM else pose[1]
                if pm < 0:
                    continue
                var rm = POSE_DIM * pm + m % POSE_DIM
                var value = Float64(row[unsafe_offset=n * (n + 1) // 2 + m])
                # The triangle of the [T_i, T_j] block covers both off-diagonal blocks once; H keeps its lower half.
                if rn >= rm:
                    h[unsafe_offset=base[rn] + rm] += value
                else:
                    h[unsafe_offset=base[rm] + rn] += value

    var ok = True
    for r in range(dim):
        var fr = first[r]
        var row = h.unsafe_offset(base[r])
        for c in range(fr, r + 1):
            var s = row[unsafe_offset=c] - _dot(row, h.unsafe_offset(base[c]), max(fr, first[c]), c)
            if c == r:
                if not (s > 0.0):
                    ok = False
                    break
                row[unsafe_offset=r] = sqrt(s)
            else:
                row[unsafe_offset=c] = s / h[unsafe_offset=base[c] + c]
        if not ok:
            break
    if not ok:
        for k in range(dim):
            dx[unsafe_offset=k] = 0.0
        return False

    var xp = x.unsafe_ptr()
    for r in range(dim):  # L y = g
        xp[unsafe_offset=r] = (xp[unsafe_offset=r] - _dot(h.unsafe_offset(base[r]), xp, first[r], r)) / h[unsafe_offset=base[r] + r]
    var r = dim - 1
    while r >= 0:  # Lᵀ x = y
        var xr = xp[unsafe_offset=r] / h[unsafe_offset=base[r] + r]
        xp[unsafe_offset=r] = xr
        for c in range(first[r], r):
            xp[unsafe_offset=c] -= h[unsafe_offset=base[r] + c] * xr
        r -= 1
    for k in range(dim):
        dx[unsafe_offset=k] = Float32(-xp[unsafe_offset=k])
    return True
