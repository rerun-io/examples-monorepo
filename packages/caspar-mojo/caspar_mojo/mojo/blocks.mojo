"""The packed layout of the tangent vector and of the block diagonal of J^T J, and the dense block algebra on it.

Groups, in this order: the free poses then the free sensor_from_rig (6 each, one group), the free calibrations
(K each), the free points (3 each). The tangent vector holds each block's D values, the block diagonal each block's
upper triangle (tri(D) values), blocks in the same order. The host allocation, every kernel and the Python test
hook (`packed_groups`) take offsets from `Packed` only, so a group is added or moved here.
"""

from std.math import sqrt

from layout import ImmTileTensor, row_major

comptime FRAME_WIDTH = 6
"""Tangent width of a pose or a sensor_from_rig: rotation, then translation."""
comptime POINT_WIDTH = 3

comptime L1 = type_of(row_major((0,)))
comptime VecIn = ImmTileTensor[DType.float32, L1, ImmUntrackedOrigin]


def tri(d: Int) -> Int:
    """Packed upper-triangle size of a d x d block."""
    return d * (d + 1) // 2


def tri_index[D: Int](i: Int, j: Int) -> Int:
    return i * D - i * (i - 1) // 2 + (j - i)


@fieldwise_init
struct Packed[K: Int](TrivialRegisterPassable):
    """Block counts of the camera-side groups (the point group comes last, so no offset depends on its count).
    `*_h(v)` map a tangent offset `v` of that group to its block's packed offset, straight-line."""

    var n6: Int
    """Free poses plus free sensor_from_rig."""
    var n_calib: Int

    def calib_start(self) -> Int:
        return FRAME_WIDTH * self.n6

    def point_start(self) -> Int:
        """Also the camera part's length, the size of the reduced camera system."""
        return self.calib_start() + Self.K * self.n_calib

    def n_tan(self, n_points: Int) -> Int:
        return self.point_start() + POINT_WIDTH * n_points

    def calib_h_start(self) -> Int:
        return tri(FRAME_WIDTH) * self.n6

    def point_h_start(self) -> Int:
        return self.calib_h_start() + tri(Self.K) * self.n_calib

    def n_h(self, n_points: Int) -> Int:
        return self.point_h_start() + tri(POINT_WIDTH) * n_points

    def h6(self, v: Int) -> Int:
        return tri(FRAME_WIDTH) * (v // FRAME_WIDTH)

    def h_calib(self, v: Int) -> Int:
        return self.calib_h_start() + tri(Self.K) * ((v - self.calib_start()) // Self.K)

    def h_point(self, v: Int) -> Int:
        return self.point_h_start() + tri(POINT_WIDTH) * ((v - self.point_start()) // POINT_WIDTH)


@always_inline
def predicted_decrease[D: Int](s: VecIn, g: VecIn, hblk: VecIn, v: Int, h: Int, lam: Float32) -> Float32:
    """One block's share of s.(g + lambda diag(B) s), CASPAR's predicted decrease times two."""
    comptime assert s.flat_rank == 1 and g.flat_rank == 1 and hblk.flat_rank == 1
    var pred: Float32 = 0.0
    comptime for i in range(D):
        var si = s[v + i][0]
        pred += si * (g[v + i][0] + lam * hblk[h + tri_index[D](i, i)][0] * si)
    return pred


@always_inline
def damped_solve[D: Int](hblk: VecIn, h: Int, rhs: Array[Float32, D], lam: Float32) -> Array[Float32, D]:
    """(B + lambda diag(B) + lambda 1e-6 I)^-1 rhs for the packed block at `h` (CASPAR's `normalize` matrix)."""
    comptime assert hblk.flat_rank == 1
    var a = Array[Float32, D * D](fill=0.0)
    comptime for i in range(D):
        comptime for j in range(i, D):
            var hij = hblk[h + tri_index[D](i, j)][0]
            a[i * D + j] = hij
            a[j * D + i] = hij
        a[i * D + i] = a[i * D + i] * (1.0 + lam) + lam * 1e-6
    # In-place Cholesky, lower triangle.
    comptime for j in range(D):
        var d = a[j * D + j]
        comptime for k in range(j):
            d -= a[j * D + k] * a[j * D + k]
        var l = sqrt(d)
        a[j * D + j] = l
        comptime for i in range(j + 1, D):
            var s = a[i * D + j]
            comptime for k in range(j):
                s -= a[i * D + k] * a[j * D + k]
            a[i * D + j] = s / l
    var y = Array[Float32, D](fill=0.0)
    comptime for i in range(D):
        var s = rhs[i]
        comptime for k in range(i):
            s -= a[i * D + k] * y[k]
        y[i] = s / a[i * D + i]
    comptime for ii in range(D):
        comptime i = D - 1 - ii
        var s = y[i]
        comptime for k in range(i + 1, D):
            s -= a[k * D + i] * y[k]
        y[i] = s / a[i * D + i]
    return y^
