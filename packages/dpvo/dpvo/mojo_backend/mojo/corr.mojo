"""DPVO's correlation and patchify kernels (`dpvo/altcorr/correlation_kernel.cu`), one source for CPU and CUDA.

Each `*_element` function computes what one CUDA thread computes, with the CUDA kernel's index
decomposition and arithmetic order, so the GPU kernels and the CPU loop share it. Half-precision
inputs accumulate like c10::Half: every product and every sum is rounded to half, as the CUDA
kernel's `scalar_t s; s += f1 * f2` does.

Shapes (C = channels, D = 2R + 2, P x P = patch grid):
- corr: fmap1 (B, N1, C, P, P), fmap2 (B, N2, C, H2, W2), coords (B, E, 2, P, P) float32,
  us/vs (E,) int64 -> corr (B, E, D, D, P, P).
- patchify: net (B, C, H, W), coords (B, M, 2) float32 -> patches (B, M, C, D, D), zero-initialised
  by the caller (out-of-bounds pixels stay 0).
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.math import floor
from std.memory import bitcast
from std.utils.static_tuple import StaticTuple

from layout import ImmTileTensor, MutTileTensor, row_major

comptime NUM_THREADS = 256

comptime L1 = type_of(row_major((0,)))
comptime L3 = type_of(row_major((0, 0, 0)))
comptime L4 = type_of(row_major((0, 0, 0, 0)))
comptime L5 = type_of(row_major((0, 0, 0, 0, 0)))
comptime L6 = type_of(row_major((0, 0, 0, 0, 0, 0)))


@always_inline
def _within_bounds(h: Int, w: Int, height: Int, width: Int) -> Bool:
    return h >= 0 and h < height and w >= 0 and w < width


@always_inline
def _round_to_half(x: Float32) -> Float32:
    """`x` rounded to the nearest float16 value (ties to even), returned as float32, with integer operations only.

    Mojo compiles with fp-contract=fast. Written with casts (`fptrunc`, `fpext`), the half-precision
    multiply-add `s + a * b` below narrows to half arithmetic and fuses into one fma.f16, which skips
    the product's rounding that c10::Half does (up to 12 half ulps apart over 128 channels on
    pool.mp4). Integer bit operations are opaque to contraction. The contraction has to stay on for
    the BA kernels, where it reproduces nvcc's FMAs (`--fp-mode contract=off` broke their parity).
    """
    var bits = bitcast[DType.uint32](x)
    var sign = bits & 0x80000000
    var mag = bits & 0x7FFFFFFF
    if mag >= 0x477FF000:  # >= 65520: rounds to inf (65504 is the largest half)
        return bitcast[DType.float32](sign | 0x7F800000)
    if mag >= 0x38800000:  # half normal (>= 2^-14): keep 10 mantissa bits
        var lsb = (mag >> 13) & 1
        return bitcast[DType.float32](sign | ((mag + 0xFFF + lsb) & 0xFFFFE000))
    var exponent = Int(mag >> 23)
    if exponent == 0:  # float32 subnormal or zero: far below half's smallest subnormal (2^-24)
        return bitcast[DType.float32](sign)
    # Half subnormal: a multiple q of 2^-24. x = m * 2^(exponent - 150) = m * 2^(exponent - 126) units of 2^-24.
    var shift = 126 - exponent
    if shift > 24:
        return bitcast[DType.float32](sign)
    var m = (mag & 0x7FFFFF) | 0x800000
    var q = m >> UInt32(shift)
    var rem = m & ((UInt32(1) << UInt32(shift)) - 1)
    var halfway = UInt32(1) << UInt32(shift - 1)
    if rem > halfway or (rem == halfway and (q & 1) == 1):
        q += 1
    var value = Float32(Int(q)) * 5.9604644775390625e-08  # q * 2^-24, exact
    return -value if sign != 0 else value


@always_inline
def _mad[dtype: DType](s: Scalar[dtype], a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """`s + a * b` as c10::Half computes it for half (product and sum each rounded to half), plain for float32."""
    comptime match dtype:
        case .float16:
            var product = _round_to_half(a.cast[.float32]() * b.cast[.float32]())
            return (s.cast[.float32]() + product).cast[dtype]()
        case _:
            return s + a * b


@always_inline
def corr_element[
    dtype: DType
](
    n_in: Int,
    radius: Int,
    fmap1: ImmTileTensor[dtype, L5, _],
    fmap2: ImmTileTensor[dtype, L5, _],
    coords: ImmTileTensor[DType.float32, L5, _],
    us: ImmTileTensor[DType.int64, L1, _],
    vs: ImmTileTensor[DType.int64, L1, _],
    corr: MutTileTensor[dtype, L6, _],
):
    """Correlation at one (batch, edge, patch row, patch col, search row, search col): `corr_forward_kernel`'s body."""
    comptime assert fmap1.flat_rank == 5 and fmap2.flat_rank == 5 and coords.flat_rank == 5
    comptime assert us.flat_rank == 1 and vs.flat_rank == 1 and corr.flat_rank == 6
    var d = 2 * radius + 2
    var num_edges = Int(coords.dim[1]())
    var grid_h = Int(coords.dim[3]())
    var grid_w = Int(coords.dim[4]())
    var channels = Int(fmap1.dim[2]())
    var h2 = Int(fmap2.dim[3]())
    var w2 = Int(fmap2.dim[4]())

    var n = n_in
    var jj = n % d
    n //= d
    var ii = n % d
    n //= d
    var j0 = n % grid_w
    n //= grid_w
    var i0 = n % grid_h
    n //= grid_h
    var m = n % num_edges
    n //= num_edges

    var ix = Int(us[m][0])
    var jx = Int(vs[m][0])
    var x = coords[n, m, 0, i0, j0][0]
    var y = coords[n, m, 1, i0, j0][0]
    var i1 = Int(floor(y)) + (ii - radius)
    var j1 = Int(floor(x)) + (jj - radius)

    var s: Scalar[dtype] = 0
    if _within_bounds(i1, j1, h2, w2):
        for c in range(channels):
            s = _mad(s, fmap1[n, ix, c, i0, j0][0], fmap2[n, jx, c, i1, j1][0])
    corr[n, m, ii, jj, i0, j0] = s


@always_inline
def patchify_element[
    dtype: DType
](
    n_in: Int,
    radius: Int,
    net: ImmTileTensor[dtype, L4, _],
    coords: ImmTileTensor[DType.float32, L3, _],
    patches: MutTileTensor[dtype, L5, _],
):
    """One (batch, patch, row offset, col offset) of `patchify_forward_kernel`: copy all channels of one pixel."""
    comptime assert net.flat_rank == 4 and coords.flat_rank == 3 and patches.flat_rank == 5
    var d = 2 * radius + 2
    var num_patches = Int(coords.dim[1]())
    var channels = Int(net.dim[1]())
    var height = Int(net.dim[2]())
    var width = Int(net.dim[3]())

    var n = n_in
    var ii = n % d
    n //= d
    var jj = n % d
    n //= d
    var m = n % num_patches
    n //= num_patches

    var x = coords[n, m, 0][0]
    var y = coords[n, m, 1][0]
    var i = Int(floor(y)) + (ii - radius)
    var j = Int(floor(x)) + (jj - radius)
    if _within_bounds(i, j, height, width):
        for k in range(channels):
            patches[n, m, k, ii, jj] = net[n, k, i, j]


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def corr_kernel[
    dtype: DType
](
    total: Int64,
    radius: Int64,
    fmap1: ImmTileTensor[dtype, L5, ImmUntrackedOrigin],
    fmap2: ImmTileTensor[dtype, L5, ImmUntrackedOrigin],
    coords: ImmTileTensor[DType.float32, L5, ImmUntrackedOrigin],
    us: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    vs: ImmTileTensor[DType.int64, L1, ImmUntrackedOrigin],
    corr: MutTileTensor[dtype, L6, MutUntrackedOrigin],
):
    var n = global_idx.x
    if n < Int(total):
        corr_element(n, Int(radius), fmap1, fmap2, coords, us, vs, corr)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def patchify_kernel[
    dtype: DType
](
    total: Int64,
    radius: Int64,
    net: ImmTileTensor[dtype, L4, ImmUntrackedOrigin],
    coords: ImmTileTensor[DType.float32, L3, ImmUntrackedOrigin],
    patches: MutTileTensor[dtype, L5, MutUntrackedOrigin],
):
    var n = global_idx.x
    if n < Int(total):
        patchify_element(n, Int(radius), net, coords, patches)
