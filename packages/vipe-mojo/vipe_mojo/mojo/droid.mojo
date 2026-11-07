"""ViPE's DROID correlation kernels (`csrc/droid_net_ext/correlation_kernels.cu`, `altcorr_kernel.cu`), one source for CPU and CUDA.

Each `*_element` function computes what one CUDA thread computes for one output pixel, with the CUDA
kernel's loop order and arithmetic, so the GPU kernels and the CPU loop share it. Tensors travel as raw
pointers to contiguous memory with their sizes.

- corr_index: volume (B, H1, W1, H2, W2), coords (B, 2, H1, W1) float32 -> corr (B, rd, rd, H1, W1) in
  the volume's dtype, rd = 2r + 1, zeroed by the caller. Half precision rounds every product and sum
  to half as c10::Half does (`s * scalar_t(w)` then `+=`).
- altcorr_index (one pyramid level per launch): fmap1 (B, N1, Hf, Wf, C), fmap2 (B, N2, H2, W2, C),
  coords (B, M, H1, W1, 2) float32, ii / jj (M,) int64 -> corr (B, M, L * rd * rd, H1, W1) float32,
  zeroed by the caller. `pix` is the CUDA fast path for half features with C % 8 == 0 and C <= 128
  (two float accumulators over even and odd channels); the staged path sums each 32-channel chunk in
  float and adds it to the output chunk by chunk.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.math import floor
from std.memory import bitcast
from std.utils.static_tuple import StaticTuple

comptime NUM_THREADS = 256
comptime CHANNEL_STRIDE = 32
comptime PIX_WIDTH = 8
"""Half channels per 16-byte load on the pix path."""
comptime PIX_VECS = 16
"""Loads per pixel on the pix path: C <= 128."""
comptime PIX_ALIGN = 16
comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime I64Ptr = MutPointer[Int64, MutUntrackedOrigin]


@always_inline
def _within_bounds(h: Int, w: Int, height: Int, width: Int) -> Bool:
    return h >= 0 and h < height and w >= 0 and w < width


@always_inline
def _round_to_half(x: Float32) -> Float32:
    """`x` rounded to the nearest float16 value (ties to even), returned as float32, with integer operations only.

    Integer bit operations are opaque to fp contraction, so the product's rounding is kept (see
    dpvo.mojo_backend's corr.mojo: written with casts, `s + a * b` fuses into one fma.f16).
    """
    var bits = bitcast[DType.uint32](x)
    var sign = bits & 0x80000000
    var mag = bits & 0x7FFFFFFF
    if mag >= 0x477FF000:  # >= 65520: rounds to inf
        return bitcast[DType.float32](sign | 0x7F800000)
    if mag >= 0x38800000:  # half normal: keep 10 mantissa bits
        var lsb = (mag >> 13) & 1
        return bitcast[DType.float32](sign | ((mag + 0xFFF + lsb) & 0xFFFFE000))
    var exponent = Int(mag >> 23)
    if exponent == 0:
        return bitcast[DType.float32](sign)
    var shift = 126 - exponent  # half subnormal: a multiple of 2^-24
    if shift > 24:
        return bitcast[DType.float32](sign)
    var m = (mag & 0x7FFFFF) | 0x800000
    var q = m >> UInt32(shift)
    var rem = m & ((UInt32(1) << UInt32(shift)) - 1)
    var halfway = UInt32(1) << UInt32(shift - 1)
    if rem > halfway or (rem == halfway and (q & 1) == 1):
        q += 1
    var value = Float32(Int(q)) * 5.9604644775390625e-08
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
def corr_index_element[
    dtype: DType
](
    n_in: Int,
    radius: Int,
    h1: Int,
    w1: Int,
    h2: Int,
    w2: Int,
    volume: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    coords: F32Ptr,
    corr: MutPointer[Scalar[dtype], MutUntrackedOrigin],
):
    """Bilinear lookup of one query pixel (b, y, x) in its correlation volume (`corr_index_forward_kernel`)."""
    var x = n_in % w1
    var y = (n_in // w1) % h1
    var b = n_in // (w1 * h1)
    var rd = 2 * radius + 1
    var plane = h1 * w1
    var x0 = coords[unsafe_offset=((b * 2 + 0) * h1 + y) * w1 + x]
    var y0 = coords[unsafe_offset=((b * 2 + 1) * h1 + y) * w1 + x]
    var dx = x0 - floor(x0)
    var dy = y0 - floor(y0)
    var w_nw = (dx * dy).cast[dtype]()
    var w_ne = (dx * (1.0 - dy)).cast[dtype]()
    var w_sw = ((1.0 - dx) * dy).cast[dtype]()
    var w_se = ((1.0 - dx) * (1.0 - dy)).cast[dtype]()
    var src = ((b * h1 + y) * w1 + x) * h2 * w2
    var dst = b * rd * rd * plane + y * w1 + x
    for i in range(rd + 1):
        for j in range(rd + 1):
            var x1 = Int(floor(x0)) - radius + i
            var y1 = Int(floor(y0)) - radius + j
            if _within_bounds(y1, x1, h2, w2):
                var s = volume[unsafe_offset=src + y1 * w2 + x1]
                if i > 0 and j > 0:
                    var o = dst + ((i - 1) * rd + (j - 1)) * plane
                    corr[unsafe_offset=o] = _mad(corr[unsafe_offset=o], s, w_nw)
                if i > 0 and j < rd:
                    var o = dst + ((i - 1) * rd + j) * plane
                    corr[unsafe_offset=o] = _mad(corr[unsafe_offset=o], s, w_ne)
                if i < rd and j > 0:
                    var o = dst + (i * rd + (j - 1)) * plane
                    corr[unsafe_offset=o] = _mad(corr[unsafe_offset=o], s, w_sw)
                if i < rd and j < rd:
                    var o = dst + (i * rd + j) * plane
                    corr[unsafe_offset=o] = _mad(corr[unsafe_offset=o], s, w_se)


@always_inline
def _bilinear_add(corr: F32Ptr, base: Int, plane: Int, rd: Int, iy: Int, ix: Int, s: Float32, dx: Float32, dy: Float32):
    """The four `*(corr_ptr + ...) += s * ...` of the altcorr kernels for search offset (iy, ix)."""
    if iy > 0 and ix > 0:
        var o = base + ((iy - 1) + rd * (ix - 1)) * plane
        corr[unsafe_offset=o] = corr[unsafe_offset=o] + s * dy * dx
    if iy > 0 and ix < rd:
        var o = base + ((iy - 1) + rd * ix) * plane
        corr[unsafe_offset=o] = corr[unsafe_offset=o] + s * dy * (1.0 - dx)
    if iy < rd and ix > 0:
        var o = base + (iy + rd * (ix - 1)) * plane
        corr[unsafe_offset=o] = corr[unsafe_offset=o] + s * (1.0 - dy) * dx
    if iy < rd and ix < rd:
        var o = base + (iy + rd * ix) * plane
        corr[unsafe_offset=o] = corr[unsafe_offset=o] + s * (1.0 - dy) * (1.0 - dx)


@always_inline
def altcorr_index_element[
    dtype: DType, pix: Bool
](
    n_in: Int,
    radius: Int,
    level_offset: Int,
    coord_scale: Float32,
    num_edges: Int,
    height1: Int,
    width1: Int,
    frames1: Int,
    hf: Int,
    wf: Int,
    frames2: Int,
    height2: Int,
    width2: Int,
    channels: Int,
    corr_channels: Int,
    fmap1: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    fmap2: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    coords: F32Ptr,
    ii: I64Ptr,
    jj: I64Ptr,
    corr: F32Ptr,
):
    """One query pixel (b, m, h1, w1) at one pyramid level (`altcorr_index_forward_kernel_pix` / `_kernel`)."""
    var w1 = n_in % width1
    var h1 = (n_in // width1) % height1
    var bm = n_in // (width1 * height1)
    var m = bm % num_edges
    var b = bm // num_edges
    var src = Int(ii[unsafe_offset=m])
    var dst = Int(jj[unsafe_offset=m])
    var c_off = (bm * height1 + h1) * width1 + w1
    var x2 = coords[unsafe_offset=c_off * 2] / coord_scale
    var y2 = coords[unsafe_offset=c_off * 2 + 1] / coord_scale
    var dx = x2 - floor(x2)
    var dy = y2 - floor(y2)
    var x0 = Int(floor(x2))
    var y0 = Int(floor(y2))
    var rd = 2 * radius + 1
    var plane = height1 * width1
    var base = (bm * corr_channels + level_offset) * plane + h1 * width1 + w1
    var f1 = (((b * frames1 + src) * hf + h1) * wf + w1) * channels

    comptime if pix:
        # As the CUDA kernel: this pixel's fmap1 channels stay in registers (16-byte loads, C <= 128), and each
        # fmap2 pixel is read 16 bytes at a time. Pixel rows are C * 2 bytes, a multiple of 16 since C % 8 == 0.
        var f1v = Array[SIMD[dtype, PIX_WIDTH], PIX_VECS](fill=SIMD[dtype, PIX_WIDTH](0))
        comptime for v in range(PIX_VECS):
            if v * PIX_WIDTH < channels:
                f1v[v] = fmap1.load[width=PIX_WIDTH, alignment=PIX_ALIGN](f1 + v * PIX_WIDTH)
        for iy in range(rd + 1):
            for ix in range(rd + 1):
                var hh = y0 - radius + iy
                var ww = x0 - radius + ix
                var s: Float32 = 0.0
                if _within_bounds(hh, ww, height2, width2):
                    var f2 = (((b * frames2 + dst) * height2 + hh) * width2 + ww) * channels
                    var acc_x: Float32 = 0.0
                    var acc_y: Float32 = 0.0
                    comptime for v in range(PIX_VECS):
                        if v * PIX_WIDTH < channels:
                            var a = f1v[v].cast[DType.float32]()
                            var g = fmap2.load[width=PIX_WIDTH, alignment=PIX_ALIGN](f2 + v * PIX_WIDTH).cast[DType.float32]()
                            comptime for u in range(0, PIX_WIDTH, 2):
                                acc_x = acc_x + a[u] * g[u]
                                acc_y = acc_y + a[u + 1] * g[u + 1]
                    s = acc_x + acc_y
                _bilinear_add(corr, base, plane, rd, iy, ix, s, dx, dy)
    else:
        for c in range(0, channels, CHANNEL_STRIDE):
            var chunk = min(CHANNEL_STRIDE, channels - c)
            for iy in range(rd + 1):
                for ix in range(rd + 1):
                    var hh = y0 - radius + iy
                    var ww = x0 - radius + ix
                    var s: Float32 = 0.0
                    if _within_bounds(hh, ww, height2, width2):
                        var f2 = (((b * frames2 + dst) * height2 + hh) * width2 + ww) * channels + c
                        for k in range(chunk):
                            s = s + fmap1[unsafe_offset=f1 + c + k].cast[DType.float32]() * fmap2[unsafe_offset=f2 + k].cast[DType.float32]()
                    _bilinear_add(corr, base, plane, rd, iy, ix, s, dx, dy)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def corr_index_kernel[
    dtype: DType
](
    total: Int64,
    radius: Int64,
    h1: Int64,
    w1: Int64,
    h2: Int64,
    w2: Int64,
    volume: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    coords: F32Ptr,
    corr: MutPointer[Scalar[dtype], MutUntrackedOrigin],
):
    var n = global_idx.x
    if n < Int(total):
        corr_index_element(n, Int(radius), Int(h1), Int(w1), Int(h2), Int(w2), volume, coords, corr)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](NUM_THREADS))
def altcorr_index_kernel[
    dtype: DType, pix: Bool
](
    total: Int64,
    radius: Int64,
    level_offset: Int64,
    coord_scale: Float32,
    num_edges: Int64,
    height1: Int64,
    width1: Int64,
    frames1: Int64,
    hf: Int64,
    wf: Int64,
    frames2: Int64,
    height2: Int64,
    width2: Int64,
    channels: Int64,
    corr_channels: Int64,
    fmap1: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    fmap2: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    coords: F32Ptr,
    ii: I64Ptr,
    jj: I64Ptr,
    corr: F32Ptr,
):
    var n = global_idx.x
    if n < Int(total):
        altcorr_index_element[dtype, pix](
            n, Int(radius), Int(level_offset), coord_scale, Int(num_edges), Int(height1), Int(width1), Int(frames1), Int(hf), Int(wf),
            Int(frames2), Int(height2), Int(width2), Int(channels), Int(corr_channels), fmap1, fmap2, coords, ii, jj, corr,
        )
