"""fused-ssim (rahul-goel/fused-ssim@1272e21 `ssim.cu`): SSIM map with an 11x11 Gaussian (sigma 1.5), zero padding.

One 32x32 block per 32x32 pixel tile of one image (grid z = batch). Per channel, the tile and its 5-pixel
halo are loaded into shared memory, and every windowed moment is a separable convolution: a horizontal
pass over the 42 halo rows into a (42, 32) scratch, then a vertical pass. The forward pass also writes the
three partial derivatives of the map that the backward pass convolves with dL/dmap.

Upstream's scratch is declared (42, 32) but indexed at column tx + 5, so its last row runs five floats past
the array; here the scratch is indexed at tx, which is the same arithmetic without the overrun. Products
and sums follow upstream's order with nvcc's default contraction of a * b + c into one fma.
"""

from layout import TileTensor, row_major
from layout import stack_allocation as tile_allocation
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.sync import barrier
from std.math import ceildiv, fma
from std.sys import llvm_intrinsic
from std.utils.static_tuple import StaticTuple

from tensors import In, Out, Planes, StackedPlanes

comptime BX = 32
comptime BY = 32
comptime HALO = 5
comptime SX = BX + 2 * HALO
comptime SY = BY + 2 * HALO
comptime HALO_ROUNDS = ceildiv(SY * SX, BX * BY)
comptime G = SIMD[.float32, 16](
    0.001028380123898387,
    0.0075987582094967365,
    0.036000773310661316,
    0.10936068743467331,
    0.21300552785396576,
    0.26601171493530273,
    0.21300552785396576,
    0.10936068743467331,
    0.036000773310661316,
    0.0075987582094967365,
    0.001028380123898387,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)


@inline(.always)
def _div(a: Float32, b: Float32) -> Float32:
    """IEEE round-to-nearest a / b (PTX div.rn.f32), as nvcc emits without --use_fast_math."""
    return llvm_intrinsic["llvm.nvvm.div.rn.f", Float32](a, b)


@inline(.always)
def _pix(img: TileTensor[.float32, ...], plane: Int, y: Int, x: Int, h: Int, w: Int) -> Float32:
    """img[plane, y, x], 0 outside the h x w image (one h, w for every image: fewer live registers than each
    tensor's own dims)."""
    comptime assert img.flat_rank == 3
    if x >= w or y >= h or x < 0 or y < 0:
        return 0.0
    return img[plane, y, x][0]


@inline(.always)
def _load_tile(tile: TileTensor[mut=True, .float32, ...], img: TileTensor[.float32, ...], plane: Int, h: Int, w: Int):
    """tile[ly, lx] = img[plane, y0 + ly - 5, x0 + lx - 5] (0 outside) for the (42, 42) halo tile."""
    comptime assert tile.flat_rank == 2
    var start_y = block_idx.y * BY
    var start_x = block_idx.x * BX
    var rank = thread_idx.y * BX + thread_idx.x
    comptime for b in range(HALO_ROUNDS):
        var tid = b * BX * BY + rank
        if tid < SY * SX:
            var ly, lx = divmod(tid, SX)
            tile[ly, lx] = _pix(img, plane, start_y + ly - HALO, start_x + lx - HALO, h, w)


@inline(.always)
def _load_stacked_tile(tile: TileTensor[mut=True, .float32, ...], stack: TileTensor[.float32, ...], k: Int, plane: Int, h: Int, w: Int):
    """`_load_tile` from plane `plane` of image `k` of a (K, planes, h, w) stack."""
    comptime assert tile.flat_rank == 2 and stack.flat_rank == 4
    var start_y = block_idx.y * BY
    var start_x = block_idx.x * BX
    var rank = thread_idx.y * BX + thread_idx.x
    comptime for b in range(HALO_ROUNDS):
        var tid = b * BX * BY + rank
        if tid < SY * SX:
            var ly, lx = divmod(tid, SX)
            var y = start_y + ly - HALO
            var x = start_x + lx - HALO
            tile[ly, lx] = stack[k, plane, y, x][0] if 0 <= y < h and 0 <= x < w else Float32(0.0)


@inline(.always)
def _multiply_tile(tile: TileTensor[mut=True, .float32, ...], other: TileTensor[.float32, ...]):
    comptime assert tile.flat_rank == 2 and other.flat_rank == 2
    var rank = thread_idx.y * BX + thread_idx.x
    comptime for b in range(HALO_ROUNDS):
        var tid = b * BX * BY + rank
        if tid < SY * SX:
            var ly, lx = divmod(tid, SX)
            tile[ly, lx] = tile[ly, lx][0] * other[ly, lx][0]


@inline(.always)
def _row_conv[square: Bool](tile: TileTensor[.float32, ...], row: Int, tx: Int) -> Float32:
    comptime assert tile.flat_rank == 2
    var val: Float32 = 0.0
    comptime for k in range(11):
        var p = tile[row, tx + k][0]
        comptime if square:
            p = p * p
        val = fma(G[k], p, val)
    return val


@inline(.always)
def _conv[square: Bool](tile: TileTensor[.float32, ...], scratch: TileTensor[mut=True, .float32, ...]) -> Float32:
    """The Gaussian-windowed mean of the tile (of its squares) at this thread's pixel; leaves all threads synced."""
    comptime assert scratch.flat_rank == 2
    var tx = thread_idx.x
    var ty = thread_idx.y
    scratch[ty, tx] = _row_conv[square](tile, ty, tx)
    if ty + BY < SY:
        scratch[ty + BY, tx] = _row_conv[square](tile, ty + BY, tx)
    barrier()
    var val: Float32 = 0.0
    comptime for k in range(11):
        val = fma(G[k], scratch[ty + k, tx][0], val)
    barrier()
    return val


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BX * BY))
def fused_ssim_kernel[train: Bool](
    channels: Int32,
    c1: Float32,
    c2: Float32,
    img1: In[.float32, Planes],
    img2: In[.float32, Planes],
    maps: Out[.float32, StackedPlanes],
):
    """Images are (batch * channels, H, W) planes; maps stacks the SSIM map and, when training, its three partials
    d/dmu1, d/dsigma1^2, d/dsigma12 (one tensor: one set of strides, not four, in registers)."""
    var h = Int(img1.dim[1]())
    var w = Int(img1.dim[2]())
    var ch = Int(channels)
    var pix_y = block_idx.y * BY + thread_idx.y
    var pix_x = block_idx.x * BX + thread_idx.x
    # Every image has img1's shape: index them all through img1's layout (one set of strides in registers).
    var image2 = TileTensor(img2.unsafe_ptr(), img1.layout)
    var buf1 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, SX]())
    var buf2 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, SX]())
    var buf3 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, BX]())
    for i in range(ch):
        var plane = block_idx.z * ch + i
        _load_tile(buf1, img1, plane, h, w)
        barrier()
        var mu1 = _conv[False](buf1, buf3)
        var sigma1_sq = fma(-mu1, mu1, _conv[True](buf1, buf3))
        _load_tile(buf2, image2, plane, h, w)
        barrier()
        var mu2 = _conv[False](buf2, buf3)
        var sigma2_sq = fma(-mu2, mu2, _conv[True](buf2, buf3))
        _multiply_tile(buf1, buf2)
        barrier()
        var sigma12 = fma(-mu1, mu2, _conv[False](buf1, buf3))

        var mu1_mu2 = mu1 * mu2
        var cc = fma(Float32(2.0), mu1_mu2, c1)
        var dd = fma(Float32(2.0), sigma12, c2)
        var aa = fma(mu1, mu1, mu2 * mu2) + c1
        var bb = sigma1_sq + sigma2_sq + c2
        if pix_x < w and pix_y < h:
            maps[0, plane, pix_y, pix_x] = _div(cc * dd, aa * bb)
            comptime if train:
                maps[1, plane, pix_y, pix_x] = (
                    _div(mu2 * 2.0 * dd, aa * bb)
                    - _div(mu2 * 2.0 * cc, aa * bb)
                    - _div(mu1 * 2.0 * cc * dd, aa * aa * bb)
                    + _div(mu1 * 2.0 * cc * dd, aa * bb * bb)
                )
                maps[2, plane, pix_y, pix_x] = _div(-cc * dd, aa * bb * bb)
                maps[3, plane, pix_y, pix_x] = _div(2.0 * cc, aa * bb)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BX * BY))
def fused_ssim_backward_kernel(
    channels: Int32,
    img1: In[.float32, Planes],
    img2: In[.float32, Planes],
    dl_dmap: In[.float32, Planes],
    dl_dimg1: Out[.float32, Planes],
    partials: In[.float32, StackedPlanes],
):
    """partials: the forward's d/dmu1, d/dsigma1^2, d/dsigma12, stacked."""
    var h = Int(img1.dim[1]())
    var w = Int(img1.dim[2]())
    var ch = Int(channels)
    var pix_y = block_idx.y * BY + thread_idx.y
    var pix_x = block_idx.x * BX + thread_idx.x
    # Every image has img1's shape: index them all through img1's layout (one set of strides in registers).
    var image2 = TileTensor(img2.unsafe_ptr(), img1.layout)
    var dmap = TileTensor(dl_dmap.unsafe_ptr(), img1.layout)
    var grad = TileTensor(dl_dimg1.unsafe_ptr(), img1.layout)
    var buf1 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, SX]())
    var buf2 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, SX]())
    var buf3 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, BX]())
    for i in range(ch):
        var plane = block_idx.z * ch + i
        var pix1 = _pix(img1, plane, pix_y, pix_x, h, w)
        var pix2 = _pix(image2, plane, pix_y, pix_x, h, w)
        _load_tile(buf1, dmap, plane, h, w)

        _load_stacked_tile(buf2, partials, 0, plane, h, w)
        barrier()
        _multiply_tile(buf2, buf1)
        barrier()
        var dl_dpix = _conv[False](buf2, buf3)

        _load_stacked_tile(buf2, partials, 1, plane, h, w)
        barrier()
        _multiply_tile(buf2, buf1)
        barrier()
        dl_dpix = fma(pix1 * 2.0, _conv[False](buf2, buf3), dl_dpix)

        _load_stacked_tile(buf2, partials, 2, plane, h, w)
        barrier()
        _multiply_tile(buf2, buf1)
        barrier()
        dl_dpix = fma(pix2, _conv[False](buf2, buf3), dl_dpix)

        if pix_x < w and pix_y < h:
            grad[plane, pix_y, pix_x] = dl_dpix
