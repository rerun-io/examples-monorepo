"""fused-ssim (rahul-goel/fused-ssim@1272e21 `ssim.cu`): SSIM map with an 11x11 Gaussian (sigma 1.5), zero padding.

One 32x32 block per 32x32 pixel tile of one image (grid z = batch). Per channel, the tile and its 5-pixel
halo are loaded into shared memory, and every windowed moment is a separable convolution: a horizontal
pass over the 42 halo rows into a (42, 32) scratch, then a vertical pass. The forward pass also writes the
three partial derivatives of the map that the backward pass convolves with dL/dmap.

Upstream's scratch is declared (42, 32) but indexed at column tx + 5, so its last row runs five floats past
the array; here the scratch is indexed at tx, which is the same arithmetic without the overrun. Products
and sums follow upstream's order with nvcc's default contraction of a * b + c into one fma.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import fma
from std.memory import stack_allocation
from std.sys import llvm_intrinsic
from std.utils.static_tuple import StaticTuple

comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime SharedPtr = MutPointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]
comptime BX = 32
comptime BY = 32
comptime HALO = 5
comptime SX = BX + 2 * HALO
comptime SY = BY + 2 * HALO
comptime G = SIMD[DType.float32, 16](
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


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    """IEEE round-to-nearest a / b (PTX div.rn.f32), as nvcc emits without --use_fast_math."""
    return llvm_intrinsic["llvm.nvvm.div.rn.f", Float32](a, b)


@always_inline
def _pix(img: F32Ptr, plane: Int, y: Int, x: Int, height: Int, width: Int) -> Float32:
    if x >= width or y >= height or x < 0 or y < 0:
        return 0.0
    return img.unsafe_load(plane + y * width + x)


@always_inline
def _load_tile(tile: SharedPtr, img: F32Ptr, plane: Int, height: Int, width: Int):
    """tile[ly, lx] = img[plane][y0 + ly - 5, x0 + lx - 5] (0 outside) for the (42, 42) halo tile."""
    var start_y = block_idx.y * BY
    var start_x = block_idx.x * BX
    var rank = thread_idx.y * BX + thread_idx.x
    comptime for b in range((SY * SX + BX * BY - 1) // (BX * BY)):
        var tid = b * BX * BY + rank
        if tid < SY * SX:
            var ly = tid // SX
            var lx = tid % SX
            tile[unsafe_offset=tid] = _pix(img, plane, start_y + ly - HALO, start_x + lx - HALO, height, width)


@always_inline
def _multiply_tile(tile: SharedPtr, other: SharedPtr):
    var rank = thread_idx.y * BX + thread_idx.x
    comptime for b in range((SY * SX + BX * BY - 1) // (BX * BY)):
        var tid = b * BX * BY + rank
        if tid < SY * SX:
            tile[unsafe_offset=tid] = tile[unsafe_offset=tid] * other[unsafe_offset=tid]


@always_inline
def _row_conv[square: Bool](tile: SharedPtr, row: Int, tx: Int) -> Float32:
    var val: Float32 = 0.0
    comptime for k in range(11):
        var p = tile[unsafe_offset=row * SX + tx + k]
        comptime if square:
            p = p * p
        val = fma(G[k], p, val)
    return val


@always_inline
def _conv[square: Bool](tile: SharedPtr, scratch: SharedPtr) -> Float32:
    """The Gaussian-windowed mean of the tile (of its squares) at this thread's pixel; leaves all threads synced."""
    var tx = thread_idx.x
    var ty = thread_idx.y
    scratch[unsafe_offset=ty * BX + tx] = _row_conv[square](tile, ty, tx)
    if ty + BY < SY:
        scratch[unsafe_offset=(ty + BY) * BX + tx] = _row_conv[square](tile, ty + BY, tx)
    barrier()
    var val: Float32 = 0.0
    comptime for k in range(11):
        val = fma(G[k], scratch[unsafe_offset=(ty + k) * BX + tx], val)
    barrier()
    return val


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BX * BY))
def fused_ssim_kernel[train: Bool](
    height: Int32,
    width: Int32,
    channels: Int32,
    c1: Float32,
    c2: Float32,
    img1: F32Ptr,
    img2: F32Ptr,
    ssim_map: F32Ptr,
    dm_dmu1: F32Ptr,
    dm_dsigma1_sq: F32Ptr,
    dm_dsigma12: F32Ptr,
):
    var h = Int(height)
    var w = Int(width)
    var ch = Int(channels)
    var pix_y = block_idx.y * BY + thread_idx.y
    var pix_x = block_idx.x * BX + thread_idx.x
    var buf1 = stack_allocation[SY * SX, Float32, address_space=AddressSpace.SHARED]()
    var buf2 = stack_allocation[SY * SX, Float32, address_space=AddressSpace.SHARED]()
    var buf3 = stack_allocation[SY * BX, Float32, address_space=AddressSpace.SHARED]()
    for i in range(ch):
        var plane = (block_idx.z * ch + i) * h * w
        _load_tile(buf1, img1, plane, h, w)
        barrier()
        var mu1 = _conv[False](buf1, buf3)
        var sigma1_sq = fma(-mu1, mu1, _conv[True](buf1, buf3))
        _load_tile(buf2, img2, plane, h, w)
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
            var at = plane + pix_y * w + pix_x
            ssim_map[unsafe_offset=at] = _div(cc * dd, aa * bb)
            comptime if train:
                dm_dmu1[unsafe_offset=at] = (
                    _div(mu2 * 2.0 * dd, aa * bb)
                    - _div(mu2 * 2.0 * cc, aa * bb)
                    - _div(mu1 * 2.0 * cc * dd, aa * aa * bb)
                    + _div(mu1 * 2.0 * cc * dd, aa * bb * bb)
                )
                dm_dsigma1_sq[unsafe_offset=at] = _div(-cc * dd, aa * bb * bb)
                dm_dsigma12[unsafe_offset=at] = _div(2.0 * cc, aa * bb)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BX * BY))
def fused_ssim_backward_kernel(
    height: Int32,
    width: Int32,
    channels: Int32,
    img1: F32Ptr,
    img2: F32Ptr,
    dl_dmap: F32Ptr,
    dl_dimg1: F32Ptr,
    dm_dmu1: F32Ptr,
    dm_dsigma1_sq: F32Ptr,
    dm_dsigma12: F32Ptr,
):
    var h = Int(height)
    var w = Int(width)
    var ch = Int(channels)
    var pix_y = block_idx.y * BY + thread_idx.y
    var pix_x = block_idx.x * BX + thread_idx.x
    var buf1 = stack_allocation[SY * SX, Float32, address_space=AddressSpace.SHARED]()
    var buf2 = stack_allocation[SY * SX, Float32, address_space=AddressSpace.SHARED]()
    var buf3 = stack_allocation[SY * BX, Float32, address_space=AddressSpace.SHARED]()
    for i in range(ch):
        var plane = (block_idx.z * ch + i) * h * w
        var pix1 = _pix(img1, plane, pix_y, pix_x, h, w)
        var pix2 = _pix(img2, plane, pix_y, pix_x, h, w)
        _load_tile(buf1, dl_dmap, plane, h, w)

        _load_tile(buf2, dm_dmu1, plane, h, w)
        barrier()
        _multiply_tile(buf2, buf1)
        barrier()
        var dl_dpix = _conv[False](buf2, buf3)

        _load_tile(buf2, dm_dsigma1_sq, plane, h, w)
        barrier()
        _multiply_tile(buf2, buf1)
        barrier()
        dl_dpix = fma(pix1 * 2.0, _conv[False](buf2, buf3), dl_dpix)

        _load_tile(buf2, dm_dsigma12, plane, h, w)
        barrier()
        _multiply_tile(buf2, buf1)
        barrier()
        dl_dpix = fma(pix2, _conv[False](buf2, buf3), dl_dpix)

        if pix_x < w and pix_y < h:
            dl_dimg1[unsafe_offset=plane + pix_y * w + pix_x] = dl_dpix
