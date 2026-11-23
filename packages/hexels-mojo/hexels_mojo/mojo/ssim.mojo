"""fused-ssim (rahul-goel/fused-ssim@a7c48d6 `ssim.cu`): SSIM map with an 11x11 Gaussian (sigma 1.5), zero padding.

One BX x BY block per BX x BY pixel tile of one image (grid z = batch). Per channel, both images' tiles and their
5-pixel halos are loaded into shared memory once; ONE horizontal pass computes the five windowed moments (x, x^2, y, y^2,
xy) of every halo row into a (5, SY, BX) scratch, and ONE vertical pass finishes them: two barriers per channel. As
upstream, sigma^2 is clamped at 0 and the map to [-1, 1]; where the map is clamped its partials are 0. The forward pass
writes the three partial derivatives of the map that the backward pass convolves with dL/dmap; the backward pass
multiplies them with dL/dmap while loading, then runs one three-channel horizontal and one vertical pass.

Sums follow upstream's order: symmetric pairs around the centre (left + right) * w, then the centre tap. 16 x 16 blocks
(256 threads, 13.7 KB of shared memory in the forward, 13.1 KB in the backward) fit every GPU Mojo targets; on the RTX 5090
they beat 32 x 32, 32 x 16 and 32 x 8 (several blocks per SM stay resident across the barriers).
"""

from layout import TileTensor, row_major
from layout import stack_allocation as tile_allocation
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from max.gpu.sync import barrier
from std.math import ceildiv, fma, max, min
from std.sys import llvm_intrinsic
from std.sys.info import is_nvidia_gpu
from std.utils.static_tuple import StaticTuple

from tensors import In, Out, Planes, StackedPlanes

comptime BX = 16
comptime BY = 16
comptime THREADS = BX * BY
comptime HALO = 5
comptime SX = BX + 2 * HALO
comptime SY = BY + 2 * HALO
comptime LOAD_ROUNDS = ceildiv(SY * SX, THREADS)
comptime ROW_ROUNDS = ceildiv(SY, BY)
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
comptime Moments = SIMD[.float32, 8]
"""x, x^2, y, y^2, xy in lanes 0-4 (lanes 5-7 unused)."""
comptime Chain = SIMD[.float32, 4]
"""The three partials times dL/dmap in lanes 0-2 (lane 3 unused)."""


@inline(.always)
def _div(a: Float32, b: Float32) -> Float32:
    """a / b as the CUDA package computes it: it builds with --use_fast_math, so nvcc emits div.approx.f32; the same
    instruction here gives its maps and gradients bit for bit. Other GPUs divide as their compiler does."""
    comptime if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.div.approx.f", Float32](a, b)
    else:
        return a / b


@inline(.always)
def _pix(img: TileTensor[.float32, ...], plane: Int, y: Int, x: Int, h: Int, w: Int) -> Float32:
    """img[plane, y, x], 0 outside the h x w image."""
    comptime assert img.flat_rank == 3
    if x >= w or y >= h or x < 0 or y < 0:
        return 0.0
    return img[plane, y, x][0]


@inline(.always)
def _tile_moments(t1: TileTensor[.float32, ...], t2: TileTensor[.float32, ...], y: Int, x: Int) -> Moments:
    comptime assert t1.flat_rank == 2 and t2.flat_rank == 2
    var a = t1[y, x][0]
    var b = t2[y, x][0]
    return Moments(a, a * a, b, b * b, a * b, 0.0, 0.0, 0.0)


@inline(.always)
def _scratch_moments(scratch: TileTensor[.float32, ...], y: Int, x: Int) -> Moments:
    comptime assert scratch.flat_rank == 3
    return Moments(scratch[0, y, x][0], scratch[1, y, x][0], scratch[2, y, x][0], scratch[3, y, x][0], scratch[4, y, x][0], 0.0, 0.0, 0.0)


@inline(.always)
def _chain(planes: TileTensor[.float32, ...], y: Int, x: Int) -> Chain:
    comptime assert planes.flat_rank == 3
    return Chain(planes[0, y, x][0], planes[1, y, x][0], planes[2, y, x][0], 0.0)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](THREADS))
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
    var tx = thread_idx.x
    var ty = thread_idx.y
    var start_y = block_idx.y * BY
    var start_x = block_idx.x * BX
    var pix_y = start_y + ty
    var pix_x = start_x + tx
    var rank = ty * BX + tx
    # Every image has img1's shape: index them all through img1's layout (one set of strides in registers).
    var image2 = TileTensor(img2.unsafe_ptr(), img1.layout)
    var tile1 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, SX]())
    var tile2 = tile_allocation[.float32, address_space=.SHARED](row_major[SY, SX]())
    var scratch = tile_allocation[.float32, address_space=.SHARED](row_major[5, SY, BX]())
    for i in range(Int(channels)):
        var plane = block_idx.z * Int(channels) + i
        # 1. Both images' halo tiles.
        comptime for r in range(LOAD_ROUNDS):
            var tid = r * THREADS + rank
            if tid < SY * SX:
                var ly, lx = divmod(tid, SX)
                tile1[ly, lx] = _pix(img1, plane, start_y + ly - HALO, start_x + lx - HALO, h, w)
                tile2[ly, lx] = _pix(image2, plane, start_y + ly - HALO, start_x + lx - HALO, h, w)
        barrier()
        # 2. Horizontal pass: the five moments of every halo row at this thread's column.
        comptime for r in range(ROW_ROUNDS):
            var row = ty + r * BY
            if row < SY:
                var lx = tx + HALO
                var acc = Moments(0.0)
                comptime for d in range(1, HALO + 1):
                    acc = fma(_tile_moments(tile1, tile2, row, lx - d) + _tile_moments(tile1, tile2, row, lx + d), G[HALO - d], acc)
                acc = fma(_tile_moments(tile1, tile2, row, lx), G[HALO], acc)
                comptime for m in range(5):
                    scratch[m, row, tx] = acc[m]
        barrier()
        # 3. Vertical pass and the map.
        var ly = ty + HALO
        var out = Moments(0.0)
        comptime for d in range(1, HALO + 1):
            out = fma(_scratch_moments(scratch, ly - d, tx) + _scratch_moments(scratch, ly + d, tx), G[HALO - d], out)
        out = fma(_scratch_moments(scratch, ly, tx), G[HALO], out)
        if pix_x < w and pix_y < h:
            var mu1 = out[0]
            var mu2 = out[2]
            var mu1_sq = mu1 * mu1
            var mu2_sq = mu2 * mu2
            var sigma1_sq = max(Float32(0.0), out[1] - mu1_sq)
            var sigma2_sq = max(Float32(0.0), out[3] - mu2_sq)
            var sigma12 = out[4] - mu1 * mu2
            var aa = mu1_sq + mu2_sq + c1
            var bb = sigma1_sq + sigma2_sq + c2
            var cc = 2.0 * mu1 * mu2 + c1
            var dd = 2.0 * sigma12 + c2
            var val = _div(cc * dd, aa * bb)
            var clamped = val < -1.0 or val > 1.0
            maps[0, plane, pix_y, pix_x] = max(Float32(-1.0), min(Float32(1.0), val))
            comptime if train:
                var dm_dmu1: Float32 = 0.0
                var dm_dsigma1_sq: Float32 = 0.0
                var dm_dsigma12: Float32 = 0.0
                if not clamped:
                    dm_dmu1 = (
                        _div(mu2 * 2.0 * dd, aa * bb)
                        - _div(mu2 * 2.0 * cc, aa * bb)
                        - _div(mu1 * 2.0 * cc * dd, aa * aa * bb)
                        + _div(mu1 * 2.0 * cc * dd, aa * bb * bb)
                    )
                    dm_dsigma1_sq = _div(-cc * dd, aa * bb * bb)
                    dm_dsigma12 = _div(2.0 * cc, aa * bb)
                maps[1, plane, pix_y, pix_x] = dm_dmu1
                maps[2, plane, pix_y, pix_x] = dm_dsigma1_sq
                maps[3, plane, pix_y, pix_x] = dm_dsigma12
        barrier()


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](THREADS))
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
    var tx = thread_idx.x
    var ty = thread_idx.y
    var start_y = block_idx.y * BY
    var start_x = block_idx.x * BX
    var pix_y = start_y + ty
    var pix_x = start_x + tx
    var rank = ty * BX + tx
    # Every image has img1's shape: index them all through img1's layout (one set of strides in registers).
    var image2 = TileTensor(img2.unsafe_ptr(), img1.layout)
    var dmap = TileTensor(dl_dmap.unsafe_ptr(), img1.layout)
    var grad = TileTensor(dl_dimg1.unsafe_ptr(), img1.layout)
    var chain = tile_allocation[.float32, address_space=.SHARED](row_major[3, SY, SX]())
    var scratch = tile_allocation[.float32, address_space=.SHARED](row_major[3, SY, BX]())
    for i in range(Int(channels)):
        var plane = block_idx.z * Int(channels) + i
        var pix1 = _pix(img1, plane, pix_y, pix_x, h, w)
        var pix2 = _pix(image2, plane, pix_y, pix_x, h, w)
        # 1. dL/dmap times each partial, halo included (0 outside the image).
        comptime for r in range(LOAD_ROUNDS):
            var tid = r * THREADS + rank
            if tid < SY * SX:
                var ly, lx = divmod(tid, SX)
                var y = start_y + ly - HALO
                var x = start_x + lx - HALO
                var dl: Float32 = 0.0
                var p = Chain(0.0)
                if 0 <= y < h and 0 <= x < w:
                    dl = dmap[plane, y, x][0]
                    p = Chain(partials[0, plane, y, x][0], partials[1, plane, y, x][0], partials[2, plane, y, x][0], 0.0)
                comptime for k in range(3):
                    chain[k, ly, lx] = p[k] * dl
        barrier()
        # 2. Horizontal pass, three channels at once.
        comptime for r in range(ROW_ROUNDS):
            var row = ty + r * BY
            if row < SY:
                var lx = tx + HALO
                var acc = Chain(0.0)
                comptime for d in range(1, HALO + 1):
                    acc = fma(_chain(chain, row, lx - d) + _chain(chain, row, lx + d), G[HALO - d], acc)
                acc = fma(_chain(chain, row, lx), G[HALO], acc)
                comptime for k in range(3):
                    scratch[k, row, tx] = acc[k]
        barrier()
        # 3. Vertical pass and dL/dimg1.
        if pix_x < w and pix_y < h:
            var ly = ty + HALO
            var out = Chain(0.0)
            comptime for d in range(1, HALO + 1):
                out = fma(_chain(scratch, ly - d, tx) + _chain(scratch, ly + d, tx), G[HALO - d], out)
            out = fma(_chain(scratch, ly, tx), G[HALO], out)
            grad[plane, pix_y, pix_x] = out[0] + (2.0 * pix1) * out[1] + pix2 * out[2]
        barrier()
