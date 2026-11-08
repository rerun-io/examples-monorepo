"""hexel-kernel's 2D hexel rasterizer, flat and color-gradient (`gsplat/cuda/csrc/hexels_{forward,backward}.cu`,
`forward.cu` tile binning), which Image-GS's 2D hexel fits (`hexels-fit-2d`, rich and flat) run every step.

Tile binning: every hexel lists the 16x16 tiles its bounding square touches (`map_gaussian_to_intersects`); PyTorch
sorts the (tile << 32) keys and `get_tile_bin_edges` turns them into one id range per tile.

Forward (`nd_rasterize_forward_topk_norm_hexels[_color_grad][_softness]`; `color_grad` selects the variant): one
16x16 block per tile, one thread per pixel. The tile's hexels are streamed through shared memory in batches of 256; each pixel keeps the 10 largest
alphas (sigmoid of the hexagon SDF, optionally times a Gaussian peak) and outputs their alpha-normalized colors
(the flat color, or base color + color_grad_scale * (delta . color gradient)) plus peak-weighted emission, the normalizer
(alpha sum + 1e-4) and the entropy of the normalized weights. Every float op is nvcc's fast-math instruction in
the order of its PTX (`fastmath`), so the outputs match the CUDA build bit for bit.

Backward (`nd_rasterize_backward_topk_norm_hexels[_color_grad]_kernel`): one thread per pixel over its
saved top-10, gradients scattered with float atomics (scheduling-order sums, as in CUDA).

Softness: stride 0 = one scalar for all hexels, 1 = one per hexel, 3 = one per hexagon edge orientation.
Per-hexel channel data (colors, gradients, emission) and images are (rows, channels) grids, channels <= 12.
"""

from layout import stack_allocation as tile_allocation
from layout import row_major
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from std.math import ceildiv, clamp
from std.memory import stack_allocation
from std.utils.static_tuple import StaticTuple

import fastmath as fm
from tensors import Grid, In, Out, Rows, atomic_add

comptime TILE = 16
comptime BLOCK_SIZE = TILE * TILE
comptime TOP_K = 10
comptime MAX_CHANNELS = 12
comptime EPS: Float64 = 1e-4  # config.h `#define EPS 1e-4`: a double
comptime ENTROPY_EPS: Float32 = 1e-7
comptime HEX_A: Float32 = 0.8660254  # 0.5 * sqrt(3), 0f3F5DB3D7
comptime HEX_B: Float32 = 0.5
comptime SCALE_EPS: Float32 = 1e-8  # 0f322BCC77
comptime BINNING_THREADS = 256
comptime HEXEL_PARAMS = 9  # staged per hexel: x, y, scale x, scale y, rotation, sharpness, softness x3


# ── tile binning ─────────────────────────────────────────────────────────────


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BINNING_THREADS))
def map_to_intersects_kernel(
    tiles_x: Int32,
    tiles_y: Int32,
    xys: In[.float32, Rows[2]],
    radii: In[.int32, Rows[1]],
    cum_tiles_hit: In[.int32, Rows[1]],
    isect_ids: Out[.int64, Rows[1]],
    ids: Out[.int32, Rows[1]],
):
    var idx = global_idx.x
    if idx >= Int(xys.dim[0]()):
        return
    var radius = radii[idx, 0][0]
    if radius <= 0:
        return
    var cx = xys[idx, 0][0] / Float32(TILE)
    var cy = xys[idx, 1][0] / Float32(TILE)
    var r = Float32(radius) / Float32(TILE)
    # (int) truncation, then min(max(0, .), tiles), as hexel-kernel's get_tile_bbox.
    var x0 = clamp(Int(cx - r), 0, Int(tiles_x))
    var x1 = clamp(Int(cx + r + 1.0), 0, Int(tiles_x))
    var y0 = clamp(Int(cy - r), 0, Int(tiles_y))
    var y1 = clamp(Int(cy + r + 1.0), 0, Int(tiles_y))
    var cur = 0 if idx == 0 else Int(cum_tiles_hit[idx - 1, 0][0])
    for i in range(y0, y1):
        for j in range(x0, x1):
            isect_ids[cur, 0] = Int64(i * Int(tiles_x) + j) << 32
            ids[cur, 0] = Int32(idx)
            cur += 1


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BINNING_THREADS))
def tile_bin_edges_kernel(isect_ids_sorted: In[.int64, Rows[1]], tile_bins: Out[.int32, Rows[2]]):
    var idx = global_idx.x
    var n = Int(isect_ids_sorted.dim[0]())
    if idx >= n:
        return
    var cur = Int(isect_ids_sorted[idx, 0][0] >> 32)
    if idx == 0:
        tile_bins[cur, 0] = 0
    if idx == n - 1:
        tile_bins[cur, 1] = Int32(n)
    if idx == 0:
        return
    var prev = Int(isect_ids_sorted[idx - 1, 0][0] >> 32)
    if prev != cur:
        tile_bins[prev, 1] = Int32(idx)
        tile_bins[cur, 0] = Int32(idx)


# ── forward ──────────────────────────────────────────────────────────────────


@inline(.always)
def _hexel_alpha(
    dx: Float32, dy: Float32, sx_: Float32, sy_: Float32, rotation: Float32, aa: Float32, s0: Float32, s1: Float32, s2: Float32, sharp: Float32
) -> Tuple[Float32, Float32]:
    """(alpha, peak) of hexels_forward.cu's `hexel_alpha`, op for op as nvcc's PTX."""
    var s = fm.sin(rotation)
    var c = fm.cos(rotation)
    var u_rot = fm.fma(dy, s, fm.mul(dx, c))
    var v_rot = fm.sub(fm.mul(dy, c), fm.mul(dx, s))
    var sx = fm.add(sx_, SCALE_EPS)
    var sy = fm.add(sy_, SCALE_EPS)
    var u = fm.div(u_rot, sx)
    var v = fm.div(v_rot, sy)
    if aa > 0.0:
        var s_u = fm.rsqrt(fm.fma(aa, fm.rcp(fm.mul(sx, sx)), 1.0))
        var s_v = fm.rsqrt(fm.fma(aa, fm.rcp(fm.mul(sy, sy)), 1.0))
        u = fm.mul(u, s_u)
        v = fm.mul(v, s_v)
    var d0 = abs(v)
    var ua = fm.mul(u, HEX_A)
    var vb = fm.mul(v, HEX_B)
    var d1 = abs(fm.add(ua, vb))
    var d2 = abs(fm.sub(ua, vb))
    var slant_is_1 = d2 < d1  # setp.ltu: edge 1 unless d2 >= d1
    var slant = d1 if slant_is_1 else d2
    var not_edge0 = d0 <= slant
    var dist = fm.add(slant if not_edge0 else d0, -1.0)
    var soft = (s1 if slant_is_1 else s2) if not_edge0 else s0
    var alpha = fm.rcp(fm.add(fm.ex2(fm.mul(fm.mul(dist, soft), fm.LOG2E)), 1.0))
    var peak: Float32 = 0.0
    if sharp > 0.0:
        var r2 = fm.fma(u, u, fm.mul(v, v))
        peak = fm.ex2(fm.mul(fm.mul(r2, sharp), -fm.LOG2E))
        alpha = fm.mul(alpha, peak)
    return (alpha, peak)


@inline(.always)
def _softness(softnesses: In[.float32, Grid], stride: Int, default: Float32, g: Int) -> Tuple[Float32, Float32, Float32]:
    """Hexel g's softness per edge orientation: (N, 3) rows, one (N, 1) value, or the scalar default."""
    match stride:
        case 3:
            return (softnesses[g, 0][0], softnesses[g, 1][0], softnesses[g, 2][0])
        case 1:
            var s = softnesses[g, 0][0]
            return (s, s, s)
        case _:
            return (default, default, default)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def forward_kernel[color_grad: Bool](
    tiles_x: Int32,
    width: Int32,
    height: Int32,
    ids_sorted: In[.int32, Rows[1]],
    tile_bins: In[.int32, Rows[2]],
    xys: In[.float32, Rows[2]],
    scales: In[.float32, Rows[2]],
    rotation: In[.float32, Rows[1]],
    colors: In[.float32, Grid],
    color_grads: In[.float32, Grid],
    color_grad_scale: Float32,
    aa: Float32,
    softnesses: In[.float32, Grid],
    softness_stride: Int32,
    softness_default: Float32,
    has_sharpness: Int32,
    sharpness: In[.float32, Rows[1]],
    has_emission: Int32,
    emission: In[.float32, Grid],
    out_img: Out[.float32, Grid],
    pixel_topk: Out[.int32, Rows[TOP_K]],
    out_weights_sum: Out[.float32, Rows[1]],
    out_entropy: Out[.float32, Rows[1]],
):
    """colors / emission (N, C), color_grads (2 N, C) = (N, 2, C), out_img (H W, C)."""
    var w = Int(width)
    var ch = Int(colors.dim[1]())
    var tile_id = block_idx.y * Int(tiles_x) + block_idx.x
    var i = block_idx.y * TILE + thread_idx.y
    var j = block_idx.x * TILE + thread_idx.x
    var px = Float32(j)
    var py = Float32(i)
    var pix = i * w + j
    var inside = i < Int(height) and j < w
    var start = Int(tile_bins[tile_id, 0][0])
    var end = Int(tile_bins[tile_id, 1][0])
    if start == end:
        if inside:
            for c in range(ch):
                out_img[pix, c] = 0.0
            comptime for k in range(TOP_K):
                pixel_topk[pix, k] = -1
            out_weights_sum[pix, 0] = Float32(EPS)
            out_entropy[pix, 0] = 0.0
        return

    var id_batch = tile_allocation[.int32, address_space=.SHARED](row_major[BLOCK_SIZE]())
    var param_batch = tile_allocation[.float32, address_space=.SHARED](row_major[BLOCK_SIZE, HEXEL_PARAMS]())
    var tr = thread_idx.y * TILE + thread_idx.x
    var stride = Int(softness_stride)

    # Per-thread top-10, indexed by a runtime slot: local memory, as nvcc places it (registers need constant indices).
    var topk_ids = stack_allocation[TOP_K, Int32]()
    var topk_val = stack_allocation[TOP_K, Float32]()
    var topk_pk = stack_allocation[TOP_K, Float32]()
    comptime for k in range(TOP_K):
        topk_ids[unsafe_offset=k] = -1
        topk_val[unsafe_offset=k] = 0.0
        topk_pk[unsafe_offset=k] = 0.0
    var min_topk_id = 0
    var min_topk_val: Float32 = -1.0

    for b in range(ceildiv(end - start, BLOCK_SIZE)):
        barrier()
        var batch_start = start + BLOCK_SIZE * b
        var idx = batch_start + tr
        if idx < end:
            var g = Int(ids_sorted[idx, 0][0])
            id_batch[tr] = Int32(g)
            param_batch[tr, 0] = xys[g, 0][0]
            param_batch[tr, 1] = xys[g, 1][0]
            param_batch[tr, 2] = scales[g, 0][0]
            param_batch[tr, 3] = scales[g, 1][0]
            param_batch[tr, 4] = rotation[g, 0][0]
            param_batch[tr, 5] = sharpness[g, 0][0] if has_sharpness != 0 else 0.0
            var s0, s1, s2 = _softness(softnesses, stride, softness_default, g)
            param_batch[tr, 6] = s0
            param_batch[tr, 7] = s1
            param_batch[tr, 8] = s2
        barrier()
        if not inside:
            continue
        for t in range(min(BLOCK_SIZE, end - batch_start)):
            var alpha, peak = _hexel_alpha(
                fm.sub(param_batch[t, 0][0], px), fm.sub(param_batch[t, 1][0], py), param_batch[t, 2][0], param_batch[t, 3][0],
                param_batch[t, 4][0], aa, param_batch[t, 6][0], param_batch[t, 7][0], param_batch[t, 8][0], param_batch[t, 5][0],
            )  # fmt: skip
            if alpha > min_topk_val:
                topk_ids[unsafe_offset=min_topk_id] = id_batch[t][0]
                topk_val[unsafe_offset=min_topk_id] = alpha
                topk_pk[unsafe_offset=min_topk_id] = peak
                if min_topk_val < 0.0 and min_topk_id + 1 < TOP_K:
                    min_topk_id += 1
                else:
                    min_topk_val = 1e30
                    for k in range(TOP_K):
                        var id_k = topk_ids[unsafe_offset=k]
                        if id_k < 0:
                            min_topk_id = k
                            min_topk_val = -1.0
                            break
                        var val_k = topk_val[unsafe_offset=k]
                        if val_k < min_topk_val:
                            min_topk_id = k
                            min_topk_val = val_k
    if not inside:
        return
    var topk = SIMD[.int32, 16](-1)
    var topk_vals = SIMD[.float32, 16](0.0)
    var topk_peak = SIMD[.float32, 16](0.0)
    comptime for k in range(TOP_K):
        topk[k] = topk_ids[unsafe_offset=k]
        topk_vals[k] = topk_val[unsafe_offset=k]
        topk_peak[k] = topk_pk[unsafe_offset=k]

    var sum_val: Float32 = 0.0
    comptime for k in range(TOP_K):
        if topk[k] >= 0:
            sum_val = fm.add(sum_val, topk_vals[k])
    var denom = Float32(Float64(sum_val) + EPS)

    var pix_out = stack_allocation[MAX_CHANNELS, Float32]()
    for c in range(ch):
        pix_out[unsafe_offset=c] = 0.0
    for c in range(ch):
        comptime for k in range(TOP_K):
            var g = Int(topk[k])
            if g >= 0:
                var vis = fm.div(topk_vals[k], denom)
                var color = colors[g, c][0]
                comptime if color_grad:
                    var dx = fm.sub(xys[g, 0][0], px)
                    var dy = fm.sub(xys[g, 1][0], py)
                    color = fm.fma(color_grad_scale, fm.fma(dx, color_grads[2 * g, c][0], fm.mul(dy, color_grads[2 * g + 1, c][0])), color)
                pix_out[unsafe_offset=c] = fm.fma(vis, color, pix_out[unsafe_offset=c])
    if has_emission != 0:
        for c in range(ch):
            comptime for k in range(TOP_K):
                var g = Int(topk[k])
                if g >= 0:
                    pix_out[unsafe_offset=c] = fm.fma(emission[g, c][0], topk_peak[k], pix_out[unsafe_offset=c])

    for c in range(ch):
        out_img[pix, c] = pix_out[unsafe_offset=c]
    comptime for k in range(TOP_K):
        pixel_topk[pix, k] = topk[k]
    out_weights_sum[pix, 0] = denom
    var entropy: Float32 = 0.0
    comptime for k in range(TOP_K):
        if topk[k] >= 0:
            var p = fm.div(topk_vals[k], denom)
            entropy = fm.fma(p, fm.mul(fm.lg2(fm.add(p, ENTROPY_EPS)), -fm.LN2), entropy)
    out_entropy[pix, 0] = entropy


# ── backward ─────────────────────────────────────────────────────────────────


@inline(.always)
def _sign(x: Float32) -> Float32:
    return Float32(1.0) if x > 0.0 else (Float32(-1.0) if x < 0.0 else Float32(0.0))


struct _Partials(TrivialRegisterPassable):
    """hexels_backward.cu's `hexel_dist_and_partials` outputs."""

    var dist: Float32
    var du: Float32
    var dv: Float32
    var u_rot: Float32
    var v_rot: Float32
    var u_base: Float32
    var v_base: Float32
    var s_u: Float32
    var s_v: Float32
    var sin_t: Float32
    var cos_t: Float32
    var u: Float32
    var v: Float32
    var edge: Int

    @inline(.always)
    def __init__(out self, dx: Float32, dy: Float32, scale_x: Float32, scale_y: Float32, rotation: Float32, aa: Float32):
        self.sin_t = fm.sin(rotation)
        self.cos_t = fm.cos(rotation)
        self.u_rot = dx * self.cos_t + dy * self.sin_t
        self.v_rot = -dx * self.sin_t + dy * self.cos_t
        var sx = scale_x + SCALE_EPS
        var sy = scale_y + SCALE_EPS
        self.u_base = fm.div(self.u_rot, sx)
        self.v_base = fm.div(self.v_rot, sy)
        self.s_u = 1.0
        self.s_v = 1.0
        if aa > 0.0:
            self.s_u = fm.rsqrt(1.0 + aa * fm.rcp(sx * sx))
            self.s_v = fm.rsqrt(1.0 + aa * fm.rcp(sy * sy))
        self.u = self.u_base * self.s_u
        self.v = self.v_base * self.s_v
        var d0 = abs(self.v)
        var t1 = HEX_A * self.u + HEX_B * self.v
        var t2 = HEX_A * self.u - HEX_B * self.v
        var d1 = abs(t1)
        var d2 = abs(t2)
        var max_slant = d1
        self.edge = 1
        if d2 >= max_slant:
            max_slant = d2
            self.edge = 2
        if d0 > max_slant:
            self.edge = 0
            self.dist = d0 - 1.0
            self.du = 0.0
            self.dv = _sign(self.v)
        else:
            self.dist = max_slant - 1.0
            if self.edge == 1:
                var s = _sign(t1)
                self.du = HEX_A * s
                self.dv = HEX_B * s
            else:
                var s = _sign(t2)
                self.du = HEX_A * s
                self.dv = -HEX_B * s


@inline(.always)
def _sigmoid_neg(dist: Float32, softness: Float32) -> Float32:
    return fm.rcp(1.0 + fm.expf(dist * softness))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE))
def backward_kernel[color_grad: Bool](
    width: Int32,
    height: Int32,
    xys: In[.float32, Rows[2]],
    scales: In[.float32, Rows[2]],
    rotation: In[.float32, Rows[1]],
    colors: In[.float32, Grid],
    color_grads: In[.float32, Grid],
    color_grad_scale: Float32,
    has_emission: Int32,
    emission: In[.float32, Grid],
    v_output: In[.float32, Grid],
    v_weights_sum: In[.float32, Rows[1]],
    v_entropy: In[.float32, Rows[1]],
    pixel_topk: In[.int32, Rows[TOP_K]],
    aa: Float32,
    softnesses: In[.float32, Grid],
    softness_stride: Int32,
    softness_default: Float32,
    has_sharpness: Int32,
    sharpness: In[.float32, Rows[1]],
    v_xy: Out[.float32, Rows[2]],
    v_scale: Out[.float32, Rows[2]],
    v_rot: Out[.float32, Rows[1]],
    v_colors: Out[.float32, Grid],
    v_color_grads: Out[.float32, Grid],
    v_emission: Out[.float32, Grid],
    v_sharpness: Out[.float32, Rows[1]],
    v_softnesses: Out[.float32, Grid],
):
    var w = Int(width)
    var ch = Int(colors.dim[1]())
    var i = global_idx.y
    var j = global_idx.x
    if i >= Int(height) or j >= w:
        return
    var px = Float32(j)
    var py = Float32(i)
    var pix = i * w + j
    var topk = SIMD[.int32, 16](-1)
    comptime for k in range(TOP_K):
        topk[k] = pixel_topk[pix, k][0]
    if topk[0] < 0:
        return
    var stride = Int(softness_stride)

    var d_local = SIMD[.float32, 16](0.0)
    var u_coord = SIMD[.float32, 16](0.0)
    var v_coord = SIMD[.float32, 16](0.0)
    var prim_sharp = SIMD[.float32, 16](0.0)
    var prim_peak = SIMD[.float32, 16](0.0)
    var v_peak_emit = SIMD[.float32, 16](0.0)
    var denom = Float32(EPS)
    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var sharp = sharpness[g, 0][0] if has_sharpness != 0 else Float32(0.0)
            prim_sharp[k] = sharp
            var p = _Partials(xys[g, 0][0] - px, xys[g, 1][0] - py, scales[g, 0][0], scales[g, 1][0], rotation[g, 0][0], aa)
            var s0, s1, s2 = _softness(softnesses, stride, softness_default, g)
            var alpha = _sigmoid_neg(p.dist, s0 if p.edge == 0 else (s1 if p.edge == 1 else s2))
            if sharp > 0.0:
                var peak = fm.expf(-sharp * (p.u * p.u + p.v * p.v))
                prim_peak[k] = peak
                alpha *= peak
            denom += alpha
            d_local[k] = alpha
            u_coord[k] = p.u
            v_coord[k] = p.v

    var v_d_local = SIMD[.float32, 16](0.0)
    var v_d_sum: Float32 = 0.0
    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var alpha = d_local[k]
            var norm_d = fm.div(alpha, denom)
            var dx = xys[g, 0][0] - px
            var dy = xys[g, 1][0] - py
            var v_norm_d: Float32 = 0.0
            var v_dx_color: Float32 = 0.0
            var v_dy_color: Float32 = 0.0
            for c in range(ch):
                var vo = v_output[pix, c][0]
                var color = colors[g, c][0]
                atomic_add(v_colors, g, c, norm_d * vo)
                comptime if color_grad:
                    var grad_x = color_grads[2 * g, c][0]
                    var grad_y = color_grads[2 * g + 1, c][0]
                    color += color_grad_scale * (dx * grad_x + dy * grad_y)
                    atomic_add(v_color_grads, 2 * g, c, norm_d * vo * color_grad_scale * dx)
                    atomic_add(v_color_grads, 2 * g + 1, c, norm_d * vo * color_grad_scale * dy)
                    v_dx_color += norm_d * vo * color_grad_scale * grad_x
                    v_dy_color += norm_d * vo * color_grad_scale * grad_y
                v_norm_d += color * vo
            if has_emission != 0 and prim_peak[k] > 0.0:
                var acc: Float32 = 0.0
                for c in range(ch):
                    var vo = v_output[pix, c][0]
                    atomic_add(v_emission, g, c, prim_peak[k] * vo)
                    acc += emission[g, c][0] * vo
                v_peak_emit[k] = acc
            comptime if color_grad:
                atomic_add(v_xy, g, 0, v_dx_color)
                atomic_add(v_xy, g, 1, v_dy_color)
            v_d_local[k] = fm.div(v_norm_d, denom)
            v_d_sum += fm.div(-alpha, denom * denom) * v_norm_d
    comptime for k in range(TOP_K):
        v_d_local[k] += v_d_sum

    # Overdraw (weights sum) and entropy regularizers.
    var v_ws = v_weights_sum[pix, 0][0]
    var v_ent = v_entropy[pix, 0][0]
    if v_ws != 0.0:
        comptime for k in range(TOP_K):
            if topk[k] >= 0:
                v_d_local[k] += v_ws
    if v_ent != 0.0:
        var dhdp = SIMD[.float32, 16](0.0)
        var avg: Float32 = 0.0
        comptime for k in range(TOP_K):
            if topk[k] >= 0:
                var p = fm.div(d_local[k], denom)
                dhdp[k] = -(fm.lg2(p + ENTROPY_EPS) * fm.LN2 + 1.0)
                avg += p * dhdp[k]
        comptime for k in range(TOP_K):
            if topk[k] >= 0:
                v_d_local[k] += v_ent * fm.div(dhdp[k] - avg, denom)

    comptime for k in range(TOP_K):
        var g = Int(topk[k])
        if g >= 0:
            var scale_x = scales[g, 0][0]
            var scale_y = scales[g, 1][0]
            var p = _Partials(xys[g, 0][0] - px, xys[g, 1][0] - py, scale_x, scale_y, rotation[g, 0][0], aa)
            var v_d = v_d_local[k]
            var s0, s1, s2 = _softness(softnesses, stride, softness_default, g)
            var soft = s0 if p.edge == 0 else (s1 if p.edge == 1 else s2)
            var sig = _sigmoid_neg(p.dist, soft)
            var sharp = prim_sharp[k]
            var v_sig = v_d
            var v_u_peak: Float32 = 0.0
            var v_v_peak: Float32 = 0.0
            if sharp > 0.0:
                var r2 = u_coord[k] * u_coord[k] + v_coord[k] * v_coord[k]
                var peak = prim_peak[k]
                v_sig = v_d * peak
                var v_peak = v_d * sig + v_peak_emit[k]
                var v_r2 = v_peak * peak * (-sharp)
                v_u_peak = v_r2 * 2.0 * u_coord[k]
                v_v_peak = v_r2 * 2.0 * v_coord[k]
                atomic_add(v_sharpness, g, 0, v_peak * peak * (-r2))
            match stride:
                case 3:
                    atomic_add(v_softnesses, g, p.edge, v_sig * (-p.dist) * sig * (1.0 - sig))
                case 1:
                    atomic_add(v_softnesses, g, 0, v_sig * (-p.dist) * sig * (1.0 - sig))
                case _:
                    pass
            var v_dist = v_sig * (-soft) * sig * (1.0 - sig)
            var inv_sx = fm.rcp(scale_x + SCALE_EPS)
            var inv_sy = fm.rcp(scale_y + SCALE_EPS)
            var v_u = v_dist * p.du + v_u_peak
            var v_v = v_dist * p.dv + v_v_peak
            var v_u_base = v_u * p.s_u
            var v_v_base = v_v * p.s_v
            var v_u_rot = v_u_base * inv_sx
            var v_v_rot = v_v_base * inv_sy
            var v_sx = v_u_base * (-p.u_rot) * inv_sx * inv_sx
            var v_sy = v_v_base * (-p.v_rot) * inv_sy * inv_sy
            if aa > 0.0:
                v_sx += v_u * p.u_base * aa * (p.s_u * p.s_u * p.s_u) * (inv_sx * inv_sx * inv_sx)
                v_sy += v_v * p.v_base * aa * (p.s_v * p.s_v * p.s_v) * (inv_sy * inv_sy * inv_sy)
            atomic_add(v_xy, g, 0, v_u_rot * p.cos_t + v_v_rot * (-p.sin_t))
            atomic_add(v_xy, g, 1, v_u_rot * p.sin_t + v_v_rot * p.cos_t)
            atomic_add(v_scale, g, 0, v_sx)
            atomic_add(v_scale, g, 1, v_sy)
            atomic_add(v_rot, g, 0, v_u_rot * p.v_rot + v_v_rot * (-p.u_rot))
