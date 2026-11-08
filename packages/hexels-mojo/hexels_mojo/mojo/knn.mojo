"""simple-knn `distCUDA2` (bkerbl/simple-knn@f155ec0 `simple_knn.cu`): mean squared distance to the 3 nearest points.

Approximate, as upstream: points are sorted by a 30-bit Morton code (10 bits per axis over the bounding box);
the 3 nearest of the 3 neighbours on either side in that order give a rejection radius, then every box of
1024 consecutive sorted points whose bounds lie within both that radius and the current 3rd-best distance is
searched exhaustively. The bounds reduction and the stable radix sort run in PyTorch (`hexels_mojo.simple_knn`);
the three kernels here are upstream's `coord2Morton`, `boxMinMax` and `boxMeanDist`.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from max.gpu.sync import barrier
from std.math import ceildiv
from std.memory import stack_allocation
from std.utils.static_tuple import StaticTuple

from tensors import In, Out, Rows

comptime BOX_SIZE = 1024
comptime MORTON_THREADS = 256
comptime FLT_MAX = Float32.MAX_FINITE
comptime Vec3 = SIMD[.float32, 4]


@inline(.always)
def _point(points: In[.float32, Rows[3]], i: Int) -> Vec3:
    return Vec3(points[i, 0][0], points[i, 1][0], points[i, 2][0], 0.0)


@inline(.always)
def _prep_morton(v: Float32) -> UInt32:
    var x = UInt32(v)  # C++ float -> uint32_t conversion: truncation
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](MORTON_THREADS))
def morton_kernel(points: In[.float32, Rows[3]], bounds: In[.float32, Rows[3]], codes: Out[.int64, Rows[1]]):
    """codes[i] = Morton code of points[i] in the box bounds = (min xyz, max xyz)."""
    var i = global_idx.x
    if i >= Int(points.dim[0]()):
        return
    var code: UInt32 = 0
    comptime for axis in range(3):
        var lo = bounds[0, axis][0]
        var hi = bounds[1, axis][0]
        code |= _prep_morton(((points[i, axis][0] - lo) / (hi - lo)) * Float32((1 << 10) - 1)) << UInt32(axis)
    codes[i, 0] = Int64(code)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BOX_SIZE))
def box_bounds_kernel(points: In[.float32, Rows[3]], order: In[.int64, Rows[1]], boxes: Out[.float32, Rows[6]]):
    """boxes[b] = (min xyz, max xyz) of the sorted points b * 1024 ... (b + 1) * 1024 - 1."""
    var tid = thread_idx.x
    var idx = block_idx.x * BOX_SIZE + tid
    var lo = Vec3(FLT_MAX)
    var hi = Vec3(-FLT_MAX)
    if idx < Int(points.dim[0]()):
        lo = _point(points, Int(order[idx, 0][0]))
        hi = lo
    # SIMD-valued scratch: a raw shared array (TileTensor elements are scalars).
    var shared_lo = stack_allocation[BOX_SIZE, Vec3, address_space=.SHARED]()
    var shared_hi = stack_allocation[BOX_SIZE, Vec3, address_space=.SHARED]()
    var off = BOX_SIZE // 2
    while off >= 1:
        if tid < 2 * off:
            shared_lo[unsafe_offset=tid] = lo
            shared_hi[unsafe_offset=tid] = hi
        barrier()
        if tid < off:
            lo = min(lo, shared_lo[unsafe_offset=tid + off])
            hi = max(hi, shared_hi[unsafe_offset=tid + off])
        barrier()
        off //= 2
    if tid == 0:
        comptime for axis in range(3):
            boxes[block_idx.x, axis] = lo[axis]
            boxes[block_idx.x, 3 + axis] = hi[axis]


@inline(.always)
def _dist2(d: Vec3) -> Float32:
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


@inline(.always)
def _update_best(ref_point: Vec3, point: Vec3, mut best: Vec3):
    """Insert |point - ref|^2 into the ascending best[0:3]."""
    var dist = _dist2(point - ref_point)
    comptime for j in range(3):
        if best[j] > dist:
            var t = best[j]
            best[j] = dist
            dist = t


@inline(.always)
def _box_dist2(boxes: In[.float32, Rows[6]], b: Int, p: Vec3) -> Float32:
    var diff = Vec3(0.0)
    comptime for axis in range(3):
        var lo = boxes[b, axis][0]
        var hi = boxes[b, 3 + axis][0]
        if p[axis] < lo or p[axis] > hi:
            diff[axis] = min(abs(p[axis] - lo), abs(p[axis] - hi))
    return _dist2(diff)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BOX_SIZE))
def box_mean_dist_kernel(
    points: In[.float32, Rows[3]], order: In[.int64, Rows[1]], boxes: In[.float32, Rows[6]], dists: Out[.float32, Rows[1]]
):
    """dists[order[i]] = mean of the 3 smallest squared distances from sorted point i (box-pruned search)."""
    var n = Int(points.dim[0]())
    var idx = global_idx.x
    if idx >= n:
        return
    var point = _point(points, Int(order[idx, 0][0]))
    var best = Vec3(FLT_MAX)
    for i in range(max(0, idx - 3), min(n - 1, idx + 3) + 1):
        if i != idx:
            _update_best(point, _point(points, Int(order[i, 0][0])), best)
    var reject = best[2]
    best = Vec3(FLT_MAX)
    for b in range(ceildiv(n, BOX_SIZE)):
        var dist = _box_dist2(boxes, b, point)
        if dist > reject or dist > best[2]:
            continue
        for i in range(b * BOX_SIZE, min(n, (b + 1) * BOX_SIZE)):
            if i != idx:
                _update_best(point, _point(points, Int(order[i, 0][0])), best)
    dists[Int(order[idx, 0][0]), 0] = (best[0] + best[1] + best[2]) / 3.0
