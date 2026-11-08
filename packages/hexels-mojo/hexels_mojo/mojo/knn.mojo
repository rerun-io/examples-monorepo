"""simple-knn `distCUDA2` (bkerbl/simple-knn@f155ec0 `simple_knn.cu`): mean squared distance to the 3 nearest points.

Approximate, as upstream: points are sorted by a 30-bit Morton code (10 bits per axis over the bounding box);
the 3 nearest of the 3 neighbours on either side in that order give a rejection radius, then every box of
1024 consecutive sorted points whose bounds lie within both that radius and the current 3rd-best distance is
searched exhaustively. The bounds reduction and the stable radix sort run in PyTorch (`hexels_mojo.simple_knn`);
the three kernels here are upstream's `coord2Morton`, `boxMinMax` and `boxMeanDist`.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_idx, global_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation
from std.utils.static_tuple import StaticTuple

comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime I64Ptr = MutPointer[Int64, MutUntrackedOrigin]
comptime BOX_SIZE = 1024
comptime MORTON_THREADS = 256
comptime FLT_MAX = Float32.MAX_FINITE
comptime Vec3 = SIMD[DType.float32, 4]


@always_inline
def _point(points: F32Ptr, i: Int) -> Vec3:
    return Vec3(points.unsafe_load(3 * i), points.unsafe_load(3 * i + 1), points.unsafe_load(3 * i + 2), 0.0)


@always_inline
def _prep_morton(v: Float32) -> UInt32:
    var x = UInt32(v)  # C++ float -> uint32_t conversion: truncation
    x = (x | (x << 16)) & 0x030000FF
    x = (x | (x << 8)) & 0x0300F00F
    x = (x | (x << 4)) & 0x030C30C3
    x = (x | (x << 2)) & 0x09249249
    return x


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](MORTON_THREADS))
def morton_kernel(count: Int32, points: F32Ptr, bounds: F32Ptr, codes: I64Ptr):
    """codes[i] = Morton code of points[i] in the box bounds = (min xyz, max xyz)."""
    var i = global_idx.x
    if i >= Int(count):
        return
    var code: UInt32 = 0
    comptime for axis in range(3):
        var lo = bounds.unsafe_load(axis)
        var hi = bounds.unsafe_load(3 + axis)
        code |= _prep_morton(((points.unsafe_load(3 * i + axis) - lo) / (hi - lo)) * Float32((1 << 10) - 1)) << UInt32(axis)
    codes[unsafe_offset=i] = Int64(code)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BOX_SIZE))
def box_bounds_kernel(count: Int32, points: F32Ptr, order: I64Ptr, boxes: F32Ptr):
    """boxes[b] = (min xyz, max xyz) of the sorted points b * 1024 ... (b + 1) * 1024 - 1."""
    var tid = thread_idx.x
    var idx = block_idx.x * BOX_SIZE + tid
    var lo = Vec3(FLT_MAX)
    var hi = Vec3(-FLT_MAX)
    if idx < Int(count):
        lo = _point(points, Int(order.unsafe_load(idx)))
        hi = lo
    var shared_lo = stack_allocation[BOX_SIZE, Vec3, address_space=AddressSpace.SHARED]()
    var shared_hi = stack_allocation[BOX_SIZE, Vec3, address_space=AddressSpace.SHARED]()
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
            boxes[unsafe_offset=6 * block_idx.x + axis] = lo[axis]
            boxes[unsafe_offset=6 * block_idx.x + 3 + axis] = hi[axis]


@always_inline
def _dist2(d: Vec3) -> Float32:
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


@always_inline
def _update_best(ref_point: Vec3, point: Vec3, mut best: Vec3):
    """Insert |point - ref|^2 into the ascending best[0:3]."""
    var dist = _dist2(point - ref_point)
    comptime for j in range(3):
        if best[j] > dist:
            var t = best[j]
            best[j] = dist
            dist = t


@always_inline
def _box_dist2(boxes: F32Ptr, b: Int, p: Vec3) -> Float32:
    var diff = Vec3(0.0)
    comptime for axis in range(3):
        var lo = boxes.unsafe_load(6 * b + axis)
        var hi = boxes.unsafe_load(6 * b + 3 + axis)
        if p[axis] < lo or p[axis] > hi:
            diff[axis] = min(abs(p[axis] - lo), abs(p[axis] - hi))
    return _dist2(diff)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BOX_SIZE))
def box_mean_dist_kernel(count: Int32, points: F32Ptr, order: I64Ptr, boxes: F32Ptr, dists: F32Ptr):
    """dists[order[i]] = mean of the 3 smallest squared distances from sorted point i (box-pruned search)."""
    var n = Int(count)
    var idx = global_idx.x
    if idx >= n:
        return
    var point = _point(points, Int(order.unsafe_load(idx)))
    var best = Vec3(FLT_MAX)
    for i in range(max(0, idx - 3), min(n - 1, idx + 3) + 1):
        if i != idx:
            _update_best(point, _point(points, Int(order.unsafe_load(i))), best)
    var reject = best[2]
    best = Vec3(FLT_MAX)
    for b in range((n + BOX_SIZE - 1) // BOX_SIZE):
        var dist = _box_dist2(boxes, b, point)
        if dist > reject or dist > best[2]:
            continue
        for i in range(b * BOX_SIZE, min(n, (b + 1) * BOX_SIZE)):
            if i != idx:
                _update_best(point, _point(points, Int(order.unsafe_load(i))), best)
    dists[unsafe_offset=Int(order.unsafe_load(idx))] = (best[0] + best[1] + best[2]) / 3.0
