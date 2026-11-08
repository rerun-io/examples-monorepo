"""Hexels' 2D prune scores (hexels-src `scripts/image_compression_prune_scores.cu`).

One thread per pixel. For each of the pixel's ten renderer-selected hexels, the change in the
pixel's MSE when that one hexel is removed (the others held fixed) is added to its score. The
thread also records the largest |denominator - rendered alpha| as a check that the weights here
match the renderer's. `hexel_alpha` is the renderer's SDF (hexel-kernel `hexels_forward.cu`)
with no anti-aliasing and no peak, which is the only form the scorer calls.

Float atomics accumulate the scores, so the summation order (and the last bits) depends on
scheduling, as in CUDA.
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx
from std.atomic import Atomic, Ordering
from std.math import cos, exp, sin
from std.memory import bitcast
from std.utils.static_tuple import StaticTuple

comptime F32Ptr = MutPointer[Float32, MutUntrackedOrigin]
comptime I32Ptr = MutPointer[Int32, MutUntrackedOrigin]
comptime U32Ptr = MutPointer[UInt32, MutUntrackedOrigin]
comptime TOP_K = 10
comptime BLOCK = 16


@always_inline
def hexel_alpha(dx: Float32, dy: Float32, sx: Float32, sy: Float32, rotation: Float32, softness: Float32) -> Float32:
    """sigmoid(-sdf * softness) of the pointy-top hexagon SDF max(|v|, |a u + b v|, |a u - b v|) - 1."""
    comptime a: Float32 = 0.5 * 1.7320508
    comptime b: Float32 = 0.5
    var s = sin(rotation)
    var c = cos(rotation)
    var u = (dx * c + dy * s) / (sx + 1e-8)
    var v = (-dx * s + dy * c) / (sy + 1e-8)
    var dist = max(abs(v), abs(a * u + b * v), abs(a * u - b * v)) - 1.0
    return 1.0 / (1.0 + exp(dist * softness))


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def removal_scores_kernel[rich: Bool](
    width: Int32,
    height: Int32,
    xy: F32Ptr,
    scales: F32Ptr,
    rotations: F32Ptr,
    colors: F32Ptr,
    gradients: F32Ptr,
    softness: F32Ptr,
    image: F32Ptr,
    target: F32Ptr,
    selected: I32Ptr,
    scores: F32Ptr,
    max_error: U32Ptr,
):
    var x = global_idx.x
    var y = global_idx.y
    if x >= Int(width) or y >= Int(height):
        return
    var pixel = y * Int(width) + x
    var weights = SIMD[DType.float32, 16](0.0)
    var local = SIMD[DType.float32, 32](0.0)
    var total: Float32 = 0.0
    comptime for k in range(TOP_K):
        var id = Int(selected.unsafe_load(pixel * TOP_K + k))
        if id >= 0:
            var dx = xy.unsafe_load(2 * id) - Float32(x)
            var dy = xy.unsafe_load(2 * id + 1) - Float32(y)
            var soft: Float32
            comptime if rich:
                soft = softness.unsafe_load(id)
            else:
                soft = 10.0
            weights[k] = hexel_alpha(dx, dy, scales.unsafe_load(2 * id), scales.unsafe_load(2 * id + 1), rotations.unsafe_load(id), soft)
            total += weights[k]
            comptime for c in range(3):
                local[k * 3 + c] = colors.unsafe_load(id * 3 + c)
                comptime if rich:
                    local[k * 3 + c] += 0.1 * (dx * gradients.unsafe_load(id * 6 + c) + dy * gradients.unsafe_load(id * 6 + 3 + c))
    var denominator = total + 1e-4
    var error = abs(denominator - image.unsafe_load(pixel * 4 + 3))
    # Non-negative floats order like their bit patterns, so an unsigned max is a float max.
    Atomic[UInt32, scope="device"].max[ordering=Ordering.RELAXED](max_error, bitcast[DType.uint32](error))
    var before: Float32 = 0.0
    comptime for c in range(3):
        var difference = image.unsafe_load(pixel * 4 + c) - target.unsafe_load(pixel * 3 + c)
        before += difference * difference / 3.0
    comptime for k in range(TOP_K):
        var id = Int(selected.unsafe_load(pixel * TOP_K + k))
        if id >= 0 and weights[k] != 0.0:
            var remaining = max(denominator - weights[k], 1e-4)
            var after: Float32 = 0.0
            comptime for c in range(3):
                var removed = (image.unsafe_load(pixel * 4 + c) * denominator - weights[k] * local[k * 3 + c]) / remaining
                var difference = removed - target.unsafe_load(pixel * 3 + c)
                after += difference * difference / 3.0
            _ = Atomic[Float32, scope="device"].fetch_add[ordering=Ordering.RELAXED](scores.unsafe_offset(id), after - before)
