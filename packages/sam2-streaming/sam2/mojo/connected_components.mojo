"""SAM 2's connected components (`sam2/csrc/connected_components.cu`, after zsef123/Connected_components_PyTorch).

8-connected labelling of (N, 1, H, W) uint8 masks, H and W even, by block union-find: every 2x2 block is one
node (any two pixels of a 2x2 block are 8-adjacent), named by the index of its top-left pixel. `merge` links
each block to its foreground neighbours above and to the left, always pointing the larger root at the smaller
one, so every component ends at its smallest block index whatever the scheduling. Labels are that index + 1
(0 = background), and counts are each pixel's component area. Grid z is the batch item (CUDA loops on the host).

Masks, labels and counts are (N, H W) tensors indexed by (item, pixel): a label is a pixel index, so the
union-find walks pixel indices.
"""

from layout import Coord, Idx, ImmTileTensor, MutTileTensor, TensorLayout, TileTensor, row_major
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_dim, block_idx, thread_idx
from std.atomic import Atomic, Ordering
from std.utils.static_tuple import StaticTuple

comptime BLOCK = 16
comptime Pixels = type_of(row_major((0, 0)))
comptime Mask = ImmTileTensor[.uint8, Pixels, ImmUntrackedOrigin]
comptime Labels = MutTileTensor[.int32, Pixels, MutUntrackedOrigin]
comptime Counts = ImmTileTensor[.int32, Pixels, ImmUntrackedOrigin]


@inline(.always)
def _find(labels: TileTensor[mut=True, .int32, ...], z: Int, var n: Int32) -> Int32:
    comptime assert labels.flat_rank == 2
    while labels[z, Int(n)][0] != n:
        n = labels[z, Int(n)][0]
    return n


@inline(.always)
def _atomic_min(labels: TileTensor[mut=True, .int32, ...], z: Int, n: Int32, value: Int32) -> Int32:
    """CUDA atomicMin on labels[z, n]: store min(labels[z, n], value) and return the old value."""
    comptime assert labels.flat_rank == 2
    var p = labels.unsafe_ptr().unsafe_offset(Int(labels.layout(Coord(z, Int(n)))))
    var old = Atomic[Int32, scope="device"].load[ordering=Ordering.RELAXED](p)
    while value < old:
        if Atomic[Int32, scope="device"].compare_exchange[success_ordering=Ordering.RELAXED, failure_ordering=Ordering.RELAXED](p, old, value):
            break
    return old


@inline(.always)
def _union(labels: TileTensor[mut=True, .int32, ...], z: Int, var a: Int32, var b: Int32):
    while True:
        a = _find(labels, z, a)
        b = _find(labels, z, b)
        if a < b:
            var old = _atomic_min(labels, z, b, a)
            if old == b:
                return
            b = old
        elif b < a:
            var old = _atomic_min(labels, z, a, b)
            if old == a:
                return
            a = old
        else:
            return


@inline(.always)
def _block_origin() -> Tuple[Int, Int]:
    """(row, col) of this thread's 2x2 block."""
    return ((block_idx.y * block_dim.y + thread_idx.y) * 2, (block_idx.x * block_dim.x + thread_idx.x) * 2)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def init_labeling_kernel(labels: Labels, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col = _block_origin()
    if row < h and col < w:
        labels[block_idx.z, row * w + col] = Int32(row * w + col)


@inline(.always)
def _bit(p: UInt32, pos: Int) -> Bool:
    return ((p >> UInt32(pos)) & 1) == 1


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def merge_kernel(mask: Mask, labels: Labels, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col = _block_origin()
    if row >= h or col >= w:
        return
    var z = block_idx.z
    var idx = row * w + col
    # P: the 4x4 neighbourhood (rows row-1 .. row+2, cols col-1 .. col+2) that the block's pixels touch.
    var p: UInt32 = 0
    if mask[z, idx][0] != 0:
        p |= 0x777
    if row + 1 < h and mask[z, idx + w][0] != 0:
        p |= 0x777 << 4
    if col + 1 < w and mask[z, idx + 1][0] != 0:
        p |= 0x777 << 1
    if col == 0:
        p &= 0xEEEE
    if col + 1 >= w:
        p &= 0x3333
    elif col + 2 >= w:
        p &= 0x7777
    if row == 0:
        p &= 0xFFF0
    if row + 1 >= h:
        p &= 0xFF
    if p == 0:
        return
    var me = Int32(idx)
    if _bit(p, 0) and mask[z, idx - w - 1][0] != 0:
        _union(labels, z, me, Int32(idx - 2 * w - 2))  # top-left block
    if (_bit(p, 1) and mask[z, idx - w][0] != 0) or (_bit(p, 2) and mask[z, idx - w + 1][0] != 0):
        _union(labels, z, me, Int32(idx - 2 * w))  # block above
    if _bit(p, 3) and mask[z, idx + 2 - w][0] != 0:
        _union(labels, z, me, Int32(idx - 2 * w + 2))  # top-right block
    if (_bit(p, 4) and mask[z, idx - 1][0] != 0) or (_bit(p, 8) and mask[z, idx + w - 1][0] != 0):
        _union(labels, z, me, Int32(idx - 2))  # block to the left


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def compression_kernel(labels: Labels, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col = _block_origin()
    if row >= h or col >= w:
        return
    var z = block_idx.z
    var id = row * w + col
    var n = Int32(id)
    while labels[z, Int(n)][0] != n:
        n = labels[z, Int(n)][0]
        labels[z, id] = n


@inline(.always)
def _label_pixel(mask: Mask, labels: Labels, z: Int, at: Int, label: Int32):
    labels[z, at] = label if mask[z, at][0] != 0 else 0


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def final_labeling_kernel(mask: Mask, labels: Labels, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col = _block_origin()
    if row >= h or col >= w:
        return
    var z = block_idx.z
    var idx = row * w + col
    var y = labels[z, idx][0] + 1
    _label_pixel(mask, labels, z, idx, y)
    if col + 1 < w:
        _label_pixel(mask, labels, z, idx + 1, y)
        if row + 1 < h:
            _label_pixel(mask, labels, z, idx + w + 1, y)
    if row + 1 < h:
        _label_pixel(mask, labels, z, idx + w, y)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def init_counting_kernel(labels: Counts, counts_init: Labels, height: Int32, width: Int32):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    if row >= Int(height) or col >= Int(width):
        return
    var z = block_idx.z
    var y = labels[z, row * Int(width) + col][0]
    if y > 0:
        var slot = counts_init.unsafe_ptr().unsafe_offset(Int(counts_init.layout(Coord(z, Int(y) - 1))))
        _ = Atomic[Int32, scope="device"].fetch_add[ordering=Ordering.RELAXED](slot, 1)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def final_counting_kernel(labels: Counts, counts_init: Counts, counts: Labels, height: Int32, width: Int32):
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    if row >= Int(height) or col >= Int(width):
        return
    var z = block_idx.z
    var at = row * Int(width) + col
    var y = labels[z, at][0]
    counts[z, at] = counts_init[z, Int(y) - 1][0] if y > 0 else 0
