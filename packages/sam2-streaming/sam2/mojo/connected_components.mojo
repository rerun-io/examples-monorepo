"""SAM 2's connected components (`sam2/csrc/connected_components.cu`, after zsef123/Connected_components_PyTorch).

8-connected labelling of (N, 1, H, W) uint8 masks, H and W even, by block union-find: every 2x2 block is one
node (any two pixels of a 2x2 block are 8-adjacent), named by the index of its top-left pixel. `merge` links
each block to its foreground neighbours above and to the left, always pointing the larger root at the smaller
one, so every component ends at its smallest block index whatever the scheduling. Labels are that index + 1
(0 = background), and counts are each pixel's component area. Grid z is the batch item (CUDA loops on the host).
"""

from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, block_dim, block_idx, thread_idx
from std.atomic import Atomic, Ordering
from std.utils.static_tuple import StaticTuple

comptime U8Ptr = MutPointer[UInt8, MutUntrackedOrigin]
comptime I32Ptr = MutPointer[Int32, MutUntrackedOrigin]
comptime BLOCK = 16


@always_inline
def _find(label: I32Ptr, n_: Int32) -> Int32:
    var n = n_
    while label.unsafe_load(Int(n)) != n:
        n = label.unsafe_load(Int(n))
    return n


@always_inline
def _atomic_min(p: I32Ptr, value: Int32) -> Int32:
    """CUDA atomicMin: store min(*p, value) and return the old *p."""
    var old = Atomic[Int32, scope="device"].load[ordering=Ordering.RELAXED](p)
    while value < old:
        if Atomic[Int32, scope="device"].compare_exchange[success_ordering=Ordering.RELAXED, failure_ordering=Ordering.RELAXED](p, old, value):
            break
    return old


@always_inline
def _union(label: I32Ptr, a_: Int32, b_: Int32):
    var a = a_
    var b = b_
    while True:
        a = _find(label, a)
        b = _find(label, b)
        if a < b:
            var old = _atomic_min(label.unsafe_offset(Int(b)), a)
            if old == b:
                return
            b = old
        elif b < a:
            var old = _atomic_min(label.unsafe_offset(Int(a)), b)
            if old == a:
                return
            a = old
        else:
            return


@always_inline
def _block_origin(height: Int, width: Int) -> Tuple[Int, Int, Int]:
    """(row, col, plane offset) of this thread's 2x2 block."""
    var row = (block_idx.y * block_dim.y + thread_idx.y) * 2
    var col = (block_idx.x * block_dim.x + thread_idx.x) * 2
    return (row, col, block_idx.z * height * width)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def init_labeling_kernel(label_: I32Ptr, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col, plane = _block_origin(h, w)
    if row < h and col < w:
        label_.unsafe_offset(plane)[unsafe_offset=row * w + col] = Int32(row * w + col)


@always_inline
def _bit(p: UInt32, pos: Int) -> Bool:
    return ((p >> UInt32(pos)) & 1) == 1


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def merge_kernel(img_: U8Ptr, label_: I32Ptr, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col, plane = _block_origin(h, w)
    if row >= h or col >= w:
        return
    var img = img_.unsafe_offset(plane)
    var label = label_.unsafe_offset(plane)
    var idx = row * w + col
    # P: the 4x4 neighbourhood (rows row-1 .. row+2, cols col-1 .. col+2) that the block's pixels touch.
    var p: UInt32 = 0
    if img.unsafe_load(idx) != 0:
        p |= 0x777
    if row + 1 < h and img.unsafe_load(idx + w) != 0:
        p |= 0x777 << 4
    if col + 1 < w and img.unsafe_load(idx + 1) != 0:
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
    if _bit(p, 0) and img.unsafe_load(idx - w - 1) != 0:
        _union(label, me, Int32(idx - 2 * w - 2))  # top-left block
    if (_bit(p, 1) and img.unsafe_load(idx - w) != 0) or (_bit(p, 2) and img.unsafe_load(idx - w + 1) != 0):
        _union(label, me, Int32(idx - 2 * w))  # block above
    if _bit(p, 3) and img.unsafe_load(idx + 2 - w) != 0:
        _union(label, me, Int32(idx - 2 * w + 2))  # top-right block
    if (_bit(p, 4) and img.unsafe_load(idx - 1) != 0) or (_bit(p, 8) and img.unsafe_load(idx + w - 1) != 0):
        _union(label, me, Int32(idx - 2))  # block to the left


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def compression_kernel(label_: I32Ptr, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col, plane = _block_origin(h, w)
    if row >= h or col >= w:
        return
    var label = label_.unsafe_offset(plane)
    var id = row * w + col
    var n = Int32(id)
    while label.unsafe_load(Int(n)) != n:
        n = label.unsafe_load(Int(n))
        label[unsafe_offset=id] = n


@always_inline
def _set(img: U8Ptr, label: I32Ptr, at: Int, y: Int32):
    label[unsafe_offset=at] = y if img.unsafe_load(at) != 0 else 0


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def final_labeling_kernel(img_: U8Ptr, label_: I32Ptr, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row, col, plane = _block_origin(h, w)
    if row >= h or col >= w:
        return
    var img = img_.unsafe_offset(plane)
    var label = label_.unsafe_offset(plane)
    var idx = row * w + col
    var y = label.unsafe_load(idx) + 1
    _set(img, label, idx, y)
    if col + 1 < w:
        _set(img, label, idx + 1, y)
        if row + 1 < h:
            _set(img, label, idx + w + 1, y)
    if row + 1 < h:
        _set(img, label, idx + w, y)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def init_counting_kernel(label_: I32Ptr, count_init_: I32Ptr, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    if row >= h or col >= w:
        return
    var plane = block_idx.z * h * w
    var y = label_.unsafe_load(plane + row * w + col)
    if y > 0:
        _ = Atomic[Int32, scope="device"].fetch_add[ordering=Ordering.RELAXED](count_init_.unsafe_offset(plane + Int(y) - 1), 1)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK * BLOCK))
def final_counting_kernel(label_: I32Ptr, count_init_: I32Ptr, count_final_: I32Ptr, height: Int32, width: Int32):
    var h = Int(height)
    var w = Int(width)
    var row = block_idx.y * block_dim.y + thread_idx.y
    var col = block_idx.x * block_dim.x + thread_idx.x
    if row >= h or col >= w:
        return
    var at = block_idx.z * h * w + row * w + col
    var y = label_.unsafe_load(at)
    count_final_[unsafe_offset=at] = count_init_.unsafe_load(block_idx.z * h * w + Int(y) - 1) if y > 0 else 0
