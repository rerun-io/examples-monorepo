"""Torch tensors as the kernels see them: row-major TileTensors with a runtime row count.

The launcher (`_hexels_mojo.mojo`) wraps every contiguous CUDA tensor as `(rows, width)`, `width` being the static
product of its trailing dimensions (3 for (P, 3), 6 for (P, 2, 3)), or with fully runtime dimensions where those
vary (channels, image planes). Inputs are immutable, outputs mutable.
"""

from layout import Coord, Idx, ImmTileTensor, MutTileTensor, TensorLayout, row_major
from std.atomic import Atomic, Ordering

comptime Rows[width: Int] = type_of(row_major((0, Idx[width])))
comptime Grid = type_of(row_major((0, 0)))
comptime Planes = type_of(row_major((0, 0, 0)))
comptime StackedPlanes = type_of(row_major((0, 0, 0, 0)))
comptime In[dtype: DType, L: TensorLayout] = ImmTileTensor[dtype, L, ImmUntrackedOrigin]
comptime Out[dtype: DType, L: TensorLayout] = MutTileTensor[dtype, L, MutUntrackedOrigin]


@inline(.always)
def atomic_add[L: TensorLayout](t: MutTileTensor[.float32, L, _], row: Int, col: Int, value: Float32):
    """Relaxed device-scope `t[row, col] += value` (CUDA's atomicAdd)."""
    comptime assert t.flat_rank == 2
    var at = t.unsafe_ptr().unsafe_offset(Int(t.layout(Coord(row, col))))
    _ = Atomic[Float32, scope="device"].fetch_add[ordering=Ordering.RELAXED](at, value)
