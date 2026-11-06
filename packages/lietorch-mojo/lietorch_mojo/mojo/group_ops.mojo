"""Every lietorch backend op as a per-element function, generic over the group.

Each op belongs to one family, which fixes the tensors it takes (inputs, then
outputs; every tensor is a (batch, width) row-major `TileTensor`):

| family          | tensors                     | ops                                    |
|-----------------|-----------------------------|----------------------------------------|
| UNARY           | x -> y                      | exp, log, inv, as_matrix, projector    |
| UNARY_BACKWARD  | grad, x -> dx               | exp, log, inv backward                 |
| BINARY          | X, operand -> y             | mul, adj, adjT, act, act4, Jinv        |
| BINARY_BACKWARD | grad, X, operand -> dX, d_operand | mul, adj, adjT, act, act4 backward |

`row_widths` gives every tensor's width. The GPU kernels and the CPU loop in
`launch.mojo` call the same `*_row` functions, so the two devices run the
same code. Each branch transcribes the matching `*_kernel` template body of
lietorch `src/lietorch_gpu.cu`.

Gradient conventions are lietorch's: the gradient with respect to a group
element is a row vector in the tangent space (left perturbation), stored in
the first K slots of an N-wide row; the remaining N-K slots are zero.
Gradient inputs that belong to a group-valued output are N wide and only
their first K entries are read, as the C++ does.
"""

from layout import ImmTileTensor, MutTileTensor

from lie_group import LieGroup
from lie_linalg import Mat

# Op codes; the extension exports them (`op_codes`), and lietorch_mojo/backends.py reads them from there.
comptime OP_EXP = 0
comptime OP_EXP_BACKWARD = 1
comptime OP_LOG = 2
comptime OP_LOG_BACKWARD = 3
comptime OP_INV = 4
comptime OP_INV_BACKWARD = 5
comptime OP_MUL = 6
comptime OP_MUL_BACKWARD = 7
comptime OP_ADJ = 8
comptime OP_ADJ_BACKWARD = 9
comptime OP_ADJT = 10
comptime OP_ADJT_BACKWARD = 11
comptime OP_ACT = 12
comptime OP_ACT_BACKWARD = 13
comptime OP_ACT4 = 14
comptime OP_ACT4_BACKWARD = 15
comptime OP_AS_MATRIX = 16
comptime OP_PROJECTOR = 17
comptime OP_JINV = 18
comptime NUM_OPS = 19

comptime UNARY = 0
comptime UNARY_BACKWARD = 1
comptime BINARY = 2
comptime BINARY_BACKWARD = 3


def family(op: Int) -> Int:
    match op:
        case == OP_EXP | == OP_LOG | == OP_INV | == OP_AS_MATRIX | == OP_PROJECTOR:
            return UNARY
        case == OP_EXP_BACKWARD | == OP_LOG_BACKWARD | == OP_INV_BACKWARD:
            return UNARY_BACKWARD
        case == OP_MUL | == OP_ADJ | == OP_ADJT | == OP_ACT | == OP_ACT4 | == OP_JINV:
            return BINARY
        case _:
            return BINARY_BACKWARD


def num_tensors(op: Int) -> Int:
    match family(op):
        case == UNARY:
            return 2
        case == BINARY_BACKWARD:
            return 5
        case _:
            return 3


def row_widths[G: LieGroup, op: Int]() -> Array[Int, 5]:
    """Width of each of the op's tensors, inputs then outputs (unused slots are 0)."""
    comptime N = G.N
    comptime K = G.K
    comptime match op:
        case == OP_EXP:
            return [K, N, 0, 0, 0]
        case == OP_LOG:
            return [N, K, 0, 0, 0]
        case == OP_INV:
            return [N, N, 0, 0, 0]
        case == OP_AS_MATRIX:
            return [N, 16, 0, 0, 0]
        case == OP_PROJECTOR:
            return [N, N * N, 0, 0, 0]
        case == OP_EXP_BACKWARD:
            return [N, K, K, 0, 0]
        case == OP_LOG_BACKWARD:
            return [K, N, N, 0, 0]
        case == OP_INV_BACKWARD:
            return [N, N, N, 0, 0]
        case _:
            pass
    # Binary families: y, grad and d_operand have the operand's width.
    comptime w = operand_width[G, op]()
    comptime if family(op) == BINARY:
        return [N, w, w, 0, 0]
    else:
        return [w, N, w, N, w]


def operand_width[G: LieGroup, op: Int]() -> Int:
    """Row width of a binary op's second input: a group element, a 3D or homogeneous point, or a tangent vector."""
    comptime match op:
        case == OP_MUL | == OP_MUL_BACKWARD:
            return G.N
        case == OP_ACT | == OP_ACT_BACKWARD:
            return 3
        case == OP_ACT4 | == OP_ACT4_BACKWARD:
            return 4
        case _:
            return G.K


@inline(.always)
def load_element[G: LieGroup](rows: ImmTileTensor[G.dtype, ...], i: Int) -> G:
    return G.from_data(Mat[G.dtype, G.N, 1].load(rows, i))


@inline(.always)
def unary_row[G: LieGroup, op: Int](i: Int, x: ImmTileTensor[G.dtype, ...], y: MutTileTensor[G.dtype, ...]):
    comptime match op:
        case == OP_EXP:
            G.Exp(Mat[G.dtype, G.K, 1].load(x, i)).data().store(y, i)
        case == OP_LOG:
            load_element[G](x, i).Log().store(y, i)
        case == OP_INV:
            load_element[G](x, i).inv().data().store(y, i)
        case == OP_AS_MATRIX:
            load_element[G](x, i).Matrix4x4().store(y, i)
        case == OP_PROJECTOR:
            load_element[G](x, i).orthogonal_projector().store(y, i)
        case _:
            comptime assert False, "not a UNARY op"


@inline(.always)
def unary_backward_row[
    G: LieGroup, op: Int
](i: Int, grad: ImmTileTensor[G.dtype, ...], x: ImmTileTensor[G.dtype, ...], dx: MutTileTensor[G.dtype, ...]):
    comptime match op:
        case == OP_EXP_BACKWARD:
            var dX = Mat[G.dtype, 1, G.K].load_head(grad, i)
            (dX @ G.left_jacobian(Mat[G.dtype, G.K, 1].load(x, i))).store(dx, i)
        case == OP_LOG_BACKWARD:
            var a = load_element[G](x, i).Log()
            (Mat[G.dtype, 1, G.K].load(grad, i) @ G.left_jacobian_inverse(a)).store_padded(dx, i)
        case == OP_INV_BACKWARD:
            var Y = load_element[G](x, i).inv()
            var dY = Mat[G.dtype, 1, G.K].load_head(grad, i)
            ((-dY) @ Y.Adj()).store_padded(dx, i)
        case _:
            comptime assert False, "not a UNARY_BACKWARD op"


@inline(.always)
def binary_row[G: LieGroup, op: Int](i: Int, X: ImmTileTensor[G.dtype, ...], operand: ImmTileTensor[G.dtype, ...], y: MutTileTensor[G.dtype, ...]):
    var g = load_element[G](X, i)
    comptime match op:
        case == OP_MUL:
            g.mul(load_element[G](operand, i)).data().store(y, i)
        case == OP_ADJ:
            (g.Adj() @ Mat[G.dtype, G.K, 1].load(operand, i)).store(y, i)
        case == OP_ADJT:
            (g.Adj().T() @ Mat[G.dtype, G.K, 1].load(operand, i)).store(y, i)
        case == OP_ACT:
            g.act(Mat[G.dtype, 3, 1].load(operand, i)).store(y, i)
        case == OP_ACT4:
            g.act4(Mat[G.dtype, 4, 1].load(operand, i)).store(y, i)
        case == OP_JINV:
            (G.left_jacobian_inverse(g.Log()) @ Mat[G.dtype, G.K, 1].load(operand, i)).store(y, i)
        case _:
            comptime assert False, "not a BINARY op"


@inline(.always)
def binary_backward_row[
    G: LieGroup, op: Int
](
    i: Int,
    grad: ImmTileTensor[G.dtype, ...],
    X: ImmTileTensor[G.dtype, ...],
    operand: ImmTileTensor[G.dtype, ...],
    dX: MutTileTensor[G.dtype, ...],
    d_operand: MutTileTensor[G.dtype, ...],
):
    var g = load_element[G](X, i)
    comptime match op:
        case == OP_MUL_BACKWARD:
            var dZ = Mat[G.dtype, 1, G.K].load_head(grad, i)
            dZ.store_padded(dX, i)
            (dZ @ g.Adj()).store_padded(d_operand, i)
        case == OP_ADJ_BACKWARD:
            var db = Mat[G.dtype, 1, G.K].load(grad, i)
            var Ad = g.Adj()
            var b = Ad @ Mat[G.dtype, G.K, 1].load(operand, i)
            (db @ Ad).store(d_operand, i)
            ((-db) @ G.adj(b)).store_padded(dX, i)
        case == OP_ADJT_BACKWARD:
            var Ad_db = g.Adj() @ Mat[G.dtype, G.K, 1].load(grad, i)
            Ad_db.store(d_operand, i)
            ((-Mat[G.dtype, 1, G.K].load(operand, i)) @ G.adj(Ad_db)).store_padded(dX, i)
        case == OP_ACT_BACKWARD:
            var p = Mat[G.dtype, 3, 1].load(operand, i)
            var dq = Mat[G.dtype, 1, 3].load(grad, i)
            (dq @ g.Matrix4x4().block[0, 0, 3, 3]()).store(d_operand, i)
            (dq @ G.act_jacobian(g.act(p))).store_padded(dX, i)
        case == OP_ACT4_BACKWARD:
            var p = Mat[G.dtype, 4, 1].load(operand, i)
            var dq = Mat[G.dtype, 1, 4].load(grad, i)
            (dq @ g.Matrix4x4()).store(d_operand, i)
            (dq @ G.act4_jacobian(g.act4(p))).store_padded(dX, i)
        case _:
            comptime assert False, "not a BINARY_BACKWARD op"
