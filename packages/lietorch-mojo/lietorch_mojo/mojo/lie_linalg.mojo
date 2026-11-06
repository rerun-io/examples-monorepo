"""Fixed-size row-major matrices for per-element Lie group math.

`Mat[dtype, R, C]` stands in for the Eigen fixed-size matrices of lietorch's
C++ headers. Every loop is a `comptime for`, so after inlining each entry is a
compile-time index and the whole matrix lives in registers on the GPU. Column
vectors are `Mat[dtype, R, 1]`, row vectors (gradients) are `Mat[dtype, 1, C]`.
Products sum k = 0, 1, ... in order, as Eigen's coefficient-based small-matrix
product does.

Memory is read and written only through `load`/`load_head`/`store`/
`store_padded`, which move one row of a (batch, width) `TileTensor`.
"""

from layout import ImmTileTensor, MutTileTensor


struct Mat[dtype: DType, R: Int, C: Int](ImplicitlyCopyable):
    var values: Array[Scalar[Self.dtype], Self.R * Self.C]

    @inline(.always)
    def __init__(out self):
        """Zero matrix."""
        self.values = Array[Scalar[Self.dtype], Self.R * Self.C](fill=0)

    @inline(.always)
    def __init__(out self, *, copy: Self):
        # `Array` is only explicitly copyable; a register matrix is a value type.
        self.values = copy.values.copy()

    @staticmethod
    @inline(.always)
    def identity() -> Self:
        var m = Self()
        comptime for i in range(min(Self.R, Self.C)):
            m[i, i] = 1
        return m

    @staticmethod
    @inline(.always)
    def load(rows: ImmTileTensor[Self.dtype, ...], i: Int) -> Self:
        """Row `i` of `rows`, whose width is exactly R*C (row-major)."""
        comptime assert rows.flat_rank == 2, "expected a (batch, width) tensor"
        comptime assert rows.static_shape[1] == Self.R * Self.C, "row width must be R*C"
        return Self.load_head(rows, i)

    @staticmethod
    @inline(.always)
    def load_head(rows: ImmTileTensor[Self.dtype, ...], i: Int) -> Self:
        """The first R*C values of row `i` (the tangent part of an embedded-size gradient row)."""
        comptime assert rows.flat_rank == 2, "expected a (batch, width) tensor"
        comptime assert rows.static_shape[1] >= Self.R * Self.C, "row too narrow"
        var m = Self()
        comptime for j in range(Self.R * Self.C):
            # Plain loads: the invariant (ld.global.nc) loads of an immutable tensor made
            # SE3 inv and mul 19-25% slower on the RTX 5090 (kernel time, n = 2^20).
            m.values[j] = rows.load[invariant=False]((i, j))[0]
        return m

    @inline(.always)
    def store(self, rows: MutTileTensor[Self.dtype, ...], i: Int):
        """Write row `i` of `rows`, whose width is exactly R*C."""
        comptime assert rows.flat_rank == 2, "expected a (batch, width) tensor"
        comptime assert rows.static_shape[1] == Self.R * Self.C, "row width must be R*C"
        comptime for j in range(Self.R * Self.C):
            rows[i, j] = self.values[j]

    @inline(.always)
    def store_padded(self, rows: MutTileTensor[Self.dtype, ...], i: Int):
        """Write the R*C values, then zeros to the end of row `i` (a tangent gradient in an embedded-size row)."""
        comptime assert rows.flat_rank == 2, "expected a (batch, width) tensor"
        comptime width = rows.static_shape[1]
        comptime assert width >= Self.R * Self.C, "row too narrow"
        comptime for j in range(Self.R * Self.C):
            rows[i, j] = self.values[j]
        comptime for j in range(Self.R * Self.C, width):
            rows[i, j] = 0

    @inline(.always)
    def __getitem__(self, i: Int, j: Int) -> Scalar[Self.dtype]:
        return self.values[i * Self.C + j]

    @inline(.always)
    def __setitem__(mut self, i: Int, j: Int, v: Scalar[Self.dtype]):
        self.values[i * Self.C + j] = v

    @inline(.always)
    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        return self.values[i]

    @inline(.always)
    def __setitem__(mut self, i: Int, v: Scalar[Self.dtype]):
        self.values[i] = v

    @inline(.always)
    def __matmul__[C2: Int](self, o: Mat[Self.dtype, Self.C, C2]) -> Mat[Self.dtype, Self.R, C2]:
        var r = Mat[Self.dtype, Self.R, C2]()
        comptime for i in range(Self.R):
            comptime for j in range(C2):
                var s: Scalar[Self.dtype] = 0
                comptime for k in range(Self.C):
                    s += self[i, k] * o[k, j]
                r[i, j] = s
        return r

    @inline(.always)
    def __add__(self, o: Self) -> Self:
        var r = Self()
        comptime for i in range(Self.R * Self.C):
            r.values[i] = self.values[i] + o.values[i]
        return r

    @inline(.always)
    def __sub__(self, o: Self) -> Self:
        var r = Self()
        comptime for i in range(Self.R * Self.C):
            r.values[i] = self.values[i] - o.values[i]
        return r

    @inline(.always)
    def __neg__(self) -> Self:
        var r = Self()
        comptime for i in range(Self.R * Self.C):
            r.values[i] = -self.values[i]
        return r

    @inline(.always)
    def __mul__(self, s: Scalar[Self.dtype]) -> Self:
        var r = Self()
        comptime for i in range(Self.R * Self.C):
            r.values[i] = self.values[i] * s
        return r

    @inline(.always)
    def __rmul__(self, s: Scalar[Self.dtype]) -> Self:
        var r = Self()
        comptime for i in range(Self.R * Self.C):
            r.values[i] = s * self.values[i]
        return r

    @inline(.always)
    def T(self) -> Mat[Self.dtype, Self.C, Self.R]:
        var r = Mat[Self.dtype, Self.C, Self.R]()
        comptime for i in range(Self.R):
            comptime for j in range(Self.C):
                r[j, i] = self[i, j]
        return r

    @inline(.always)
    def squared_norm(self) -> Scalar[Self.dtype]:
        var s: Scalar[Self.dtype] = 0
        comptime for i in range(Self.R * Self.C):
            s += self.values[i] * self.values[i]
        return s

    @inline(.always)
    def block[r0: Int, c0: Int, BR: Int, BC: Int](self) -> Mat[Self.dtype, BR, BC]:
        comptime assert r0 + BR <= Self.R and c0 + BC <= Self.C, "block out of range"
        var r = Mat[Self.dtype, BR, BC]()
        comptime for i in range(BR):
            comptime for j in range(BC):
                r[i, j] = self[r0 + i, c0 + j]
        return r

    @inline(.always)
    def set_block[BR: Int, BC: Int, //, r0: Int, c0: Int](mut self, b: Mat[Self.dtype, BR, BC]):
        comptime assert r0 + BR <= Self.R and c0 + BC <= Self.C, "block out of range"
        comptime for i in range(BR):
            comptime for j in range(BC):
                self[r0 + i, c0 + j] = b[i, j]
