"""The `LieGroup` trait: what a group must provide so the generic ops can run.

The trait mirrors the member functions that lietorch's CUDA/CPU kernel
templates (`lietorch_gpu.cu`, `lietorch_cpu.cpp`) call on their `Group`
template parameter. A group implementation is a struct parameterised on the
scalar dtype, e.g. `SE3[DType.float32]`, and declares:

- `dtype` — the scalar type,
- `N` — the embedded dimension (length of one element's data row),
- `K` — the manifold (tangent) dimension.

`group_ops.mojo` writes every backend op once against this trait, so SO3,
RxSO3, SE3 and Sim3 drop in by implementing it. A group is pure register math:
`group_ops.mojo` moves the data rows, which use lietorch's layout exactly
(SE3: tx ty tz qx qy qz qw; scalar-last quaternion).
"""

from linalg import Mat


trait LieGroup(ImplicitlyCopyable, Deinitable):
    comptime dtype: DType
    comptime N: Int
    comptime K: Int

    @staticmethod
    def from_data(x: Mat[Self.dtype, Self.N, 1]) -> Self:
        """Construct from one data row (normalising the quaternion, as lietorch's constructors do)."""
        ...

    def data(self) -> Mat[Self.dtype, Self.N, 1]:
        """The N-value data row (`data()` in lietorch)."""
        ...

    @staticmethod
    def Exp(a: Mat[Self.dtype, Self.K, 1]) -> Self:
        ...

    def Log(self) -> Mat[Self.dtype, Self.K, 1]:
        ...

    def inv(self) -> Self:
        ...

    def mul(self, other: Self) -> Self:
        """Group product `self * other`."""
        ...

    def act(self, p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, 1]:
        """Action on a 3D point (`operator*(Point)`)."""
        ...

    def act4(self, p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, 1]:
        """Action on a homogeneous point."""
        ...

    def Adj(self) -> Mat[Self.dtype, Self.K, Self.K]:
        ...

    def Matrix4x4(self) -> Mat[Self.dtype, 4, 4]:
        ...

    def orthogonal_projector(self) -> Mat[Self.dtype, Self.N, Self.N]:
        ...

    @staticmethod
    def adj(a: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        """Lie bracket matrix (small adjoint)."""
        ...

    @staticmethod
    def left_jacobian(a: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        ...

    @staticmethod
    def left_jacobian_inverse(a: Mat[Self.dtype, Self.K, 1]) -> Mat[Self.dtype, Self.K, Self.K]:
        ...

    @staticmethod
    def act_jacobian(p: Mat[Self.dtype, 3, 1]) -> Mat[Self.dtype, 3, Self.K]:
        ...

    @staticmethod
    def act4_jacobian(p: Mat[Self.dtype, 4, 1]) -> Mat[Self.dtype, 4, Self.K]:
        ...
