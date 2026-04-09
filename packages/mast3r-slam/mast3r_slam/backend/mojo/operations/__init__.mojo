# MASt3R-SLAM Mojo GPU operations for CustomOpLibrary.
# This directory is loaded by max.experimental.torch.CustomOpLibrary
# and auto-compiled at runtime. No explicit mojo build step needed.
#
# Vec3 is defined here so it's available to all ops in the package
# (matching_ops.mojo, gn_ops.mojo) as part of the package namespace.


@fieldwise_init
struct Vec3(TrivialRegisterPassable):
    """3D vector stored as three Float32 scalars.

    TrivialRegisterPassable = zero overhead, lives entirely in GPU registers.
    """
    var x: Float32
    var y: Float32
    var z: Float32

    @always_inline
    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    @always_inline
    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    @always_inline
    def __mul__(self, s: Float32) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    @always_inline
    def dot(self, other: Vec3) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @always_inline
    def squared_norm(self) -> Float32:
        return self.x * self.x + self.y * self.y + self.z * self.z

    @always_inline
    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
