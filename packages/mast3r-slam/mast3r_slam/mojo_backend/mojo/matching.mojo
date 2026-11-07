"""MASt3R-SLAM's matching kernels (paper §3.1–3.2; `matching_kernels.cu`): one thread per query point.

- `iter_proj_kernel`: Levenberg-Marquardt over the pixel (u, v) of image i whose interpolated ray best matches
  the unit direction of a point of image j. rays (B, H, W, 9) holds per pixel the ray and its derivatives in u
  and v; pts (B, N, 3), p_init / p_new (B, N, 2) float32, converged (B, N) bool.
- `refine_matches_kernel`: coarse-to-fine search around p1 (B, N, 2) int64 for the pixel of D11 (B, H, W, F)
  float16 whose descriptor has the largest dot product with the query's D21 (B, N, F) float16. The score is
  computed as the CUDA kernel's `scalar_t` (c10::Half) arithmetic does: every product and every partial sum is
  rounded to half, in descriptor order, and the best score starts at the smallest positive normal half, so a
  candidate must score above it to move the match. This keeps the CUDA kernel's choice between near-tied
  candidates (float32 sums pick another pixel for 3 % of the points of a real frame).

Grid (ceil(N / BLOCK), B), BLOCK threads, as the CUDA kernels.
"""

from std.math import clamp, floor, sqrt

from max.gpu import block_idx, global_idx

from gn import F32Ptr, I64Ptr, U8Ptr, Vec3

comptime BLOCK = 16
comptime F16Ptr = MutPointer[Float16, MutUntrackedOrigin]
comptime DESC_WIDTH = 8
comptime DESC_ALIGN = 16


struct _Bilinear(TrivialRegisterPassable):
    """The 4 pixels around (u, v) in a (H, W, 9) ray image and their weights; the pixels are opposite the area
    weights, as in the CUDA kernel."""

    var p11: F32Ptr
    var p12: F32Ptr
    var p21: F32Ptr
    var p22: F32Ptr
    var w: SIMD[DType.float32, 4]

    @always_inline
    def __init__(out self, img: F32Ptr, width: Int, u: Float32, v: Float32):
        var u11 = Int(floor(u))
        var v11 = Int(floor(v))
        var du = u - Float32(u11)
        var dv = v - Float32(v11)
        self.p22 = img.unsafe_offset((v11 * width + u11) * 9)
        self.p21 = self.p22.unsafe_offset(9)
        self.p12 = self.p22.unsafe_offset(width * 9)
        self.p11 = self.p12.unsafe_offset(9)
        self.w = SIMD[DType.float32, 4](du * dv, (1.0 - du) * dv, du * (1.0 - dv), (1.0 - du) * (1.0 - dv))

    @always_inline
    def vec3[c: Int](self) -> Vec3:
        """Channels c .. c + 2, interpolated."""
        return Vec3(self.channel[c](), self.channel[c + 1](), self.channel[c + 2]())

    @always_inline
    def channel[c: Int](self) -> Float32:
        return self.w[0] * self.p11[unsafe_offset=c] + self.w[1] * self.p12[unsafe_offset=c] + self.w[2] * self.p21[unsafe_offset=c] + self.w[3] * self.p22[unsafe_offset=c]


def iter_proj_kernel(
    num_points: Int64, height: Int64, width: Int64, max_iter: Int64, lambda_init: Float32, cost_thresh: Float32,
    rays: F32Ptr, pts: F32Ptr, p_init: F32Ptr, p_new: F32Ptr, converged: U8Ptr,
):  # fmt: skip
    var n = global_idx.x
    var b = block_idx.y
    var num = Int(num_points)
    if n >= num:
        return
    var w = Int(width)
    var img = rays.unsafe_offset(b * Int(height) * w * 9)
    var row = b * num + n
    var u_max = Float32(width - 2)
    var v_max = Float32(height - 2)
    var u = clamp(p_init[unsafe_offset=2 * row], 1.0, u_max)
    var v = clamp(p_init[unsafe_offset=2 * row + 1], 1.0, v_max)
    var target = Vec3(pts[unsafe_offset=3 * row], pts[unsafe_offset=3 * row + 1], pts[unsafe_offset=3 * row + 2])
    var lam = lambda_init
    var conv: UInt8 = 0

    for _ in range(Int(max_iter)):
        var px = _Bilinear(img, w, u, v)
        var r = px.vec3[0]()
        var gx = px.vec3[3]()
        var gy = px.vec3[6]()
        var err = r * (1.0 / sqrt(r.squared_norm())) - target
        var cost = err.squared_norm()

        var a00 = gx.squared_norm() + lam
        var a01 = gx.dot(gy)
        var a11 = gy.squared_norm() + lam
        var b0 = -err.dot(gx)
        var b1 = -err.dot(gy)
        var det_inv: Float32 = 1.0 / (a00 * a11 - a01 * a01)
        var u_new = clamp(u + det_inv * (a11 * b0 - a01 * b1), 1.0, u_max)
        var v_new = clamp(v + det_inv * (-a01 * b0 + a00 * b1), 1.0, v_max)
        var rn = _Bilinear(img, w, u_new, v_new).vec3[0]()
        var new_cost = (rn * (1.0 / sqrt(rn.squared_norm())) - target).squared_norm()

        # lambda is scaled in double precision, as the CUDA kernel's `lambda *= 0.1` does.
        if new_cost < cost:
            u = u_new
            v = v_new
            lam = Float32(Float64(lam) * 0.1)
            conv = 1 if new_cost < cost_thresh else 0
        else:
            lam = Float32(Float64(lam) * 10.0)
            conv = 1 if cost < cost_thresh else 0

    p_new[unsafe_offset=2 * row] = u
    p_new[unsafe_offset=2 * row + 1] = v
    converged[unsafe_offset=row] = conv


@always_inline
def _half_add(a: Float16, b: Float16) -> Float16:
    """c10::Half's `a + b`: the float sum (exact for two halves) rounded to half."""
    return (Float32(a) + Float32(b)).cast[DType.float16]()


def refine_matches_kernel[fdim: Int](
    num_points: Int64, height: Int64, width: Int64, feat_dim: Int64, radius: Int64, dilation_max: Int64,
    d11: F16Ptr, d21: F16Ptr, p1: I64Ptr, p_new: I64Ptr,
):  # fmt: skip
    """fdim > 0: the descriptor width at compile time (the query stays in registers; both descriptor arrays must be
    16-byte aligned); fdim = 0: `feat_dim` at run time."""
    var n = global_idx.x
    var b = block_idx.y
    var num = Int(num_points)
    if n >= num:
        return
    var h = Int(height)
    var w = Int(width)
    var f = fdim if fdim > 0 else Int(feat_dim)
    var row = b * num + n
    var query = d21.unsafe_offset(row * f)
    var image = d11.unsafe_offset(b * h * w * f)
    comptime chunks = fdim // DESC_WIDTH
    var q = Array[SIMD[DType.float32, DESC_WIDTH], max(chunks, 1)](fill=0)
    comptime if fdim > 0:
        comptime assert fdim % DESC_WIDTH == 0, "fdim must be a multiple of 8"
        comptime for c in range(chunks):
            q[c] = query.unsafe_load[width=DESC_WIDTH, alignment=DESC_ALIGN](c * DESC_WIDTH).cast[DType.float32]()
    # c10::Half's numeric_limits::min(): the smallest positive normal half.
    var best = Float32(6.103515625e-05)

    var u0 = Int(p1[unsafe_offset=2 * row])
    var v0 = Int(p1[unsafe_offset=2 * row + 1])
    var u_best = u0
    var v_best = v0
    var r = Int(radius)
    var d = Int(dilation_max)
    while d > 0:
        var rd = r * d
        for i in range(0, 2 * rd + 1, d):
            for j in range(0, 2 * rd + 1, d):
                var u = u0 - rd + i
                var v = v0 - rd + j
                if v < 0 or v >= h or u < 0 or u >= w:
                    continue
                var cand = image.unsafe_offset((v * w + u) * f)
                var score: Float16 = 0.0
                comptime if fdim > 0:
                    comptime for c in range(chunks):
                        var x = cand.unsafe_load[width=DESC_WIDTH, alignment=DESC_ALIGN](c * DESC_WIDTH).cast[DType.float32]()
                        var prod = (q[c] * x).cast[DType.float16]()
                        comptime for k in range(DESC_WIDTH):
                            score = _half_add(score, prod[k])
                else:
                    for k in range(f):
                        score = _half_add(score, (Float32(query[unsafe_offset=k]) * Float32(cand[unsafe_offset=k])).cast[DType.float16]())
                if Float32(score) > best:
                    best = Float32(score)
                    u_best = u
                    v_best = v
        u0 = u_best
        v0 = v_best
        d -= 1
    p_new[unsafe_offset=2 * row] = Int64(u_best)
    p_new[unsafe_offset=2 * row + 1] = Int64(v_best)
