/*
 * ba_cuda.cu -- CUDA implementation of bundle adjustment and reprojection
 *               for the DPVO visual odometry pipeline.
 *
 * Part of DPVO (Deep Patch Visual Odometry, Teed et al., 2022).
 *
 * This file contains the core Gauss-Newton bundle adjustment solver that
 * jointly optimizes camera poses (SE3) and patch inverse-depths.  It also
 * provides a reprojection kernel used to update target coordinates after
 * pose/depth changes.
 *
 * ========================================================================
 * SE3 CONVENTIONS
 * ========================================================================
 *
 * Poses are stored as 7-vectors: [tx, ty, tz, qx, qy, qz, qw]
 *   - (tx, ty, tz) = translation component
 *   - (qx, qy, qz, qw) = unit quaternion (Hamilton convention, scalar-last)
 *
 * The relative transform from frame i to frame j is:
 *   Gᵢⱼ = Tⱼ · Tᵢ⁻¹
 * which maps points from frame i's coordinate system into frame j's.
 *
 * Quaternion layout in memory (arrays of length 4):
 *   q[0] = qx,  q[1] = qy,  q[2] = qz,  q[3] = qw
 *
 * ========================================================================
 * BUNDLE ADJUSTMENT OVERVIEW
 * ========================================================================
 *
 * The system minimizes a weighted reprojection error:
 *
 *   E = Σₙ wₙ · ||targetₙ - π(Gᵢⱼ · backproject(patch_kk[center]))||²
 *
 * where n iterates over edges (ii[n], jj[n], kk[n]) in the factor graph,
 * pi() is the pinhole projection, and backproject lifts the patch center
 * to 3D using its inverse depth.
 *
 * Variables: poses[t0..t1) as SE3 Lie algebra updates (6-DOF each),
 *            inverse-depths for each unique patch.
 *
 * The Gauss-Newton normal equations are:
 *
 *   [ B   E ] [dX]   [v]
 *   [ Eᵀ  C ] [dZ] = [u]
 *
 * where:
 *   B  = pose-pose Hessian block       (6N × 6N, N = t1 - t0 active poses)
 *   E  = pose-depth cross-term         (6N × M,  M = number of unique patches)
 *   C  = depth-depth diagonal block    (M × 1, diagonal so stored as vector)
 *   v  = pose RHS                      (6N × 1)
 *   u  = depth RHS                     (M × 1)
 *   dX = pose updates                  (6N × 1, in Lie algebra se(3))
 *   dZ = inverse-depth updates         (M × 1)
 *
 * The Schur complement eliminates the depth variables:
 *   Q = 1 / (C + λ)                    (element-wise, M × 1)
 *   S = B - E · Q · Eᵀ                 (6N × 6N reduced system)
 *   y = v - E · Q · u
 *   Solve S · dX = y via Cholesky
 *   Then dZ = Q · (u - Eᵀ · dX)        (back-substitution for depths)
 *
 * After solving, poses and patches are retracted on their respective
 * manifolds (SE3 for poses, positive reals for inverse depth).
 *
 * ========================================================================
 * FILE ORGANIZATION
 * ========================================================================
 *
 * Device helper functions (Lie group operations):
 *   actSO3()    -- rotate a point by a quaternion
 *   actSE3()    -- apply an SE3 transform to a homogeneous point
 *   adjSE3()    -- adjoint representation of SE3 (maps twists)
 *   relSE3()    -- relative transform Tᵢⱼ = Tⱼ · Tᵢ⁻¹
 *   expSO3()    -- axis-angle to quaternion (SO3 exponential map)
 *   expSE3()    -- Lie algebra to (t, q) (SE3 exponential map)
 *   retrSE3()   -- retraction: apply a Lie algebra update to a pose
 *
 * CUDA kernels:
 *   pose_retr_kernel                      -- per-pose retraction
 *   patch_retr_kernel                     -- per-patch inverse-depth update
 *   reprojection_residuals_and_hessian    -- THE MAIN BA KERNEL
 *   reproject                             -- per-edge patch reprojection
 *
 * Host functions:
 *   cuda_ba()        -- orchestrates the full BA loop
 *   cuda_reproject() -- batch reprojection
 */

#include <torch/extension.h>
#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>


/*
 * Grid-stride loop macro: each thread processes elements starting at its
 * global index and stepping by the total number of threads.  This is the
 * standard CUDA pattern for handling workloads larger than the grid size.
 */
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += blockDim.x * gridDim.x)


#define NUM_THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + NUM_THREADS - 1) / NUM_THREADS)


/* =========================================================================
 * LIE GROUP HELPER FUNCTIONS (device-only)
 * =========================================================================
 * These small functions implement the core SE3/SO3 operations needed by
 * the BA kernel.  They operate on raw float arrays (not tensors) for
 * maximum performance in device code.
 * ======================================================================= */

/**
 * Rotate a 3D point X by quaternion q, storing the result in Y.
 *
 * Uses the efficient Rodrigues-like formula that avoids building the full
 * 3×3 rotation matrix:
 *
 *   uv = 2 · (q_xyz × X)                   -- cross product, scaled by 2
 *   Y  = X + q_w · uv + (q_xyz × uv)       -- add rotation correction
 *
 * This is mathematically equivalent to R(q) · X where R(q) is the 3×3
 * rotation matrix corresponding to quaternion q, but uses fewer operations.
 *
 * @param q  Quaternion [qx, qy, qz, qw] (unit quaternion, scalar-last).
 * @param X  Input 3D point [x, y, z].
 * @param Y  Output rotated point [x', y', z'].
 */
__device__ void
actSO3(const float *q, const float *X, float *Y) {
  // uv = 2 * (q_xyz x X)
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  // Y = X + q_w * uv + (q_xyz x uv)
  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

/**
 * Apply an SE3 rigid-body transform to a homogeneous 4D point.
 *
 * Given a pose (t, q) and a point X = [x, y, z, w] in homogeneous
 * coordinates, compute:
 *   Y = [R(q) · [x, y, z] + w · t,  w]
 *
 * The homogeneous coordinate w acts as a scaling factor on the translation.
 * For a standard 3D point, w = inverse_depth (from DPVO's patch
 * representation), so the translation is scaled by the inverse depth.
 *
 * @param t  Translation [tx, ty, tz].
 * @param q  Quaternion [qx, qy, qz, qw].
 * @param X  Input homogeneous point [x, y, z, w].
 * @param Y  Output transformed point [x', y', z', w'].
 */
__device__  void
actSE3(const float *t, const float *q, const float *X, float *Y) {
  // First rotate the spatial part [x, y, z]
  actSO3(q, X, Y);
  // Preserve the homogeneous coordinate
  Y[3] = X[3];
  // Add translation scaled by homogeneous weight
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

/**
 * Compute the adjoint representation of the INVERSE SE3 transform.
 *
 * The adjoint maps a 6D twist (velocity) from one frame to another.
 * Given a pose (t, q) representing Tij, this computes:
 *
 *   Y = Ad(Tᵢⱼ⁻¹) · X
 *
 * where X = [v; omega] is a 6D twist (first 3 = translational, last 3 =
 * rotational), and:
 *
 *   Ad(T⁻¹) = [ Rᵀ       0   ]
 *             [ -Rᵀ[t]×  Rᵀ ]
 *
 * Here Rᵀ is the inverse rotation (`q_inv`), and [t]× is the skew-symmetric
 * matrix of t.  The computation proceeds as:
 *   Y[0:3] = Rᵀ · X[0:3]                         (rotate translational part)
 *   Y[3:6] = Rᵀ · X[3:6] + Rᵀ · (t × X[0:3])    (rotate + cross-couple)
 *
 * In the BA context, this is used to transform the Jacobian w.r.t. pose j
 * into the Jacobian w.r.t. pose i:  Jᵢ = -Adj(Gᵢⱼ)ᵀ · Jⱼ.
 * (Note: actually Ji = adjSE3(Gij, Jj) with the specific sign convention
 * used in the BA kernel.)
 *
 * @param t  Translation [tx, ty, tz] of the SE3 transform.
 * @param q  Quaternion [qx, qy, qz, qw] of the SE3 transform.
 * @param X  Input 6D twist [v0, v1, v2, w0, w1, w2].
 * @param Y  Output 6D twist after adjoint transformation.
 */
__device__ void
adjSE3(const float *t, const float *q, const float *X, float *Y) {
  // Conjugate quaternion = inverse rotation (for unit quaternions)
  float qinv[4] = {-q[0], -q[1], -q[2], q[3]};

  // Rotate the translational part: Y[0:3] = Rᵀ · X[0:3]
  actSO3(qinv, &X[0], &Y[0]);
  // Rotate the rotational part: Y[3:6] = Rᵀ · X[3:6]
  actSO3(qinv, &X[3], &Y[3]);

  // Compute u = t × X[0:3]  (cross product of translation with input's translational part)
  float u[3], v[3];
  u[0] = t[2]*X[1] - t[1]*X[2];
  u[1] = t[0]*X[2] - t[2]*X[0];
  u[2] = t[1]*X[0] - t[0]*X[1];

  // Rotate the cross-coupling term: v = Rᵀ · u
  actSO3(qinv, u, v);

  // Add cross-coupling to the rotational output part
  Y[3] += v[0];
  Y[4] += v[1];
  Y[5] += v[2];
}

/**
 * Compute the relative SE3 transform: Tᵢⱼ = Tⱼ · Tᵢ⁻¹.
 *
 * This gives the transform that takes points from frame i to frame j.
 * The computation is:
 *   qᵢⱼ = qⱼ · qᵢ⁻¹          (quaternion multiplication with qᵢ conjugated)
 *   tᵢⱼ = tⱼ - R(qᵢⱼ) · tᵢ   (rotate tᵢ by the relative rotation, subtract)
 *
 * The quaternion product qj * qi^{-1} is expanded using Hamilton's formula
 * with qi^{-1} = (-qx_i, -qy_i, -qz_i, qw_i) for a unit quaternion.
 *
 * @param ti   Translation of pose i [tx, ty, tz].
 * @param qi   Quaternion of pose i [qx, qy, qz, qw].
 * @param tj   Translation of pose j [tx, ty, tz].
 * @param qj   Quaternion of pose j [qx, qy, qz, qw].
 * @param tij  Output relative translation [tx, ty, tz].
 * @param qij  Output relative quaternion [qx, qy, qz, qw].
 */
__device__ void
relSE3(const float *ti, const float *qi, const float *tj, const float *qj, float *tij, float *qij) {
  /*
   * Quaternion multiplication: qᵢⱼ = qⱼ · qᵢ⁻¹
   * With qᵢ⁻¹ = (-qi[0], -qi[1], -qi[2], qi[3]) for a unit quaternion.
   *
   * Hamilton product (scalar-last convention):
   *   qij.x = -qj.w*qi.x + qj.x*qi.w - qj.y*qi.z + qj.z*qi.y
   *   qij.y = -qj.w*qi.y + qj.y*qi.w - qj.z*qi.x + qj.x*qi.z
   *   qij.z = -qj.w*qi.z + qj.z*qi.w - qj.x*qi.y + qj.y*qi.x
   *   qij.w =  qj.w*qi.w + qj.x*qi.x + qj.y*qi.y + qj.z*qi.z
   */
  qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1],
  qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2],
  qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0],
  qij[3] =  qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2],

  // tij = tj - R(qij) * ti
  // First compute R(qij) * ti, then subtract from tj
  actSO3(qij, ti, tij);
  tij[0] = tj[0] - tij[0];
  tij[1] = tj[1] - tij[1];
  tij[2] = tj[2] - tij[2];
}


/**
 * SO3 exponential map: convert an axis-angle vector φ to a unit quaternion q.
 *
 * Given φ = θ · axis (where θ = ||φ|| is the rotation angle
 * and axis is the unit rotation axis), the quaternion is:
 *
 *   q = [sin(θ/2) / θ · φ,  cos(θ/2)]
 *     = [imag · φ,  real]
 *
 * For small angles (θ² < 1e-8), a Taylor expansion is used to avoid
 * numerical instability in sin(θ/2)/θ:
 *   imag ≈ 0.5 - θ²/48 + θ⁴/3840
 *   real ≈ 1.0 - θ²/8 + θ⁴/384
 *
 * @param phi  Axis-angle rotation vector [φₓ, φᵧ, φ_z].
 * @param q    Output unit quaternion [qx, qy, qz, qw].
 */
__device__ void
expSO3(const float *phi, float* q) {
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta_p4 = theta_sq * theta_sq;

  float theta = sqrtf(theta_sq);
  float imag, real;

  if (theta_sq < 1e-8) {
    // Taylor expansion for small angles to avoid division by near-zero
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

/**
 * Compute the cross product a × b and store the result back into b (in-place).
 *
 * @param a  First vector [a0, a1, a2] (read-only).
 * @param b  Second vector [b0, b1, b2]; overwritten with a x b on return.
 */
__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

/**
 * SE3 exponential map: convert a Lie algebra element ξ to a pose (t, q).
 *
 * The Lie algebra of SE3 is se(3), represented as a 6-vector:
 *   ξ = [v0, v1, v2, w0, w1, w2]
 * where v = ξ[0:3] is the translational part and w = ξ[3:6] is the
 * rotational part (axis-angle).
 *
 * The exponential map computes:
 *   q = exp_SO3(w)           -- rotation quaternion
 *   t = J(w) · v             -- translation via the left Jacobian
 *
 * where J(w) is the left Jacobian of SO3:
 *   J(w) = I + a · [w]× + b · [w]×²
 * with:
 *   a = (1 - cos(θ)) / θ²
 *   b = (θ - sin(θ)) / θ³
 *
 * The translation computation uses two successive cross products to build
 * J(w) · v without explicitly forming the 3×3 matrix:
 *   tau = v
 *   tau += a · (w × tau)
 *   tau += b · (w × (w × v))   [the second cross is applied to the already-crossed result]
 *
 * For small angles (θ < 1e-4), the first-order approximation J ≈ I
 * is used (i.e., t ≈ v).
 *
 * @param xi  6D Lie algebra element [v0, v1, v2, w0, w1, w2].
 * @param t   Output translation [tx, ty, tz].
 * @param q   Output quaternion [qx, qy, qz, qw].
 */
__device__ void
expSE3(const float *xi, float* t, float* q) {
  // Compute the rotation quaternion from the rotational part xi[3:6]
  expSO3(xi + 3, q);

  float tau[3] = {xi[0], xi[1], xi[2]};   // translational part (copy)
  float phi[3] = {xi[3], xi[4], xi[5]};   // rotational part (axis-angle)

  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  // Start with t = v (identity Jacobian)
  t[0] = tau[0];
  t[1] = tau[1];
  t[2] = tau[2];

  if (theta > 1e-4) {
    // a = (1 - cos(θ)) / θ²
    float a = (1 - cosf(theta)) / theta_sq;
    // tau = φ × tau (in-place), then t += a · (φ × v)
    crossInplace(phi, tau);
    t[0] += a * tau[0];
    t[1] += a * tau[1];
    t[2] += a * tau[2];

    // b = (θ - sin(θ)) / θ³
    float b = (theta - sinf(theta)) / (theta * theta_sq);
    // tau = φ × tau (in-place again), so tau is now φ × (φ × v)
    crossInplace(phi, tau);
    t[0] += b * tau[0];
    t[1] += b * tau[1];
    t[2] += b * tau[2];
  }
  // For theta < 1e-4, the left Jacobian is approximately I, so t ~ v
}


/**
 * Retraction on the SE3 manifold: apply a small Lie algebra update to an
 * existing pose.
 *
 * Computes:
 *   (t1, q1) = Exp(ξ) · (t, q)
 *
 * This is the standard left-multiplication retraction used in Lie-group
 * optimization.  The update xi is first converted to a small SE3 transform
 * (dt, dq) via the exponential map, then composed with the current pose
 * via quaternion multiplication and rotation of the translation.
 *
 * Quaternion composition: q1 = dq · q  (Hamilton product, scalar-last).
 * Translation: t1 = R(dq) · t + dt.
 *
 * @param xi  6D Lie algebra update [v0, v1, v2, w0, w1, w2].
 * @param t   Current translation [tx, ty, tz].
 * @param q   Current quaternion [qx, qy, qz, qw].
 * @param t1  Output updated translation [tx', ty', tz'].
 * @param q1  Output updated quaternion [qx', qy', qz', qw'].
 */
__device__ void
retrSE3(const float *xi, const float* t, const float* q, float* t1, float* q1) {
  // Convert the Lie algebra update to an incremental SE3 transform
  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};   // identity quaternion

  expSE3(xi, dt, dq);

  // Quaternion composition: q1 = dq * q (Hamilton product, scalar-last)
  q1[0] = dq[3] * q[0] + dq[0] * q[3] + dq[1] * q[2] - dq[2] * q[1];
  q1[1] = dq[3] * q[1] + dq[1] * q[3] + dq[2] * q[0] - dq[0] * q[2];
  q1[2] = dq[3] * q[2] + dq[2] * q[3] + dq[0] * q[1] - dq[1] * q[0];
  q1[3] = dq[3] * q[3] - dq[0] * q[0] - dq[1] * q[1] - dq[2] * q[2];

  // Translation: t1 = R(dq) * t + dt
  actSO3(dq, t, t1);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
}



/* =========================================================================
 * RETRACTION KERNELS
 * =========================================================================
 * After solving the Gauss-Newton linear system, these kernels apply the
 * updates to poses and inverse-depths respectively.
 * ======================================================================= */

/**
 * Per-pose retraction kernel: applies a 6-DOF Lie algebra update to each
 * active camera pose.
 *
 * Each thread handles one pose.  The update vector update[i] contains the
 * 6D twist (v, ω) that is applied via left-multiplication:
 *   pose[t0 + i] ← Exp(update[i]) · pose[t0 + i]
 *
 * @param t0      Start of the active pose window (global pose index).
 * @param t1      End of the active pose window (exclusive).
 * @param poses   All camera poses [N_total, 7], modified in-place.
 * @param update  Pose updates [N_active, 6] where N_active = t1 - t0.
 */
__global__ void pose_retr_kernel(const int t0, const int t1,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(i, t1 - t0) {
    const float t = t0 + i;  // global pose index

    // Read current pose: translation t0[3], quaternion q0[4]
    float t1[3], t0[3] = { poses[t][0], poses[t][1], poses[t][2] };
    float q1[4], q0[4] = { poses[t][3], poses[t][4], poses[t][5], poses[t][6] };

    // Read the 6-DOF Lie algebra update for this pose
    float xi[6] = {
      update[i][0],
      update[i][1],
      update[i][2],
      update[i][3],
      update[i][4],
      update[i][5],
    };

    // Apply retraction: (t1, q1) = Exp(xi) * (t0, q0)
    retrSE3(xi, t0, q0, t1, q1);

    // Write updated pose back to global memory
    poses[t][0] = t1[0];
    poses[t][1] = t1[1];
    poses[t][2] = t1[2];
    poses[t][3] = q1[0];
    poses[t][4] = q1[1];
    poses[t][5] = q1[2];
    poses[t][6] = q1[3];
  }
}


/**
 * Per-patch inverse-depth retraction kernel: applies a scalar update to
 * each patch's inverse depth.
 *
 * The inverse depth is stored in patches[kx][2][i][j] for all (i,j) in the
 * patch grid.  Since DPVO assumes a fronto-parallel patch, all pixels
 * share the same depth, so we read from [0][0], update, clamp, and write
 * the result to all grid positions.
 *
 * Clamping:
 *   - If d > 20 after the update, reset to 1.0 (safety against divergence).
 *   - d is clamped to a minimum of 1e-4 (no zero/negative depth).
 *
 * @param index   Mapping from sequential unique-patch index to the global
 *                patch index kx, shape [M].
 * @param patches All patch data [N_patches, 3, P, P].  Channel 2 is inverse
 *                depth.  Modified in-place.
 * @param update  Inverse-depth updates [M], one per unique patch.
 */
__global__ void patch_retr_kernel(
    torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> index,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> update)
{
  GPU_1D_KERNEL_LOOP(n, index.size(0)) {
    const int p = patches.size(2);   // patch grid dimension (e.g. 3)
    const int ix = index[n];         // global patch index

    // Read current inverse depth from the patch center
    float d = patches[ix][2][0][0];
    d = d + update[n];

    // Clamp: if depth exploded beyond 20, reset to 1.0
    d = (d > 20) ? 1.0 : d;
    // Clamp to minimum positive value to prevent degenerate geometry
    d = max(d, 1e-4);

    // Write the uniform inverse depth to all positions in the patch grid
    for (int i=0; i<p; i++) {
      for (int j=0; j<p; j++) {
        patches[ix][2][i][j] = d;
      }
    }
  }
}


/* =========================================================================
 * MAIN BUNDLE ADJUSTMENT KERNEL
 * =========================================================================
 * This is the central kernel that computes reprojection residuals and
 * accumulates the Gauss-Newton Hessian blocks for every edge in the
 * factor graph.  It runs once per Gauss-Newton iteration (the host
 * function zeros the accumulators and re-launches for each iteration).
 * ======================================================================= */

/**
 * Compute reprojection residuals and accumulate the Gauss-Newton Hessian.
 *
 * For each edge n in the factor graph (connecting source frame ii[n] to
 * target frame jj[n] via patch kk[n]), this kernel:
 *
 *   1. Computes the relative pose: Gᵢⱼ = Tⱼⱼ · Tᵢᵢ⁻¹
 *
 *   2. Back-projects the patch center to 3D in frame ii's coordinates:
 *        Xi = [(px - cx)/fx,  (py - cy)/fy,  1,  d]
 *      where (px, py) = patches[kk][0:2][1][1] (center pixel) and
 *      d = patches[kk][2][1][1] (inverse depth).  The 4th component
 *      stores the inverse depth (homogeneous representation).
 *
 *   3. Transforms to the target frame:  Xⱼ = Gᵢⱼ · Xᵢ
 *
 *   4. Projects to 2D:
 *        x₁ = fₓ · Xⱼ.x / Xⱼ.z + cₓ
 *        y₁ = fᵧ · Xⱼ.y / Xⱼ.z + cᵧ
 *
 *   5. Computes the 2D reprojection residual:
 *        r = target[n] - (x₁, y₁)
 *
 *   6. Computes analytical Jacobians (derivatives of the projected point
 *      w.r.t. the optimization variables):
 *
 *      Jⱼ = d(x₁, y₁)/d(ξⱼ)   -- 2×6 Jacobian w.r.t. target pose (Lie algebra)
 *      Jᵢ = -Adj(Gᵢⱼ)ᵀ · Jⱼ   -- 2×6 Jacobian w.r.t. source pose (via adjoint)
 *      Jz = d(x₁, y₁)/d(d)    -- 2×1 Jacobian w.r.t. inverse depth
 *
 *      The Jacobian Jⱼ for the x-component is derived from the pinhole
 *      projection model and SE3 action:
 *        Jⱼₓ = [fₓ·W/Z, 0, -fₓ·X·W/Z², -fₓ·X·Y/Z², fₓ·(1+X²/Z²), -fₓ·Y/Z]
 *      where X, Y, Z, W are the components of Xⱼ.
 *      (Similarly for the y-component with fᵧ.)
 *
 *   7. Accumulates into the Hessian blocks using atomicAdd:
 *        B  += Jᵀ · W · J            (pose-pose block, 6×6 per pose pair)
 *        E  += J_poseᵀ · W · Jz      (pose-depth cross term)
 *        C  += Jzᵀ · W · Jz          (depth-depth diagonal)
 *        v  += J_poseᵀ · W · r       (pose RHS)
 *        u  += Jzᵀ · W · r           (depth RHS)
 *
 *      The x and y components of the residual are processed in separate
 *      blocks with their own Jacobians and weights.
 *
 * Outlier rejection:
 *   An edge is masked out (weight set to 0) if:
 *   - The reprojection error exceeds 128 pixels
 *   - The depth Z is too small (< 0.2, i.e., behind or very close to camera)
 *   - The projected point is far outside the image (> 64px beyond borders)
 *
 * Thread indexing:
 *   Uses the GPU_1D_KERNEL_LOOP grid-stride pattern, so each thread
 *   processes one or more edges.
 *
 * Shared memory:
 *   Camera intrinsics (fx, fy, cx, cy) are loaded once per block into
 *   __shared__ memory by thread 0, then broadcast to all threads via
 *   __syncthreads().
 *
 * Sign conventions for the Hessian accumulation:
 *   Because Ji comes from the adjoint (which negates), the B matrix
 *   cross-terms between pose i and pose j carry a minus sign:
 *     B[ix, jx] -= w · Jᵢ · Jⱼᵀ
 *     B[jx, ix] -= w · Jⱼ · Jᵢᵀ
 *   Similarly, E[ix, k] accumulates -w · Jz · Jᵢ and E[jx, k] accumulates +w · Jz · Jⱼ.
 *   And v[ix] accumulates -w · r · Jᵢ, v[jx] accumulates +w · r · Jⱼ.
 *
 * @param poses       Camera poses [N_total, 7].
 * @param patches     Patch data [N_patches, 3, P, P].
 * @param intrinsics  Camera intrinsics [1, 4] as (fx, fy, cx, cy).
 * @param target      Target 2D positions per edge [N_edges, 2].
 * @param weight      Per-edge confidence weights [N_edges, 2].
 * @param lmbda       Levenberg-Marquardt damping (unused in kernel, but passed through).
 * @param ii          Source frame indices [N_edges].
 * @param jj          Target frame indices [N_edges].
 * @param kk          Global patch indices [N_edges].
 * @param ku          Unique-patch mapping: ku[n] = index into the M-length
 *                    unique-patch array for edge n.
 * @param B           Pose-pose Hessian block [6N, 6N], accumulated via atomicAdd.
 * @param E           Pose-depth cross-term [6N, M], accumulated via atomicAdd.
 * @param C           Depth-depth diagonal [M], accumulated via atomicAdd.
 * @param v           Pose RHS [6N], accumulated via atomicAdd.
 * @param u           Depth RHS [M], accumulated via atomicAdd.
 * @param t0          Start of the active pose window.  Pose indices are
 *                    shifted by -t0 so that B, E, v use zero-based indexing
 *                    for the active window.  Poses with index < t0 are
 *                    treated as fixed (ix < 0 after shifting).
 */
__global__ void reprojection_residuals_and_hessian(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> target,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> weight,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> lmbda,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ku,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> E,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> C,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> u, const int t0)
{

  /* Load camera intrinsics into shared memory (one load per block). */
  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
    cx = intrinsics[0][2];
    cy = intrinsics[0][3];
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(n, ii.size(0)) {
    int k = ku[n];   // index into the unique-patch array (for C, u, E columns)
    int ix = ii[n];  // source frame global index
    int jx = jj[n];  // target frame global index
    int kx = kk[n];  // global patch index (for reading patch data)

    /* --- Step 1: Load poses for source frame i and target frame j --- */
    float ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    float tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    float qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    float qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    /* --- Step 2: Back-project patch center to homogeneous 3D point --- */
    float Xi[4], Xj[4];
    // patches[kx][0][1][1] = x pixel coord of patch center
    // patches[kx][1][1][1] = y pixel coord of patch center
    // patches[kx][2][1][1] = inverse depth at patch center
    Xi[0] = (patches[kx][0][1][1] - cx) / fx;   // normalized x
    Xi[1] = (patches[kx][1][1][1] - cy) / fy;   // normalized y
    Xi[2] = 1.0;                                  // z = 1 (unit depth plane)
    Xi[3] = patches[kx][2][1][1];                // inverse depth (homogeneous w)

    /* --- Step 3: Compute relative transform and transform point --- */
    float tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);  // Gij = Tj * Ti^{-1}
    actSE3(tij, qij, Xi, Xj);          // Xⱼ = Gᵢⱼ · Xᵢ

    const float X = Xj[0];   // X coordinate in target frame
    const float Y = Xj[1];   // Y coordinate in target frame
    const float Z = Xj[2];   // Z coordinate in target frame (depth)
    const float W = Xj[3];   // homogeneous weight (= inverse depth)

    // Precompute 1/Z and 1/Z² for projection derivatives.
    // If Z < 0.2, set d = 0 to effectively zero out the Jacobians
    // (treats near/behind-camera points as degenerate).
    const float d = (Z >= 0.2) ? 1.0 / Z : 0.0;
    const float d2 = d * d;

    /* --- Step 4: Project to 2D in the target frame --- */
    const float x1 = fx * (X / Z) + cx;
    const float y1 = fy * (Y / Z) + cy;

    /* --- Step 5: Compute reprojection residual --- */
    const float rx = target[n][0] - x1;
    const float ry = target[n][1] - y1;

    /* --- Outlier rejection mask --- */
    // Reject if: residual too large, depth too small, or point too far outside image
    const bool in_bounds = (sqrt(rx*rx + ry*ry) < 128) && (Z > 0.2) &&
      (x1 > -64) && (y1 > -64) && (x1 < 2*cx + 64) && (y1 < 2*cy + 64);

    const float mask = in_bounds ? 1.0 : 0.0;

    // Shift pose indices to zero-based for the active window [t0, t1).
    // If ix < 0 after shifting, this pose is fixed (not optimized).
    ix = ix - t0;
    jx = jx - t0;

    /* --- Step 6-7: X-component residual and Jacobians --- */
    {
      const float r = target[n][0] - x1;         // x-residual
      const float w = mask * weight[n][0];        // x-weight (masked)

      /*
       * Jz: Jacobian of the x-projection w.r.t. inverse depth.
       *   x₁ = fₓ · X/Z + cₓ
       *   The inverse depth d_inv only affects X and Z through the
       *   translation part of Gᵢⱼ (tᵢⱼ · d_inv), so:
       *   dx₁/d(d_inv) = fₓ · (tᵢⱼₓ / Z - tᵢⱼ_z · X / Z²)
       */
      float Jz = fx * (tij[0] * d - tij[2] * (X * d2));

      /*
       * Jⱼ: 6D Jacobian of x₁ w.r.t. the Lie algebra update of pose j.
       *
       * The SE3 action on the point and the pinhole projection give:
       *   Jⱼ = [fₓ·W/Z, 0, -fₓ·X·W/Z², -fₓ·X·Y/Z², fₓ·(1+X²/Z²), -fₓ·Y/Z]
       *
       * Components correspond to: [d/dtₓ, d/dtᵧ, d/dt_z, d/dωₓ, d/dωᵧ, d/dω_z]
       * where (tₓ, tᵧ, t_z) are the translational and (ωₓ, ωᵧ, ω_z) the rotational
       * parts of the Lie algebra.
       */
      float Ji[6], Jj[6] = {fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};

      /*
       * Jᵢ: Jacobian w.r.t. source pose i, computed via the adjoint.
       *   Jᵢ = Adj(Gᵢⱼ⁻¹)ᵀ · Jⱼ  (with appropriate sign from the chain rule)
       * The adjSE3 function computes Y = Ad(Gᵢⱼ⁻¹) · X.
       */
      adjSE3(tij, qij, Jj, Ji);

      /* Accumulate into Hessian blocks (all via atomicAdd for thread safety) */
      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          // B[ii, ii] += w · Jᵢ · Jᵢᵀ  (source-source pose block)
          if (ix >= 0)
            atomicAdd(&B[6*ix+i][6*ix+j],  w * Ji[i] * Ji[j]);
          // B[jj, jj] += w · Jⱼ · Jⱼᵀ  (target-target pose block)
          if (jx >= 0)
            atomicAdd(&B[6*jx+i][6*jx+j],  w * Jj[i] * Jj[j]);
          // B[ii, jj] -= w · Jᵢ · Jⱼᵀ  (cross-block, negative from chain rule)
          // B[jj, ii] -= w · Jⱼ · Jᵢᵀ  (symmetric counterpart)
          if (ix >= 0 && jx >= 0) {
            atomicAdd(&B[6*ix+i][6*jx+j], -w * Ji[i] * Jj[j]);
            atomicAdd(&B[6*jx+i][6*ix+j], -w * Jj[i] * Ji[j]);
          }
        }
      }

      /* Pose-depth cross-term E */
      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&E[6*ix+i][k], -w * Jz * Ji[i]);
        if (jx >= 0)
          atomicAdd(&E[6*jx+i][k],  w * Jz * Jj[i]);
      }

      /* Pose RHS vector v */
      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&v[6*ix+i], -w * r * Ji[i]);
        if (jx >= 0)
          atomicAdd(&v[6*jx+i],  w * r * Jj[i]);
      }

      /* Depth-depth diagonal C and depth RHS u */
      atomicAdd(&C[k], w * Jz * Jz);
      atomicAdd(&u[k], w *  r * Jz);
    }

    /* --- Step 6-7 (repeated): Y-component residual and Jacobians --- */
    {
      const float r = target[n][1] - y1;         // y-residual
      const float w = mask * weight[n][1];        // y-weight (masked)

      /*
       * Jz for y: same structure as x, but with fᵧ and Y instead of fₓ and X.
       *   dy₁/d(d_inv) = fᵧ · (tᵢⱼᵧ / Z - tᵢⱼ_z · Y / Z²)
       */
      float Jz = fy * (tij[1] * d - tij[2] * (Y * d2));

      /*
       * Jⱼ for y-component:
       *   [0, fᵧ·W/Z, -fᵧ·Y·W/Z², -fᵧ·(1+Y²/Z²), fᵧ·X·Y/Z², fᵧ·X/Z]
       */
      float Ji[6], Jj[6] = {0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};

      adjSE3(tij, qij, Jj, Ji);

      /* Accumulate into Hessian blocks (same pattern as x-component) */
      for (int i=0; i<6; i++) {
        for (int j=0; j<6; j++) {
          if (ix >= 0)
            atomicAdd(&B[6*ix+i][6*ix+j],  w * Ji[i] * Ji[j]);
          if (jx >= 0)
            atomicAdd(&B[6*jx+i][6*jx+j],  w * Jj[i] * Jj[j]);
          if (ix >= 0 && jx >= 0) {
            atomicAdd(&B[6*ix+i][6*jx+j], -w * Ji[i] * Jj[j]);
            atomicAdd(&B[6*jx+i][6*ix+j], -w * Jj[i] * Ji[j]);
          }
        }
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&E[6*ix+i][k], -w * Jz * Ji[i]);
        if (jx >= 0)
          atomicAdd(&E[6*jx+i][k],  w * Jz * Jj[i]);
      }

      for (int i=0; i<6; i++) {
        if (ix >= 0)
          atomicAdd(&v[6*ix+i], -w * r * Ji[i]);
        if (jx >= 0)
          atomicAdd(&v[6*jx+i],  w * r * Jj[i]);
      }

      atomicAdd(&C[k], w * Jz * Jz);
      atomicAdd(&u[k], w *  r * Jz);
    }
  }
}


/* =========================================================================
 * REPROJECTION KERNEL
 * =========================================================================
 * Reprojects all pixels of each patch from the source frame into the
 * target frame using the current pose estimates.  Unlike the BA kernel
 * (which only uses the patch center), this kernel processes the full
 * P × P patch grid.
 * ======================================================================= */

/**
 * Per-edge full-patch reprojection kernel.
 *
 * For each edge (ii[n], jj[n], kk[n]) and each pixel (i, j) in the P × P
 * patch grid:
 *   1. Back-project using pixel coords and inverse depth from the patch.
 *   2. Apply relative SE3 transform Gᵢⱼ = Tⱼ · Tᵢ⁻¹.
 *   3. Project into the target frame using pinhole model.
 *
 * @param poses       Camera poses [N_total, 7].
 * @param patches     Patch data [N_patches, 3, P, P].
 * @param intrinsics  Camera intrinsics [1, 4].
 * @param ii          Source frame indices [N_edges].
 * @param jj          Target frame indices [N_edges].
 * @param kk          Patch indices [N_edges].
 * @param coords      Output reprojected 2D coords [N_edges, 2, P, P].
 *                    coords[n][0][i][j] = x (column),
 *                    coords[n][1][i][j] = y (row).
 */
__global__ void reproject(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> patches,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> kk,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords) {

  /* Load intrinsics into shared memory once per block */
  __shared__ float fx, fy, cx, cy;
  if (threadIdx.x == 0) {
    fx = intrinsics[0][0];
    fy = intrinsics[0][1];
    cx = intrinsics[0][2];
    cy = intrinsics[0][3];
  }

  __syncthreads();

  GPU_1D_KERNEL_LOOP(n, ii.size(0)) {
    int ix = ii[n];
    int jx = jj[n];
    int kx = kk[n];

    /* Load poses for source and target frames */
    float ti[3] = { poses[ix][0], poses[ix][1], poses[ix][2] };
    float tj[3] = { poses[jx][0], poses[jx][1], poses[jx][2] };
    float qi[4] = { poses[ix][3], poses[ix][4], poses[ix][5], poses[ix][6] };
    float qj[4] = { poses[jx][3], poses[jx][4], poses[jx][5], poses[jx][6] };

    /* Compute the relative transform once per edge */
    float tij[3], qij[4];
    relSE3(ti, qi, tj, qj, tij, qij);

    /* Iterate over all pixels in the P x P patch grid */
    float Xi[4], Xj[4];
    for (int i=0; i<patches.size(2); i++) {
      for (int j=0; j<patches.size(3); j++) {

        // Back-project this patch pixel to a homogeneous 3D point
        Xi[0] = (patches[kx][0][i][j] - cx) / fx;   // normalized x
        Xi[1] = (patches[kx][1][i][j] - cy) / fy;   // normalized y
        Xi[2] = 1.0;                                  // unit depth
        Xi[3] = patches[kx][2][i][j];                // inverse depth

        // Transform from source frame to target frame
        actSE3(tij, qij, Xi, Xj);

        // Project to 2D in the target frame (pinhole model)
        coords[n][0][i][j] = fx * (Xj[0] / Xj[2]) + cx;  // x (column)
        coords[n][1][i][j] = fy * (Xj[1] / Xj[2]) + cy;  // y (row)

      }
    }
  }
}



/* =========================================================================
 * HOST FUNCTIONS
 * =========================================================================
 * These functions are called from the C++ binding layer (ba.cpp).  They
 * manage tensor allocation, kernel launches, and the Gauss-Newton
 * iteration loop including the Schur complement linear solve.
 * ======================================================================= */

/**
 * Run Gauss-Newton bundle adjustment on camera poses and patch depths.
 *
 * This is the main entry point for BA from the Python side.  It:
 *   1. Maps duplicate patch indices to unique sequential indices.
 *   2. Allocates Hessian blocks B, E, C and RHS vectors v, u.
 *   3. Iterates the Gauss-Newton loop:
 *      a. Zero all accumulators.
 *      b. Launch the reprojection_residuals_and_hessian kernel.
 *      c. Solve the reduced system via the Schur complement:
 *         - Q = 1 / (C + λ)
 *         - S = B - E · Q · Eᵀ    (reduced Hessian for poses only)
 *         - y = v - E · Q · u     (reduced RHS)
 *         - Regularize: S += (1e-4 · S + I) to ensure positive-definiteness.
 *         - Solve S · dX = y via Cholesky factorization.
 *         - Back-substitute: dZ = Q · (u - Eᵀ · dX)
 *      d. Retract: apply dX to poses, dZ to inverse depths.
 *
 * Special case: if t1 - t0 == 0, no poses are active (only depth is
 * optimized).  In this case the pose blocks are skipped and only the
 * depth update dZ = Q · u is applied.
 *
 * @param poses       Camera poses [N_total, 7].  Modified in-place.
 * @param patches     Patch data [N_patches, 3, P, P].  Modified in-place.
 * @param intrinsics  Camera intrinsics [1, 4] as (fx, fy, cx, cy).
 * @param target      Target 2D positions per edge [N_edges, 2].
 * @param weight      Per-edge confidence weights [N_edges, 2].
 * @param lmbda       Levenberg-Marquardt damping scalar tensor.
 * @param ii          Source frame index per edge [N_edges].
 * @param jj          Target frame index per edge [N_edges].
 * @param kk          Patch index per edge [N_edges].
 * @param t0          Start of the active pose window (inclusive).
 * @param t1          End of the active pose window (exclusive).
 * @param iterations  Number of Gauss-Newton iterations to run.
 *
 * @return Empty vector (poses and patches are updated in-place).
 */
std::vector<torch::Tensor> cuda_ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk,
    const int t0, const int t1, const int iterations)
{

  /*
   * Map patch indices kk to unique sequential indices.
   * kx = unique patch indices (sorted), shape [M]
   * ku = inverse mapping such that kx[ku[n]] == kk[n], shape [N_edges]
   * This is needed because different edges may reference the same patch,
   * and the Hessian depth blocks are indexed by unique patch index.
   */
  auto ktuple = torch::_unique(kk, true, true);
  torch::Tensor kx = std::get<0>(ktuple);   // unique patch indices
  torch::Tensor ku = std::get<1>(ktuple);   // inverse mapping (edge -> unique index)

  const int N = t1 - t0;    // number of active (optimizable) poses
  const int M = kx.size(0); // number of unique patches
  const int P = patches.size(3); // patch grid dimension (e.g. 3)

  auto opts = torch::TensorOptions()
    .dtype(torch::kFloat32).device(torch::kCUDA);

  // Flatten tensors to 2D for efficient kernel access
  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  target = target.view({-1, 2});
  weight = weight.view({-1, 2});

  /*
   * Allocate the Gauss-Newton system matrices:
   *   B: pose-pose Hessian      [6N × 6N]
   *   E: pose-depth cross-term  [6N × M]
   *   C: depth-depth diagonal   [M]     (stored as a flat vector)
   *   v: pose RHS               [6N]
   *   u: depth RHS              [M]
   */
  const int num = ii.size(0);
  torch::Tensor B = torch::empty({6*N, 6*N}, opts);
  torch::Tensor E = torch::empty({6*N, 1*M}, opts);
  torch::Tensor C = torch::empty({M}, opts);

  torch::Tensor v = torch::empty({6*N}, opts);
  torch::Tensor u = torch::empty({1*M}, opts);

  /* === Gauss-Newton iteration loop === */
  for (int itr=0; itr < iterations; itr++) {

    // Zero all accumulators before each iteration (the kernel uses atomicAdd)
    B.zero_();
    E.zero_();
    C.zero_();
    v.zero_();
    u.zero_();

    // Ensure v and u are 1D for the kernel's PackedTensorAccessor
    v = v.view({6*N});
    u = u.view({1*M});

    /* Launch the main BA kernel: computes residuals and accumulates Hessian */
    reprojection_residuals_and_hessian<<<NUM_BLOCKS(ii.size(0)), NUM_THREADS>>>(
      poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      target.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      weight.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      lmbda.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      ku.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      B.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      E.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      C.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      v.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
      u.packed_accessor32<float,1,torch::RestrictPtrTraits>(), t0);

    // Reshape v and u to column vectors for matrix operations
    v = v.view({6*N, 1});
    u = u.view({1*M, 1});

    /*
     * Q = 1 / (C + λ): element-wise inverse of the damped depth diagonal.
     * This is the "inverse" of the depth block in the Schur complement.
     * Shape: [1, M] (row vector for broadcasting in matrix multiplies).
     */
    torch::Tensor Q = 1.0 / (C + lmbda).view({1, M});

    if (t1 - t0 == 0) {
      /*
       * Special case: no active poses (only depth optimization).
       * Skip the Schur complement and just solve for depth:
       *   dZ = Q · u
       */

      torch::Tensor Qt = torch::transpose(Q, 0, 1);  // [M, 1]
      torch::Tensor dZ = Qt * u;

      dZ = dZ.view({M});

      // Apply the depth update
      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
        kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        dZ.packed_accessor32<float,1,torch::RestrictPtrTraits>());

    }

    else {
      /*
       * Full Schur complement solve.
       *
       * The joint Gauss-Newton system is:
       *   [ B   E ] [dX]   [v]
       *   [ Eᵀ  C ] [dZ] = [u]
       *
       * Eliminate dZ via the Schur complement:
       *   S  = B - E · diag(Q) · Eᵀ    (reduced system for poses)
       *   y  = v - E · diag(Q) · u
       *   Solve: S · dX = y
       *   Back-sub: dZ = diag(Q) · (u - Eᵀ · dX)
       */

      // EQ = E · diag(Q)  -- broadcasting Q as a row vector across E's columns
      torch::Tensor EQ = E * Q;
      torch::Tensor Et = torch::transpose(E, 0, 1);   // [M, 6N]
      torch::Tensor Qt = torch::transpose(Q, 0, 1);   // [M, 1]

      // S = B - E · Q · Eᵀ  (Schur complement matrix)
      torch::Tensor S = B - torch::matmul(EQ, Et);
      // y = v - E · Q · u    (Schur complement RHS)
      torch::Tensor y = v - torch::matmul(EQ,  u);

      /*
       * Regularization: S += (1e-4 · diag(S) + I)
       * This adds a small multiple of the diagonal plus the identity
       * to ensure S is positive-definite for the Cholesky solver.
       * The 1e-4 * S term provides relative damping, while +1.0
       * provides absolute damping.
       */
      torch::Tensor I = torch::eye(6*N, opts);
      S += I * (1e-4 * S + 1.0);

      /* Solve S · dX = y via Cholesky factorization */
      torch::Tensor U = torch::linalg_cholesky(S);
      torch::Tensor dX = torch::cholesky_solve(y, U);

      /* Back-substitute for depth updates: dZ = Q · (u - Eᵀ · dX) */
      torch::Tensor dZ = Qt * (u - torch::matmul(Et, dX));

      dX = dX.view({N, 6});   // reshape to [N_poses, 6]
      dZ = dZ.view({M});      // reshape to [M_patches]

      /* Apply updates via retraction kernels */
      pose_retr_kernel<<<NUM_BLOCKS(N), NUM_THREADS>>>(t0, t1,
          poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
          dX.packed_accessor32<float,2,torch::RestrictPtrTraits>());

      patch_retr_kernel<<<NUM_BLOCKS(M), NUM_THREADS>>>(
          kx.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
          patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
          dZ.packed_accessor32<float,1,torch::RestrictPtrTraits>());
    }
  }

  return {};
}


/**
 * Batch reprojection of all patches into their target frames.
 *
 * This host function launches the `reproject` kernel to compute the
 * projected 2D coordinates for every pixel in every patch, using the
 * current pose estimates.
 *
 * @param poses       Camera poses [N_total, 7].
 * @param patches     Patch data [N_patches, 3, P, P].
 * @param intrinsics  Camera intrinsics [1, 4].
 * @param ii          Source frame index per edge [N_edges].
 * @param jj          Target frame index per edge [N_edges].
 * @param kk          Patch index per edge [N_edges].
 *
 * @return Reprojected 2D coordinates, shape [1, N_edges, 2, P, P].
 *         The leading dimension of 1 is for batch compatibility.
 */
torch::Tensor cuda_reproject(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk)
{

  const int N = ii.size(0);
  const int P = patches.size(3); // patch grid dimension

  // Flatten inputs for kernel access
  poses = poses.view({-1, 7});
  patches = patches.view({-1,3,P,P});
  intrinsics = intrinsics.view({-1, 4});

  auto opts = torch::TensorOptions()
    .dtype(torch::kFloat32).device(torch::kCUDA);

  // Output: 2D coordinates (x, y) for each pixel in each patch
  torch::Tensor coords = torch::empty({N, 2, P, P}, opts);

  reproject<<<NUM_BLOCKS(N), NUM_THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    patches.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    kk.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
    coords.packed_accessor32<float,4,torch::RestrictPtrTraits>());

  // Add a leading batch dimension for downstream compatibility
  return coords.view({1, N, 2, P, P});

}
