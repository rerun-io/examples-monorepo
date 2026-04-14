/*
 * ba.cpp -- PyBind11 module exposing bundle adjustment, reprojection,
 *           temporal neighbor-finding, and sparse PGO solve to Python.
 *
 * Part of DPV-SLAM (Deep Patch Visual SLAM, Lipson et al., 2024),
 * extended from DPVO (Teed et al., 2022).
 *
 * This file provides four functions to the Python layer:
 *
 *   1. ba()           -- Runs Gauss-Newton bundle adjustment on camera poses
 *                        and patch inverse-depths.  Delegates to the CUDA
 *                        implementation in ba_cuda.cu.  Supports both
 *                        sliding-window (dense E) and global (efficient
 *                        E-block) modes via the eff_impl flag.
 *
 *   2. reproject()    -- Projects patches from their source frames into target
 *                        frames using the current pose estimates.  Also
 *                        delegates to ba_cuda.cu.
 *
 *   3. neighbors()    -- A CPU-side function that computes temporal neighbor
 *                        indices for the GRU update network.  For each
 *                        observation of a patch, it finds the previous and
 *                        next observations (of the same patch) ordered by
 *                        target frame index.  This temporal linking enables
 *                        the GRU to aggregate information across time.
 *
 *   4. solve_system() -- Sparse Sim(3) pose-graph optimization for classical
 *                        loop closure.  Builds a sparse Jacobian from per-edge
 *                        7x7 blocks, forms normal equations (JᵀJ), and solves
 *                        via Cholesky factorization.
 *
 * The module is typically imported as `_cuda_ba` in Python.
 */

#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>


/* -------------------------------------------------------------------------
 * Forward declarations of CUDA functions (implemented in ba_cuda.cu).
 * ---------------------------------------------------------------------- */

/**
 * CUDA bundle adjustment: jointly optimizes camera poses and patch
 * inverse-depths via Gauss-Newton with Schur complement elimination.
 *
 * @param poses       Camera poses [N_total, 7] as (tx, ty, tz, qx, qy, qz, qw).
 *                    Modified in-place.
 * @param patches     Patch data [N_patches, 3, P, P].  Channel 0 = x coords,
 *                    channel 1 = y coords, channel 2 = inverse depth.
 *                    Modified in-place.
 * @param intrinsics  Camera intrinsics [1, 4] as (fx, fy, cx, cy).
 * @param target      Target 2D coordinates per edge [N_edges, 2].
 * @param weight      Confidence weights per edge [N_edges, 2].
 * @param lmbda       Levenberg-Marquardt damping factor (scalar tensor).
 * @param ii          Source frame index per edge [N_edges].
 * @param jj          Target frame index per edge [N_edges].
 * @param kk          Patch index per edge [N_edges].
 * @param PPF         Patches per frame (used for E-block allocation when eff_impl=true).
 * @param t0          Start of the active pose window (poses before t0 are fixed).
 * @param t1          End of the active pose window (exclusive).
 * @param iterations  Number of Gauss-Newton iterations to run.
 * @param eff_impl    If true, use efficient E-block (block_e.cu) for global BA.
 *                    If false, use dense E matrix for sliding-window BA.
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
    const int PPF,
    int t0, int t1, int iterations, bool eff_impl);


/**
 * CUDA reprojection: projects each patch from its source frame into its
 * target frame using the current pose estimates.
 *
 * @param poses       Camera poses [N_total, 7].
 * @param patches     Patch data [N_patches, 3, P, P].
 * @param intrinsics  Camera intrinsics [1, 4].
 * @param ii          Source frame index per edge [N_edges].
 * @param jj          Target frame index per edge [N_edges].
 * @param kk          Patch index per edge [N_edges].
 *
 * @return Reprojected 2D coordinates, shape [1, N_edges, 2, P, P].
 */
torch::Tensor cuda_reproject(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk);

/* -------------------------------------------------------------------------
 * Thin C++ wrappers that forward to the CUDA implementations.
 * ---------------------------------------------------------------------- */

std::vector<torch::Tensor> ba(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor target,
    torch::Tensor weight,
    torch::Tensor lmbda,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk,
    int PPF,
    int t0, int t1, int iterations, bool eff_impl) {
  return cuda_ba(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, PPF, t0, t1, iterations, eff_impl);
}


torch::Tensor reproject(
    torch::Tensor poses,
    torch::Tensor patches,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor kk) {
  return cuda_reproject(poses, patches, intrinsics, ii, jj, kk);
}

/* -------------------------------------------------------------------------
 * neighbors() -- CPU implementation of temporal neighbor linking.
 * ---------------------------------------------------------------------- */

/**
 * Build temporal neighbor indices for the GRU update network.
 *
 * Each edge in the factor graph connects a source patch (indexed by ii)
 * to a target frame (indexed by jj).  A single patch may be observed in
 * multiple target frames.  The GRU update step needs to pass messages
 * along the temporal chain of observations for the same patch, ordered
 * by target frame index.
 *
 * This function groups all edges by their patch index (ii), sorts each
 * group by target frame (jj), and then for each edge records:
 *   - ix[n]: the index of the previous observation of the same patch
 *            (the one with the next-smaller jj value), or -1 if none.
 *   - jx[n]: the index of the next observation of the same patch
 *            (the one with the next-larger jj value), or -1 if none.
 *
 * @param ii  Patch (source) index per edge, shape [N_edges].
 * @param jj  Target frame index per edge, shape [N_edges].
 *
 * @return {ix, jx} both shape [N_edges] on CUDA.
 *         ix[n] = index of the previous temporal neighbor, or -1.
 *         jx[n] = index of the next temporal neighbor, or -1.
 */
std::vector<torch::Tensor> neighbors(torch::Tensor ii, torch::Tensor jj)
{
  /* Step 1: Find unique patch indices and map each edge to its group. */
  auto tup = torch::_unique(ii, true, true);
  torch::Tensor uniq = std::get<0>(tup).to(torch::kCPU);
  torch::Tensor perm = std::get<1>(tup).to(torch::kCPU);

  jj = jj.to(torch::kCPU);
  auto jj_accessor = jj.accessor<long,1>();

  /* Step 2: Group edge indices by their patch. */
  auto perm_accessor = perm.accessor<long,1>();
  std::vector<std::vector<long>> index(uniq.size(0));
  for (int i=0; i < ii.size(0); i++) {
    index[perm_accessor[i]].push_back(i);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor ix = torch::empty({ii.size(0)}, opts);
  torch::Tensor jx = torch::empty({ii.size(0)}, opts);

  auto ix_accessor = ix.accessor<long,1>();
  auto jx_accessor = jx.accessor<long,1>();

  /* Step 3: Sort each group by target frame and link into a chain. */
  for (int i=0; i<uniq.size(0); i++) {
    std::vector<long>& idx = index[i];

    // Sort edge indices within this group by ascending target frame index
    std::stable_sort(idx.begin(), idx.end(),
       [&jj_accessor](size_t i, size_t j) {return jj_accessor[i] < jj_accessor[j];});

    // Link each edge to its predecessor and successor in temporal order
    for (int i=0; i < idx.size(); i++) {
      ix_accessor[idx[i]] = (i > 0) ? idx[i-1] : -1;
      jx_accessor[idx[i]] = (i < idx.size() - 1) ? idx[i+1] : -1;
    }
  }

  ix = ix.to(torch::kCUDA);
  jx = jx.to(torch::kCUDA);

  return {ix, jx};
}

/* -------------------------------------------------------------------------
 * solve_system() -- Sparse least-squares solve for pose-graph optimization.
 *
 * Used by loop closure PGO.  Builds sparse Jacobian from per-edge 7x7
 * blocks, forms normal equations, solves via Cholesky.
 * ---------------------------------------------------------------------- */

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

Eigen::VectorXd solve(const SpMat &A, const Eigen::VectorXd &b, int freen){

  if (freen < 0){
    const Eigen::SimplicialCholesky<SpMat> chol(A);
    return chol.solve(b);
  }

  const SpMat A_sub = A.topLeftCorner(freen, freen);
  const Eigen::VectorXd b_sub = b.topRows(freen);
  const Eigen::VectorXd delta = solve(A_sub, b_sub, -7);

  Eigen::VectorXd delta2(b.rows());
  delta2.setZero();
  delta2.topRows(freen) = delta;

  return delta2;
}

/**
 * Sparse Sim(3) pose-graph optimization for classical loop closure.
 *
 * Builds a sparse Jacobian J from per-edge 7x7 blocks, forms the normal
 * equations A = JᵀJ, b = -Jᵀr, adds LM damping, and solves via Cholesky.
 *
 * @param J_Ginv_i  Jacobian blocks w.r.t. pose i [N_edges, 7, 7].
 * @param J_Ginv_j  Jacobian blocks w.r.t. pose j [N_edges, 7, 7].
 * @param ii        Source pose index per edge [N_edges].
 * @param jj        Target pose index per edge [N_edges].
 * @param res       Residual vectors [N_edges, 7].
 * @param ep        Epsilon for diagonal regularization.
 * @param lm        Levenberg-Marquardt damping factor.
 * @param freen     Number of free poses (first freen are optimized,
 *                  rest are fixed).  Set to -1 to optimize all.
 *
 * @return {delta} pose updates [N_poses, 7] on the input device.
 */
std::vector<torch::Tensor> solve_system(torch::Tensor J_Ginv_i, torch::Tensor J_Ginv_j, torch::Tensor ii, torch::Tensor jj, torch::Tensor res, float ep, float lm, int freen)
{

  const torch::Device device = res.device();
  J_Ginv_i = J_Ginv_i.to(torch::kCPU);
  J_Ginv_j = J_Ginv_j.to(torch::kCPU);
  ii = ii.to(torch::kCPU);
  jj = jj.to(torch::kCPU);
  res = res.clone().to(torch::kCPU);

  const int r = res.size(0);
  const int n = std::max(ii.max().item<long>(), jj.max().item<long>()) + 1;

  res.resize_({r*7});
  float *res_ptr = res.data_ptr<float>();
  Eigen::Map<Eigen::VectorXf> v(res_ptr, r*7);

  SpMat J(r*7, n*7);
  std::vector<T> tripletList;
  tripletList.reserve(r*7*7*2);

  auto ii_acc = ii.accessor<long,1>();
  auto jj_acc = jj.accessor<long,1>();
  auto J_Ginv_i_acc = J_Ginv_i.accessor<float,3>();
  auto J_Ginv_j_acc = J_Ginv_j.accessor<float,3>();

  for (int x=0; x<r; x++){
    const int i = ii_acc[x];
    const int j = jj_acc[x];
    for (int k=0; k<7; k++){
      for (int l=0; l<7; l++){
        if (i == j)
          exit(1);
        const float val_i = J_Ginv_i_acc[x][k][l];
        tripletList.emplace_back(x*7 + k, i*7 + l, val_i);
        const float val_j = J_Ginv_j_acc[x][k][l];
        tripletList.emplace_back(x*7 + k, j*7 + l, val_j);
      }
    }
  }

  J.setFromTriplets(tripletList.begin(), tripletList.end());
  const SpMat Jt = J.transpose();
  Eigen::VectorXd b = -(Jt * v.cast<double>());
  SpMat A = Jt * J;

  A.diagonal() += (A.diagonal() * lm);
  A.diagonal().array() += ep;
  Eigen::VectorXf delta = solve(A, b, freen*7).cast<float>();

  torch::Tensor delta_tensor = torch::from_blob(delta.data(), {n*7}).clone().to(device);
  delta_tensor.resize_({n, 7});
  return {delta_tensor};
}


/* -------------------------------------------------------------------------
 * Python module registration.
 *
 * Exposes four functions:
 *   forward(...)      -> ba (bundle adjustment)
 *   neighbors(...)    -> temporal neighbor indices
 *   reproject(...)    -> patch reprojection
 *   solve_system(...) -> sparse Sim(3) PGO for loop closure
 * ---------------------------------------------------------------------- */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ba, "BA forward operator");
  m.def("neighbors", &neighbors, "temporal neighbor indices");
  m.def("reproject", &reproject, "reproject patches into target frames");
  m.def("solve_system", &solve_system, "sparse pose-graph solve for loop closure PGO");

}
