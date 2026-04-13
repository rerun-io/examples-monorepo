/*
 * ba.cpp -- PyBind11 module exposing bundle adjustment, reprojection, and
 *           temporal neighbor-finding operations to Python.
 *
 * Part of DPVO (Deep Patch Visual Odometry, Teed et al., 2022).
 *
 * This file provides three functions to the Python layer:
 *
 *   1. ba()         -- Runs Gauss-Newton bundle adjustment on camera poses
 *                      and patch inverse-depths.  Delegates to the CUDA
 *                      implementation in ba_cuda.cu.
 *
 *   2. reproject()  -- Projects patches from their source frames into target
 *                      frames using the current pose estimates.  Also
 *                      delegates to ba_cuda.cu.
 *
 *   3. neighbors()  -- A CPU-side function that computes temporal neighbor
 *                      indices for the GRU update network.  For each
 *                      observation of a patch, it finds the previous and
 *                      next observations (of the same patch) ordered by
 *                      target frame index.  This temporal linking enables
 *                      the GRU to aggregate information across time.
 *
 * The module is typically imported as `fastba_cuda` in Python.
 */

#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>


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
 * @param t0          Start of the active pose window (poses before t0 are fixed).
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
    int t0, int t1, int iterations);


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
    int t0, int t1, int iterations) {
  return cuda_ba(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, iterations);
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

/*
 * NOTE: The commented-out block below is an older implementation of the same
 * algorithm using std::unordered_map.  It has been replaced by the version
 * below that uses torch::_unique for cleaner grouping, but is preserved
 * here for reference.
 */

// std::vector<torch::Tensor> neighbors(torch::Tensor ii, torch::Tensor jj)
// {
//   ii = ii.to(torch::kCPU);
//   jj = jj.to(torch::kCPU);
//   auto ii_data = ii.accessor<long,1>();
//   auto jj_data = jj.accessor<long,1>();

//   std::unordered_map<long, std::vector<long>> graph;
//   std::unordered_map<long, std::vector<long>> index;
//   for (int i=0; i < ii.size(0); i++) {
//     const long ix = ii_data[i];
//     const long jx = jj_data[i];
//     if (graph.find(ix) == graph.end()) {
//       graph[ix] = std::vector<long>();
//       index[ix] = std::vector<long>();
//     }
//     graph[ix].push_back(jx);
//     index[ix].push_back( i);
//   }

//   auto opts = torch::TensorOptions().dtype(torch::kInt64);
//   torch::Tensor ix = torch::empty({ii.size(0)}, opts);
//   torch::Tensor jx = torch::empty({jj.size(0)}, opts);

//   auto ix_data = ix.accessor<long,1>();
//   auto jx_data = jx.accessor<long,1>();

//   for (std::pair<long, std::vector<long>> element : graph) {
//     std::vector<long>& v = graph[element.first];
//     std::vector<long>& idx = index[element.first];

//     std::stable_sort(idx.begin(), idx.end(),
//        [&v](size_t i, size_t j) {return v[i] < v[j];});

//     ix_data[idx.front()] = -1;
//     jx_data[idx.back()]  = -1;

//     for (int i=0; i < idx.size(); i++) {
//       ix_data[idx[i]] = (i > 0) ? idx[i-1] : -1;
//       jx_data[idx[i]] = (i < idx.size() - 1) ? idx[i+1] : -1;
//     }
//   }

//   ix = ix.to(torch::kCUDA);
//   jx = jx.to(torch::kCUDA);

//   return {ix, jx};
// }


/**
 * Build temporal neighbor indices for the GRU update network.
 *
 * In DPVO, each edge in the factor graph connects a source patch (indexed
 * by ii) to a target frame (indexed by jj).  A single patch may be
 * observed in multiple target frames.  The GRU update step needs to pass
 * messages along the temporal chain of observations for the same patch,
 * ordered by target frame index.
 *
 * This function groups all edges by their patch index (ii), sorts each
 * group by target frame (jj), and then for each edge records:
 *   - ix[n]: the index of the previous observation of the same patch
 *            (the one with the next-smaller jj value), or -1 if none.
 *   - jx[n]: the index of the next observation of the same patch
 *            (the one with the next-larger jj value), or -1 if none.
 *
 * Example:
 *   Suppose edges 3, 7, 12 all observe patch #5, with target frames
 *   jj[3]=10, jj[7]=20, jj[12]=15.
 *   After sorting by jj: order is [3, 12, 7] (frames 10, 15, 20).
 *   Then: ix[3]=-1, jx[3]=12   (edge 3 has no predecessor, successor is 12)
 *         ix[12]=3, jx[12]=7   (edge 12's predecessor is 3, successor is 7)
 *         ix[7]=12, jx[7]=-1   (edge 7's predecessor is 12, no successor)
 *
 * @param ii  Patch (source) index per edge, shape [N_edges].  Must be on
 *            GPU (will be moved to CPU internally for processing).
 * @param jj  Target frame index per edge, shape [N_edges].  Must be on GPU.
 *
 * @return {ix, jx} both shape [N_edges] on CUDA.
 *         ix[n] = index of the previous temporal neighbor for edge n, or -1.
 *         jx[n] = index of the next temporal neighbor for edge n, or -1.
 */
std::vector<torch::Tensor> neighbors(torch::Tensor ii, torch::Tensor jj)
{
  /*
   * Step 1: Find unique patch indices and get a mapping from each edge
   * to its group.  torch::_unique returns (unique_values, inverse_indices).
   * perm[n] tells us which group (unique patch) edge n belongs to.
   */
  auto tup = torch::_unique(ii, true, true);
  torch::Tensor uniq = std::get<0>(tup).to(torch::kCPU);  // unique patch IDs
  torch::Tensor perm = std::get<1>(tup).to(torch::kCPU);  // group membership for each edge

  jj = jj.to(torch::kCPU);
  auto jj_accessor = jj.accessor<long,1>();

  /*
   * Step 2: Group edge indices by their patch.
   * index[g] = list of edge indices that belong to patch group g.
   */
  auto perm_accessor = perm.accessor<long,1>();
  std::vector<std::vector<long>> index(uniq.size(0));
  for (int i=0; i < ii.size(0); i++) {
    index[perm_accessor[i]].push_back(i);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor ix = torch::empty({ii.size(0)}, opts);  // previous neighbor
  torch::Tensor jx = torch::empty({ii.size(0)}, opts);  // next neighbor

  auto ix_accessor = ix.accessor<long,1>();
  auto jx_accessor = jx.accessor<long,1>();

  /*
   * Step 3: For each patch group, sort the edge indices by target frame
   * (jj) and then link them into a doubly-linked list.
   */
  for (int i=0; i<uniq.size(0); i++) {
    std::vector<long>& idx = index[i];

    // Sort edge indices within this group by ascending target frame index
    std::stable_sort(idx.begin(), idx.end(),
       [&jj_accessor](size_t i, size_t j) {return jj_accessor[i] < jj_accessor[j];});

    // Link each edge to its predecessor and successor in temporal order
    for (int i=0; i < idx.size(); i++) {
      ix_accessor[idx[i]] = (i > 0) ? idx[i-1] : -1;              // previous, or -1
      jx_accessor[idx[i]] = (i < idx.size() - 1) ? idx[i+1] : -1; // next, or -1
    }
  }

  // Move results back to GPU for downstream consumption
  ix = ix.to(torch::kCUDA);
  jx = jx.to(torch::kCUDA);

  return {ix, jx};
}


/* -------------------------------------------------------------------------
 * Python module registration.
 *
 * Exposes three functions:
 *   forward(...)   -> ba (bundle adjustment)
 *   neighbors(...) -> temporal neighbor indices
 *   reproject(...) -> patch reprojection
 * ---------------------------------------------------------------------- */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &ba, "BA forward operator");
  m.def("neighbors", &neighbors, "temporal neighbor indices");
  m.def("reproject", &reproject, "reproject patches into target frames");

}
