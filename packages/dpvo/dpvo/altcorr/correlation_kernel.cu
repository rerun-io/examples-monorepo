/*
 * correlation_kernel.cu -- CUDA kernels for patch extraction and local
 *                          correlation in the DPVO visual odometry pipeline.
 *
 * Part of DPVO (Deep Patch Visual Odometry, Teed et al., 2022).
 *
 * This file implements four GPU kernels and their host-side launch wrappers:
 *
 *   Patchify (extract / scatter):
 *     - patchify_forward_kernel  : gather D × D patches from a feature map
 *     - patchify_backward_kernel : scatter gradients back via atomicAdd
 *
 *   Correlation (dot-product matching):
 *     - corr_forward_kernel  : integer-grid dot-product correlation volume
 *     - corr_backward_kernel : gradient distribution to both feature maps
 *
 *   Host wrappers:
 *     - patchify_cuda_forward / patchify_cuda_backward
 *     - corr_cuda_forward  (kernel + bilinear interpolation on host)
 *     - corr_cuda_backward (bilinear gradient split + kernel)
 *
 * Notation used throughout:
 *   B  = batch size
 *   M  = number of patches (edges in the factor graph)
 *   C  = feature channels (typically 128)
 *   H, W   = spatial dimensions of the feature map
 *   H2, W2 = spatial dimensions of the target feature map (may differ)
 *   R  = search radius (a hyperparameter, e.g. 3)
 *   D  = 2R + 2    -- the "diameter" including one extra pixel for bilinear
 *                      interpolation.  After interpolation the effective
 *                      neighborhood becomes (2R + 1) × (2R + 1).
 *   H_p, W_p = spatial dimensions of the 3×3 patch grid (typically 3×3)
 */

#include <torch/extension.h>
#include <THC/THCAtomics.cuh>
#include <vector>
#include <iostream>

using namespace torch::indexing;

/* Number of CUDA threads per block (a common choice for occupancy). */
#define THREADS 256

/* Compute the number of thread blocks needed to cover `n` work items. */
#define BLOCKS(n) (n + THREADS - 1) / THREADS

/**
 * Bounds check: returns true if pixel (h, w) lies inside [0, H) x [0, W).
 * Used by every kernel to guard against out-of-bounds memory accesses when
 * a patch extends beyond the feature map boundary.
 */
__forceinline__ __device__
bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

/* =========================================================================
 * PATCHIFY KERNELS
 * =========================================================================
 * "Patchify" extracts small D × D windows from a dense feature map at
 * floating-point (x, y) locations.  The window is centered on floor(x),
 * floor(y) and extends from -R to R+1 in each axis.  The extra pixel on
 * each side (D = 2R + 2 instead of 2R + 1) lets the caller perform bilinear
 * interpolation between four neighboring integer-offset patches to get
 * sub-pixel accurate features.
 *
 * Parallelization: one CUDA thread per (batch, patch_index, row, col)
 * tuple, i.e. B · M · D · D total threads.  Each thread copies one pixel
 * across all C channels.
 * ======================================================================= */

/**
 * Forward kernel: gather feature values from `net` into `patches`.
 *
 * @tparam scalar_t  Feature map dtype (float, half, etc.).
 *
 * @param R        Search radius.
 * @param net      Dense feature map, shape [B, C, H, W].
 *                 Accessed as net[batch][channel][row][col].
 * @param coords   Patch center coordinates, shape [B, M, 2].
 *                 coords[b][m] = (x, y) in feature-map pixel space.
 *                 x is the horizontal (column) coordinate,
 *                 y is the vertical (row) coordinate.
 * @param patches  Output tensor, shape [B, M, C, D, D].
 *                 Each (b, m) slice is a D × D patch with C channels.
 *
 * Thread indexing:
 *   The linear thread index `n` is decomposed in column-major order:
 *     ii = n % D        → column within the D × D patch
 *     jj = (n/D) % D    → row within the D × D patch
 *     m  = ... % M       → patch index
 *     b  = ...           → batch index
 *
 *   For each thread:
 *     i = floor(y) + (ii - R)   → absolute row in the feature map
 *     j = floor(x) + (jj - R)   → absolute column in the feature map
 *   Then for every channel k: patches[b][m][k][ii][jj] = net[b][k][i][j]
 */
template <typename scalar_t>
__global__ void patchify_forward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> net,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> patches)
{
  // D = 2R + 2: patch side length including the extra border for bilinear interp
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = net.size(1);
  const int H = net.size(2);
  const int W = net.size(3);

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < B * M * D * D) {
    // Decompose linear index into (ii, jj, m, batch) in column-major order
    const int ii = n % D; n /= D;   // column offset within the patch
    const int jj = n % D; n /= D;   // row offset within the patch
    const int  m = n % M; n /= M;   // patch index within the batch
    // After all divisions, `n` is the batch index

    const float x = coords[n][m][0]; // horizontal center (column)
    const float y = coords[n][m][1]; // vertical center (row)

    // Compute the absolute feature-map position for this patch pixel.
    // The patch covers [floor(y)-R, floor(y)+R+1] in rows,
    //                  [floor(x)-R, floor(x)+R+1] in cols.
    const int i = static_cast<int>(floor(y)) + (ii - R);
    const int j = static_cast<int>(floor(x)) + (jj - R);

    if (within_bounds(i, j, H, W)) {
      // Copy all C feature channels for this spatial position
      for (int k=0; k<C; k++)
        patches[n][m][k][ii][jj] = net[n][k][i][j];
    }
    // Out-of-bounds positions remain at the zero-initialized value
  }
}

/**
 * Backward kernel: scatter patch gradients back onto the dense feature map.
 *
 * This is the reverse of patchify_forward_kernel.  Multiple patches may
 * overlap on the same feature-map pixel, so gradients are accumulated
 * with atomicAdd to avoid race conditions between threads.
 *
 * @tparam scalar_t        Feature map dtype.
 *
 * @param R                Search radius.
 * @param patch_gradient   Upstream gradient w.r.t. patches, shape [B, M, C, D, D].
 * @param coords           Patch center coordinates (same as forward), [B, M, 2].
 * @param gradient         Output gradient w.r.t. net, shape [B, C, H, W].
 *                         Accumulated into via atomicAdd (must be zero-initialized).
 */
template <typename scalar_t>
__global__ void patchify_backward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> patch_gradient,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> gradient)
{
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = gradient.size(1);
  const int H = gradient.size(2);
  const int W = gradient.size(3);

  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < B * M * D * D) {
    // Same index decomposition as the forward kernel
    const int ii = n % D; n /= D;
    const int jj = n % D; n /= D;
    const int  m = n % M; n /= M;

    const float x = coords[n][m][0];
    const float y = coords[n][m][1];
    const int i = static_cast<int>(floor(y)) + (ii - R);
    const int j = static_cast<int>(floor(x)) + (jj - R);

    if (within_bounds(i, j, H, W)) {
      for (int k=0; k<C; k++)
        // atomicAdd because multiple patches may contribute to the
        // same feature-map pixel
        atomicAdd(&gradient[n][k][i][j], patch_gradient[n][m][k][ii][jj]);
    }
  }
}

/* =========================================================================
 * CORRELATION KERNELS
 * =========================================================================
 * Correlation computes the dot-product similarity between:
 *   - A source patch: features from fmap1 at the patch's own (i0, j0) grid
 *     positions in the source frame.
 *   - A target neighborhood: features from fmap2 at integer offsets around
 *     the reprojected (x, y) position in the target frame.
 *
 * The output is a "correlation volume" of shape [B, M, D, D, H_p, W_p]
 * where D × D is the search neighborhood and H_p × W_p is the 3×3 patch
 * grid.  The host wrapper then applies bilinear interpolation across the
 * D × D grid to produce sub-pixel-accurate correlation at the fractional
 * (x, y) position, yielding shape [B, M, (2R + 1), (2R + 1), H_p, W_p].
 *
 * Parallelization: one thread per (batch, patch, patch_row, patch_col,
 * search_row, search_col) = B · M · H_p · W_p · D · D total threads.
 * Each thread computes one dot product over C channels.
 * ======================================================================= */

/**
 * Forward kernel: compute integer-grid dot-product correlations.
 *
 * For each edge (source frame ii → target frame jj) and each pixel in the
 * 3×3 patch grid (i0, j0), this kernel evaluates the dot product:
 *
 *   corr[b][m][ii_off][jj_off][i0][j0] =
 *       Σ_c fmap1[b][us[m]][c][i0][j0] · fmap2[b][vs[m]][c][i1][j1]
 *
 * where:
 *   (i1, j1) = ( floor(y) + ii_off - R,  floor(x) + jj_off - R )
 *   (x, y)   = coords[b][m][:][i0][j0]   -- the reprojected position
 *   us[m] / vs[m] = source / target frame indices for edge m
 *
 * The dot product over C = 128 channels is unrolled 8-at-a-time for better
 * instruction-level parallelism (ILP).  The #pragma unroll 8 hint tells
 * the compiler to process the outer loop in chunks of 8 channels.
 *
 * @tparam scalar_t  Feature dtype (float or half).
 *
 * @param R      Search radius.
 * @param fmap1  Source feature maps, shape [B, N_frames, C, H, W].
 * @param fmap2  Target feature maps, shape [B, N_frames, C, H2, W2].
 * @param coords Reprojected coordinates for each patch pixel in the target
 *               frame, shape [B, M, 2, H_p, W_p].
 *               coords[b][m][0][i0][j0] = x (column),
 *               coords[b][m][1][i0][j0] = y (row).
 * @param us     Source frame index per edge, shape [M].
 * @param vs     Target frame index per edge, shape [M].
 * @param corr   Output correlation volume, shape [B, M, D, D, H_p, W_p].
 */
template <typename scalar_t>
__global__ void corr_forward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> us,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> vs,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> corr)
{
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int H = coords.size(3);  // patch grid height (typically 3)
  const int W = coords.size(4);  // patch grid width  (typically 3)

  const int C = fmap1.size(2);   // feature channels (e.g. 128)
  const int H2 = fmap2.size(3);  // target feature map height
  const int W2 = fmap2.size(4);  // target feature map width

  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < B * M * H * W * D * D) {
    // Decompose linear thread index into 6D coordinates
    const int jj = n % D; n /= D;   // search neighborhood column offset
    const int ii = n % D; n /= D;   // search neighborhood row offset
    const int j0 = n % W; n /= W;   // patch grid column (0..W_p-1)
    const int i0 = n % H; n /= H;   // patch grid row    (0..H_p-1)
    const int  m = n % M; n /= M;   // edge (patch) index
    // `n` is now the batch index

    const int ix = us[m];  // source frame index for this edge
    const int jx = vs[m];  // target frame index for this edge

    // Reprojected position of this patch pixel in the target frame
    const float x = coords[n][m][0][i0][j0]; // column
    const float y = coords[n][m][1][i0][j0]; // row

    // Integer offset in the search neighborhood: floor + offset - R
    const int i1 = static_cast<int>(floor(y)) + (ii - R);
    const int j1 = static_cast<int>(floor(x)) + (jj - R);

    // Compute dot product over all C feature channels
    scalar_t s = 0;
    if (within_bounds(i1, j1, H2, W2)) {

      /*
       * Loop unrolling: process 8 channels at a time.
       * This reduces loop overhead and allows the compiler to schedule
       * more independent multiply-add instructions in parallel.
       * IMPORTANT: C must be a multiple of 8 for this to be correct
       * (C=128 satisfies this).
       */
      #pragma unroll 8
      for (int i=0; i<C; i+=8) {
        // Load 8 channels from the source patch at grid position (i0, j0)
        scalar_t f1[8]; for (int j=0; j<8; j++) f1[j] = fmap1[n][ix][i+j][i0][j0];
        // Load 8 channels from the target feature map at search position (i1, j1)
        scalar_t f2[8]; for (int j=0; j<8; j++) f2[j] = fmap2[n][jx][i+j][i1][j1];

        // Accumulate element-wise products
        #pragma unroll
        for (int j=0; j<8; j++) s += f1[j] * f2[j];
      }
    }
    // Out-of-bounds positions get correlation = 0

    corr[n][m][ii][jj][i0][j0] = s;
  }
}


/**
 * Backward kernel: distribute correlation gradient to both feature maps.
 *
 * Given dL/d(corr), this kernel computes dL/d(fmap1) and dL/d(fmap2).
 * Since corr = Σ_c fmap1[c] · fmap2[c], the chain rule gives:
 *   dL/d(fmap1[c]) += dL/d(corr) · fmap2[c]
 *   dL/d(fmap2[c]) += dL/d(corr) · fmap1[c]
 *
 * Multiple threads may write to the same feature-map location (because
 * different patch-grid positions or search offsets can map to the same
 * pixel), so atomicAdd is required for correctness.
 *
 * @tparam scalar_t    Feature dtype.
 *
 * @param R            Search radius.
 * @param fmap1        Source features, shape [B, N_frames, C, H, W].
 * @param fmap2        Target features, shape [B, N_frames, C, H2, W2].
 * @param coords       Reprojected coordinates, shape [B, M, 2, H_p, W_p].
 * @param us           Source frame indices, shape [M].
 * @param vs           Target frame indices, shape [M].
 * @param corr_grad    Upstream gradient w.r.t. corr, shape [B, M, D, D, H_p, W_p].
 * @param fmap1_grad   Output: gradient w.r.t. fmap1 (same shape), accumulated via atomicAdd.
 * @param fmap2_grad   Output: gradient w.r.t. fmap2 (same shape), accumulated via atomicAdd.
 */
template <typename scalar_t>
__global__ void corr_backward_kernel(int R,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap1,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap2,
    const torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> us,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> vs,
    const torch::PackedTensorAccessor32<float,6,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap1_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> fmap2_grad)
{
  const int D = 2*R + 2;

  const int B = coords.size(0);
  const int M = coords.size(1);
  const int H = coords.size(3);
  const int W = coords.size(4);

  const int C = fmap1.size(2);
  const int H2 = fmap2.size(3);
  const int W2 = fmap2.size(4);

  int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n < B * M * H * W * D * D) {
    // Same 6D decomposition as the forward kernel
    const int jj = n % D; n /= D;
    const int ii = n % D; n /= D;
    const int j0 = n % W; n /= W;
    const int i0 = n % H; n /= H;
    const int  m = n % M; n /= M;

    const int ix = us[m];
    const int jx = vs[m];

    const float x = coords[n][m][0][i0][j0];
    const float y = coords[n][m][1][i0][j0];

    const int i1 = static_cast<int>(floor(y)) + (ii - R);
    const int j1 = static_cast<int>(floor(x)) + (jj - R);

    // The scalar gradient for this particular (edge, search_offset, grid_pos)
    const scalar_t g = (scalar_t) corr_grad[n][m][ii][jj][i0][j0];

    if (within_bounds(i1, j1, H2, W2)) {
      /*
       * For each channel c:
       *   d(corr)/d(fmap1[c]) = fmap2[c]  =>  fmap1_grad[c] += g * fmap2[c]
       *   d(corr)/d(fmap2[c]) = fmap1[c]  =>  fmap2_grad[c] += g * fmap1[c]
       */
      #pragma unroll 32
      for (int i=0; i<C; i++) {
        atomicAdd(&fmap1_grad[n][ix][i][i0][j0], g * fmap2[n][jx][i][i1][j1]);
        atomicAdd(&fmap2_grad[n][jx][i][i1][j1], g * fmap1[n][ix][i][i0][j0]);
      }
    }
  }
}


/* =========================================================================
 * HOST WRAPPER FUNCTIONS
 * =========================================================================
 * These are called from the C++ side (via correlation.cpp).  They allocate
 * output tensors, launch kernels, and (for correlation) perform bilinear
 * interpolation on the host side using PyTorch tensor ops.
 * ======================================================================= */

/**
 * Host function: launch the correlation forward kernel, then apply bilinear
 * interpolation over the search neighborhood to get sub-pixel accuracy.
 *
 * The kernel computes correlation at integer grid offsets, producing a
 * D × D volume per patch pixel.  Bilinear interpolation between the four
 * nearest integer positions reduces this to (D - 1) × (D - 1) = (2R + 1) × (2R + 1),
 * weighted by the fractional part of the reprojected coordinates.
 *
 * Bilinear interpolation formula (for each output position):
 *   out = (1 - dx) · (1 - dy) · corr[0:D-1, 0:D-1]   (top-left)
 *       +      dx · (1 - dy) · corr[0:D-1, 1:D]      (top-right)
 *       + (1 - dx) ·      dy · corr[1:D,   0:D-1]    (bottom-left)
 *       +      dx ·      dy · corr[1:D,   1:D]       (bottom-right)
 * where dx = x - floor(x), dy = y - floor(y).
 *
 * After interpolation the output is permuted to [B, M, (2R + 1), (2R + 1), H_p, W_p]
 * with the search dimensions swapped to match the expected downstream layout.
 *
 * @param fmap1   Source features [B, N_frames, C, H, W].
 * @param fmap2   Target features [B, N_frames, C, H2, W2].
 * @param coords  Reprojected coordinates [B, M, 2, H_p, W_p].
 * @param ii      Source frame index per edge [M].
 * @param jj      Target frame index per edge [M].
 * @param radius  Search radius R.
 *
 * @return {out} where out has shape [B, M, (2R + 1), (2R + 1), H_p, W_p].
 */
std::vector<torch::Tensor> corr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor ii,
  torch::Tensor jj,
  int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);

  const int H = coords.size(3);  // patch grid height
  const int W = coords.size(4);  // patch grid width
  const int D = 2 * radius + 2;  // expanded diameter (includes bilinear border)

  auto opts = fmap1.options();
  auto corr = torch::empty({B, M, D, D, H, W}, opts);

  // Launch the CUDA kernel: one thread per (batch, edge, search_row, search_col, grid_row, grid_col)
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "corr_forward_kernel", ([&] {
      corr_forward_kernel<scalar_t><<<BLOCKS(B * M * H * W * D * D), THREADS>>>(radius,
        fmap1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        fmap2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        corr.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>());
  }));

  /*
   * Bilinear interpolation on the host (PyTorch tensor operations).
   *
   * The kernel produced correlation at integer grid offsets.  The actual
   * reprojected position has a fractional part (dx, dy).  We interpolate
   * between the four surrounding integer-grid correlation values:
   *
   *     (floor(x), floor(y))  -----  (floor(x)+1, floor(y))
   *            |                              |
   *            |         * (x, y)             |
   *            |                              |
   *     (floor(x), floor(y)+1) --- (floor(x)+1, floor(y)+1)
   *
   * This is why D = 2R + 2: we need one extra row and column beyond the
   * 2R + 1 search window so we have neighbors to interpolate with.
   */
  torch::Tensor x = coords.index({Slice(), Slice(), 0, None, None});
  torch::Tensor y = coords.index({Slice(), Slice(), 1, None, None});
  torch::Tensor dx = x - x.floor(); dx = dx.to(fmap1.dtype()); // fractional x offset
  torch::Tensor dy = y - y.floor(); dy = dy.to(fmap2.dtype()); // fractional y offset

  // Weighted sum of four corners → (D - 1) × (D - 1) = (2R + 1) × (2R + 1) output
  torch::Tensor out;
  out  = (1 - dx) * (1 - dy) * corr.index({Slice(), Slice(), Slice(0, D-1), Slice(0, D-1)});
  out +=     (dx) * (1 - dy) * corr.index({Slice(), Slice(), Slice(0, D-1), Slice(1, D-0)});
  out += (1 - dx) *     (dy) * corr.index({Slice(), Slice(), Slice(1, D-0), Slice(0, D-1)});
  out +=     (dx) *     (dy) * corr.index({Slice(), Slice(), Slice(1, D-0), Slice(1, D-0)});

  // Permute search dimensions: swap dims 2 and 3 so output is
  // [B, M, search_y, search_x, patch_row, patch_col]
  return { out.permute({0,1,3,2,4,5}) };
}


/**
 * Host function: backward pass for correlation.
 *
 * This reverses both the bilinear interpolation and the CUDA dot-product
 * kernel.  The steps are:
 *
 *   1. Un-permute the incoming gradient to match the kernel's layout.
 *   2. Distribute the gradient through the bilinear interpolation:
 *      each of the four corners receives grad · (its bilinear weight),
 *      placed into the appropriate D × D sub-region.
 *   3. Sum the four contributions into a single D × D gradient tensor.
 *   4. Launch the CUDA backward kernel, which uses the chain rule to
 *      compute dL/d(fmap1) and dL/d(fmap2) via atomicAdd.
 *
 * @param fmap1   Source features [B, N_frames, C, H, W].
 * @param fmap2   Target features [B, N_frames, C, H2, W2].
 * @param coords  Reprojected coordinates [B, M, 2, H_p, W_p].
 * @param ii      Source frame index per edge [M].
 * @param jj      Target frame index per edge [M].
 * @param grad    Upstream gradient, shape [B, M, (2R + 1), (2R + 1), H_p, W_p].
 * @param radius  Search radius R.
 *
 * @return {fmap1_grad, fmap2_grad}, each the same shape as the input fmaps.
 */
std::vector<torch::Tensor> corr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor grad,
  int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);

  const int H = coords.size(3);
  const int W = coords.size(4);
  const int D = 2 * radius + 2;

  // Undo the permute from the forward pass so the search dimensions
  // are back in the kernel's expected order
  grad = grad.permute({0,1,3,2,4,5}).contiguous();

  // Extract fractional offsets for bilinear gradient distribution
  torch::Tensor x = coords.index({Slice(), Slice(), 0, None, None});
  torch::Tensor y = coords.index({Slice(), Slice(), 1, None, None});
  torch::Tensor dx = x - x.floor();
  torch::Tensor dy = y - y.floor();

  /*
   * Distribute the (2R + 1) × (2R + 1) gradient back to the D × D grid.
   *
   * In the forward pass:
   *   out = (1 - dx) · (1 - dy) · corr[0:D-1, 0:D-1] + dx · (1 - dy) · corr[0:D-1, 1:D]
   *       + (1 - dx) · dy · corr[1:D, 0:D-1] + dx · dy · corr[1:D, 1:D]
   *
   * So d(out)/d(corr) distributes the gradient to the four corners:
   *   g1[0:D-1, 0:D-1] = (1 - dx) · (1 - dy) · grad   (top-left contribution)
   *   g2[0:D-1, 1:D]   =      dx · (1 - dy) · grad    (top-right)
   *   g3[1:D,   0:D-1] = (1 - dx) ·      dy · grad    (bottom-left)
   *   g4[1:D,   1:D]   =      dx ·      dy · grad     (bottom-right)
   */
  auto opts = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
  torch::Tensor g1 = torch::zeros({B, M, D, D, H, W}, grad.options());
  torch::Tensor g2 = torch::zeros({B, M, D, D, H, W}, grad.options());
  torch::Tensor g3 = torch::zeros({B, M, D, D, H, W}, grad.options());
  torch::Tensor g4 = torch::zeros({B, M, D, D, H, W}, grad.options());

  g1.index_put_({Slice(), Slice(), Slice(0, D-1), Slice(0, D-1)}, (1 - dx) * (1 - dy) * grad);
  g2.index_put_({Slice(), Slice(), Slice(0, D-1), Slice(1, D-0)},     (dx) * (1 - dy) * grad);
  g3.index_put_({Slice(), Slice(), Slice(1, D-0), Slice(0, D-1)}, (1 - dx) *     (dy) * grad);
  g4.index_put_({Slice(), Slice(), Slice(1, D-0), Slice(1, D-0)},     (dx) *     (dy) * grad);

  // Sum to get the full D × D gradient for the kernel's backward pass
  torch::Tensor corr_grad = g1 + g2 + g3 + g4;

  // Allocate zero-initialized gradient tensors for both feature maps
  auto fmap1_grad = torch::zeros_like(fmap1);
  auto fmap2_grad = torch::zeros_like(fmap2);

  // Launch the backward kernel to distribute gradients to fmap1 and fmap2
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(fmap1.scalar_type(), "corr_backward_kernel", ([&] {
    corr_backward_kernel<scalar_t><<<BLOCKS(B * M * H * W * D * D), THREADS>>>(radius,
      fmap1.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      fmap2.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      corr_grad.packed_accessor32<float,6,torch::RestrictPtrTraits>(),
      fmap1_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      fmap2_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));

  return {fmap1_grad, fmap2_grad};
}

/**
 * Host function: launch the patchify forward kernel.
 *
 * Allocates the output tensor and dispatches the CUDA kernel across all
 * (batch, patch, row, col) work items.
 *
 * @param net     Dense feature map [B, C, H, W].
 * @param coords  Patch center coordinates [B, M, 2].
 * @param radius  Patch half-size (output is D × D where D = 2R + 2).
 *
 * @return {patches} with shape [B, M, C, D, D].
 */
std::vector<torch::Tensor> patchify_cuda_forward(
  torch::Tensor net, torch::Tensor coords, int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = net.size(1);
  const int D = 2 * radius + 2;

  auto opts = net.options();
  // Zero-init so out-of-bounds patch pixels default to 0
  auto patches = torch::zeros({B, M, C, D, D}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(net.scalar_type(), "patchify_forward_kernel", ([&] {
      patchify_forward_kernel<scalar_t><<<BLOCKS(B * M * D * D), THREADS>>>(radius,
        net.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
        coords.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        patches.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));

  return { patches };
}


/**
 * Host function: launch the patchify backward kernel.
 *
 * @param net       Dense feature map [B, C, H, W] (used for shape).
 * @param coords    Patch center coordinates [B, M, 2].
 * @param gradient  Upstream gradient w.r.t. patches [B, M, C, D, D].
 * @param radius    Patch half-size.
 *
 * @return {net_gradient} with shape [B, C, H, W].
 */
std::vector<torch::Tensor> patchify_cuda_backward(
  torch::Tensor net,
  torch::Tensor coords,
  torch::Tensor gradient,
  int radius)
{
  const int B = coords.size(0);
  const int M = coords.size(1);
  const int C = net.size(1);
  const int H = net.size(2);
  const int W = net.size(3);
  const int D = 2 * radius + 2;

  // Zero-init: the kernel accumulates into this via atomicAdd
  torch::Tensor net_gradient = torch::zeros_like(net);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(net.scalar_type(), "patchify_backward_kernel", ([&] {
    patchify_backward_kernel<scalar_t><<<BLOCKS(B * M * D * D), THREADS>>>(radius,
      gradient.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      net_gradient.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
  }));

  return { net_gradient };
}
