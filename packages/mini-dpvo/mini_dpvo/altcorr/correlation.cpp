/*
 * correlation.cpp -- PyBind11 module exposing CUDA correlation and patchify
 *                    kernels to Python via PyTorch's C++ extension mechanism.
 *
 * Part of the DPVO (Deep Patch Visual Odometry) pipeline (Teed et al., 2022).
 *
 * This file is the "glue" between Python and the CUDA kernels defined in
 * correlation_kernel.cu.  It declares thin C++ wrappers that simply forward
 * their arguments to the corresponding `*_cuda_*` functions, then registers
 * those wrappers as Python-callable functions through the PYBIND11_MODULE
 * macro.
 *
 * Two families of operations are exposed:
 *
 *   1. **Correlation** (corr_forward / corr_backward)
 *      Computes a local dot-product correlation volume between source-frame
 *      feature patches and a target-frame feature map inside a search
 *      neighborhood of radius `radius`.  This is the key building block for
 *      matching patches across frames.
 *
 *   2. **Patchify** (patchify_forward / patchify_backward)
 *      Extracts small square patches from a dense feature map at given (x, y)
 *      coordinates.  Each patch has size D x D where D = 2*radius + 2 (the
 *      extra column/row is needed for bilinear interpolation performed later
 *      on the host side).
 *
 * The Python module is imported as `altcorr_cuda` (name set at build time by
 * setup.py / JIT compilation) and exposes:
 *   - forward(fmap1, fmap2, coords, ii, jj, radius)  -> [corr]
 *   - backward(fmap1, fmap2, coords, ii, jj, corr_grad, radius) -> [fmap1_grad, fmap2_grad]
 *   - patchify_forward(net, coords, radius)  -> [patches]
 *   - patchify_backward(net, coords, gradient, radius) -> [net_gradient]
 */

#include <torch/extension.h>
#include <vector>

/* -------------------------------------------------------------------------
 * Forward declarations of CUDA entry-point functions.
 * These are implemented in correlation_kernel.cu and compiled by nvcc.
 * The signatures use torch::Tensor so they integrate naturally with
 * PyTorch's autograd and GPU memory management.
 * ---------------------------------------------------------------------- */

std::vector<torch::Tensor> corr_cuda_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor ii,
    torch::Tensor jj,
    int radius);

std::vector<torch::Tensor> corr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor ii,
  torch::Tensor jj,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> patchify_cuda_forward(
    torch::Tensor net, torch::Tensor coords, int radius);

std::vector<torch::Tensor> patchify_cuda_backward(
    torch::Tensor net, torch::Tensor coords, torch::Tensor gradient, int radius);

/* -------------------------------------------------------------------------
 * Thin C++ wrappers.
 * These exist only to provide a clean boundary between the pybind layer
 * and the CUDA translation unit.  No logic happens here.
 * ---------------------------------------------------------------------- */

/**
 * Compute the local correlation volume between patches in fmap1 and a
 * search neighborhood in fmap2.
 *
 * @param fmap1   Source feature map patches, shape [B, N_frames, C, H, W].
 *                `C` is the feature dimension (typically 128).
 * @param fmap2   Target feature map, same layout as fmap1.
 * @param coords  Reprojected (x, y) coordinates for each patch grid point
 *                in the target frame, shape [B, M, 2, H_patch, W_patch].
 *                These define where in fmap2 each patch grid point lands.
 * @param ii      Index into fmap1 (source frame index per edge), shape [M].
 * @param jj      Index into fmap2 (target frame index per edge), shape [M].
 * @param radius  Search radius.  The kernel evaluates a (2*radius+1) x
 *                (2*radius+1) neighborhood around each reprojected point.
 *                Internally an extra border pixel is included (D = 2*R+2)
 *                to enable bilinear interpolation on the host side.
 *
 * @return A vector containing a single tensor: the correlation volume with
 *         shape [B, M, (2R+1), (2R+1), H_patch, W_patch] after bilinear
 *         interpolation is applied on the host side.
 */
std::vector<torch::Tensor> corr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor ii,
    torch::Tensor jj, int radius) {
  return corr_cuda_forward(fmap1, fmap2, coords, ii, jj, radius);
}

/**
 * Backward pass for the correlation operation.
 *
 * Distributes the gradient of the correlation volume back to both feature
 * maps (fmap1 and fmap2) using atomicAdd for thread-safe accumulation.
 *
 * @param fmap1      Source feature map (same as forward).
 * @param fmap2      Target feature map (same as forward).
 * @param coords     Reprojected coordinates (same as forward).
 * @param ii         Source frame indices (same as forward).
 * @param jj         Target frame indices (same as forward).
 * @param corr_grad  Gradient of the loss w.r.t. the correlation output,
 *                   shape [B, M, (2R+1), (2R+1), H_patch, W_patch].
 * @param radius     Search radius (same as forward).
 *
 * @return A vector of two tensors: [fmap1_grad, fmap2_grad], each the same
 *         shape as the corresponding input feature map.
 */
std::vector<torch::Tensor> corr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor corr_grad, int radius) {
  return corr_cuda_backward(fmap1, fmap2, coords, ii, jj, corr_grad, radius);
}

/**
 * Extract D x D feature patches from a dense feature map at specified
 * (x, y) locations.
 *
 * The extracted patch is centered on floor(x), floor(y) and extends
 * from -radius to +radius+1 in each direction (total D = 2*radius + 2).
 * The extra pixel allows the caller to perform bilinear interpolation
 * between the four nearest integer-grid patches.
 *
 * @param net     Dense feature map, shape [B, C, H, W].
 * @param coords  Patch center coordinates, shape [B, M, 2] where
 *                coords[b][m] = (x, y) in feature-map pixel units.
 * @param radius  Half-size of the patch (actual size is 2*radius + 2).
 *
 * @return A vector containing a single tensor: the extracted patches with
 *         shape [B, M, C, D, D].
 */
std::vector<torch::Tensor> patchify_forward(
    torch::Tensor net, torch::Tensor coords, int radius) {
  return patchify_cuda_forward(net, coords, radius);
}

/**
 * Backward pass for patchify: scatters the gradient from patches back
 * onto the dense feature map using atomicAdd.
 *
 * @param net       Dense feature map (same as forward, used for shape info).
 * @param coords    Patch center coordinates (same as forward).
 * @param gradient  Gradient of the loss w.r.t. the extracted patches,
 *                  shape [B, M, C, D, D].
 * @param radius    Half-size of the patch (same as forward).
 *
 * @return A vector containing a single tensor: the gradient w.r.t. the
 *         dense feature map `net`, same shape as `net` [B, C, H, W].
 */
std::vector<torch::Tensor> patchify_backward(
    torch::Tensor net, torch::Tensor coords, torch::Tensor gradient, int radius) {
  return patchify_cuda_backward(net, coords, gradient, radius);
}

/* -------------------------------------------------------------------------
 * Python module registration via PyBind11.
 *
 * TORCH_EXTENSION_NAME is a macro defined at build time (typically set to
 * "altcorr_cuda" or similar).  The resulting .so can be imported in Python
 * as:
 *     import altcorr_cuda
 *     corr = altcorr_cuda.forward(fmap1, fmap2, coords, ii, jj, radius)
 * ---------------------------------------------------------------------- */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &corr_forward, "CORR forward");
  m.def("backward", &corr_backward, "CORR backward");

  m.def("patchify_forward", &patchify_forward, "PATCHIFY forward");
  m.def("patchify_backward", &patchify_backward, "PATCHIFY backward");
}
