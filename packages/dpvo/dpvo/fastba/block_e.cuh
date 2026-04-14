/*
 * block_e.cuh -- Header for efficient E-block computation in global BA.
 *
 * Part of DPV-SLAM (Deep Patch Visual SLAM, Lipson et al., 2024).
 *
 * The EfficentE class pre-computes lookup tables for the cross-term
 * (pose × depth) Hessian block E in the Schur complement:
 *
 *     S = B - E · C⁻¹ · Eᵀ
 *
 * Instead of materializing the dense E matrix (6N × M), which is
 * prohibitively large for global BA with many patches, this class
 * stores E in a compressed form (per unique (frame, frame) pair)
 * and provides GPU kernels for the three operations needed:
 *
 *   - E · Q · Eᵀ   (computeEQEt)  -- for the Schur complement
 *   - E · v         (computeEv)    -- for the right-hand side
 *   - Eᵀ · v        (computeEtv)   -- for the depth update
 *
 * where Q = diag(1 / (C + λ)) is the inverse depth Hessian.
 */

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

const auto mdtype = torch::dtype(torch::kFloat32).device(torch::kCUDA);

typedef float mtype;

/**
 * Efficient E-block for global bundle adjustment.
 *
 * Stores the pose-depth cross-term E in a compressed per-(frame, patch)
 * lookup table instead of materializing the dense 6N x M matrix.
 * The lookup table is indexed by unique (source_frame, target_frame)
 * pairs and patch-within-frame index (k % patches_per_frame).
 */
class EfficentE
{
private:
    torch::Tensor block_index_tensor, index_tensor, patch_to_ku;
    const int t0;

public:
    const int ppf;
    torch::Tensor E_lookup, ij_xself;

    /** Construct from edge indices, building all lookup tables on GPU. */
    EfficentE(const torch::Tensor &ii, const torch::Tensor &jj, const torch::Tensor &ku, const int patches_per_frame, const int t0);

    /** Default constructor (empty, zero-sized). */
    EfficentE();

    /** Compute S_contribution = E · Q · Eᵀ (Schur complement pose-pose block). */
    torch::Tensor computeEQEt(const int N, const torch::Tensor &Q) const;

    /** Compute E · vec (project depth-space vector to pose-space). */
    torch::Tensor computeEv(const int N, const torch::Tensor &vec) const;

    /** Compute Eᵀ · vec (project pose-space vector to depth-space). */
    torch::Tensor computeEtv(const int M, const torch::Tensor &vec) const;
};
