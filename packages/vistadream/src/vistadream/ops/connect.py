from copy import deepcopy

import cv2
import numpy as np
from jaxtyping import Bool, Float

from vistadream.ops.gs.basic import Frame, Gaussian_Scene
from vistadream.ops.utils import dpt2xyz, transform_points


class Connect_Tool:
    """
    Aligns depth maps from different sources using scale and shift transformations.

    This tool is used to align inpainted depth maps with rendered depth maps from the
    Gaussian scene to ensure depth consistency across different estimation methods.
    Uses linear regression to find optimal scale and shift parameters.
    """

    def __init__(self) -> None:
        """Initialize the connection tool."""
        pass

    def _align_scale_shift_numpy(self, pred: np.ndarray, target: np.ndarray):
        """
        Compute scale and shift parameters to align predicted depth to target depth.

        Uses linear regression (polyfit) to find the best linear transformation:
        aligned_pred = pred * scale + shift

        Args:
            pred: Predicted depth values (e.g., from depth estimation model)
            target: Target depth values (e.g., from rendered Gaussian scene)

        Returns:
            tuple: (scale, shift) parameters for alignment

        Algorithm:
            1. Filter valid depth correspondences (target > 0, pred < 199)
            2. Use linear regression if sufficient samples (>10 points)
            3. Fall back to median ratio if regression fails (negative scale)
            4. Return identity transform if insufficient data
        """
        # Filter valid depth correspondences for alignment
        mask = (target > 0) & (pred < 199)  # 199: max valid depth threshold
        target_mask = target[mask]
        pred_mask = pred[mask]

        if np.sum(mask) > 10:  # Minimum sample size for reliable estimation
            # Linear regression: target = pred * scale + shift
            scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
            if scale < 0:  # Fallback if regression produces invalid scale
                # Use median ratio scaling with zero shift
                scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
                shift = 0
        else:
            # Identity transform if insufficient data
            scale = 1
            shift = 0
        return scale, shift

    def __call__(self, render_dpt, inpaint_dpt, inpaint_msk):
        """
        Align inpainted depth map to rendered depth map using scale and shift.

        Args:
            render_dpt: Depth map rendered from Gaussian scene
            inpaint_dpt: Depth map from inpainting/estimation process
            inpaint_msk: Binary mask indicating inpainted regions (True = inpainted)

        Returns:
            np.ndarray: Aligned inpainted depth map

        Process:
            1. Extract non-inpainted areas for alignment reference
            2. Compute scale/shift from overlapping valid regions
            3. Apply transformation to entire inpainted depth map
        """
        if np.sum(inpaint_msk > 0.5) < 1.0:  # No inpainted areas to align
            return render_dpt

        # Extract overlapping regions for alignment calculation
        render_dpt_valid = render_dpt[~inpaint_msk]  # Non-inpainted rendered depth
        inpaint_dpt_valid = inpaint_dpt[~inpaint_msk]  # Non-inpainted estimated depth

        # Compute alignment parameters and apply transformation
        scale, shift = self._align_scale_shift_numpy(inpaint_dpt_valid, render_dpt_valid)
        inpaint_dpt = inpaint_dpt * scale + shift
        return inpaint_dpt


class Smooth_Connect_Tool:
    """
    Advanced depth alignment tool that provides smooth transitions between rendered
    and inpainted depth regions using iterative diffusion-based refinement.

    This tool extends basic depth alignment with spatial smoothing to avoid sharp
    discontinuities at the boundaries between rendered and inpainted areas.
    Implements the method from https://arxiv.org/pdf/2311.13384
    """

    def __init__(self) -> None:
        """Initialize the smooth connection tool with a basic alignment component."""
        self.coarse_align = Connect_Tool()

    def _coarse_alignment(self, render_dpt, ipaint_dpt, ipaint_msk):
        """
        Perform initial coarse alignment using basic scale and shift transformation.

        Args:
            render_dpt: Depth map rendered from Gaussian scene
            ipaint_dpt: Depth map from inpainting process
            ipaint_msk: Binary mask indicating inpainted regions

        Returns:
            np.ndarray: Coarsely aligned inpainted depth map
        """
        # determine the scale and shift of inpaint_dpt to coarsely align it to render_dpt
        inpaint_dpt = self.coarse_align(render_dpt, ipaint_dpt, ipaint_msk)
        return inpaint_dpt

    def _refine_movements(self, render_dpt, ipaint_dpt, ipaint_msk):
        """
        Refine depth alignment using iterative diffusion to create smooth transitions.

        This method implements spatial smoothing by:
        1. Computing depth adjustments in non-inpainted regions
        2. Iteratively diffusing these adjustments across the entire image
        3. Creating smooth transitions at inpaint boundaries

        Reference: https://arxiv.org/pdf/2311.13384

        Args:
            render_dpt: Depth map rendered from Gaussian scene
            ipaint_dpt: Depth map from inpainting process
            ipaint_msk: Binary mask indicating inpainted regions

        Returns:
            np.ndarray: Smoothly refined depth map with seamless transitions

        Algorithm:
            - Computes depth corrections in known (non-inpainted) areas
            - Uses iterative Gaussian blurring (100 iterations, 15x15 kernel)
            - Propagates corrections into inpainted regions smoothly
            - Maintains known depth values while smoothing boundaries
        """
        # Follow https://arxiv.org/pdf/2311.13384

        # Convert mask to binary and get image dimensions
        ipaint_msk = ipaint_msk > 0.5
        H, W = ipaint_msk.shape[0:2]

        # Compute depth adjustments in non-inpainted (known) areas
        keep_render_dpt = render_dpt[~ipaint_msk]  # Known rendered depths
        keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]  # Known inpainted depths
        keep_adjust_dpt = keep_render_dpt - keep_ipaint_dpt  # Required corrections

        # Iterative diffusion refinement to smooth transitions
        complete_adjust = np.zeros_like(ipaint_dpt)
        for _i in range(100):  # 100 iterations for stable convergence
            # Reset known corrections (boundary conditions)
            complete_adjust[~ipaint_msk] = keep_adjust_dpt
            # Diffuse corrections spatially using Gaussian blur
            complete_adjust = cv2.blur(complete_adjust, (15, 15))  # 15x15 kernel

        # Apply smoothed corrections to inpainted depth
        ipaint_dpt = ipaint_dpt + complete_adjust
        return ipaint_dpt

    def _affine_dpt_to_GS(
        self,
        render_dpt: Float[np.ndarray, "H W"],
        inpaint_dpt: Float[np.ndarray, "H W"],
        inpaint_msk: Bool[np.ndarray, "H W"],
    ) -> Float[np.ndarray, "H W"]:
        """
        Full two-stage depth alignment: coarse alignment + smooth refinement.

        Args:
            render_dpt: Depth map rendered from Gaussian scene
            inpaint_dpt: Depth map from inpainting process
            inpaint_msk: Binary mask indicating inpainted regions

        Returns:
            np.ndarray: Fully aligned and smoothed depth map
        """
        if np.sum(inpaint_msk > 0.5) < 1.0:  # No inpainted areas
            return render_dpt
        # Two-stage alignment process
        inpaint_dpt = self._coarse_alignment(render_dpt, inpaint_dpt, inpaint_msk)
        inpaint_dpt = self._refine_movements(render_dpt, inpaint_dpt, inpaint_msk)
        return inpaint_dpt

    def _scale_dpt_to_GS(self, render_dpt, inpaint_dpt, inpaint_msk):
        """
        Apply only the smooth refinement step without initial coarse alignment.

        Used when the depth maps are already roughly aligned and only smoothing
        is needed at the boundaries.

        Args:
            render_dpt: Depth map rendered from Gaussian scene
            inpaint_dpt: Depth map from inpainting process
            inpaint_msk: Binary mask indicating inpainted regions

        Returns:
            np.ndarray: Smoothed depth map with refined boundaries
        """
        if np.sum(inpaint_msk > 0.5) < 1.0:  # No inpainted areas
            return render_dpt
        # Apply only smooth refinement (skip coarse alignment)
        inpaint_dpt = self._refine_movements(render_dpt, inpaint_dpt, inpaint_msk)
        return inpaint_dpt


class Occlusion_Removal:
    """
    Performs occlusion culling for 3D scene reconstruction by removing inpainted points
    that are occluded when viewed from previously reconstructed camera viewpoints.

    This class prevents "floating" or occluded geometry from being added to the Gaussian
    scene by testing the visibility of newly inpainted 3D points against existing frames.
    """

    def __init__(self) -> None:
        """Initialize the occlusion removal tool."""
        pass

    def __call__(self, scene: Gaussian_Scene, frame: Frame):
        """
        Remove occluded points from the inpaint mask of a frame.

        This method performs multi-view occlusion testing by:
        1. Converting inpainted depth pixels to 3D world coordinates
        2. Projecting these points to all existing camera views
        3. Comparing depths to detect occlusions
        4. Removing occluded pixels from the inpaint mask

        Args:
            scene: The Gaussian scene containing existing frames
            frame: The current frame with inpainted areas to validate

        Returns:
            frame: The input frame with updated inpaint mask (occluded pixels removed)

        Algorithm Details:
            - Only tests points in the inpaint mask (newly added geometry)
            - Uses perspective projection to map 3D points to 2D camera views
            - Applies frustum culling to only test visible projections
            - Uses depth tolerance of ~6.7% to handle estimation noise
            - Points are considered occluded if existing depth is significantly closer
        """
        # first get xyz of the newly added frame
        xyz = dpt2xyz(frame.dpt, frame.intrinsic)
        # we only check newly added areas
        xyz = xyz[frame.inpaint]
        # move these xyzs to world coor system
        inv_extrinsic = np.linalg.inv(frame.cam_T_world)
        xyz = transform_points(xyz, inv_extrinsic)
        # we will add which pixels to the gaussian scene
        msk = np.ones_like(xyz[..., 0])  # Start with all points visible

        # project the xyzs to already built frames
        for former_frame in scene.frames:
            # Transform 3D world points to camera coordinate system
            xyz_camera = transform_points(deepcopy(xyz), former_frame.cam_T_world)
            # Project 3D camera points to homogeneous 2D coordinates (uvz)
            uvz_camera = np.einsum("ab,pb->pa", former_frame.intrinsic, xyz_camera)
            # Perform perspective division to get 2D pixel coords (uv) and depth (d)
            uv, d = uvz_camera[..., :2] / uvz_camera[..., -1:], uvz_camera[..., -1]

            # Frustum culling: only test points that project within camera bounds
            valid_msk = (
                (uv[..., 0] > 0)
                & (uv[..., 0] < former_frame.W)
                & (uv[..., 1] > 0)
                & (uv[..., 1] < former_frame.H)
                & (d > 1e-2)  # Minimum depth threshold (1cm) to avoid numerical issues
            )
            valid_idx = np.where(valid_msk)[0]
            uv, d = uv[valid_idx].astype(np.uint32), d[valid_idx]

            # Occlusion test: compare projected depth with existing depth at same pixel
            # If existing depth is significantly closer, the new point is occluded
            compare_d = former_frame.dpt[uv[:, 1], uv[:, 0]]  # Existing depth at projection
            # Tolerance threshold: ~6.7% of average depth to handle estimation noise
            remove_msk = (compare_d - d) > (d + compare_d) / 2.0 / 15.0

            # Mark occluded points as invalid
            invalid_idx = valid_idx[remove_msk]
            msk[invalid_idx] = 0.0

        # Update frame inpaint mask: remove occluded pixels from inpainting
        # Convert back from flattened 3D points to 2D pixel coordinates
        inpaint_idx_v, inpaint_idx_u = np.where(frame.inpaint)
        inpaint_idx_v = inpaint_idx_v[msk < 0.5]  # Select occluded points
        inpaint_idx_u = inpaint_idx_u[msk < 0.5]
        frame.inpaint[inpaint_idx_v, inpaint_idx_u] = False  # Remove from inpaint mask
        return frame
