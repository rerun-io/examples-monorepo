import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

import mast3r_slam.image as img_utils
from mast3r_slam.config import config

# Matching kernel backend selection: prefer Mojo (faster, no C++ build step),
# fall back to CUDA C++ if Mojo .so is not built. Both provide identical
# iter_proj() and refine_matches() APIs.
# Note: global_opt.py imports mast3r_slam._backends directly for the
# Gauss-Newton solvers which have not been ported to Mojo.
try:
    import mast3r_slam_mojo_backends as _matching_backends
except ImportError:
    from mast3r_slam import _backends as _matching_backends


def match(
    X11: Float[Tensor, "b h w 3"],
    X21: Float[Tensor, "b h w 3"],
    D11: Float[Tensor, "b h w d"],
    D21: Float[Tensor, "b h w d"],
    idx_1_to_2_init: Int[Tensor, "b hw"] | None = None,
) -> tuple[Int[Tensor, "b hw"], Bool[Tensor, "b hw 1"]]:
    """Compute dense correspondences from frame 2 to frame 1 via iterative projection.

    Args:
        X11: 3D point map of frame 1 in frame 1 coordinates.
        X21: 3D point map of frame 2 in frame 1 coordinates.
        D11: Descriptor map for frame 1.
        D21: Descriptor map for frame 2.
        idx_1_to_2_init: Optional initial linear-index correspondence guess.

    Returns:
        A tuple of (idx_1_to_2, valid_match2) where idx_1_to_2 are linear
        pixel indices and valid_match2 is a per-pixel validity mask.
    """
    idx_1_to_2: Int[Tensor, "b hw"]
    valid_match2: Bool[Tensor, "b hw 1"]
    idx_1_to_2, valid_match2 = match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init)
    return idx_1_to_2, valid_match2


def pixel_to_lin(
    p1: Int[Tensor, "... 2"],
    w: int,
) -> Int[Tensor, "..."]:
    """Convert (u, v) pixel coordinates to linear indices (u + w * v).

    Args:
        p1: Pixel coordinates with last dim (u, v).
        w: Image width.

    Returns:
        Linear indices into a flattened (h*w) array.
    """
    idx_1_to_2: Int[Tensor, "..."] = p1[..., 0] + (w * p1[..., 1])
    return idx_1_to_2


def lin_to_pixel(
    idx_1_to_2: Int[Tensor, "..."],
    w: int,
) -> Int[Tensor, "... 2"]:
    """Convert linear indices back to (u, v) pixel coordinates.

    Args:
        idx_1_to_2: Linear indices into a flattened (h*w) array.
        w: Image width.

    Returns:
        Pixel coordinates with last dim (u, v).
    """
    u: Int[Tensor, "..."] = idx_1_to_2 % w
    v: Int[Tensor, "..."] = idx_1_to_2 // w
    p: Int[Tensor, "... 2"] = torch.stack((u, v), dim=-1)
    return p


def prep_for_iter_proj(
    X11: Float[Tensor, "b h w 3"],
    X21: Float[Tensor, "b h w 3"],
    idx_1_to_2_init: Int[Tensor, "b hw"] | None,
) -> tuple[
    Float[Tensor, "b h w 9"],
    Float[Tensor, "b hw 3"],
    Float[Tensor, "b hw 2"],
]:
    """Prepare inputs for the iterative projection matching kernel.

    Builds a ray image with spatial gradients, normalises the 3D points,
    and generates initial pixel guesses.

    Args:
        X11: 3D point map of frame 1 in frame 1 coordinates.
        X21: 3D point map of frame 2 in frame 1 coordinates.
        idx_1_to_2_init: Optional initial correspondence indices; identity if None.

    Returns:
        A tuple of (rays_with_grad_img, pts3d_norm, p_init).
    """
    b: int
    h: int
    w: int
    b, h, w, _ = X11.shape
    device: torch.device = X11.device

    # Ray image
    rays_img: Float[Tensor, "b h w 3"] = F.normalize(X11, dim=-1)
    rays_img = rays_img.permute(0, 3, 1, 2)  # (b,3,h,w)
    gx_img: Float[Tensor, "b 3 h w"]
    gy_img: Float[Tensor, "b 3 h w"]
    gx_img, gy_img = img_utils.img_gradient(rays_img)
    rays_with_grad_img: Float[Tensor, "b 9 h w"] = torch.cat((rays_img, gx_img, gy_img), dim=1)
    rays_with_grad_img = rays_with_grad_img.permute(
        0, 2, 3, 1
    ).contiguous()  # (b,h,w,c)

    # 3D points to project
    X21_vec: Float[Tensor, "b hw 3"] = X21.view(b, -1, 3)
    pts3d_norm: Float[Tensor, "b hw 3"] = F.normalize(X21_vec, dim=-1)

    # Initial guesses of projections
    if idx_1_to_2_init is None:
        # Reset to identity mapping
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    p_init_int: Int[Tensor, "b hw 2"] = lin_to_pixel(idx_1_to_2_init, w)
    p_init: Float[Tensor, "b hw 2"] = p_init_int.float()

    return rays_with_grad_img, pts3d_norm, p_init


def match_iterative_proj(
    X11: Float[Tensor, "b h w 3"],
    X21: Float[Tensor, "b h w 3"],
    D11: Float[Tensor, "b h w d"],
    D21: Float[Tensor, "b h w d"],
    idx_1_to_2_init: Int[Tensor, "b hw"] | None = None,
) -> tuple[Int[Tensor, "b hw"], Bool[Tensor, "b hw 1"]]:
    """Run iterative-projection matching followed by optional descriptor refinement.

    Args:
        X11: 3D point map of frame 1 in frame 1 coordinates.
        X21: 3D point map of frame 2 in frame 1 coordinates.
        D11: Descriptor map for frame 1.
        D21: Descriptor map for frame 2.
        idx_1_to_2_init: Optional initial correspondence guess.

    Returns:
        A tuple of (idx_1_to_2, valid_proj2) with linear correspondence indices
        and a boolean validity mask.
    """
    cfg: dict = config["matching"]
    b: int
    h: int
    w: int
    b, h, w = X21.shape[:3]
    device: torch.device = X11.device

    rays_with_grad_img: Float[Tensor, "b h w 9"]
    pts3d_norm: Float[Tensor, "b hw 3"]
    p_init: Float[Tensor, "b hw 2"]
    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(
        X11, X21, idx_1_to_2_init
    )
    p1: Float[Tensor, "b hw 2"]
    valid_proj2: Bool[Tensor, "b hw"]
    p1, valid_proj2 = _matching_backends.iter_proj(
        rays_with_grad_img,
        pts3d_norm,
        p_init,
        cfg["max_iter"],
        cfg["lambda_init"],
        cfg["convergence_thresh"],
    )
    p1 = p1.long()

    # Check for occlusion based on distances
    batch_inds: Int[Tensor, "b hw"] = torch.arange(b, device=device)[:, None].repeat(1, h * w)
    dists2: Float[Tensor, "b h w"] = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21, dim=-1
    )
    valid_dists2: Bool[Tensor, "b hw"] = (dists2 < cfg["dist_thresh"]).view(b, -1)
    valid_proj2 = valid_proj2 & valid_dists2

    if cfg["radius"] > 0:
        (p1,) = _matching_backends.refine_matches(
            D11.half(),
            D21.view(b, h * w, -1).half(),
            p1,
            cfg["radius"],
            cfg["dilation_max"],
        )

    # Convert to linear index
    idx_1_to_2: Int[Tensor, "b hw"] = pixel_to_lin(p1, w)

    return idx_1_to_2, valid_proj2.unsqueeze(-1)
