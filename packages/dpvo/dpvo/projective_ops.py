"""Projective camera geometry operations for DPVO.

Implements the core camera model used by the bundle adjustment and the
recurrent update operator:

- :func:`iproj` -- Inverse projection: lifts 2-D image coordinates plus
  inverse depth to 3-D homogeneous points ``(X/Z, Y/Z, 1, 1/Z)``.
- :func:`proj` -- Forward projection: projects 3-D homogeneous points back
  to pixel coordinates via the pinhole camera model.
- :func:`transform` -- Full projective transform pipeline: backproject a
  patch from frame *i*, apply the SE3 relative pose from frame *i* to
  frame *j*, then project into frame *j*.  Optionally computes analytical
  Jacobians for the Gauss-Newton bundle adjustment.
- :func:`point_cloud` -- Backproject patches to world-space 3-D points.
- :func:`flow_mag` -- Compute optical flow magnitude between frames,
  used for keyframe selection.

The 4-vector convention for 3-D points is ``(X/Z, Y/Z, 1, 1/Z)`` --
a homogeneous representation that stores inverse depth in the last
component.  This avoids explicit depth normalisation and simplifies the
Jacobian computation.  See Sec. 3 of Teed et al. (2022).
"""

from typing import Any, Literal, overload

import torch
from jaxtyping import Float, Int
from lietorch import SE3
from torch import Tensor

MIN_DEPTH: float = 0.2
"""Minimum depth (in meters) used for validity checks.  Points closer
than this are considered unreliable or behind the camera."""


def extract_intrinsics(intrinsics: Float[Tensor, "... 4"]) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]:
    """Unpack intrinsic parameters ``(fx, fy, cx, cy)`` with spatial broadcast dims.

    Args:
        intrinsics: Camera intrinsics, last dim is ``[fx, fy, cx, cy]``.

    Returns:
        4-tuple of ``(fx, fy, cx, cy)`` each with two trailing size-1
        dims for broadcasting over spatial (patch) dimensions.
    """
    fx: Float[Tensor, "..."]
    fy: Float[Tensor, "..."]
    cx: Float[Tensor, "..."]
    cy: Float[Tensor, "..."]
    fx, fy, cx, cy = intrinsics[...,None,None,:].unbind(dim=-1)
    return (fx, fy, cx, cy)


def coords_grid(ht: int, wd: int, **kwargs: Any) -> Float[Tensor, "h w 2"]:
    """Create a pixel coordinate grid of shape ``(h, w, 2)`` with ``(x, y)`` ordering.

    Args:
        ht: Grid height (number of rows).
        wd: Grid width (number of columns).
        **kwargs: Passed to tensor creation (e.g. ``device``).

    Returns:
        Coordinate grid where ``grid[y, x] = (x, y)``.
    """
    y: Float[Tensor, "h"]
    x: Float[Tensor, "w"]
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)


def iproj(patches: Float[Tensor, "... 3 ps ps"], intrinsics: Float[Tensor, "... 4"]) -> Float[Tensor, "... ps ps 4"]:
    """Inverse projection: lift patches from image coordinates to 3-D homogeneous rays.

    Converts each patch pixel ``(u, v, d)`` (where ``d`` is inverse depth)
    to the 4-vector ``((u - cₓ) / fₓ,  (v - cᵧ) / fᵧ,  1,  d)``.

    This representation stores the point as ``(X/Z, Y/Z, 1, 1/Z)`` in
    camera coordinates, which is convenient because SE3 group actions in
    lietorch operate directly on these homogeneous coordinates.

    Args:
        patches: Patch tensor with channels ``(x, y, inverse_depth)`` and
            spatial dims ``(ps, ps)`` where ``ps`` is the patch size
            (typically 3).
        intrinsics: Camera intrinsics ``(fx, fy, cx, cy)``.

    Returns:
        4-D homogeneous points of shape ``(..., ps, ps, 4)``.
    """
    x: Float[Tensor, "... ps ps"]
    y: Float[Tensor, "... ps ps"]
    d: Float[Tensor, "... ps ps"]
    x, y, d = patches.unbind(dim=2)

    fx: Float[Tensor, "..."]
    fy: Float[Tensor, "..."]
    cx: Float[Tensor, "..."]
    cy: Float[Tensor, "..."]
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    # Normalised image coordinates: (u - cx) / fx, (v - cy) / fy
    i: Float[Tensor, "... ps ps"] = torch.ones_like(d)
    xn: Float[Tensor, "... ps ps"] = (x - cx) / fx
    yn: Float[Tensor, "... ps ps"] = (y - cy) / fy

    # Homogeneous 4-vector: (X/Z, Y/Z, 1, 1/Z)
    X: Float[Tensor, "... ps ps 4"] = torch.stack([xn, yn, i, d], dim=-1)
    return X


def proj(X: Float[Tensor, "... 4"], intrinsics: Float[Tensor, "... 4"], depth: bool = False) -> Float[Tensor, "..."]:
    """Forward projection: project 3-D homogeneous points to pixel coordinates.

    Given a point ``(X, Y, Z, W)`` in the homogeneous representation and
    intrinsics ``(fₓ, fᵧ, cₓ, cᵧ)``, computes:

    - ``d = 1 / max(Z, 0.1)``  (inverse depth, clamped for stability)
    - ``u = fₓ · d · X + cₓ``
    - ``v = fᵧ · d · Y + cᵧ``

    The depth clamping at ``Z = 0.1`` prevents division-by-zero for points
    near or behind the camera plane.

    Args:
        X: 3-D points in homogeneous coordinates ``(X, Y, Z, W)``.
        intrinsics: Camera intrinsics ``(fx, fy, cx, cy)``.
        depth: If True, return ``(u, v, d)`` including the inverse depth.

    Returns:
        Projected 2-D coordinates ``(u, v)`` or ``(u, v, d)`` if
        ``depth=True``.
    """
    Y: Float[Tensor, "..."]
    Z: Float[Tensor, "..."]
    W: Float[Tensor, "..."]
    X, Y, Z, W = X.unbind(dim=-1)

    fx: Float[Tensor, "..."]
    fy: Float[Tensor, "..."]
    cx: Float[Tensor, "..."]
    cy: Float[Tensor, "..."]
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    # Inverse depth, clamped to avoid division by zero
    d: Float[Tensor, "..."] = 1.0 / Z.clamp(min=0.1)
    x: Float[Tensor, "..."] = fx * (d * X) + cx
    y: Float[Tensor, "..."] = fy * (d * Y) + cy

    if depth:
        return torch.stack([x, y, d], dim=-1)

    return torch.stack([x, y], dim=-1)


@overload
def transform(
    poses: SE3,
    patches: Float[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float[Tensor, "1 n_frames 4"],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    depth: bool = False,
    valid: bool = False,
    *,
    jacobian: Literal[True],
    tonly: bool = False,
) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."], tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]]:
    ...


@overload
def transform(
    poses: SE3,
    patches: Float[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float[Tensor, "1 n_frames 4"],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    depth: bool = False,
    valid: bool = False,
    jacobian: bool = False,
    tonly: bool = False,
) -> Float[Tensor, "..."]:
    ...


def transform(
    poses: SE3,
    patches: Float[Tensor, "1 n_patches 3 ps ps"],
    intrinsics: Float[Tensor, "1 n_frames 4"],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    kk: Int[Tensor, "n_edges"],
    depth: bool = False,
    valid: bool = False,
    jacobian: bool = False,
    tonly: bool = False,
) -> Float[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]] | tuple[Float[Tensor, "..."], Float[Tensor, "..."], tuple[Float[Tensor, "..."], Float[Tensor, "..."], Float[Tensor, "..."]]]:
    """Full projective transform: reproject patch kk from frame ii into frame jj.

    This is the central geometric operation in DPVO.  For each measurement
    edge ``(ii[e], jj[e], kk[e])``:

    1. **Backproject** patch ``kk[e]`` using intrinsics of frame ``ii[e]``
       to get 3-D homogeneous point ``X0`` via :func:`iproj`.
    2. **Transform** ``X0`` by the relative pose ``Gjj · Gii⁻¹`` to
       get ``X1`` in frame ``jj[e]``'s coordinate system.
    3. **Project** ``X1`` into frame ``jj[e]`` using its intrinsics via
       :func:`proj`.

    When ``jacobian=True``, analytical Jacobians are computed for the
    Gauss-Newton bundle adjustment (see :func:`~dpvo.ba.BA`):

    - ``Jᵢ``: Jacobian of pixel coords w.r.t. pose ``ii`` (shape ``[1, E, 2, 6]``).
    - ``Jⱼ``: Jacobian of pixel coords w.r.t. pose ``jj`` (shape ``[1, E, 2, 6]``).
    - ``Jz``: Jacobian of pixel coords w.r.t. inverse depth (shape ``[1, E, 2, 1]``).

    These are evaluated at the center pixel of the patch (``p//2, p//2``).
    See Sec. 3.3 of Teed et al. (2022) for the BA formulation.

    Args:
        poses: SE3 poses for all frames, shape ``(1, N, 7)``.
        patches: Patch coordinates ``(x, y, inv_depth)``, shape
            ``(1, n_patches, 3, ps, ps)``.
        intrinsics: Camera intrinsics per frame ``(fx, fy, cx, cy)``,
            shape ``(1, n_frames, 4)``.
        ii: Source frame indices for each edge.
        jj: Target frame indices for each edge.
        kk: Patch indices for each edge.
        depth: If True, return 3-channel ``(u, v, inv_depth)`` coordinates.
        valid: If True, return a validity mask (``Z > 0.2``) alongside
            coordinates.
        jacobian: If True, compute and return analytical Jacobians
            ``(Ji, Jj, Jz)`` for bundle adjustment.
        tonly: If True, zero out the rotation component of the relative
            pose (translation-only transform).  Used by :func:`flow_mag`
            to decompose flow into translational and rotational components.

    Returns:
        Depending on flags:
        - Default: projected coordinates ``(1, n_edges, ps, ps, 2)``.
        - ``valid=True``: ``(coords, validity_mask)``.
        - ``jacobian=True``: ``(coords, validity_mask, (Ji, Jj, Jz))``.
    """
    # Step 1: backproject patches from frame ii into 3D
    X0: Float[Tensor, "1 n_edges ps ps 4"] = iproj(patches[:,kk], intrinsics[:,ii])

    # Step 2: compute the relative transform Gⱼ · Gᵢ⁻¹ and apply it
    Gij_result = poses[:, jj] * poses[:, ii].inv()
    assert isinstance(Gij_result, SE3)
    Gij: SE3 = Gij_result

    if tonly:
        # Zero out rotation (set quaternion to identity [0,0,0,1])
        Gij[...,3:] = torch.as_tensor([0,0,0,1], device=Gij.device)

    # Apply SE3 transform: X1 = G_ij * X0 (broadcasts over patch spatial dims)
    X1_result = Gij[:,:,None,None] * X0
    assert isinstance(X1_result, Tensor)
    X1: Float[Tensor, "1 n_edges ps ps 4"] = X1_result

    # Step 3: project transformed points into frame jj
    x1: Float[Tensor, "..."] = proj(X1, intrinsics[:,jj], depth)


    if jacobian:
        # Compute analytical Jacobians at the center pixel of each patch.
        # These are used by the Gauss-Newton solver in ba.py.
        p: int = X1.shape[2]
        # Extract center-pixel 3D coordinates
        X: Float[Tensor, "1 n_edges"]
        Y: Float[Tensor, "1 n_edges"]
        Z: Float[Tensor, "1 n_edges"]
        H: Float[Tensor, "1 n_edges"]
        X, Y, Z, H = X1[...,p//2,p//2,:].unbind(dim=-1)
        o: Float[Tensor, "1 n_edges"] = torch.zeros_like(H)
        _i: Float[Tensor, "1 n_edges"] = torch.zeros_like(H)

        fx: Float[Tensor, "1 n_edges"]
        fy: Float[Tensor, "1 n_edges"]
        cx: Float[Tensor, "1 n_edges"]
        cy: Float[Tensor, "1 n_edges"]
        fx, fy, cx, cy = intrinsics[:,jj].unbind(dim=-1)

        # Safe inverse depth for Jacobian (avoid division by zero)
        d: Float[Tensor, "1 n_edges"] = torch.zeros_like(Z)
        d[Z.abs() > 0.2] = 1.0 / Z[Z.abs() > 0.2]

        # Jₐ: Jacobian of the SE3 action on the homogeneous point X₁
        # w.r.t. the 6-DoF Lie algebra perturbation (translation, rotation).
        # For the action X′ = exp(ξ) · X on homogeneous coords (X, Y, Z, W):
        #   dX′/dξ is a 4×6 matrix per edge.
        Ja: Float[Tensor, "1 n_edges 4 6"] = torch.stack([
            H,  o,  o,  o,  Z, -Y,
            o,  H,  o, -Z,  o,  X,
            o,  o,  H,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(1, len(ii), 4, 6)

        # Jₚ: Jacobian of the pinhole projection w.r.t. 3D point (X, Y, Z, W).
        # Maps from 4D homogeneous coords to 2D pixel coords.
        Jp: Float[Tensor, "1 n_edges 2 4"] = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
        ], dim=-1).view(1, len(ii), 2, 4)

        # Jⱼ = Jₚ @ Jₐ: Jacobian of pixel coords w.r.t. target pose jj
        Jj: Float[Tensor, "1 n_edges 2 6"] = torch.matmul(Jp, Ja)
        # Jᵢ: Jacobian w.r.t. source pose ii, computed via the adjoint:
        #   d(coords)/dξᵢ = -Gij.adjT @ d(coords)/dξⱼ
        Ji: Float[Tensor, "1 n_edges 2 6"] = -Gij[:,:,None].adjT(Jj)

        # Jz: Jacobian of pixel coords w.r.t. inverse depth.
        # The inverse depth enters via the last column of the SE3 matrix
        # (the translation component acts on the W = 1/Z component).
        Jz: Float[Tensor, "1 n_edges 2 1"] = torch.matmul(Jp, Gij.matrix()[...,:,3:])

        return x1, (Z > 0.2).float(), (Ji, Jj, Jz)

    if valid:
        return x1, (X1[...,2] > 0.2).float()

    return x1


def point_cloud(poses: SE3, patches: Float[Tensor, "..."], intrinsics: Float[Tensor, "..."], ix: Int[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Generate a world-space point cloud from patches.

    Backprojects each patch into its source frame's camera coordinates
    using :func:`iproj`, then transforms to world coordinates using the
    inverse of the source frame's pose.

    Args:
        poses: SE3 poses for all frames.
        patches: Patch coordinates ``(x, y, inv_depth)``.
        intrinsics: Camera intrinsics per frame.
        ix: Source frame index for each patch.

    Returns:
        World-space homogeneous points with the same spatial layout as
        the input patches.
    """
    result = poses[:,ix,None,None].inv() * iproj(patches, intrinsics[:,ix])
    assert isinstance(result, Tensor)
    return result


def flow_mag(poses: SE3, patches: Float[Tensor, "..."], intrinsics: Float[Tensor, "..."], ii: Int[Tensor, "n_edges"], jj: Int[Tensor, "n_edges"], kk: Int[Tensor, "n_edges"], beta: float = 0.3) -> tuple[Float[Tensor, "..."], Float[Tensor, "..."]]:
    """Compute pixel flow magnitude between frames, decomposed into translation and full motion.

    Measures how much each patch moves in pixel space when reprojected
    from frame ``ii`` to frame ``jj``.  The flow is a weighted combination
    of two components:

    - ``flow1``: Full SE3 flow (translation + rotation).
    - ``flow2``: Translation-only flow (rotation zeroed out).

    The blending weight ``beta`` controls the mix.  This decomposition
    is used by the keyframe selection heuristic (:meth:`DPVO.motionmag`)
    to avoid counting pure rotation as useful parallax.  See Sec. 3.4
    of Teed et al. (2022).

    Args:
        poses: SE3 poses for all frames.
        patches: Patch coordinates.
        intrinsics: Camera intrinsics per frame.
        ii: Source frame indices.
        jj: Target frame indices.
        kk: Patch indices.
        beta: Blending weight between full flow and translation-only flow.

    Returns:
        Tuple of ``(flow, valid)`` where:
        - ``flow``: Per-edge blended flow magnitude (in pixels).
        - ``valid``: Boolean mask indicating valid reprojections (finite flow).
    """
    # Project patches from ii -> ii (identity, gives reference coordinates)
    coords0: Float[Tensor, "..."] = transform(poses, patches, intrinsics, ii, ii, kk)
    # Project patches from ii -> jj with full SE3 motion
    coords1: Float[Tensor, "..."] = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False)
    # Project patches from ii -> jj with translation only (no rotation)
    coords2: Float[Tensor, "..."] = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    flow1: Float[Tensor, "..."] = (coords1 - coords0).norm(dim=-1)
    flow2: Float[Tensor, "..."] = (coords2 - coords0).norm(dim=-1)

    flow: Float[Tensor, "..."] = beta * flow1 + (1 - beta) * flow2
    valid: Float[Tensor, "..."] = (flow1.isfinite() & flow2.isfinite()).float()

    return flow, valid


def induced_flow(
    poses: SE3,
    disps: Float[Tensor, "1 n h w"],
    intrinsics: Float[Tensor, "1 n 4"],
    ii: Int[Tensor, "n_edges"],
    jj: Int[Tensor, "n_edges"],
    tonly: bool = False,
) -> tuple[Float[Tensor, "1 n_edges h w 2"], Float[Tensor, "1 n_edges h w"]]:
    """Compute the optical flow induced by camera motion between frames.

    For each pair ``(ii[e], jj[e])``, computes the 2-D pixel flow field
    by backprojecting the disparity map of frame ``ii[e]`` to 3-D,
    transforming to frame ``jj[e]``'s coordinate system, and projecting
    back to 2-D.  The flow is the difference between the projected
    coordinates and the original pixel grid.

    This function operates on *dense* disparity maps rather than sparse
    patches, making it suitable for building co-visibility frame graphs
    during dataset construction.

    Args:
        poses: SE3 poses for all frames.
        disps: Per-frame inverse depth (disparity) maps.
        intrinsics: Per-frame camera intrinsics ``[fx, fy, cx, cy]``.
        ii: Source frame indices.
        jj: Target frame indices.
        tonly: If True, zero out the rotation component.

    Returns:
        A 2-tuple of ``(flow, valid)`` where ``flow`` has shape
        ``(1, n_edges, h, w, 2)`` and ``valid`` has shape
        ``(1, n_edges, h, w)`` with 1.0 for valid pixels.
    """
    ht: int = disps.shape[2]
    wd: int = disps.shape[3]

    y: Float[Tensor, "h w"]
    x: Float[Tensor, "h w"]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )
    coords0: Float[Tensor, "h w 2"] = torch.stack([x, y], dim=-1)

    # Backproject using disparity as inverse depth
    fx: Float[Tensor, "1 n_edges 1 1"]
    fy: Float[Tensor, "1 n_edges 1 1"]
    cx: Float[Tensor, "1 n_edges 1 1"]
    cy: Float[Tensor, "1 n_edges 1 1"]
    fx, fy, cx, cy = intrinsics[:, ii, :, None, None].unbind(dim=2)

    d: Float[Tensor, "1 n_edges h w"] = disps[:, ii]
    i_ones: Float[Tensor, "1 n_edges h w"] = torch.ones_like(d)
    xn: Float[Tensor, "1 n_edges h w"] = (x - cx) / fx
    yn: Float[Tensor, "1 n_edges h w"] = (y - cy) / fy

    X0: Float[Tensor, "1 n_edges h w 4"] = torch.stack([xn, yn, i_ones, d], dim=-1)

    # Compute relative transform
    Gij_result = poses[:, jj] * poses[:, ii].inv()
    assert isinstance(Gij_result, SE3)
    Gij: SE3 = Gij_result

    if tonly:
        Gij[..., 3:] = torch.as_tensor([0, 0, 0, 1], device=Gij.device)

    # Apply SE3 transform
    X1 = Gij[:, :, None, None] * X0
    assert isinstance(X1, Tensor)

    # Project to 2D
    X_: Float[Tensor, "1 n_edges h w"]
    Y_: Float[Tensor, "1 n_edges h w"]
    Z_: Float[Tensor, "1 n_edges h w"]
    _W: Float[Tensor, "1 n_edges h w"]
    X_, Y_, Z_, _W = X1.unbind(dim=-1)

    fx2: Float[Tensor, "1 n_edges 1 1"]
    fy2: Float[Tensor, "1 n_edges 1 1"]
    cx2: Float[Tensor, "1 n_edges 1 1"]
    cy2: Float[Tensor, "1 n_edges 1 1"]
    fx2, fy2, cx2, cy2 = intrinsics[:, jj, :, None, None].unbind(dim=2)

    d1: Float[Tensor, "1 n_edges h w"] = 1.0 / Z_.clamp(min=0.1)
    x1: Float[Tensor, "1 n_edges h w"] = fx2 * (d1 * X_) + cx2
    y1: Float[Tensor, "1 n_edges h w"] = fy2 * (d1 * Y_) + cy2

    coords1: Float[Tensor, "1 n_edges h w 2"] = torch.stack([x1, y1], dim=-1)
    valid: Float[Tensor, "1 n_edges h w"] = ((Z_ > MIN_DEPTH) & (d > 0)).float()

    return coords1 - coords0, valid
