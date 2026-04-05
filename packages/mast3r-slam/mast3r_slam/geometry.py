import lietorch
import torch
from jaxtyping import Bool, Float


def skew_sym(x: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 3 3"]:
    """Build a 3x3 skew-symmetric matrix from a 3-vector."""
    b: tuple[int, ...] = x.shape[:-1]
    x_comp: Float[torch.Tensor, "..."]
    y_comp: Float[torch.Tensor, "..."]
    z_comp: Float[torch.Tensor, "..."]
    x_comp, y_comp, z_comp = x.unbind(dim=-1)
    o: Float[torch.Tensor, "..."] = torch.zeros_like(x_comp)
    return torch.stack([o, -z_comp, y_comp, z_comp, o, -x_comp, -y_comp, x_comp, o], dim=-1).view(*b, 3, 3)


def point_to_dist(X: Float[torch.Tensor, "... 3"]) -> Float[torch.Tensor, "... 1"]:
    """Compute Euclidean distance (norm) of 3D points."""
    d: Float[torch.Tensor, "... 1"] = torch.linalg.norm(X, dim=-1, keepdim=True)
    return d


def point_to_ray_dist(
    X: Float[torch.Tensor, "... 3"],
    jacobian: bool = False,
) -> Float[torch.Tensor, "... 4"] | tuple[Float[torch.Tensor, "... 4"], Float[torch.Tensor, "... 4 3"]]:
    """Convert 3D points to ray-distance representation (unit_ray, distance).

    Returns a 4-vector [rx, ry, rz, d] where r is the unit ray and d is the distance.
    If jacobian=True, also returns the 4x3 Jacobian drd/dX.
    """
    b: tuple[int, ...] = X.shape[:-1]

    d: Float[torch.Tensor, "... 1"] = point_to_dist(X)
    d_inv: Float[torch.Tensor, "... 1"] = 1.0 / d
    r: Float[torch.Tensor, "... 3"] = d_inv * X
    rd: Float[torch.Tensor, "... 4"] = torch.cat((r, d), dim=-1)
    if not jacobian:
        return rd

    d_inv_2: Float[torch.Tensor, "... 1"] = d_inv**2
    I: Float[torch.Tensor, "... 3 3"] = torch.eye(3, device=X.device, dtype=X.dtype).repeat(*b, 1, 1)
    dr_dX: Float[torch.Tensor, "... 3 3"] = d_inv.unsqueeze(-1) * (I - d_inv_2.unsqueeze(-1) * (X.unsqueeze(-1) @ X.unsqueeze(-2)))
    dd_dX: Float[torch.Tensor, "... 1 3"] = r.unsqueeze(-2)
    drd_dX: Float[torch.Tensor, "... 4 3"] = torch.cat((dr_dX, dd_dX), dim=-2)
    return rd, drd_dX


def constrain_points_to_ray(
    img_size: tuple[int, int],
    Xs: Float[torch.Tensor, "b h w 3"],
    K: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "b h w 3"]:
    """Re-project points onto calibrated rays, preserving depth."""
    uv: Float[torch.Tensor, "b h w 2"] = get_pixel_coords(Xs.shape[0], img_size, device=Xs.device, dtype=Xs.dtype).view(*Xs.shape[:-1], 2)
    Xs = backproject(uv, Xs[..., 2:3], K)
    return Xs


def act_Sim3(
    X: lietorch.Sim3,
    pC: Float[torch.Tensor, "... 3"],
    jacobian: bool = False,
) -> Float[torch.Tensor, "... 3"] | tuple[Float[torch.Tensor, "... 3"], Float[torch.Tensor, "... 3 7"]]:
    """Apply a Sim3 transformation to 3D points.

    If jacobian=True, also returns the 3x7 Jacobian d(pW)/d(xi) w.r.t. the Sim3 tangent.
    """
    pW: Float[torch.Tensor, "... 3"] = X.act(pC)
    if not jacobian:
        return pW
    dpC_dt: Float[torch.Tensor, "... 3 3"] = torch.eye(3, device=pW.device).repeat(*pW.shape[:-1], 1, 1)
    dpC_dR: Float[torch.Tensor, "... 3 3"] = -skew_sym(pW)
    dpc_ds: Float[torch.Tensor, "... 3 1"] = pW.reshape(*pW.shape[:-1], -1, 1)
    return pW, torch.cat([dpC_dt, dpC_dR, dpc_ds], dim=-1)


def decompose_K(
    K: Float[torch.Tensor, "... 3 3"],
) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."], Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
    """Extract (fx, fy, cx, cy) from a 3x3 intrinsics matrix."""
    fx: Float[torch.Tensor, "..."] = K[..., 0, 0]
    fy: Float[torch.Tensor, "..."] = K[..., 1, 1]
    cx: Float[torch.Tensor, "..."] = K[..., 0, 2]
    cy: Float[torch.Tensor, "..."] = K[..., 1, 2]
    return fx, fy, cx, cy


def project_calib(
    P: Float[torch.Tensor, "... 3"],
    K: Float[torch.Tensor, "3 3"],
    img_size: tuple[int, int],
    jacobian: bool = False,
    border: float = 0,
    z_eps: float = 0.0,
) -> (
    tuple[Float[torch.Tensor, "... 3"], Bool[torch.Tensor, "... 1"]]
    | tuple[Float[torch.Tensor, "... 3"], Float[torch.Tensor, "... 3 3"], Bool[torch.Tensor, "... 1"]]
):
    """Project 3D points to pixel coordinates (u, v, log_z) using calibrated camera.

    Returns (pz, valid) or (pz, dpz_dP, valid) if jacobian=True.
    pz is [u, v, log(z)], valid is a boolean mask for in-frame points.
    """
    b: tuple[int, ...] = P.shape[:-1]
    K_rep: Float[torch.Tensor, "... 3 3"] = K.repeat(*b, 1, 1)

    p: Float[torch.Tensor, "... 3"] = (K_rep @ P[..., None]).squeeze(-1)
    p = p / p[..., 2:3]
    p = p[..., :2]

    u: Float[torch.Tensor, "... 1"]
    v: Float[torch.Tensor, "... 1"]
    u, v = p.split([1, 1], dim=-1)
    x: Float[torch.Tensor, "... 1"]
    y: Float[torch.Tensor, "... 1"]
    z: Float[torch.Tensor, "... 1"]
    x, y, z = P.split([1, 1, 1], dim=-1)

    valid_u: Bool[torch.Tensor, "... 1"] = (u > border) & (u < img_size[1] - 1 - border)
    valid_v: Bool[torch.Tensor, "... 1"] = (v > border) & (v < img_size[0] - 1 - border)
    valid_z: Bool[torch.Tensor, "... 1"] = z > z_eps
    valid: Bool[torch.Tensor, "... 1"] = valid_u & valid_v & valid_z

    logz: Float[torch.Tensor, "... 1"] = torch.log(z)
    invalid_z: Bool[torch.Tensor, "... 1"] = torch.logical_not(valid_z)
    logz[invalid_z] = 0.0

    pz: Float[torch.Tensor, "... 3"] = torch.cat((p, logz), dim=-1)

    if not jacobian:
        return pz, valid

    fx: Float[torch.Tensor, "..."]
    fy: Float[torch.Tensor, "..."]
    fx, fy, _, _ = decompose_K(K)
    z_inv: Float[torch.Tensor, "..."] = 1.0 / z[..., 0]
    dpz_dP: Float[torch.Tensor, "... 3 3"] = torch.zeros(*b + (3, 3), device=P.device, dtype=P.dtype)
    dpz_dP[..., 0, 0] = fx
    dpz_dP[..., 1, 1] = fy
    dpz_dP[..., 0, 2] = -fx * x[..., 0] * z_inv
    dpz_dP[..., 1, 2] = -fy * y[..., 0] * z_inv
    dpz_dP *= z_inv[..., None, None]
    dpz_dP[..., 2, 2] = z_inv
    return pz, dpz_dP, valid


def backproject(
    p: Float[torch.Tensor, "... 2"],
    z: Float[torch.Tensor, "... 1"],
    K: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "... 3"]:
    """Back-project pixel coordinates to 3D points given depth and intrinsics."""
    tmp1: Float[torch.Tensor, "..."] = (p[..., 0] - K[0, 2]) / K[0, 0]
    tmp2: Float[torch.Tensor, "..."] = (p[..., 1] - K[1, 2]) / K[1, 1]
    dP_dz: Float[torch.Tensor, "... 3 1"] = torch.empty(p.shape[:-1] + (3, 1), device=z.device, dtype=K.dtype)
    dP_dz[..., 0, 0] = tmp1
    dP_dz[..., 1, 0] = tmp2
    dP_dz[..., 2, 0] = 1.0
    P: Float[torch.Tensor, "... 3"] = torch.squeeze(z[..., None, :] * dP_dz, dim=-1)
    return P


def get_pixel_coords(
    b: int,
    img_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "b h w 2"]:
    """Generate a (b, h, w, 2) grid of pixel coordinates."""
    h: int
    w: int
    h, w = img_size
    u: torch.Tensor
    v: torch.Tensor
    u, v = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    uv: Float[torch.Tensor, "b h w 2"] = torch.stack((u, v), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    uv = uv.to(device=device, dtype=dtype)
    return uv
