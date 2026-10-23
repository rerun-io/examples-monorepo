"""mamma's triangulation policy over the shared DLT kernel (``simplecv.ops.triangulate.triangulate_weighted``).

MammaNet gives every landmark a log-variance; a view's least-squares weight is ``1 / sigma^2`` with the pixel sigma
``exp(0.5 * log_variance) * crop / 2`` clamped to a sane range. Points seen by fewer than two views come back as finite
zeros with ``valid`` False, so the floor-contact gate (which reads heights) stays finite. ``triangulate_gated`` drops a
view whose landmarks belong to the other person this tick (a contact-moment mask swap).
"""

import torch
from jaxtyping import Bool, Float32
from simplecv.ops.triangulate import triangulate_weighted
from torch import Tensor

CROP_PX: float = 512.0
"""MammaNet crop size that scales the log-variance to pixels."""
SIGMA_CLAMP_PX: tuple[float, float] = (1.0, 24.0)


def triangulate_landmarks(
    xy: Float32[Tensor, "v n 2"], log_variance: Float32[Tensor, "v n"], visibility: Float32[Tensor, "v n"], P: Float32[Tensor, "v 3 4"], vis_thresh: float = 0.5
) -> tuple[Float32[Tensor, "n 3"], Bool[Tensor, " n"]]:
    """Sigma-weighted DLT of one person's landmarks from every view where ``visibility > vis_thresh`` (an all-zero ``xy`` row = not seen)."""
    seen: Bool[Tensor, "v n"] = (visibility > vis_thresh) & (xy.abs().sum(dim=-1) > 0)
    sigma_px: Float32[Tensor, "v n"] = (torch.exp(0.5 * log_variance) * (CROP_PX / 2.0)).clamp(*SIGMA_CLAMP_PX)
    weight: Float32[Tensor, "v n"] = torch.where(seen, 1.0 / (sigma_px * sigma_px), 0.0)
    points, valid = triangulate_weighted(xy.double(), weight.double(), P.double())
    return torch.where(valid[:, None], points, 0.0).float(), valid


def triangulate_gated(
    xy: Float32[Tensor, "v n 2"],
    log_variance: Float32[Tensor, "v n"],
    visibility: Float32[Tensor, "v n"],
    P: Float32[Tensor, "v 3 4"],
    reproj_thresh_px: float = 40.0,
    min_views: int = 2,
    vis_thresh: float = 0.5,
) -> tuple[Float32[Tensor, "v n"], Float32[Tensor, "n 3"], Bool[Tensor, " n"]]:
    """Triangulate, then drop the worst view while its confident landmarks reproject more than ``reproj_thresh_px`` away
    on average (at least 4 of them) and more than ``min_views`` views remain; re-triangulate after each drop.

    Returns the gated visibility (dropped views zeroed: they neither triangulate nor pull the fit), the points and their validity.
    A no-op when the views agree, so single-person fits are unchanged.
    """
    gated: Float32[Tensor, "v n"] = visibility.clone()
    active: list[int] = list(range(xy.shape[0]))
    points, valid = triangulate_landmarks(xy, log_variance, gated, P, vis_thresh)
    while len(active) > min_views:
        homogeneous: Float32[Tensor, "n 4"] = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        worst, worst_px = -1, reproj_thresh_px
        for v in active:
            confident: Bool[Tensor, " n"] = (gated[v] > vis_thresh) & valid
            if int(confident.sum().item()) < 4:
                continue
            uvw: Float32[Tensor, "n 3"] = homogeneous @ P[v].T
            error_px: float = float((uvw[:, :2] / (uvw[:, 2:3].abs() + 1e-6) - xy[v]).norm(dim=-1)[confident].mean().item())
            if error_px > worst_px:
                worst, worst_px = v, error_px
        if worst < 0:
            break
        active.remove(worst)
        gated[worst] = 0.0
        points, valid = triangulate_landmarks(xy, log_variance, gated, P, vis_thresh)
    return gated, points, valid
