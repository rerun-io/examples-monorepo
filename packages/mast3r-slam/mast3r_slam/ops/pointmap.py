"""Point-map fusion: how a frame's canonical point map absorbs a new observation (``TrackingConfig.filtering_mode``)."""

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from mast3r_slam.types import FilteringMode, Frame, TrackingConfig


def _cartesian_to_spherical(P: Float[Tensor, "hw 3"]) -> Float[Tensor, "hw 3"]:
    r: Float[Tensor, "hw 1"] = torch.linalg.norm(P, dim=-1, keepdim=True)
    x, y, z = torch.tensor_split(P, 3, dim=-1)
    return torch.cat((r, torch.atan2(y, x), torch.acos(z / r)), dim=-1)


def _spherical_to_cartesian(spherical: Float[Tensor, "hw 3"]) -> Float[Tensor, "hw 3"]:
    r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
    return torch.cat((r * torch.sin(theta) * torch.cos(phi), r * torch.sin(theta) * torch.sin(phi), r * torch.cos(theta)), dim=-1)


def update_pointmap(frame: Frame, X: Float[Tensor, "hw 3"], C: Float[Tensor, "hw 1"], cfg: TrackingConfig) -> None:
    """Absorb a new observation into ``frame``'s canonical point map with ``cfg.filtering_mode``."""
    mode: FilteringMode = cfg.filtering_mode
    score = torch.median if cfg.filtering_score == "median" else torch.mean
    if frame.N == 0:  # the first observation is always accepted
        frame.X_canon, frame.C, frame.N, frame.N_updates = X.clone(), C.clone(), 1, 1
        if mode == "best_score":
            frame.score = score(C)
        return
    assert frame.C is not None and frame.X_canon is not None
    if mode == "first" and frame.N_updates == 1 or mode == "recent":  # first: only the second observation
        frame.X_canon, frame.C, frame.N = X.clone(), C.clone(), 1
    elif mode == "best_score":
        new_score: Float[Tensor, ""] = score(C)
        assert frame.score is not None
        if new_score > frame.score:
            frame.X_canon, frame.C, frame.N, frame.score = X.clone(), C.clone(), 1, new_score
    elif mode == "indep_conf":  # per pixel: keep the more confident point
        new_mask: Bool[Tensor, "hw 1"] = C > frame.C
        frame.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
        frame.C[new_mask] = C[new_mask]
        frame.N = 1
    elif mode == "weighted_pointmap":  # confidence-weighted running mean in Cartesian space
        frame.X_canon = ((frame.C * frame.X_canon) + (C * X)) / (frame.C + C)
        frame.C, frame.N = frame.C + C, frame.N + 1
    elif mode == "weighted_spherical":  # the same in spherical (r, phi, theta) space
        spherical = ((frame.C * _cartesian_to_spherical(frame.X_canon)) + (C * _cartesian_to_spherical(X))) / (frame.C + C)
        frame.X_canon = _spherical_to_cartesian(spherical)
        frame.C, frame.N = frame.C + C, frame.N + 1
    frame.N_updates += 1
