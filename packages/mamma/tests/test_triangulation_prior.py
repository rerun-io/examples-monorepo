"""triangulate_gated with the previous tick's points: cameras whose masks slid onto another person are dropped first."""

from __future__ import annotations

import math

import torch


def _ring(n_cams: int, radius: float = 4.0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Cameras on a ring at 1.2 m height, looking at the origin; 500 px focal, 1024x752 image."""
    ks: list[torch.Tensor] = []
    w2cs: list[torch.Tensor] = []
    for i in range(n_cams):
        a: float = 2.0 * math.pi * i / n_cams
        eye = torch.tensor([radius * math.cos(a), radius * math.sin(a), 1.2])
        z = -eye / eye.norm()
        x = torch.linalg.cross(z, torch.tensor([0.0, 0.0, 1.0]))
        x = x / x.norm()
        y = torch.linalg.cross(z, x)
        r = torch.stack([x, y, z])
        w2c = torch.eye(4)
        w2c[:3, :3] = r
        w2c[:3, 3] = -r @ eye
        ks.append(torch.tensor([[500.0, 0.0, 512.0], [0.0, 500.0, 376.0], [0.0, 0.0, 1.0]]))
        w2cs.append(w2c)
    return ks, w2cs


def _project(points: torch.Tensor, k: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    uv = cam @ k.T
    return torch.cat([uv[:, :2] / uv[:, 2:3], torch.full((len(points), 1), -6.0)], dim=1)  # log-variance: confident


def test_previous_points_drop_cameras_on_another_person() -> None:
    from mamma.fitting.triangulation import triangulate_gated

    generator = torch.Generator().manual_seed(0)
    person = torch.rand(40, 3, generator=generator) * torch.tensor([0.4, 0.3, 1.7])
    other = person + torch.tensor([0.6, 0.0, 0.0])  # the person they bumped into
    ks, w2cs = _ring(8)
    swapped: set[int] = {1, 2, 5, 6}  # half the cameras track the other person
    pts2d = [_project(other if c in swapped else person, ks[c], w2cs[c]) for c in range(8)]
    vis = [torch.ones(40) for _ in range(8)]

    gated, points3d, valid = triangulate_gated(pts2d, ks, w2cs, vis, prior=(person + 0.01, torch.ones(40, dtype=torch.bool)))

    assert bool(valid.all())
    assert float((points3d - person).norm(dim=1).max()) < 1e-3
    assert {c for c in range(8) if float(gated[c].sum()) == 0.0} == swapped


def test_prior_far_from_every_camera_is_ignored() -> None:
    from mamma.fitting.triangulation import triangulate_gated

    generator = torch.Generator().manual_seed(1)
    person = torch.rand(40, 3, generator=generator) * torch.tensor([0.4, 0.3, 1.7])
    ks, w2cs = _ring(6)
    pts2d = [_project(person, ks[c], w2cs[c]) for c in range(6)]
    vis = [torch.ones(40) for _ in range(6)]

    gated, points3d, _ = triangulate_gated(pts2d, ks, w2cs, vis, prior=(person + 3.0, torch.ones(40, dtype=torch.bool)))

    assert all(float(g.sum()) == 40.0 for g in gated)  # a stale prior (person reappeared elsewhere) drops nobody
    assert float((points3d - person).norm(dim=1).max()) < 1e-3
