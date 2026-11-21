"""triangulate_gated with the previous tick's points: cameras whose masks slid onto another person are dropped first."""

from __future__ import annotations

import math

import torch

OFFSET = 0.6 * torch.tensor([math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0])
"""0.6 m between two people, at 22.5 deg so it lies along no ring camera's optical axis (each camera sees >= 0.23 m)."""


def _ring(n_cams: int, radius: float = 4.0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Cameras on a ring at 1.2 m height, looking at the origin; 1000 px focal, 1024x752 image."""
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
        ks.append(torch.tensor([[1000.0, 0.0, 512.0], [0.0, 1000.0, 376.0], [0.0, 0.0, 1.0]]))
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
    other = person + OFFSET  # the person they bumped into
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


def test_a_prior_lagging_a_moving_majority_loses_the_vote() -> None:
    """Six cameras see the person who moved 0.6 m; two still see someone at the old position (thermo #703 probe 1)."""
    from mamma.fitting.triangulation import triangulate_gated

    generator = torch.Generator().manual_seed(2)
    old = torch.rand(40, 3, generator=generator) * torch.tensor([0.4, 0.3, 1.7])
    moved = old + OFFSET
    ks, w2cs = _ring(8)
    pts2d = [_project(old if c in (0, 4) else moved, ks[c], w2cs[c]) for c in range(8)]
    vis = [torch.ones(40) for _ in range(8)]

    _, points3d, valid = triangulate_gated(pts2d, ks, w2cs, vis, prior=(old, torch.ones(40, dtype=torch.bool)))

    # Not locked onto the stale position (0.6 m off); the residual is the unseeded gate's own (a camera at 35 px stays in).
    assert bool(valid.all()) and float((points3d - moved).norm(dim=1).mean()) < 0.1


def test_two_seed_cameras_without_a_shared_landmark_fall_back() -> None:
    """The prior's two survivors see disjoint landmarks: no seeded point is valid, so the all-camera gate wins (probe 2)."""
    from mamma.fitting.triangulation import triangulate_gated

    generator = torch.Generator().manual_seed(3)
    old = torch.rand(40, 3, generator=generator) * torch.tensor([0.4, 0.3, 1.7])
    moved = old + OFFSET * (0.8 / 0.6)
    ks, w2cs = _ring(8)
    pts2d = [_project(old if c in (0, 4) else moved, ks[c], w2cs[c]) for c in range(8)]
    vis = [torch.ones(40) for _ in range(8)]
    vis[0][20:] = 0.0
    vis[4][:20] = 0.0

    _, points3d, valid = triangulate_gated(pts2d, ks, w2cs, vis, prior=(old, torch.ones(40, dtype=torch.bool)))

    assert int(valid.sum()) == 40 and float((points3d - moved).norm(dim=1).mean()) < 0.1


def test_the_prior_tolerance_does_not_depend_on_the_engine_resolution() -> None:
    """Halving the image and the intrinsics keeps the same camera choice (probe 5)."""
    from mamma.fitting.triangulation import triangulate_gated

    generator = torch.Generator().manual_seed(4)
    person = torch.rand(40, 3, generator=generator) * torch.tensor([0.4, 0.3, 1.7])
    other = person + OFFSET
    ks, w2cs = _ring(8)
    half = torch.diag(torch.tensor([0.5, 0.5, 1.0]))
    dropped: list[set[int]] = []
    for scale, size in ((torch.eye(3), (1024, 752)), (half, (512, 376))):
        ks_s = [scale @ k for k in ks]
        pts2d = [_project(other if c in {1, 2, 5, 6} else person, ks_s[c], w2cs[c]) for c in range(8)]
        prior = (person + torch.tensor([0.05, 0.0, 0.0]), torch.ones(40, dtype=torch.bool))
        gated, _, _ = triangulate_gated(
            pts2d, ks_s, w2cs, [torch.ones(40) for _ in range(8)], reproj_thresh_px=40.0 * size[0] / 1024, prior=prior, image_wh_per_cam=[size] * 8
        )
        dropped.append({c for c in range(8) if float(gated[c].sum()) == 0.0})
    assert dropped[0] == dropped[1] == {1, 2, 5, 6}


def test_a_rejected_camera_does_not_pull_the_fit() -> None:
    """Moving a rejected camera's detections leaves the reprojection loss unchanged (thermo #703 probe 3)."""
    from mamma.fitting.losses import reprojection_loss
    from mamma.fitting.stage import drop_rejected_observations

    generator = torch.Generator().manual_seed(5)
    person = torch.rand(40, 3, generator=generator) * torch.tensor([0.4, 0.3, 1.7])
    ks, w2cs = _ring(8)
    vis = [torch.ones(40) for _ in range(8)]
    gated = [v.clone() for v in vis]
    gated[3] = torch.zeros(40)  # the gate rejected camera 3
    losses: list[float] = []
    for shift in (0.0, 80.0):
        pts2d = [_project(person, ks[c], w2cs[c]) for c in range(8)]
        pts2d[3] = pts2d[3] + torch.tensor([shift, 0.0, 0.0])
        fit_pts = drop_rejected_observations(pts2d, vis, gated)
        losses.append(float(reprojection_loss([p[None] for p in fit_pts], person[None], ks, w2cs, [g[None] for g in gated])))
    assert losses[0] == losses[1] == 0.0


def test_a_person_missing_for_a_tick_loses_the_prior() -> None:
    """Observed -> not fitted -> reacquired starts like a fresh person, without the old prior (probe 4)."""
    from mamma.fitting.stage import FittingStage

    stage: FittingStage = FittingStage.__new__(FittingStage)
    stage._previous_points = {0: (torch.zeros(4, 3), torch.ones(4, dtype=torch.bool)), 1: (torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))}
    from mamma.landmarks.estimator import LandmarkResult

    seen = LandmarkResult(obj_id=0, joints2d=torch.zeros(4, 3), visibility=torch.ones(4), contact=torch.zeros(4), floor_contact=torch.zeros(4))
    stage._expire_priors([{0: seen, 1: seen}, {0: seen}])  # person 1 is seen by one camera only: not fitted
    assert set(stage._previous_points) == {0}
