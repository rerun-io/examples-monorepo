"""MoGe-2 prior loss: sign conventions on analytic geometry, Pearson invariance, gates, schedule (CPU, no model)."""

import math

import pytest
import torch

pytest.importorskip("gsplat", reason="depth_to_normal comes from gsplat (open4dgs envs)")

H, W, F_PX = 64, 80, 70.0


def _K() -> torch.Tensor:
    return torch.tensor([[F_PX, 0.0, (W - 1) / 2], [0.0, F_PX, (H - 1) / 2], [0.0, 0.0, 1.0]])


def _rays() -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    K = _K()
    return torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], dim=-1)


def _plane_depth(normal: torch.Tensor, offset: float) -> torch.Tensor:
    """z-depth of the plane {X : n.X = offset}, camera frame."""
    return offset / (_rays() @ normal)


def _sphere(centre: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """z-depth, outward (camera-facing) unit normals and hit mask of a sphere, analytically."""
    rays = _rays()
    d = rays / rays.norm(dim=-1, keepdim=True)
    b = (d * centre).sum(-1)
    disc = b**2 - (centre @ centre - radius**2)
    hit = disc > 0
    t = b - disc.clamp(min=0).sqrt()
    points = d * t[..., None]
    normals = (points - centre) / radius
    depth = torch.where(hit, points[..., 2], torch.zeros_like(t))
    return depth, normals, hit


def test_fronto_parallel_plane_faces_the_camera() -> None:
    from open4dgs.prior import camera_normals

    n = camera_normals(torch.full((H, W), 2.0), _K())
    inner = n[1:-1, 1:-1]
    assert torch.allclose(inner, torch.tensor([0.0, 0.0, -1.0]).expand_as(inner), atol=1e-5)
    assert float(n[0].abs().sum()) == 0.0  # 1-px zero border


def test_tilted_plane_normal_matches_moge_rdf_convention() -> None:
    from open4dgs.prior import camera_normals

    # Plane tilted about y: the surface normal pointing back at the camera is -(sin a, 0, cos a) (MoGe RDF: faces camera).
    a = math.radians(30.0)
    facing = torch.tensor([-math.sin(a), 0.0, -math.cos(a)])
    depth = _plane_depth(-facing, 2.0)
    n = camera_normals(depth, _K())[2:-2, 2:-2]
    assert float((n * facing).sum(-1).min()) > 0.9999


def test_sphere_depth_normals_agree_with_analytic_normals() -> None:
    from open4dgs.prior import camera_normals, erode

    depth, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    n = camera_normals(depth, _K())
    inner = erode(hit, 2)
    cos = (n[inner] * normals[inner]).sum(-1)
    assert float(cos.median()) > 0.999
    assert float(normals[inner][:, 2].max()) < 0.0  # visible normals face the camera (-z)


def test_normal_png_round_trip() -> None:
    from open4dgs.prior import decode_normals, encode_normals

    _, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    decoded, valid = decode_normals(encode_normals(normals, hit))
    assert torch.equal(valid, hit)
    assert float((decoded[hit] * normals[hit]).sum(-1).min()) > 0.9998  # < 1.2 deg quantisation


def test_pearson_is_invariant_to_scale_and_shift() -> None:
    from open4dgs.prior import pearson_loss

    g = torch.Generator().manual_seed(0)
    d = torch.rand(500, generator=g) + 1.0
    assert float(pearson_loss(d, d)) == pytest.approx(0.0, abs=1e-5)
    assert float(pearson_loss(3.7 * d + 0.4, d)) == pytest.approx(0.0, abs=1e-5)
    assert float(pearson_loss(-d, d)) == pytest.approx(2.0, abs=1e-5)


def _terms(depth: torch.Tensor, prior_depth: torch.Tensor, prior_normals: torch.Tensor, hit: torch.Tensor, step: int = 10_000):
    from open4dgs.prior import PriorConfig, prior_terms

    return prior_terms(depth, hit.float(), hit.float(), prior_depth, prior_normals, _K(), step, PriorConfig(depth_edge_max=1.0))


def test_loss_is_zero_for_true_geometry_and_grows_with_tilt() -> None:
    depth, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    # MoGe depth with a wrong metric scale and shift: still the same shape.
    prior_depth = torch.where(hit, 1.3 * depth + 0.2, torch.zeros_like(depth))
    true = _terms(depth, prior_depth, normals, hit)
    assert float(true.depth) < 1e-4
    assert float(true.normal) < 2e-3

    plane_normal = torch.tensor([0.0, 0.0, 1.0])
    previous_depth, previous_normal = -1.0, -1.0
    for deg in (10.0, 25.0, 40.0):
        a = math.radians(deg)
        tilted = _plane_depth(torch.tensor([math.sin(a), 0.0, math.cos(a)]), 2.5)
        flat = _plane_depth(plane_normal, 2.5)
        # Rendered geometry = sphere + a tilted ramp (a wrong shape): both terms must grow with the tilt.
        rendered = depth + (tilted - flat)
        terms = _terms(rendered, prior_depth, normals, hit, step=0)
        assert float(terms.depth) > previous_depth
        assert float(terms.normal) > previous_normal
        previous_depth, previous_normal = float(terms.depth), float(terms.normal)
    assert previous_normal > 0.05


def test_terms_are_differentiable_and_empty_set_gives_zero() -> None:
    from open4dgs.prior import PriorConfig, prior_terms

    depth, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    rendered = (depth + 0.05 * _rays()[..., 0]).requires_grad_(True)
    terms = _terms(rendered, depth, normals, hit)
    (terms.depth + terms.normal).backward()
    assert rendered.grad is not None and float(rendered.grad.abs().sum()) > 0.0

    empty = torch.zeros(H, W, requires_grad=True)
    out = prior_terms(empty, torch.zeros(H, W), torch.zeros(H, W), depth, normals, _K(), 0, PriorConfig())
    assert out.valid_fraction == 0.0 and float(out.depth) == 0.0
    out.depth.backward()  # differentiable zero


def test_gates_drop_low_alpha_mask_border_and_depth_steps() -> None:
    from open4dgs.prior import depth_edges, erode

    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[10:30, 10:40] = True
    eroded = erode(mask, 3)
    assert int(eroded.sum()) == (20 - 6) * (30 - 6)
    step = torch.full((H, W), 2.0)
    step[:, W // 2 :] = 2.5  # log step 0.22 > 0.05
    edges = depth_edges(step, 0.05)
    assert bool(edges[:, W // 2 - 2 : W // 2 + 2].all()) and not bool(edges[:, : W // 2 - 3].any())


def test_ramp_schedule() -> None:
    from open4dgs.prior import PriorConfig

    config = PriorConfig(start_step=500, ramp_end_step=1500)
    assert config.ramp(0) == 0.0 and config.ramp(499) == 0.0
    assert config.ramp(1000) == pytest.approx(0.5)
    assert config.ramp(1500) == 1.0 and config.ramp(9000) == 1.0


def test_hook_skips_cameras_without_maps_and_logs_terms() -> None:
    from open4dgs.prior import ClipPriors, MonoPriorLoss, PriorConfig, encode_normals

    depth, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    priors = ClipPriors(("rig_00/cam_00",), torch.tensor([0]), (depth.half()[None],), (encode_normals(normals, hit)[None],))
    loss = MonoPriorLoss(
        PriorConfig(start_step=0, ramp_end_step=0, log_every=1, depth_edge_max=1.0), priors, (hit.float()[None],), (1,), _K().expand(2, 3, 3)
    )
    info = {"render_depth": (depth + 0.1 * _rays()[..., 0])[None, ..., None].requires_grad_(True)}
    alphas = hit.float()[None, ..., None]
    skipped = loss(step=0, frame_idx=0, cam_idx=0, colors=torch.zeros(1), alphas=alphas, info=info, cameras=None)
    assert float(skipped) == 0.0
    value = loss(step=0, frame_idx=0, cam_idx=1, colors=torch.zeros(1), alphas=alphas, info=info, cameras=None)
    assert float(value) > 0.0
    loss.finish()
    assert set(loss.rows) == {"prior/depth", "prior/normal", "prior/valid_fraction", "prior/weight"}


def test_rasterized_plane_depth_and_geometry_metric() -> None:
    from open4dgs.evaluate import geometry_frame, rasterize_depth

    # A tilted quad at z = 2 + 0.3 x, two triangles, covering the whole image: the z-buffer equals the analytic plane depth.
    a = math.atan(0.3)
    plane = _plane_depth(torch.tensor([-math.sin(a), 0.0, math.cos(a)]), 2.0 * math.cos(a))
    corners = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]])
    verts = torch.cat([corners, (2.0 + 0.3 * corners[:, :1])], dim=1)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    depth = rasterize_depth(verts, faces, _K(), (W, H))
    assert bool((depth > 0).all())
    assert float((depth - plane).abs().max()) < 1e-4
    ones = torch.ones(H, W)
    same = geometry_frame("c", 0, plane, depth, ones, _K())
    assert same.median_abs_cm < 0.01 and same.within_3cm == 1.0 and same.median_normal_deg < 0.5
    off = geometry_frame("c", 0, plane + 0.05, depth, ones, _K())
    assert off.median_abs_cm == pytest.approx(5.0, abs=0.01) and off.within_3cm == 0.0
