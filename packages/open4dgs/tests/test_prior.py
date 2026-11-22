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
    from open4dgs.prior import PriorConfig, prior_terms, static_valid

    config = PriorConfig(depth_edge_max=1.0)
    return prior_terms(depth, hit.float(), static_valid(hit.float(), prior_depth, config), prior_depth, prior_normals, _K(), step, config)


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
    from open4dgs.prior import PriorConfig, prior_terms, static_valid

    depth, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    rendered = (depth + 0.05 * _rays()[..., 0]).requires_grad_(True)
    terms = _terms(rendered, depth, normals, hit)
    (terms.depth + terms.normal).backward()
    assert rendered.grad is not None and float(rendered.grad.abs().sum()) > 0.0

    empty = torch.zeros(H, W, requires_grad=True)
    out = prior_terms(empty, torch.zeros(H, W), static_valid(torch.zeros(H, W), depth, PriorConfig()), depth, normals, _K(), 0, PriorConfig())
    assert float(out.valid_fraction) == 0.0 and float(out.depth) == 0.0
    out.depth.backward()  # differentiable zero


def _reference_terms(depth, alpha, mask, prior_depth, prior_normals, K, step, config):
    """The P1 implementation (boolean indexing, Python-side skip), kept to pin the sync-free one to it."""
    from open4dgs.prior import camera_normals, depth_edges, erode, pearson_loss

    valid = erode(mask > 0.5, config.erode_px) & (prior_depth > 0) & (alpha.detach() > config.alpha_min)
    valid = valid & ~depth_edges(prior_depth, config.depth_edge_max)
    valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False
    fraction = float(valid.float().mean())
    if fraction < config.min_valid_fraction:
        return 0.0, 0.0, 0.0
    l_depth = pearson_loss(depth[valid], prior_depth[valid])
    rendered = camera_normals(depth, K)
    normal_valid = valid
    if step >= config.contradiction_start_step:
        cos = (rendered.detach() * prior_normals).sum(dim=-1)
        normal_valid = valid & (cos >= math.cos(math.radians(config.contradiction_deg)))
    if not bool(normal_valid.any()):
        return float(l_depth), 0.0, fraction
    one_minus_cos = (1.0 - (rendered[normal_valid] * prior_normals[normal_valid]).sum(dim=-1)).clamp(min=0.0)
    if config.normal_form == "spirula":
        half = one_minus_cos / 2.0
        one_minus_cos = half + (half + 1e-8).sqrt()
    return float(l_depth), float(one_minus_cos.mean()), fraction


@pytest.mark.parametrize("form", ["cos", "spirula"])
@pytest.mark.parametrize("step", [0, 10_000])
@pytest.mark.parametrize("min_valid_fraction", [0.005, 0.5])
def test_sync_free_terms_equal_the_reference(form: str, step: int, min_valid_fraction: float) -> None:
    from open4dgs.prior import PriorConfig, prior_terms, static_valid

    depth, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    g = torch.Generator().manual_seed(1)
    rendered = depth + 0.05 * _rays()[..., 0] + 0.01 * torch.rand(H, W, generator=g)
    prior_depth = torch.where(hit, 1.3 * depth + 0.2, torch.zeros_like(depth))
    alpha = hit.float() * torch.rand(H, W, generator=g)
    config = PriorConfig(depth_edge_max=1.0, normal_form=form, min_valid_fraction=min_valid_fraction, contradiction_deg=5.0)  # type: ignore[arg-type]
    ref = _reference_terms(rendered, alpha, hit.float(), prior_depth, normals, _K(), step, config)
    new = prior_terms(rendered, alpha, static_valid(hit.float(), prior_depth, config), prior_depth, normals, _K(), step, config)
    assert (float(new.depth), float(new.normal), float(new.valid_fraction)) == pytest.approx(ref, abs=1e-6)
    if min_valid_fraction == 0.5:
        assert ref == (0.0, 0.0, 0.0)  # the skip branch is covered too
    else:
        assert ref[0] > 0.0 and ref[1] > 0.0


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


def _roi_clip(pad: float | None):
    import numpy as np

    from open4dgs.catalog_clip import ClipRoi, MultiViewClip
    from open4dgs.roi_cameras import fit_virtual_camera

    source_K = np.array([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]])
    world_T_cam = np.eye(4)
    roi = None
    K = torch.from_numpy(source_K).float()[None]
    size = (640, 480)
    if pad is not None:
        points = np.array([[0.3, 0.1, 4.0], [0.6, 0.5, 4.2], [0.4, -0.2, 3.9]], dtype=np.float32)
        camera = fit_virtual_camera(points, source_K, world_T_cam, size, pad=pad)
        roi = ClipRoi(pad=pad, max_side=None, cameras=(camera,))
        K = torch.from_numpy(camera.K).float()[None]
        size = camera.size
    return MultiViewClip(
        camera_names=("rig_01/cam_00",),
        frames=(torch.zeros(1, size[1], size[0], 3, dtype=torch.uint8),),
        valid=(torch.ones(size[1], size[0], dtype=torch.bool),),
        K=K,
        world_T_cam=torch.from_numpy(world_T_cam).float()[None],
        fps=30.0,
        frame_indices=torch.tensor([0]),
        people=None,
        roi=roi,
    )


def test_prior_layer_roi_check_accepts_its_own_grid_and_refuses_another() -> None:
    from open4dgs.prior import check_roi, roi_properties

    full, roi, wider = _roi_clip(None), _roi_clip(0.15), _roi_clip(0.3)
    names = ("rig_01/cam_00",)
    check_roi(full, names, None, "old layer")  # a layer from before ROI cameras holds real cameras
    check_roi(full, names, roi_properties(full)["roi_cameras"], "full")
    check_roi(roi, names, roi_properties(roi)["roi_cameras"], "roi")
    for clip, stored in (
        (roi, None),
        (roi, roi_properties(full)["roi_cameras"]),
        (full, roi_properties(roi)["roi_cameras"]),
        (wider, roi_properties(roi)["roi_cameras"]),
    ):
        with pytest.raises(ValueError, match="grid"):
            check_roi(clip, names, stored, "mismatch")


def test_schedule_follows_the_run_length_by_default() -> None:
    from open4dgs.prior import PriorConfig

    config = PriorConfig().at_steps(2500)  # the p2 experiment's schedule: 375 / 1000 / 1875
    assert (config.start_step, config.ramp_end_step, config.contradiction_start_step) == (375, 1000, 1875)
    assert (config.lambda_depth, config.lambda_normal) == (0.05, 0.05)
    assert PriorConfig().at_steps(5500).start_step == 825 and PriorConfig().at_steps(5500).ramp_end_step == 2200
    fixed = PriorConfig(schedule="steps", start_step=10, ramp_end_step=20, contradiction_start_step=30)
    assert fixed.at_steps(9000) == fixed


def test_as_stored_equals_the_png_round_trip() -> None:
    import cv2
    import numpy as np

    from open4dgs.prior import ClipPriors, as_stored, encode_normals

    depth, normals, hit = _sphere(torch.tensor([0.0, 0.0, 3.0]), 0.8)
    priors = ClipPriors(("rig_00/cam_00",), torch.tensor([0]), ((depth * 1.2345).half()[None],), (encode_normals(normals, hit)[None],))
    # write_prior_layer's encode + read_clip_priors' decode, without the catalog
    mm = (priors.depth[0].float() * 1000.0).round().clamp(0, 65535).numpy().astype(np.uint16)
    _, png = cv2.imencode(".png", mm[0])
    back = np.empty((1, H, W), dtype=np.float16)
    back[0] = cv2.imdecode(png, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    stored = as_stored(priors)
    assert torch.equal(stored.depth[0], torch.from_numpy(back)) and torch.equal(stored.normals[0], priors.normals[0])


def _train_config(**kwargs):  # noqa: ANN003, ANN202
    from open4dgs.apis.train import Config

    return Config(segment_id="seg", mask_catalog_url="rerun+http://pablo-dl-server:51235", mask_dataset="open4dgs-mamma", **kwargs)


def _priors_for(names):  # noqa: ANN001, ANN202
    from open4dgs.prior import ClipPriors

    return ClipPriors(tuple(names), torch.tensor([0]), tuple(torch.full((1, 2, 2), float(i), dtype=torch.float16) for i in range(len(names))), tuple(torch.zeros(1, 2, 2, 3, dtype=torch.uint8) for _ in names))


def test_prior_auto_uses_a_matching_layer_and_computes_otherwise() -> None:
    from open4dgs.apis.train import clip_priors

    clip = _roi_clip(None)
    calls: list[str] = []

    def read_ok(clip, url, dataset, segment, names):  # noqa: ANN001, ANN202
        calls.append(f"read {url} {dataset} {segment} {names}")
        return _priors_for(names)

    def read_mismatch(clip, url, dataset, segment, names):  # noqa: ANN001, ANN202
        calls.append("read")
        raise ValueError("prior layer camera rig_01/cam_00 was computed on grid 'none'")

    def read_no_dataset(clip, url, dataset, segment, names):  # noqa: ANN001, ANN202
        raise LookupError("No dataset found")

    def compute(clip, url, dataset):  # noqa: ANN001, ANN202
        calls.append(f"compute {url} {dataset}")
        return _priors_for(("a", "b", "c"))

    train, holdout, source = clip_priors(_train_config(), clip, ("a", "b"), ("c",), read=read_ok, compute=compute)
    assert source == "layer:rerun+http://pablo-dl-server:51235/open4dgs-mamma" and train.camera_names == ("a", "b")
    assert holdout is not None and holdout.camera_names == ("c",)
    assert not any(c.startswith("compute") for c in calls)

    calls.clear()
    train, holdout, source = clip_priors(_train_config(), clip, ("a", "b"), ("c",), read=read_mismatch, compute=compute)
    assert source.startswith("computed:") and calls == ["read", "compute rerun+http://pablo-dl-server:51235 open4dgs-mamma"]
    assert train.camera_names == ("a", "b") and float(train.depth[1][0, 0, 0]) == 1.0  # camera b of the computed maps
    assert holdout is not None and holdout.camera_names == ("c",)
    assert clip_priors(_train_config(), clip, ("a",), (), read=read_no_dataset, compute=compute)[2].startswith("computed:")

    with pytest.raises(ValueError, match="grid"):  # --prior layer keeps the strict behaviour
        clip_priors(_train_config(prior="layer"), clip, ("a", "b"), ("c",), read=read_mismatch, compute=compute)


def test_prior_defaults_and_location() -> None:
    from open4dgs.apis.train import Config, prior_location

    assert Config().prior == "auto"
    assert prior_location(_train_config()) == ("rerun+http://pablo-dl-server:51235", "open4dgs-mamma")
    # A source catalog that is not the local one (the read-only Spark): the local catalog and companion dataset
    spark = Config(catalog_url="rerun+http://dgx-spark:51235", dataset="dataforge-mamma-markerless-dance", segment_id="seg")
    assert prior_location(spark) == ("rerun+http://pablo-dl-server:51235", "open4dgs-mamma")
    explicit = Config(prior_catalog_url="rerun+http://elsewhere:1", prior_dataset="d")
    assert prior_location(explicit) == ("rerun+http://elsewhere:1", "d")
