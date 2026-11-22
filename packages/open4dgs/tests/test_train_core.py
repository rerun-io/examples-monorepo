"""Body-surface sampling and velocity, metric functions and camera conversion on synthetic inputs (CPU)."""

import math

import pytest
import torch


def _tetra() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return vertices, faces


def test_samples_are_fixed_body_points_and_velocity_is_per_video_frame() -> None:
    from open4dgs.body_init import central_velocity, mesh_keyframes, sample_surface, surface_points

    vertices, faces = _tetra()
    samples = sample_surface(vertices, faces, 500, seed=7)
    again = sample_surface(vertices, faces, 500, seed=7)
    assert torch.equal(samples.faces, again.faces) and torch.equal(samples.barycentric, again.barycentric)
    assert torch.all(samples.barycentric >= 0) and torch.allclose(samples.barycentric.sum(-1), torch.ones(500))

    # The mesh translates by (0.01, 0, 0) per video frame and rotates 90 degrees about z over 12 frames; meshes on every 3rd frame.
    present = torch.zeros(13, dtype=torch.bool)
    present[::3] = True
    keyframes = mesh_keyframes(present, keyframe_step=3)
    assert keyframes.tolist() == [0, 3, 6, 9, 12]
    shift = torch.tensor([0.01, 0.0, 0.0])
    positions = torch.stack([surface_points(vertices + shift * float(f), faces, samples) for f in keyframes.tolist()])
    velocity = central_velocity(positions, keyframes)
    assert torch.allclose(velocity, shift.expand_as(velocity), atol=1e-6)
    # Same body point at every keyframe: offsets from the moved mesh equal the frame-0 offsets.
    assert torch.allclose(positions[4] - shift * 12.0, positions[0], atol=1e-6)

    angle = torch.tensor(math.pi / 2 / 12)
    rot = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    spun = torch.stack([surface_points(vertices @ torch.linalg.matrix_power(rot, f).T, faces, samples) for f in keyframes.tolist()])
    v = central_velocity(spun, keyframes)
    expected = (spun[2] - spun[0]) / 6.0
    assert torch.allclose(v[1], expected, atol=1e-6)


def test_mesh_keyframes_starts_at_first_mesh() -> None:
    from open4dgs.body_init import mesh_keyframes

    present = torch.zeros(10, dtype=torch.bool)
    present[1::3] = True
    assert mesh_keyframes(present, 3).tolist() == [1, 4, 7]
    full = torch.ones(10, dtype=torch.bool)
    assert mesh_keyframes(full, 3).tolist() == [0, 3, 6, 9]


def test_interpolate_meshes_fills_off_stride_frames() -> None:
    from open4dgs.body_init import interpolate_meshes

    verts = torch.zeros(7, 2, 3)
    present = torch.zeros(7, dtype=torch.bool)
    present[[0, 3, 6]] = True
    verts[3] = 3.0
    verts[6] = 6.0
    out = interpolate_meshes(verts, present)
    assert torch.allclose(out[:, 0, 0], torch.arange(7.0))


def test_psnr_and_masked_psnr_known_values() -> None:
    from open4dgs.evaluate import mask_box, masked_psnr, psnr, ssim

    target = torch.zeros(20, 30, 3)
    pred = torch.full((20, 30, 3), 0.1)
    assert psnr(pred, target) == pytest.approx(20.0)
    mask = torch.zeros(20, 30, dtype=torch.bool)
    mask[5:10, 10:20] = True
    pred2 = target.clone()
    pred2[mask] = 0.01
    assert masked_psnr(pred2, target, mask) == pytest.approx(40.0)
    assert psnr(target, target) == math.inf
    assert ssim(target + 0.5, target + 0.5) == pytest.approx(1.0, abs=1e-5)
    assert mask_box(mask, pad=0.1) == (9, 5, 21, 10)  # x pad 1 px, y pad round(0.5) = 0 px
    assert mask_box(torch.zeros(4, 4, dtype=torch.bool)) is None


def test_orbit_poses_ring_and_look_at() -> None:
    from open4dgs.evaluate import look_at, orbit_poses

    up = torch.tensor([0.0, 0.0, 1.0])
    centre = torch.tensor([1.0, 2.0, 0.0])
    poses = torch.stack([look_at(centre + torch.tensor([3.0 * math.cos(a), 3.0 * math.sin(a), 1.5]), centre, up) for a in (0.0, 2.0, 4.0)])
    ring = orbit_poses(poses, centre, 8)
    offsets = ring[:, :3, 3] - centre
    assert torch.allclose(offsets[:, 2], torch.full((8,), 1.5), atol=1e-5)
    assert torch.allclose(offsets[:, :2].norm(dim=-1), torch.full((8,), 3.0), atol=1e-5)
    # z axis (forward) points at the centre; -y (up in image) has a positive world-up component.
    forward = ring[:, :3, 2]
    assert torch.allclose(forward, -offsets / offsets.norm(dim=-1, keepdim=True), atol=1e-5)
    assert torch.all(-ring[:, 2, 1] > 0)


def test_camera_config_viewmats_are_world_to_camera() -> None:
    pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")
    from open4dgs.evaluate import look_at
    from open4dgs.trigs_adapter import camera_config

    world_T_cam = torch.stack([look_at(torch.tensor([2.0, 0.0, 1.0]), torch.zeros(3), torch.tensor([0.0, 0.0, 1.0])), torch.eye(4)])
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]).expand(2, 3, 3)
    cameras = camera_config(world_T_cam, K, [(100, 80), (60, 120)], torch.device("cpu"))
    assert torch.allclose(cameras.viewmats @ world_T_cam, torch.eye(4).expand(2, 4, 4), atol=1e-6)
    assert cameras.widths.tolist() == [100, 60]
    # The look-at target projects to the principal point of camera 0.
    p = cameras.viewmats[0] @ torch.tensor([0.0, 0.0, 0.0, 1.0])
    uv = (cameras.Ks[0] @ p[:3]) / p[2]
    assert torch.allclose(uv[:2], torch.tensor([50.0, 40.0]), atol=1e-4)


def test_crop_raw_counts_silhouette_errors_that_crop_masked_hides() -> None:
    from open4dgs.evaluate import baseline_renders, score_frame

    gt = torch.full((40, 40, 3), 0.8)  # background 0.8 everywhere; the person is the 20x20 square at rows/cols 10..29
    mask = torch.zeros(40, 40)
    mask[10:30, 10:30] = 1.0
    gt[10:30, 10:30] = 0.5
    # The render is the person exactly plus a stray 2x20 strip (value 0.4) just outside it, inside the padded box.
    render = gt * mask[..., None]
    render[30:31, 10:30] = 0.4
    alpha = mask.clone()
    alpha[30:31, 10:30] = 1.0
    row = score_frame("cam", 7, render, alpha, gt, mask, lpips=lambda _p, _t: 0.0)
    # Box = 20x20 padded by round(0.05 * 20) = 1 px per side -> 22x22 = 484 px; the strip is 20 px of error 0.4 on 3 channels.
    assert row.crop_masked_psnr == math.inf
    assert row.crop_raw_psnr == pytest.approx(-10.0 * math.log10(20 * 0.16 / 484))
    assert row.full_psnr == pytest.approx(-10.0 * math.log10(20 * 0.16 / 1600))
    assert row.fg_psnr == math.inf  # the person's own pixels are exact
    assert row.alpha_iou == pytest.approx(400 / 420)
    assert row.crop_raw_lpips == 0.0 and row.frame_index == 7

    floors = baseline_renders(gt, mask)
    black = score_frame("cam", 7, *floors["black"], gt, mask, lpips=lambda _p, _t: 0.0)
    assert black.crop_raw_psnr == pytest.approx(-10.0 * math.log10(400 * 0.25 / 484))
    assert black.alpha_iou == 0.0
    mean = score_frame("cam", 7, *floors["mean"], gt, mask, lpips=lambda _p, _t: 0.0)
    assert mean.crop_raw_psnr == math.inf and mean.alpha_iou == 1.0  # a flat person: its mean colour is exact

    empty = score_frame("cam", 0, render, alpha, gt, torch.zeros(40, 40), lpips=lambda _p, _t: 0.0)
    assert math.isnan(empty.crop_raw_psnr) and math.isnan(empty.alpha_iou)


def test_window_keyframes_cover_the_last_frame() -> None:
    from open4dgs.body_init import central_velocity, sample_surface, surface_points, window_keyframes

    vertices, faces = _tetra()
    shift = torch.tensor([0.01, 0.0, 0.0])
    track = torch.stack([vertices + shift * float(f) for f in range(60)])  # 60 frames, 0.01 per video frame
    # MAMMA-like: meshes every 3rd frame, the window ends 2 frames past the last mesh (57) -> 59 is extrapolated.
    present = torch.zeros(60, dtype=torch.bool)
    present[::3] = True
    keyframes, verts, extrapolated = window_keyframes(track, present, 3)
    assert extrapolated and keyframes.tolist() == [*range(0, 58, 3), 59]
    assert torch.allclose(verts[-1], track[59], atol=1e-6)  # linear motion: extrapolation is exact
    samples = sample_surface(vertices, faces, 50, seed=0)
    positions = torch.stack([surface_points(v, faces, samples) for v in verts])
    velocity = central_velocity(positions, keyframes)
    assert torch.allclose(velocity[..., 0], torch.full(velocity.shape[:2], 0.01), atol=1e-5)  # irregular last interval of 2
    # 4DAnyone-like: a mesh on every frame, 30 frames, grid 0..27 -> frame 29 comes from its own mesh.
    keyframes, verts, extrapolated = window_keyframes(track[:30], torch.ones(30, dtype=torch.bool), 3)
    assert not extrapolated and keyframes.tolist() == [*range(0, 28, 3), 29] and torch.equal(verts[-1], track[29])
    # On the grid (31 frames): unchanged.
    keyframes, _, extrapolated = window_keyframes(track[:31], torch.ones(31, dtype=torch.bool), 3)
    assert not extrapolated and keyframes.tolist() == list(range(0, 31, 3))
