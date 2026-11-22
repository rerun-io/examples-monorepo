"""Init cloud and crop boxes for a person with no body mesh: SelfCap's triangulated joints plus MoGe-2 depth.

``joint_box_points``: the points whose projected box is the BiRefNet crop (the joints, plus head-top proxies because
the detector loses the head under the Quest headset).

``moge_lift_init`` (init option (a)): per training camera and keyframe, the masked MoGe-2 metric depth is aligned to
the triangulated joints by a robust affine fit ``z = a d + b`` on the joints' depths, unprojected with the calibrated
pinhole, carved by the other training cameras' masks (a point that lands on background in another camera that sees
it is dropped), coloured from its pixel, and given the velocity of its nearest joint (world metres per video frame).
"""

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from torch import Tensor

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.init_cloud import InitCloud

COCO_BODY: tuple[int, ...] = tuple(range(5, 17))
"""COCO-133 shoulders .. ankles (0-4, the face, sit under the headset)."""
COCO_HANDS_FEET: tuple[int, ...] = tuple(range(17, 23)) + tuple(range(91, 133))
HEAD_TOP_M: float = 0.32
"""Height of the head-top proxy above the shoulder midpoint (world Y up)."""


def joint_box_points(joints: Float32[ndarray, "t j 3"], up: Float64[ndarray, "3"]) -> Float32[ndarray, "t k 3"]:
    """Body, hand and foot joints plus three head proxies (shoulder midpoint raised by 0.16 / 0.32 m, and +-0.1 m sideways)."""
    keep: list[int] = list(COCO_BODY) + list(COCO_HANDS_FEET)
    shoulders: Float32[ndarray, "t 3"] = 0.5 * (joints[:, 5] + joints[:, 6])
    side: Float32[ndarray, "t 3"] = joints[:, 6] - joints[:, 5]
    side = side / np.maximum(np.linalg.norm(side, axis=-1, keepdims=True), 1e-6)
    lift: Float32[ndarray, "3"] = up.astype(np.float32)
    head: list[Float32[ndarray, "t 3"]] = [
        shoulders + 0.5 * HEAD_TOP_M * lift,
        shoulders + HEAD_TOP_M * lift + 0.1 * side,
        shoulders + HEAD_TOP_M * lift - 0.1 * side,
    ]
    return np.concatenate([joints[:, keep], np.stack(head, axis=1)], axis=1)


def robust_affine(depth_moge: Float64[ndarray, " n"], depth_joint: Float64[ndarray, " n"], iterations: int = 10) -> tuple[float, float]:
    """``a, b`` minimising a Huber-weighted (5 cm) ``|a d + b - z|``; falls back to scale only (``b = 0``) with fewer than 4 pairs
    or a non-positive scale."""
    if len(depth_moge) < 4:
        scale: float = float(np.median(depth_joint / np.maximum(depth_moge, 1e-6))) if len(depth_moge) else 1.0
        return scale, 0.0
    weights: Float64[ndarray, " n"] = np.ones_like(depth_moge)
    a, b = 1.0, 0.0
    for _ in range(iterations):
        A: Float64[ndarray, "n 2"] = np.stack([depth_moge, np.ones_like(depth_moge)], axis=1) * weights[:, None]
        a, b = (float(v) for v in np.linalg.lstsq(A, depth_joint * weights, rcond=None)[0])
        residual: Float64[ndarray, " n"] = np.abs(a * depth_moge + b - depth_joint)
        weights = np.sqrt(np.where(residual < 0.05, 1.0, 0.05 / np.maximum(residual, 1e-9)))
    if a <= 0.0:
        return float(np.median(depth_joint / np.maximum(depth_moge, 1e-6))), 0.0
    return a, b


def _project(points: Float32[Tensor, "n 3"], world_T_cam: Float32[Tensor, "4 4"], K: Float32[Tensor, "3 3"]) -> tuple[Tensor, Tensor]:
    cam_T_world: Tensor = torch.linalg.inv(world_T_cam.double()).float().to(points.device)
    p: Tensor = points @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
    uv: Tensor = (p @ K.to(points.device).T)[:, :2] / p[:, 2:3].clamp_min(1e-6)
    return uv, p[:, 2]


def _sample(image: Tensor, uv: Tensor, size: tuple[int, int]) -> Tensor:
    """Bilinear sample of a (c, h, w) image at pixel ``uv`` (centres at integers); zero outside."""
    width, height = size
    grid: Tensor = torch.stack([(uv[:, 0] + 0.5) / width * 2 - 1, (uv[:, 1] + 0.5) / height * 2 - 1], dim=-1)
    return F.grid_sample(image[None].float(), grid[None, None], mode="bilinear", align_corners=False, padding_mode="zeros")[0, :, 0].T


def moge_lift_init(
    clip: MultiViewClip,
    train_names: tuple[str, ...],
    masks: tuple[Float32[Tensor, "frames h w"], ...],
    depths: dict[str, Tensor],
    joints_per_camera: Float32[ndarray, "cams frames j 3"],
    *,
    dataset: str,
    segment_id: str,
    keyframe_step: int = 3,
    points_per_view: int = 12_000,
    carve_below: float = 0.3,
    occlusion_margin_m: float = 0.15,
    fit_scope: Literal["camera", "keyframe"] = "camera",
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[InitCloud, dict[str, list[tuple[float, float, int]]]]:
    """The MoGe-2 lifted init (module docstring); also returns, per camera, ``(a, b, n_joints)`` of each keyframe's fit, then the
    pooled per-camera fit (``n_joints = -1``). ``fit_scope``: ``camera`` uses the pooled fit (the joints span ~0.5 m of depth, so a
    single keyframe's slope is ill-conditioned), ``keyframe`` the per-keyframe one. A point is carved when it lands on
    background in another training camera, unless that camera's aligned depth there is nearer by ``occlusion_margin_m``
    (an occluder such as the table: the visual hull alone removes legs hidden in one view).

    ``depths[name]`` is the camera's (frames, h, w) MoGe-2 z-depth on the clip grid (0 = invalid);
    ``joints_per_camera[c, f]`` are the joints at camera ``c``'s own time of frame ``f``.
    """
    dev: torch.device = device if device is not None else torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cams: list[int] = [clip.camera_names.index(name) for name in train_names]
    num_frames: int = int(clip.frame_indices.shape[0])
    keyframes: list[int] = list(range(0, num_frames, keyframe_step))
    body: list[int] = list(COCO_BODY) + list(COCO_HANDS_FEET)
    fits: dict[str, list[tuple[float, float, int]]] = {name: [] for name in train_names}
    rows_p, rows_v, rows_c, rows_f = [], [], [], []
    # Pass 1: joint-depth pairs per camera and keyframe, fitted per keyframe and pooled per camera.
    pairs: dict[tuple[int, int], tuple[ndarray, ndarray]] = {}
    for frame in keyframes:
        for cam in cams:
            name: str = clip.camera_names[cam]
            width, height = clip.sizes[cam]
            depth: Tensor = depths[name][frame].float().to(dev)
            eroded: Tensor = -F.max_pool2d(-(masks[cam][frame].to(dev) > 0.5).float()[None, None], 5, 1, 2)[0, 0] > 0.5
            joints: Tensor = torch.from_numpy(joints_per_camera[cam, frame][body]).to(dev)
            finite: Tensor = torch.isfinite(joints).all(dim=-1)
            uv, z = _project(joints[finite], clip.world_T_cam[cam], clip.K[cam])
            d_at: Tensor = _sample(depth[None], uv, (width, height))[:, 0]
            m_at: Tensor = _sample(eroded.float()[None], uv, (width, height))[:, 0]
            usable: Tensor = (z > 0.1) & (d_at > 0.1) & (m_at > 0.99)
            pairs[(cam, frame)] = (d_at[usable].double().cpu().numpy(), z[usable].double().cpu().numpy())
    affine: dict[tuple[int, int], tuple[float, float]] = {}
    for cam in cams:
        pooled = robust_affine(np.concatenate([pairs[(cam, f)][0] for f in keyframes]), np.concatenate([pairs[(cam, f)][1] for f in keyframes]))
        for frame in keyframes:
            local = robust_affine(*pairs[(cam, frame)])
            fits[clip.camera_names[cam]].append((local[0], local[1], len(pairs[(cam, frame)][0])))
            affine[(cam, frame)] = pooled if fit_scope == "camera" else local
        fits[clip.camera_names[cam]].append((pooled[0], pooled[1], -1))
    # Pass 2: unproject, carve (occlusion-aware), colour, velocity.
    for frame in keyframes:
        before, after = max(frame - 1, 0), min(frame + 1, num_frames - 1)
        for cam in cams:
            name = clip.camera_names[cam]
            depth = depths[name][frame].float().to(dev)
            eroded = -F.max_pool2d(-(masks[cam][frame].to(dev) > 0.5).float()[None, None], 5, 1, 2)[0, 0] > 0.5
            joints = torch.from_numpy(joints_per_camera[cam, frame][body]).to(dev)
            a, b = affine[(cam, frame)]
            valid: Tensor = eroded & (depth > 0)
            ys, xs = torch.nonzero(valid, as_tuple=True)
            if len(ys) == 0:
                continue
            pick: Tensor = torch.randperm(len(ys), generator=generator)[:points_per_view].to(dev)
            ys, xs = ys[pick], xs[pick]
            zc: Tensor = a * depth[ys, xs] + b
            K: Tensor = clip.K[cam].to(dev)
            cam_points: Tensor = torch.stack([(xs - K[0, 2]) / K[0, 0] * zc, (ys - K[1, 2]) / K[1, 1] * zc, zc], dim=-1)
            world_T_cam: Tensor = clip.world_T_cam[cam].to(dev)
            world: Tensor = cam_points @ world_T_cam[:3, :3].T + world_T_cam[:3, 3]
            keep: Tensor = zc > 0.1
            for other in cams:
                if other == cam:
                    continue
                ow, oh = clip.sizes[other]
                ouv, oz = _project(world, clip.world_T_cam[other], clip.K[other])
                inside: Tensor = (oz > 0.1) & (ouv[:, 0] >= 0) & (ouv[:, 0] <= ow - 1) & (ouv[:, 1] >= 0) & (ouv[:, 1] <= oh - 1)
                fg: Tensor = _sample(masks[other][frame].to(dev)[None], ouv, (ow, oh))[:, 0]
                oa, ob = affine[(other, frame)]
                other_raw: Tensor = _sample(depths[clip.camera_names[other]][frame].float().to(dev)[None], ouv, (ow, oh))[:, 0]
                occluded: Tensor = (other_raw > 0.1) & (oa * other_raw + ob < oz - occlusion_margin_m)
                keep &= ~inside | (fg >= carve_below) | occluded
            world, ys, xs = world[keep], ys[keep], xs[keep]
            colors: Tensor = clip.frames[cam][frame].to(dev)[ys, xs].float() / 255.0
            j_prev: Tensor = torch.from_numpy(joints_per_camera[cam, before][body]).to(dev)
            j_next: Tensor = torch.from_numpy(joints_per_camera[cam, after][body]).to(dev)
            velocity_j: Tensor = torch.nan_to_num((j_next - j_prev) / max(after - before, 1), nan=0.0)
            j_now: Tensor = torch.nan_to_num(joints, nan=1e6)
            nearest: Int64[Tensor, " n"] = torch.cdist(world, j_now).argmin(dim=1)
            rows_p.append(world)
            rows_v.append(velocity_j[nearest])
            rows_c.append(colors)
            rows_f.append(torch.full((len(world),), frame, dtype=torch.int64))
    positions: Tensor = torch.cat(rows_p)
    frames_all: Tensor = torch.cat(rows_f)
    n: int = len(frames_all)
    frame_end: int = num_frames - 1
    cloud = InitCloud(
        positions=positions.cpu().numpy().astype(np.float32),
        velocities=torch.cat(rows_v).cpu().numpy().astype(np.float32),
        colors=torch.cat(rows_c).clamp(0, 1).cpu().numpy().astype(np.float32),
        times=(frames_all.numpy().astype(np.float32) / np.float32(max(frame_end, 1)))[:, None],
        frames=frames_all.numpy().astype(np.int32),
        frame_start=0,
        frame_end=frame_end,
        keyframe_step=keyframe_step,
        fps=float(clip.fps),
        source_frame_start=int(clip.frame_indices[0]),
        dataset=dataset,
        segment_id=segment_id,
        method="moge2_lift_joints",
        has_velocity=np.ones(n, dtype=bool),
        is_static=np.zeros(n, dtype=bool),
        dynamic_allowed=np.ones(n, dtype=bool),
    )
    return cloud, fits


def capsule_init(
    clip: MultiViewClip,
    train_names: tuple[str, ...],
    joints: Float32[ndarray, "frames j 3"],
    *,
    dataset: str,
    segment_id: str,
    keyframe_step: int = 3,
    points_per_bone: int = 1500,
    radius_m: float = 0.07,
    seed: int = 0,
) -> InitCloud:
    """Init option (c): points on capsules (radius ``radius_m``) around the COCO body bones, grey colour, joint velocity."""
    bones: list[tuple[int, int]] = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    rng = np.random.default_rng(seed)
    num_frames: int = joints.shape[0]
    rows_p, rows_v, rows_f = [], [], []
    for frame in range(0, num_frames, keyframe_step):
        before, after = max(frame - 1, 0), min(frame + 1, num_frames - 1)
        for a, b in bones:
            pa_, pb_ = joints[frame, a], joints[frame, b]
            if not (np.isfinite(pa_).all() and np.isfinite(pb_).all()):
                continue
            s: Float64[ndarray, " n"] = rng.random(points_per_bone)
            direction: Float64[ndarray, "n 3"] = rng.normal(size=(points_per_bone, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            points = pa_[None] * (1 - s[:, None]) + pb_[None] * s[:, None] + radius_m * direction
            va = np.nan_to_num((joints[after, a] - joints[before, a]) / max(after - before, 1))
            vb = np.nan_to_num((joints[after, b] - joints[before, b]) / max(after - before, 1))
            rows_p.append(points)
            rows_v.append(va[None] * (1 - s[:, None]) + vb[None] * s[:, None])
            rows_f.append(np.full(points_per_bone, frame))
    frames_all: Int64[ndarray, " n"] = np.concatenate(rows_f).astype(np.int64)
    n: int = len(frames_all)
    frame_end: int = num_frames - 1
    _ = train_names
    return InitCloud(
        positions=np.concatenate(rows_p).astype(np.float32),
        velocities=np.concatenate(rows_v).astype(np.float32),
        colors=np.full((n, 3), 0.3, dtype=np.float32),
        times=(frames_all.astype(np.float32) / np.float32(max(frame_end, 1)))[:, None],
        frames=frames_all.astype(np.int32),
        frame_start=0,
        frame_end=frame_end,
        keyframe_step=keyframe_step,
        fps=float(clip.fps),
        source_frame_start=int(clip.frame_indices[0]),
        dataset=dataset,
        segment_id=segment_id,
        method="capsules_joints",
        has_velocity=np.ones(n, dtype=bool),
        is_static=np.zeros(n, dtype=bool),
        dynamic_allowed=np.ones(n, dtype=bool),
    )
