"""Body-surface init: points on the SMPL-X mesh that follow the body, coloured from the training cameras.

Each person gets ``samples_per_person`` fixed barycentric samples (area-weighted on the first mesh, seeded), so a
sample is the same body point at every keyframe. Keyframes are the clip frames that hold a mesh, every
``keyframe_step`` frames (MAMMA stores meshes every 3rd frame, 4DAnyone every frame). Velocity is the central
difference of the sample's position between neighbouring mesh frames, in world units per VIDEO frame.

Colours come from the TRAINING cameras only: a point is seen by a camera when its face normal faces the camera and
its depth passes a z-buffer built from all samples of that keyframe (quarter resolution, 2 cm tolerance); the colour
is the mean over the cameras that see it. Points no camera sees take the mean colour of their person.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.init_cloud import InitCloud


@dataclass(frozen=True, slots=True)
class SurfaceSamples:
    """Fixed points on a triangle mesh: face index plus barycentric weights."""

    faces: Int64[Tensor, " n"]
    """Face of each sample."""
    barycentric: Float32[Tensor, "n 3"]
    """Weights of the face's three vertices (non-negative, sum 1)."""


def sample_surface(vertices: Float32[Tensor, "v 3"], faces: Int64[Tensor, "f 3"], num_samples: int, seed: int) -> SurfaceSamples:
    """Area-weighted uniform samples on the mesh ``(vertices, faces)``; the same seed gives the same samples."""
    generator: torch.Generator = torch.Generator(device="cpu").manual_seed(seed)
    tri: Float32[Tensor, "f 3 3"] = vertices.detach().cpu().float()[faces.cpu()]
    areas: Float32[Tensor, " f"] = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=-1)
    face_idx: Int64[Tensor, " n"] = torch.multinomial(areas.clamp_min(0.0), num_samples, replacement=True, generator=generator)
    uv: Float32[Tensor, "n 2"] = torch.rand(num_samples, 2, generator=generator)
    sqrt_u: Float32[Tensor, " n"] = uv[:, 0].sqrt()
    barycentric: Float32[Tensor, "n 3"] = torch.stack([1.0 - sqrt_u, sqrt_u * (1.0 - uv[:, 1]), sqrt_u * uv[:, 1]], dim=-1)
    return SurfaceSamples(faces=face_idx, barycentric=barycentric)


def surface_points(vertices: Float32[Tensor, "v 3"], faces: Int64[Tensor, "f 3"], samples: SurfaceSamples) -> Float32[Tensor, "n 3"]:
    """World positions of ``samples`` on the posed mesh ``vertices``."""
    corners: Float32[Tensor, "n 3 3"] = vertices[faces[samples.faces.to(vertices.device)]]
    return (corners * samples.barycentric.to(vertices.device)[:, :, None]).sum(dim=1)


def surface_normals(vertices: Float32[Tensor, "v 3"], faces: Int64[Tensor, "f 3"], samples: SurfaceSamples) -> Float32[Tensor, "n 3"]:
    """Unit face normals (right-hand winding) at ``samples`` on the posed mesh."""
    tri: Float32[Tensor, "n 3 3"] = vertices[faces[samples.faces.to(vertices.device)]]
    normal: Float32[Tensor, "n 3"] = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def mesh_keyframes(present: Bool[Tensor, " frames"], keyframe_step: int) -> Int64[Tensor, " k"]:
    """Frames that hold a mesh, thinned to one every ``keyframe_step`` frames (counted from the first mesh frame)."""
    frames: Int64[Tensor, " m"] = torch.nonzero(present.cpu()).reshape(-1)
    if frames.numel() == 0:
        return frames
    return frames[(frames - frames[0]) % keyframe_step == 0]


def central_velocity(positions: Float32[Tensor, "k n 3"], frames: Int64[Tensor, " k"]) -> Float32[Tensor, "k n 3"]:
    """Per-video-frame velocity at each keyframe: central difference between neighbours, one-sided at the ends."""
    if positions.shape[0] < 2:
        return torch.zeros_like(positions)
    t: Float32[Tensor, " k"] = frames.to(positions.device).float()
    prev: Int64[Tensor, " k"] = torch.clamp(torch.arange(positions.shape[0], device=positions.device) - 1, min=0)
    nxt: Int64[Tensor, " k"] = torch.clamp(torch.arange(positions.shape[0], device=positions.device) + 1, max=positions.shape[0] - 1)
    dt: Float32[Tensor, " k"] = t[nxt] - t[prev]
    return (positions[nxt] - positions[prev]) / dt[:, None, None]


def visible_colors(
    points: Float32[Tensor, "n 3"],
    normals: Float32[Tensor, "n 3"],
    images: list[Tensor],
    world_T_cam: Float32[Tensor, "c 4 4"],
    K: Float32[Tensor, "c 3 3"],
    *,
    zbuffer_scale: int = 4,
    depth_tolerance: float = 0.02,
) -> tuple[Float32[Tensor, "n 3"], Int64[Tensor, " n"]]:
    """Mean colour of each point over the cameras that see it, and how many cameras saw it.

    ``images`` are uint8 HWC frames of the cameras in ``world_T_cam``/``K`` order (training cameras only).
    """
    device: torch.device = points.device
    color_sum: Float32[Tensor, "n 3"] = torch.zeros_like(points)
    count: Int64[Tensor, " n"] = torch.zeros(points.shape[0], dtype=torch.int64, device=device)
    for cam, image in enumerate(images):
        cam_T_world: Float32[Tensor, "4 4"] = torch.linalg.inv(world_T_cam[cam].to(device))
        p_cam: Float32[Tensor, "n 3"] = points @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
        centre: Float32[Tensor, "3"] = world_T_cam[cam, :3, 3].to(device)
        facing: Bool[Tensor, " n"] = ((centre - points) * normals).sum(dim=-1) > 0.0
        z: Float32[Tensor, " n"] = p_cam[:, 2]
        uvw: Float32[Tensor, "n 3"] = p_cam @ K[cam].to(device).T
        u: Float32[Tensor, " n"] = uvw[:, 0] / z.clamp_min(1e-6)
        v: Float32[Tensor, " n"] = uvw[:, 1] / z.clamp_min(1e-6)
        height, width = int(image.shape[0]), int(image.shape[1])
        ui: Int64[Tensor, " n"] = u.round().long()
        vi: Int64[Tensor, " n"] = v.round().long()
        inside: Bool[Tensor, " n"] = (z > 1e-3) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        zb_w, zb_h = (width + zbuffer_scale - 1) // zbuffer_scale, (height + zbuffer_scale - 1) // zbuffer_scale
        cell: Int64[Tensor, " n"] = (vi.clamp(0, height - 1) // zbuffer_scale) * zb_w + ui.clamp(0, width - 1) // zbuffer_scale
        zbuf: Float32[Tensor, " cells"] = torch.full((zb_w * zb_h,), float("inf"), device=device)
        zbuf.scatter_reduce_(0, cell[inside], z[inside], reduce="amin")
        seen: Bool[Tensor, " n"] = inside & facing & (z <= zbuf[cell] + depth_tolerance)
        pixels: Float32[Tensor, "n 3"] = image.to(device)[vi.clamp(0, height - 1), ui.clamp(0, width - 1)].float() / 255.0
        color_sum += pixels * seen[:, None]
        count += seen.long()
    return color_sum / count.clamp_min(1)[:, None], count


def body_surface_init(
    clip: MultiViewClip,
    train_names: tuple[str, ...],
    *,
    dataset: str,
    segment_id: str,
    samples_per_person: int = 30_000,
    keyframe_step: int = 3,
    seed: int = 0,
    device: torch.device | None = None,
) -> InitCloud:
    """Body-surface init cloud for every person of ``clip``, coloured from the ``train_names`` cameras only.

    Raises:
        ValueError: When the clip has no SMPL-X meshes or a person has no mesh frame in the window.
    """
    if clip.people is None:
        raise ValueError(f"{segment_id}: the clip has no body meshes; body-surface init needs them")
    dev: torch.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    people = clip.people
    faces: Int64[Tensor, "f 3"] = people.faces.long().to(dev)
    cams: list[int] = [clip.camera_names.index(name) for name in train_names]
    num_frames: int = int(clip.frame_indices.shape[0])
    rows: dict[int, list[tuple[Tensor, Tensor, Tensor]]] = {}
    for person in range(len(people.names)):
        keyframes: Int64[Tensor, " k"] = mesh_keyframes(people.present[:, person], keyframe_step)
        if keyframes.numel() == 0:
            raise ValueError(f"{segment_id}: {people.names[person]} has no mesh in the window")
        verts: Float32[Tensor, "k v 3"] = people.vertices[keyframes, person].to(dev)
        samples: SurfaceSamples = sample_surface(verts[0], faces, samples_per_person, seed + person)
        positions: Float32[Tensor, "k n 3"] = torch.stack([surface_points(verts[k], faces, samples) for k in range(len(keyframes))])
        velocities: Float32[Tensor, "k n 3"] = central_velocity(positions, keyframes)
        for k, frame in enumerate(keyframes.tolist()):
            normals: Float32[Tensor, "n 3"] = surface_normals(verts[k], faces, samples)
            colors, count = visible_colors(positions[k], normals, [clip.frames[c][frame] for c in cams], clip.world_T_cam[cams], clip.K[cams])
            seen: Bool[Tensor, " n"] = count > 0
            fallback: Float32[Tensor, "3"] = colors[seen].mean(dim=0) if bool(seen.any()) else torch.full((3,), 0.5, device=dev)
            colors = torch.where(seen[:, None], colors, fallback)
            rows.setdefault(frame, []).append((positions[k], velocities[k], colors))
    frame_end: int = num_frames - 1
    ordered: list[int] = sorted(rows)
    positions_all: Float32[Tensor, "r 3"] = torch.cat([p for f in ordered for p, _, _ in rows[f]])
    velocities_all: Float32[Tensor, "r 3"] = torch.cat([v for f in ordered for _, v, _ in rows[f]])
    colors_all: Float32[Tensor, "r 3"] = torch.cat([c for f in ordered for _, _, c in rows[f]])
    frames_all: Int64[Tensor, " r"] = torch.cat([torch.full((p.shape[0],), f, dtype=torch.int64) for f in ordered for p, _, _ in rows[f]])
    n: int = int(frames_all.shape[0])
    return InitCloud(
        positions=positions_all.cpu().numpy().astype(np.float32),
        velocities=velocities_all.cpu().numpy().astype(np.float32),
        colors=colors_all.clamp(0.0, 1.0).cpu().numpy().astype(np.float32),
        times=(frames_all.numpy().astype(np.float32) / np.float32(max(frame_end, 1)))[:, None],
        frames=frames_all.numpy().astype(np.int32),
        frame_start=0,
        frame_end=frame_end,
        keyframe_step=keyframe_step,
        fps=float(clip.fps),
        source_frame_start=int(clip.frame_indices[0]),
        dataset=dataset,
        segment_id=segment_id,
        method="body_surface",
        has_velocity=np.ones(n, dtype=bool),
        is_static=np.zeros(n, dtype=bool),
        dynamic_allowed=np.ones(n, dtype=bool),
    )


def interpolate_meshes(vertices: Float32[Tensor, "frames v 3"], present: Bool[Tensor, " frames"]) -> Float32[Tensor, "frames v 3"]:
    """Fill frames without a mesh by linear interpolation between the nearest mesh frames (held at the ends)."""
    have: Int64[Tensor, " m"] = torch.nonzero(present.cpu()).reshape(-1)
    if have.numel() == 0:
        raise ValueError("no mesh frame to interpolate from")
    frames: Int64[Tensor, " t"] = torch.arange(vertices.shape[0])
    right: Int64[Tensor, " t"] = torch.searchsorted(have, frames).clamp(max=have.numel() - 1)
    left: Int64[Tensor, " t"] = torch.where(have[right] > frames, (right - 1).clamp(min=0), right)
    t0, t1 = have[left], have[right]
    weight: Float32[Tensor, " t"] = torch.where(t1 > t0, (frames - t0).float() / (t1 - t0).clamp(min=1).float(), torch.zeros(frames.shape))
    weight = weight.clamp(0.0, 1.0).to(vertices.device)
    return vertices[t0] * (1.0 - weight)[:, None, None] + vertices[t1] * weight[:, None, None]


def mesh_silhouette_masks(
    clip: MultiViewClip, *, num_samples: int = 200_000, dilate_frac: float = 0.01, seed: int = 0
) -> tuple[Float32[Tensor, "frames h w"], ...]:
    """STAND-IN person masks (development only): the projected SMPL-X surface of every person, dilated.

    Dilation radius is ``dilate_frac`` of the camera's longer side (at least 2 px); frames without a mesh use linearly
    interpolated vertices. SMPL-X has no hair or loose clothing, so these masks miss them.
    """
    if clip.people is None:
        raise ValueError("mesh silhouette masks need the clip's body meshes")
    dev: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    people = clip.people
    faces: Int64[Tensor, "f 3"] = people.faces.long().to(dev)
    tracks: list[Tensor] = []
    for person in range(len(people.names)):
        verts: Tensor = interpolate_meshes(people.vertices[:, person].to(dev), people.present[:, person])
        samples: SurfaceSamples = sample_surface(verts[0], faces, num_samples, seed + person)
        tracks.append(torch.stack([surface_points(verts[f], faces, samples) for f in range(verts.shape[0])]))
    points: Float32[Tensor, "frames n 3"] = torch.cat(tracks, dim=1)
    masks: list[Tensor] = []
    for cam in range(len(clip.camera_names)):
        width, height = clip.sizes[cam]
        cam_T_world: Tensor = torch.linalg.inv(clip.world_T_cam[cam].to(dev))
        K: Tensor = clip.K[cam].to(dev)
        radius: int = max(2, round(dilate_frac * max(width, height)))
        out: Tensor = torch.zeros(points.shape[0], height * width, device=dev)
        p_cam: Tensor = points @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
        z: Tensor = p_cam[..., 2]
        uvw: Tensor = p_cam @ K.T
        u: Tensor = (uvw[..., 0] / z.clamp_min(1e-6)).round().long()
        v: Tensor = (uvw[..., 1] / z.clamp_min(1e-6)).round().long()
        ok: Tensor = (z > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        index: Tensor = (v.clamp(0, height - 1) * width + u.clamp(0, width - 1)) * ok
        out.scatter_(1, index, ok.float())
        out[:, 0] = 0.0  # index 0 collects the rejected points
        dilated: Tensor = F.max_pool2d(out.reshape(-1, 1, height, width), kernel_size=2 * radius + 1, stride=1, padding=radius)
        masks.append(dilated[:, 0].cpu())
    return tuple(masks)
