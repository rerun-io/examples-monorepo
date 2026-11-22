"""Rendered depth against the SMPL-X mesh depth at the held-out cameras (with/without-prior comparisons).

The scoring runs in full float32 (``evaluate.ieee_fp32``), with the person mask threshold of ``evaluate``.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float32
from torch import Tensor

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.evaluate import MASK_THRESHOLD, ieee_fp32
from open4dgs.trigs_adapter import TrainedClip

GEOMETRY_TOLERANCE_M: float = 0.03
"""Depth error counted as "within" in ``GeometryMetrics.within_3cm``."""
RASTER_CANDIDATES: int = 4_000_000
"""Face x pixel candidates evaluated per chunk by ``rasterize_depth``."""


def rasterize_depth(
    vertices_cam: Float32[Tensor, "v 3"], faces: Tensor, K: Float32[Tensor, "3 3"], size: tuple[int, int]
) -> Float32[Tensor, "h w"]:
    """Z-buffer of a triangle mesh in camera coordinates (OpenCV RDF), 0 where no face covers the pixel centre.

    A plain torch rasterizer (no mesh renderer is in the open4dgs envs: no Open3D, nvdiffrast or PyTorch3D): every
    face is tested against the pixel centres of its bounding box (chunked), the z of the barycentric interpolation
    of camera-space z is kept with ``scatter_reduce(amin)``. Faces behind the camera (any vertex z <= 1 cm) are dropped.
    Exact for pixel-centre coverage; perspective-correct, because z is interpolated as 1/z in screen space.
    """
    width, height = size
    device: torch.device = vertices_cam.device
    tri: Tensor = vertices_cam[faces.long()]  # f 3 3
    z: Tensor = tri[..., 2]
    keep: Tensor = (z > 0.01).all(dim=-1)
    tri, z = tri[keep], z[keep]
    uv: Tensor = (tri[..., :2] / z[..., None]) * torch.stack([K[0, 0], K[1, 1]]) + torch.stack([K[0, 2], K[1, 2]])
    lo: Tensor = uv.amin(dim=1).ceil().clamp(min=0)
    hi: Tensor = uv.amax(dim=1).floor()
    hi = torch.minimum(hi, torch.tensor([width - 1.0, height - 1.0], device=device))
    on_screen: Tensor = (hi >= lo).all(dim=-1)
    uv, z, lo, hi = uv[on_screen], z[on_screen], lo[on_screen], hi[on_screen]
    buffer: Tensor = torch.full((height * width,), float("inf"), device=device)
    if uv.shape[0] == 0:
        return torch.zeros((height, width), device=device)
    extent: Tensor = (hi - lo + 1).long()
    bw, bh = int(extent[:, 0].max()), int(extent[:, 1].max())
    oy, ox = torch.meshgrid(torch.arange(bh, device=device), torch.arange(bw, device=device), indexing="ij")
    offsets: Tensor = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1).float()  # k 2
    chunk: int = max(1, RASTER_CANDIDATES // offsets.shape[0])
    for start in range(0, uv.shape[0], chunk):
        t_uv, t_z, t_lo, t_hi = uv[start : start + chunk], z[start : start + chunk], lo[start : start + chunk], hi[start : start + chunk]
        px: Tensor = t_lo[:, None, :] + offsets[None]  # c k 2
        inside_box: Tensor = (px <= t_hi[:, None, :]).all(dim=-1)
        a, b, c = t_uv[:, 0, None, :], t_uv[:, 1, None, :], t_uv[:, 2, None, :]

        def edge(p: Tensor, q: Tensor, r: Tensor) -> Tensor:
            return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

        area: Tensor = edge(a, b, c)  # c 1
        w0, w1, w2 = edge(b, c, px) / area, edge(c, a, px) / area, edge(a, b, px) / area
        inside: Tensor = inside_box & (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (area.abs() > 1e-12)
        inv_z: Tensor = w0 / t_z[:, 0, None] + w1 / t_z[:, 1, None] + w2 / t_z[:, 2, None]
        depth: Tensor = 1.0 / inv_z
        index: Tensor = px[..., 1].long() * width + px[..., 0].long()
        buffer.scatter_reduce_(0, index[inside], depth[inside], reduce="amin")
    buffer = torch.where(torch.isfinite(buffer), buffer, torch.zeros_like(buffer))
    return buffer.reshape(height, width)


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    """Rendered depth vs the SMPL-X mesh depth at one camera and mesh frame, inside ``GT mask > 0.5 & mesh hit``.

    SMPL-X has no clothes or hair, so the absolute numbers carry the clothing offset (centimetres); only the
    DIFFERENCE between two runs on the same frames and cameras (e.g. with and without the prior) means something.
    """

    camera: str
    frame_index: int
    median_abs_cm: float
    """Median |D_render - D_mesh|, centimetres."""
    within_3cm: float
    """Share of the pixels with |D_render - D_mesh| <= 3 cm."""
    median_normal_deg: float
    """Median angle between depth_to_normal of the render and of the mesh depth (same operator on both), degrees."""
    num_pixels: int


GEOMETRY_NAMES: tuple[str, ...] = ("median_abs_cm", "within_3cm", "median_normal_deg")


@ieee_fp32()
def geometry_frame(
    camera: str,
    frame_index: int,
    depth: Float32[Tensor, "h w"],
    mesh_depth: Float32[Tensor, "h w"],
    mask: Float32[Tensor, "h w"],
    K: Float32[Tensor, "3 3"],
) -> GeometryMetrics:
    """``GeometryMetrics`` of one rendered depth map against the mesh depth (nan when no pixel qualifies; full float32)."""
    from open4dgs.prior import camera_normals, erode

    pixels: Tensor = (mask > MASK_THRESHOLD) & (mesh_depth > 0)
    if not bool(pixels.any()):
        return GeometryMetrics(camera, frame_index, math.nan, math.nan, math.nan, 0)
    error: Tensor = (depth - mesh_depth)[pixels].abs()
    normal_pixels: Tensor = erode(pixels, 1)  # both normals need valid neighbours
    if bool(normal_pixels.any()):
        cos: Tensor = (camera_normals(depth, K)[normal_pixels] * camera_normals(mesh_depth, K)[normal_pixels]).sum(-1).clamp(-1.0, 1.0)
        normal_deg: float = float(torch.rad2deg(torch.acos(cos)).median())
    else:
        normal_deg = math.nan
    return GeometryMetrics(
        camera=camera,
        frame_index=frame_index,
        median_abs_cm=float(error.median()) * 100.0,
        within_3cm=float((error <= GEOMETRY_TOLERANCE_M).float().mean()),
        median_normal_deg=normal_deg,
        num_pixels=int(pixels.sum()),
    )


@dataclass(frozen=True)
class CameraGeometry:
    """Depth renders of one camera at the clip's mesh frames, and their metrics."""

    camera: str
    frame_slots: list[int]
    """Clip frame positions that hold a mesh."""
    depth: list[Tensor]
    """Rendered ED depth per mesh frame, CPU float32 (h, w)."""
    mesh_depth: list[Tensor]
    """SMPL-X depth per mesh frame, CPU float32 (h, w), 0 = no hit."""
    metrics: list[GeometryMetrics]


def evaluate_geometry(
    model: TrainedClip, clip: MultiViewClip, masks: tuple[Float32[Tensor, "frames h w"], ...], camera_names: tuple[str, ...], device: torch.device
) -> list[CameraGeometry]:
    """Rendered ED depth vs the rasterized SMPL-X depth (every person of the clip) at the clip frames that hold a mesh.

    MAMMA meshes are 10 Hz (every 3rd frame), so only those frames are scored. See ``GeometryMetrics`` for why only
    differences between runs are meaningful.
    """
    if clip.people is None:
        return []
    people = clip.people
    faces: Tensor = people.faces.to(device)
    slots: list[int] = [f for f in range(model.num_frames) if bool(people.present[f].any())]
    out: list[CameraGeometry] = []
    for name in camera_names:
        cam: int = clip.camera_names.index(name)
        cam_T_world: Tensor = torch.linalg.inv(clip.world_T_cam[cam].double()).float().to(device)
        K: Tensor = clip.K[cam].to(device)
        depths: list[Tensor] = []
        meshes: list[Tensor] = []
        rows: list[GeometryMetrics] = []
        for f in slots:
            depth, _ = model.render_depth(model.cameras, cam, f)
            mesh_depth: Tensor = torch.zeros_like(depth)
            for person in torch.nonzero(people.present[f]).reshape(-1).tolist():
                verts: Tensor = people.vertices[f, person].to(device)
                verts_cam: Tensor = verts @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
                person_depth: Tensor = rasterize_depth(verts_cam, faces, K, clip.sizes[cam])
                closer: Tensor = (person_depth > 0) & ((mesh_depth == 0) | (person_depth < mesh_depth))
                mesh_depth = torch.where(closer, person_depth, mesh_depth)
            rows.append(geometry_frame(name, int(clip.frame_indices[f]), depth, mesh_depth, masks[cam][f].to(device).float(), K))
            depths.append(depth.cpu())
            meshes.append(mesh_depth.cpu())
        out.append(CameraGeometry(name, slots, depths, meshes, rows))
    return out


def mean_geometry(rows: list[GeometryMetrics]) -> dict[str, float]:
    """Mean of each geometry metric over finite values (per-frame medians averaged)."""
    out: dict[str, float] = {}
    for name in GEOMETRY_NAMES:
        values: list[float] = [getattr(row, name) for row in rows if math.isfinite(getattr(row, name))]
        out[name] = float(np.mean(values)) if values else math.nan
    return out
