"""Perspective KeyNet crops: UmeTrack's crop cameras, batched on the device.

A crop camera shares the source camera's centre and looks at the hand: the minimal rotation that takes the optical axis to
the centre of the crop points' bounding box, then a roll about the new axis by the camera's mounting angle (UmeTrack's
``make_look_at_matrix`` and ``camera_angle``), then an optional x mirror (right hands, so KeyNet sees left hands only). Its
pinhole focal fits the crop points inside the 96 x 96 crop with a margin. Every crop pixel's ray goes back through the source
camera's own lens model (Fisheye62 or pinhole) and samples the native image, so a crop has no lens distortion and a hand looks
the same anywhere in any camera, at the camera's own resolution. Labels map exactly: a camera-frame point goes through the
crop camera's rotation and pinhole. Crop pixel centres are 0..95 with the centre at 47.5, as for the affine crops.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int64, UInt8
from torch import Tensor

from handtrack.geometry.camera import CameraRig, project
from handtrack.geometry.letterbox import Letterbox

CROP_SIZE: int = 96
CROP_CENTRE: float = (CROP_SIZE - 1) / 2.0
CROP_MARGIN: float = 1.2
"""The farthest crop point sits at 1/1.2 of the half-side from the centre (the affine crops' 20 % box enlargement)."""
MIN_AXIS_COSINE: float = 0.05
"""A crop camera must look less than ~87 degrees off its source camera's axis (the minimal rotation degenerates at 180)."""
_SAMPLE_CHUNK: int = 16
"""Frames converted to float and sampled per grid_sample call (a 1.3 MP frame is 5.2 MB as float)."""


@dataclass(frozen=True, slots=True)
class CropCameras:
    """One pinhole crop camera per crop, sharing its source camera's centre."""

    rotation: Float32[Tensor, "n 3 3"]
    """crop_from_camera, roll included, mirror not."""
    focal: Float32[Tensor, "n"]
    """Crop pixels per unit of normalised image coordinate (square pixels)."""
    mirror: Bool[Tensor, "n"]
    """Mirror the crop's x axis (right hands)."""

    def select(self, index: Int64[Tensor, "k"] | Bool[Tensor, "n"]) -> "CropCameras":
        return CropCameras(self.rotation[index], self.focal[index], self.mirror[index])


def _minimal_rotation(direction: Float32[Tensor, "n 3"]) -> Float32[Tensor, "n 3 3"]:
    """The smallest rotation taking +z to ``direction`` (unit), by Rodrigues' formula; UmeTrack's ``from_two_vectors``."""
    z: Float32[Tensor, "n 3"] = torch.zeros_like(direction)
    z[:, 2] = 1.0
    axis: Float32[Tensor, "n 3"] = torch.linalg.cross(z, direction)
    cosine: Float32[Tensor, "n"] = direction[:, 2]
    skew: Float32[Tensor, "n 3 3"] = torch.zeros((direction.shape[0], 3, 3), dtype=direction.dtype, device=direction.device)
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = axis[:, 2], -axis[:, 1], axis[:, 0]
    eye: Float32[Tensor, "3 3"] = torch.eye(3, dtype=direction.dtype, device=direction.device)
    # Directions at or behind the image plane get no usable crop camera; 1 + cos is floored so they stay finite (callers drop them).
    return eye + skew + skew @ skew / (1.0 + cosine).clamp_min(1e-6)[:, None, None]


def _roll(angle: Float32[Tensor, "n"]) -> Float32[Tensor, "n 3 3"]:
    c: Float32[Tensor, "n"] = angle.cos()
    s: Float32[Tensor, "n"] = angle.sin()
    out: Float32[Tensor, "n 3 3"] = torch.zeros((angle.shape[0], 3, 3), dtype=angle.dtype, device=angle.device)
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1], out[:, 2, 2] = c, -s, s, c, 1.0
    return out


def look_at(direction: Float32[Tensor, "n 3"], roll_rad: Float32[Tensor, "n"]) -> Float32[Tensor, "n 3 3"]:
    """crop_from_camera for a crop camera aimed along ``direction`` (camera frame) and rolled by ``roll_rad`` about its axis.

    UmeTrack builds camera_from_crop = minimal(z -> direction) @ Rz(camera_angle); this returns its transpose.
    """
    unit: Float32[Tensor, "n 3"] = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-12)
    return (_minimal_rotation(unit) @ _roll(roll_rad)).transpose(-1, -2)


def to_crop(cameras: CropCameras, points_cam: Float32[Tensor, "n k 3"]) -> tuple[Float32[Tensor, "n k 2"], Float32[Tensor, "n k"]]:
    """Camera-frame points into crop pixels (mirror applied) and their depth along the crop camera's axis."""
    local: Float32[Tensor, "n k 3"] = torch.einsum("nij,nkj->nki", cameras.rotation, points_cam)
    depth: Float32[Tensor, "n k"] = local[..., 2]
    safe: Float32[Tensor, "n k"] = torch.where(depth.abs() < 1e-9, torch.full_like(depth, 1e-9), depth)
    uv: Float32[Tensor, "n k 2"] = local[..., :2] / safe[..., None] * cameras.focal[:, None, None] + CROP_CENTRE
    u: Float32[Tensor, "n k"] = torch.where(cameras.mirror[:, None], (CROP_SIZE - 1) - uv[..., 0], uv[..., 0])
    return torch.stack([u, uv[..., 1]], dim=-1), depth


def crop_cameras(points_cam: Float32[Tensor, "n k 3"], valid: Bool[Tensor, "n k"], roll_rad: Float32[Tensor, "n"], mirror: Bool[Tensor, "n"],
                 margin: float = CROP_MARGIN) -> CropCameras:
    """UmeTrack's ``gen_crop_parameters_from_points``: aim at the valid points' bounding-box centre, then fit them with a margin.

    A row without valid points, or with a valid point behind its crop camera, gets a non-finite focal (callers drop it).
    """
    big: float = 1e9
    low: Float32[Tensor, "n 3"] = torch.where(valid[..., None], points_cam, torch.full_like(points_cam, big)).amin(dim=1)
    high: Float32[Tensor, "n 3"] = torch.where(valid[..., None], points_cam, torch.full_like(points_cam, -big)).amax(dim=1)
    centre: Float32[Tensor, "n 3"] = (low + high) * 0.5
    rotation: Float32[Tensor, "n 3 3"] = look_at(centre, roll_rad)
    local: Float32[Tensor, "n k 3"] = torch.einsum("nij,nkj->nki", rotation, points_cam)
    ahead: Bool[Tensor, "n k"] = local[..., 2] > 1e-4
    ndc: Float32[Tensor, "n k"] = (local[..., :2] / local[..., 2:].clamp_min(1e-4)).abs().amax(dim=-1)
    extent: Float32[Tensor, "n"] = torch.where(valid, ndc, torch.zeros_like(ndc)).amax(dim=-1)
    forward: Bool[Tensor, "n"] = centre[:, 2] > MIN_AXIS_COSINE * torch.linalg.vector_norm(centre, dim=-1)
    usable: Bool[Tensor, "n"] = (valid.any(dim=-1) & forward & (ahead | ~valid).all(dim=-1) & (extent > 1e-6)
                                 & torch.isfinite(points_cam.where(valid[..., None], 0.0)).all(dim=(1, 2)))
    focal: Float32[Tensor, "n"] = torch.where(usable, CROP_CENTRE / (extent.clamp_min(1e-6) * margin), torch.full_like(extent, math.nan))
    return CropCameras(rotation, focal, mirror)


def aim(rotation: Float32[Tensor, "n 3 3"], offset_px: Float32[Tensor, "n 2"], focal: Float32[Tensor, "n"], roll_rad: Float32[Tensor, "n"]) -> Float32[Tensor, "n 3 3"]:
    """Re-aim crop cameras at the crop pixel CENTRE + ``offset_px`` (unmirrored crop axes) and add ``roll_rad`` about the new axis."""
    ray: Float32[Tensor, "n 3"] = torch.cat([offset_px / focal[:, None], torch.ones_like(focal)[:, None]], dim=-1)
    ray = ray / torch.linalg.vector_norm(ray, dim=-1, keepdim=True)
    # In the crop frame: the minimal rotation from +z to the ray, then the extra roll; crop_from_camera composes on the left.
    return (_minimal_rotation(ray) @ _roll(roll_rad)).transpose(-1, -2) @ rotation


def jitter(cameras: CropCameras, generator: torch.Generator, max_rotation: float, scale_range: tuple[float, float], max_shift: float) -> CropCameras:
    """The affine crops' jitter for crop cameras: shift the aim by up to ``max_shift`` crop sides per axis, roll by
    +-``max_rotation`` radians, and zoom out by a factor drawn from ``scale_range`` (> 1 zooms out)."""
    n: int = cameras.focal.shape[0]
    device: torch.device = cameras.focal.device
    shift: Float32[Tensor, "n 2"] = (torch.rand((n, 2), generator=generator, device=device) * 2 - 1) * max_shift * CROP_SIZE
    roll: Float32[Tensor, "n"] = (torch.rand(n, generator=generator, device=device) * 2 - 1) * max_rotation
    scale: Float32[Tensor, "n"] = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(n, generator=generator, device=device)
    return CropCameras(aim(cameras.rotation, shift, cameras.focal, roll), cameras.focal / scale, cameras.mirror)


def from_crop(cameras: CropCameras, uv: Float32[Tensor, "n k 2"]) -> Float32[Tensor, "n k 3"]:
    """Camera-frame rays through crop pixels ``uv`` (mirror undone): the inverse of ``to_crop`` up to depth."""
    u: Float32[Tensor, "n k"] = torch.where(cameras.mirror[:, None], (CROP_SIZE - 1) - uv[..., 0], uv[..., 0])
    local: Float32[Tensor, "n k 3"] = torch.stack([(u - CROP_CENTRE) / cameras.focal[:, None], (uv[..., 1] - CROP_CENTRE) / cameras.focal[:, None],
                                                    torch.ones_like(u)], dim=-1)
    return torch.einsum("nji,nkj->nki", cameras.rotation, local)


def crop_rays(cameras: CropCameras) -> Float32[Tensor, "n p 3"]:
    """The camera-frame ray of every crop pixel centre, row-major (p = 96 * 96)."""
    device: torch.device = cameras.focal.device
    axis: Float32[Tensor, "s"] = torch.arange(CROP_SIZE, dtype=torch.float32, device=device)
    v, u = torch.meshgrid(axis, axis, indexing="ij")
    u_flat: Float32[Tensor, "p"] = u.reshape(-1)
    v_flat: Float32[Tensor, "p"] = v.reshape(-1)
    u_src: Float32[Tensor, "n p"] = torch.where(cameras.mirror[:, None], (CROP_SIZE - 1) - u_flat[None], u_flat[None])
    local: Float32[Tensor, "n p 3"] = torch.stack([(u_src - CROP_CENTRE) / cameras.focal[:, None], ((v_flat[None] - CROP_CENTRE) / cameras.focal[:, None]).expand_as(u_src),
                                                    torch.ones_like(u_src)], dim=-1)
    return torch.einsum("nji,npj->npi", cameras.rotation, local)  # camera = rotationᵀ · crop


def sample_crops(frames: UInt8[Tensor, "m h w"], image: Int64[Tensor, "n"], cameras: CropCameras, camera: CameraRig) -> Float32[Tensor, "n 1 96 96"]:
    """Sample each crop from its native frame through the one-camera rig ``camera`` (bilinear; zero outside the image).

    Crops are grouped by frame: each frame is converted to float once and sampled by one grid holding all of its crops
    stacked along the grid's height (converting a 1.3 MP frame per crop cost 4x the whole build's decode)."""
    n: int = image.shape[0]
    height, width = frames.shape[1], frames.shape[2]
    out: Float32[Tensor, "n 1 96 96"] = torch.zeros((n, 1, CROP_SIZE, CROP_SIZE), dtype=torch.float32, device=frames.device)
    if n == 0:
        return out
    size: Float32[Tensor, "2"] = torch.tensor([width, height], dtype=torch.float32, device=frames.device)
    rays: Float32[Tensor, "n p 3"] = crop_rays(cameras)
    pixels: Float32[Tensor, "n p 2"] = project(camera, rays[:, None])[:, 0]
    grid: Float32[Tensor, "n p 2"] = (pixels + 0.5) / size * 2.0 - 1.0
    grid = torch.where((rays[..., 2:] > 1e-6) & torch.isfinite(grid), grid, torch.full_like(grid, 2.0))  # behind the camera: outside, so zero
    # Slot each crop into its frame's column of a padded (frames, most crops per frame) layout.
    used, slot_of = torch.unique(image, return_inverse=True)
    order: Int64[Tensor, "n"] = torch.argsort(slot_of, stable=True)
    counts: Int64[Tensor, "u"] = torch.bincount(slot_of, minlength=used.shape[0])
    starts: Int64[Tensor, "u"] = torch.cumsum(counts, dim=0) - counts
    rank: Int64[Tensor, "n"] = torch.empty_like(order)
    rank[order] = torch.arange(n, device=image.device) - starts[slot_of[order]]
    most: int = int(counts.max())
    padded: Float32[Tensor, "u k p 2"] = torch.full((used.shape[0], most, CROP_SIZE * CROP_SIZE, 2), 2.0, device=frames.device)
    padded[slot_of, rank] = grid
    for begin in range(0, used.shape[0], _SAMPLE_CHUNK):
        part: slice = slice(begin, min(used.shape[0], begin + _SAMPLE_CHUNK))
        source: Float32[Tensor, "g 1 h w"] = frames[used[part]].float()[:, None] / 255.0
        stacked: Float32[Tensor, "g kh 96 2"] = padded[part].reshape(source.shape[0], most * CROP_SIZE, CROP_SIZE, 2)
        sampled: Float32[Tensor, "g 1 kh 96"] = F.grid_sample(source, stacked, mode="bilinear", padding_mode="zeros", align_corners=False)
        crops: Float32[Tensor, "g k 96 96"] = sampled.reshape(source.shape[0], most, CROP_SIZE, CROP_SIZE)
        mine: Bool[Tensor, "n"] = (slot_of >= begin) & (slot_of < part.stop)
        out[mine, 0] = crops[slot_of[mine] - begin, rank[mine]]
    return out


def unproject(camera: CameraRig, pixels: Float32[Tensor, "n 2"], iterations: int = 10) -> Float32[Tensor, "n 3"]:
    """Unit camera-frame rays through native pixels of the one-camera rig ``camera``: exact for a pinhole, Newton steps on
    the normalised coordinates for Fisheye62 (finite-difference Jacobian of ``project``)."""
    normalised: Float32[Tensor, "n 2"] = (pixels - camera.principal[0]) / camera.focal[0]
    guess: Float32[Tensor, "n 3"] = torch.cat([normalised, torch.ones_like(normalised[:, :1])], dim=-1)
    if camera.fisheye62 is not None:
        # Solve project([a, b, 1]) = pixel for (a, b), starting from the pinhole guess.
        ab: Float32[Tensor, "n 2"] = normalised.clone()
        step: float = 1e-4
        for _ in range(iterations):
            points: Float32[Tensor, "n 3 3"] = torch.stack([
                torch.cat([ab, torch.ones_like(ab[:, :1])], dim=-1),
                torch.cat([ab + torch.tensor([step, 0.0], device=ab.device), torch.ones_like(ab[:, :1])], dim=-1),
                torch.cat([ab + torch.tensor([0.0, step], device=ab.device), torch.ones_like(ab[:, :1])], dim=-1),
            ], dim=1)
            projected: Float32[Tensor, "n 3 2"] = project(camera, points[None])[0]
            residual: Float32[Tensor, "n 2"] = projected[:, 0] - pixels
            jacobian: Float32[Tensor, "n 2 2"] = torch.stack([(projected[:, 1] - projected[:, 0]) / step, (projected[:, 2] - projected[:, 0]) / step], dim=-1)
            # An explicit 2x2 inverse: where the lens model saturates (Fisheye62 clips r^2 at pi^2) the Jacobian is singular,
            # and those rays keep their last estimate instead of failing the whole batch.
            a, b, c, d = jacobian[:, 0, 0], jacobian[:, 0, 1], jacobian[:, 1, 0], jacobian[:, 1, 1]
            det: Float32[Tensor, "n"] = a * d - b * c
            ok: Bool[Tensor, "n"] = det.abs() > 1e-9
            safe_det: Float32[Tensor, "n"] = torch.where(ok, det, torch.ones_like(det))
            delta: Float32[Tensor, "n 2"] = torch.stack([d * residual[:, 0] - b * residual[:, 1], a * residual[:, 1] - c * residual[:, 0]], dim=-1) / safe_det[:, None]
            ab = ab - torch.where(ok[:, None] & torch.isfinite(delta), delta, torch.zeros_like(delta))
        guess = torch.cat([ab, torch.ones_like(ab[:, :1])], dim=-1)
    return guess / torch.linalg.vector_norm(guess, dim=-1, keepdim=True)


def local_crop_from_net(cameras: CropCameras, camera: CameraRig, letterbox: Letterbox) -> Float32[Tensor, "n 3 3"]:
    """The crop's linearisation at its centre as a net-frame -> crop affine (mirror included): validation metrics map crop
    errors back to net-frame pixels with it, as they do for the affine crops."""
    offsets: Float32[Tensor, "3 2"] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=cameras.focal.device)
    u: Float32[Tensor, "n 3"] = (CROP_CENTRE + offsets[:, 0])[None].expand(cameras.focal.shape[0], 3)
    v: Float32[Tensor, "n 3"] = (CROP_CENTRE + offsets[:, 1])[None].expand_as(u)
    u_src: Float32[Tensor, "n 3"] = torch.where(cameras.mirror[:, None], (CROP_SIZE - 1) - u, u)
    local: Float32[Tensor, "n 3 3"] = torch.stack([(u_src - CROP_CENTRE) / cameras.focal[:, None], (v - CROP_CENTRE) / cameras.focal[:, None], torch.ones_like(u)], dim=-1)
    rays: Float32[Tensor, "n 3 3"] = torch.einsum("nji,nqj->nqi", cameras.rotation, local)
    native: Float32[Tensor, "n 3 2"] = project(camera, rays[:, None])[:, 0]
    net: Float32[Tensor, "n 3 2"] = letterbox.to_net(native.reshape(-1, 2)).reshape(-1, 3, 2)
    # net_from_crop columns: d(net)/du, d(net)/dv at the centre, then the centre itself.
    net_from_crop: Float32[Tensor, "n 3 3"] = torch.zeros((u.shape[0], 3, 3), device=u.device)
    net_from_crop[:, :2, 0] = net[:, 1] - net[:, 0]
    net_from_crop[:, :2, 1] = net[:, 2] - net[:, 0]
    net_from_crop[:, :2, 2] = net[:, 0] - CROP_CENTRE * (net_from_crop[:, :2, 0] + net_from_crop[:, :2, 1])
    net_from_crop[:, 2, 2] = 1.0
    return torch.linalg.inv(net_from_crop)
