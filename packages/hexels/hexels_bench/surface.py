"""Surfaces from rendered depth, following upstream Hexels' mesh extraction line by line.

Reference: hexels@698d1af `scripts/extract_hexels_dtu_mesh.py` (L.22-54 cameras and post-process,
L.96-170 fusion) and the 2DGS camera loader it uses (`scene/dataset_readers.py` readColmapCameras,
`utils/camera_utils.py` loadCam, `utils/graphics_utils.py` getWorld2View2, focal2fov). Depth and color
come from `gsplat render --surface` instead of upstream's CUDA renderer; cameras, masks, TSDF
settings and post-processing are upstream's.
"""

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import open3d as o3d
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from PIL import Image

from hexels_bench.cameras import ContractCamera

DepthMode: TypeAlias = Literal["median", "expected"]

_COLMAP_PARAMS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}
"""COLMAP camera model id -> (name, parameter count); 2DGS accepts only the two pinhole models."""


@dataclass(frozen=True, slots=True)
class TsdfSettings:
    """Open3D ScalableTSDFVolume settings of one scene (upstream docs/datasets_2dgs.md, paper App. F)."""

    voxel_size: float
    """Voxel edge in scene units."""
    sdf_trunc: float
    """SDF truncation distance."""
    depth_trunc: float
    """Depth beyond this is not fused."""
    clusters_to_keep: int = 1
    """Post-processing keeps clusters at least as large as the n-th largest (upstream --num_cluster 1)."""


def tsdf_settings(dataset: str, scene: str) -> TsdfSettings:
    """DTU and T&T Barn/Caterpillar/Ignatius/Truck: 0.004/0.016/3.0; T&T Courthouse/Meetingroom: 0.006/0.024/4.5.
    Mip-NeRF 360 has no upstream mesh protocol: 0.01/0.04/6.0 and the 50 largest clusters (ours, visual only), so
    the 3-unit DTU cut and the single-cluster rule do not drop everything but the centre object."""
    if dataset == "mipnerf360":
        return TsdfSettings(voxel_size=0.01, sdf_trunc=0.04, depth_trunc=6.0, clusters_to_keep=50)
    if scene in ("Courthouse", "Meetingroom"):
        return TsdfSettings(voxel_size=0.006, sdf_trunc=0.024, depth_trunc=4.5)
    return TsdfSettings(voxel_size=0.004, sdf_trunc=0.016, depth_trunc=3.0)


def _read_colmap(
    sparse_dir: Path,
) -> tuple[dict[int, tuple[str, int, int, tuple[float, ...]]], list[tuple[str, tuple[float, ...], tuple[float, ...], int]]]:
    """COLMAP cameras {id: (model, width, height, params)} and images [(name, qvec wxyz, tvec, camera id)], binary first."""
    cameras: dict[int, tuple[str, int, int, tuple[float, ...]]] = {}
    images: list[tuple[str, tuple[float, ...], tuple[float, ...], int]] = []
    if (sparse_dir / "images.bin").is_file():
        data: bytes = (sparse_dir / "cameras.bin").read_bytes()
        offset: int = 8
        for _ in range(struct.unpack_from("<Q", data, 0)[0]):
            camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
            model, count = _COLMAP_PARAMS[model_id]
            cameras[camera_id] = (model, width, height, struct.unpack_from(f"<{count}d", data, offset + 24))
            offset += 24 + 8 * count
        data = (sparse_dir / "images.bin").read_bytes()
        offset = 8
        for _ in range(struct.unpack_from("<Q", data, 0)[0]):
            record: tuple = struct.unpack_from("<i7di", data, offset)
            end: int = data.index(b"\x00", offset + 64)
            images.append((data[offset + 64 : end].decode(), record[1:5], record[5:8], record[8]))
            offset = end + 1 + 8 + 24 * struct.unpack_from("<Q", data, end + 1)[0]
        return cameras, images
    for line in (sparse_dir / "cameras.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            fields: list[str] = line.split()
            cameras[int(fields[0])] = (fields[1], int(fields[2]), int(fields[3]), tuple(float(v) for v in fields[4:]))
    lines: list[str] = [line for line in (sparse_dir / "images.txt").read_text().splitlines() if not line.startswith("#")]
    for line in lines[0::2]:
        fields = line.split()
        if fields:
            images.append((fields[9], tuple(float(v) for v in fields[1:5]), tuple(float(v) for v in fields[5:8]), int(fields[8])))
    return cameras, images


def _rotation(qvec: tuple[float, ...]) -> Float64[ndarray, "3 3"]:
    """COLMAP qvec2rotmat (w, x, y, z): world-to-camera rotation."""
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def scene_cameras(scene_dir: Path, images: str, resolution: int) -> list[ContractCamera]:
    """Every registered view of a COLMAP scene as upstream's 2DGS loader builds it, sorted by image name.

    Size `round(w / r), round(h / r)` of the image file; field of view from the COLMAP focal and the
    COLMAP image size; `world_view_transform` = float32 `inv(inv([R | t]))`, transposed (row-vector form).
    """
    if resolution not in (1, 2, 4, 8):
        raise ValueError(f"resolution {resolution}: only the fixed 2DGS factors 1, 2, 4, 8 are supported")
    model: tuple[dict[int, tuple[str, int, int, tuple[float, ...]]], list[tuple[str, tuple[float, ...], tuple[float, ...], int]]] = _read_colmap(
        scene_dir / "sparse" / "0"
    )
    colmap_cameras: dict[int, tuple[str, int, int, tuple[float, ...]]] = model[0]
    records: list[ContractCamera] = []
    for path_name, qvec, tvec, camera_id in model[1]:
        camera_model, colmap_width, colmap_height, params = colmap_cameras[camera_id]
        if camera_model not in ("SIMPLE_PINHOLE", "PINHOLE"):
            raise ValueError(f"{scene_dir}: camera model {camera_model}; 2DGS loads undistorted pinhole cameras only")
        focal_x: float = params[0]
        focal_y: float = params[0] if camera_model == "SIMPLE_PINHOLE" else params[1]
        file_name: str = Path(path_name).name
        with Image.open(scene_dir / images / file_name) as image:
            width, height = image.size
        world_to_camera: Float64[ndarray, "4 4"] = np.eye(4)
        world_to_camera[:3, :3] = _rotation(qvec)
        world_to_camera[:3, 3] = tvec
        world_view: Float32[ndarray, "4 4"] = np.linalg.inv(np.linalg.inv(world_to_camera)).astype(np.float32).T
        center: Float32[ndarray, "3"] = np.linalg.inv(world_view.astype(np.float64))[3, :3].astype(np.float32)
        records.append(
            ContractCamera(
                name=file_name.split(".")[0],
                width=round(width / resolution),
                height=round(height / resolution),
                fov_x=2 * math.atan(colmap_width / (2 * focal_x)),
                fov_y=2 * math.atan(colmap_height / (2 * focal_y)),
                world_view_transform=world_view.astype(np.float64).tolist(),
                camera_center=center.astype(np.float64).tolist(),
            )
        )
    return sorted(records, key=lambda camera: camera.name)


def alpha_mask(image_path: Path, width: int, height: int) -> Bool[ndarray, "h w"] | None:
    """Pixels upstream keeps: the image's alpha band resized alone (PIL default filter), `alpha / 255 >= 0.5`.

    None for images without alpha (upstream `gt_alpha_mask` is None, nothing is masked)."""
    with Image.open(image_path) as image:
        if len(image.getbands()) <= 3:
            return None
        alpha: UInt8[ndarray, "h w"] = np.asarray(image.split()[3].resize((width, height)), dtype=np.uint8)
    return alpha.astype(np.float32) / np.float32(255.0) >= 0.5


def surface_depth(
    median: Float32[ndarray, "h w"],
    expected: Float32[ndarray, "h w"],
    alpha: Float32[ndarray, "h w"],
    mode: DepthMode,
    alpha_threshold: float,
    mask: Bool[ndarray, "h w"] | None,
) -> Float32[ndarray, "h w"]:
    """The depth upstream fuses (extract_hexels_dtu_mesh.py L.112-116): zero where alpha <= threshold or
    outside the mask. `expected` is already divided by alpha (WGSL surface target)."""
    depth: Float32[ndarray, "h w"] = np.where(alpha > alpha_threshold, median if mode == "median" else expected, np.float32(0.0))
    if mask is not None:
        depth = np.where(mask, depth, np.float32(0.0))
    return depth


def intrinsic(camera: ContractCamera) -> o3d.camera.PinholeCameraIntrinsic:
    """Upstream `_camera_parameters`: focal from the field of view, principal point (W - 1) / 2."""
    return o3d.camera.PinholeCameraIntrinsic(
        width=camera.width,
        height=camera.height,
        fx=0.5 * camera.width / math.tan(0.5 * camera.fov_x),
        fy=0.5 * camera.height / math.tan(0.5 * camera.fov_y),
        cx=(camera.width - 1) / 2,
        cy=(camera.height - 1) / 2,
    )


def camera_from_world(camera: ContractCamera) -> Float64[ndarray, "4 4"]:
    """Upstream `world_view_transform.T`: the OpenCV world-to-camera matrix."""
    return np.asarray(camera.world_view_transform, dtype=np.float32).astype(np.float64).T


@dataclass(frozen=True, slots=True)
class FusionView:
    """One view to integrate."""

    camera: ContractCamera
    """Upstream camera record."""
    rgb: UInt8[ndarray, "h w 3"]
    """Rendered color."""
    depth: Float32[ndarray, "h w"]
    """Camera z per pixel, zero where nothing is fused."""


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Raw marching-cubes mesh and the fusion counters upstream records in extraction.json."""

    mesh: o3d.geometry.TriangleMesh
    """Raw TSDF mesh (`fuse.ply`)."""
    views: int
    """Integrated views."""
    valid_depth_fraction: float
    """Fraction of pixels with 0 < depth < depth_trunc."""


def fuse_tsdf(views: Iterable[FusionView], settings: TsdfSettings) -> FusionResult:
    """Integrate RGB-D views into upstream's ScalableTSDFVolume (RGB8) and extract the triangle mesh."""
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=settings.voxel_size, sdf_trunc=settings.sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    count: int = 0
    valid_pixels: int = 0
    total_pixels: int = 0
    for view in views:
        valid_pixels += int(np.count_nonzero((view.depth > 0) & (view.depth < settings.depth_trunc)))
        total_pixels += view.depth.size
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(view.rgb)),
            o3d.geometry.Image(np.ascontiguousarray(view.depth)),
            depth_scale=1.0,
            depth_trunc=settings.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic(view.camera), camera_from_world(view.camera))
        count += 1
    if count == 0:
        raise ValueError("no views to fuse")
    return FusionResult(mesh=volume.extract_triangle_mesh(), views=count, valid_depth_fraction=valid_pixels / total_pixels)


def post_process_mesh(mesh: o3d.geometry.TriangleMesh, clusters_to_keep: int = 1) -> o3d.geometry.TriangleMesh:
    """Upstream `_post_process_mesh`: keep clusters at least as large as the `clusters_to_keep`-th largest
    (and >= 50 triangles), drop unreferenced vertices and degenerate triangles. Edits `mesh` in place."""
    clustered: tuple = mesh.cluster_connected_triangles()
    cluster_ids: ndarray = np.asarray(clustered[0])
    sizes: ndarray = np.asarray(clustered[1])
    threshold: int = max(int(np.sort(sizes)[-min(clusters_to_keep, len(sizes))]), 50)
    mesh.remove_triangles_by_mask(sizes[cluster_ids] < threshold)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


def normals_from_depth(depth: Float32[ndarray, "h w"], camera: ContractCamera) -> Float32[ndarray, "h w 3"]:
    """World normals of the depth map's surface (upstream trainer.py L.570, 2DGS depth_to_normal):
    unproject every pixel, `normalize((P[y+1,x] - P[y-1,x]) x (P[y,x+1] - P[y,x-1]))`, zero on the border."""
    height, width = depth.shape
    focal_x: float = 0.5 * width / math.tan(0.5 * camera.fov_x)
    focal_y: float = 0.5 * height / math.tan(0.5 * camera.fov_y)
    grid: Float64[ndarray, "2 h w"] = np.mgrid[0:height, 0:width].astype(np.float64)
    z: Float64[ndarray, "h w"] = depth.astype(np.float64)
    points_cam: Float64[ndarray, "h w 3"] = np.stack(
        [(grid[1] - (width - 1) / 2) / focal_x * z, (grid[0] - (height - 1) / 2) / focal_y * z, z], axis=-1
    )
    world_from_camera: Float64[ndarray, "4 4"] = np.linalg.inv(camera_from_world(camera))
    points: Float64[ndarray, "h w 3"] = points_cam @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    normal: Float64[ndarray, "h w 3"] = np.zeros((height, width, 3))
    normal[1:-1, 1:-1] = np.cross(points[2:, 1:-1] - points[:-2, 1:-1], points[1:-1, 2:] - points[1:-1, :-2])
    length: Float64[ndarray, "h w 1"] = np.linalg.norm(normal, axis=-1, keepdims=True)
    return (normal / np.maximum(length, 1e-12)).astype(np.float32)
