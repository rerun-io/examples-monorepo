"""Nerfstudio export at the end of a run: keyframe images, OpenGL poses, a merged point cloud and ``transforms.json``."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d
import torch
from dagkit.node import Emit, PortType, RunContext, Tick
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde
from serde.json import to_json
from simplecv.ops import conventions
from torch import Tensor

from mast3r_slam.nodes.shared import SharedKeyframes, SharedSlam
from mast3r_slam.types import Frame, TrackedFrame

CONFIDENCE_THRESH: float = 100.0
"""Smallest accumulated confidence of a point kept in the sparse cloud."""
VOXEL_M: float = 0.03
"""Voxel size of the sparse cloud's downsampling (canonical units)."""


@serde
class NSFrame:
    """A single frame entry in NerfStudio's ``transforms.json`` format."""

    file_path: str
    """Relative path to the frame's image file."""
    transform_matrix: Float32[np.ndarray, "4 4"]
    """4x4 camera transformation matrix in OpenGL (RUB) convention."""
    colmap_im_id: int
    """COLMAP-style image ID (sequential index)."""


@serde
class NerfstudioData:
    """Top-level NerfStudio dataset descriptor written to ``transforms.json``."""

    w: int
    """Image width in pixels."""
    h: int
    """Image height in pixels."""
    fl_x: float
    """Focal length along the x-axis."""
    fl_y: float
    """Focal length along the y-axis."""
    cx: float
    """Principal point x-coordinate."""
    cy: float
    """Principal point y-coordinate."""
    k1: float
    """First radial distortion coefficient."""
    k2: float
    """Second radial distortion coefficient."""
    p1: float
    """First tangential distortion coefficient."""
    p2: float
    """Second tangential distortion coefficient."""
    camera_model: Literal["OPENCV"]
    """Camera model identifier (always OPENCV for this exporter)."""
    frames: list[NSFrame]
    """List of per-frame pose entries."""
    applied_transform: Float32[ndarray, "3 4"]
    """3x4 affine transform applied to all poses (typically identity)."""
    ply_file_path: Literal["sparse_pc.ply"]
    """Relative path to the sparse point cloud PLY file."""


def estimate_focal(points: Float32[Tensor, "h w 3"]) -> float:
    """Focal length (pixels, square pixels, centred principal point) that best reprojects a point map: Weiszfeld IRLS."""
    h, w = points.shape[:2]
    v, u = torch.meshgrid(torch.arange(h, dtype=points.dtype), torch.arange(w, dtype=points.dtype), indexing="ij")
    pixels: Float32[Tensor, "hw 2"] = torch.stack([u - w / 2, v - h / 2], dim=-1).reshape(-1, 2)
    xy_over_z: Float32[Tensor, "hw 2"] = (points[..., :2] / points[..., 2:3]).reshape(-1, 2).nan_to_num(posinf=0, neginf=0)
    dot_xy_px: Float32[Tensor, " hw"] = (xy_over_z * pixels).sum(dim=-1)
    dot_xy_xy: Float32[Tensor, " hw"] = xy_over_z.square().sum(dim=-1)
    focal: Float32[Tensor, ""] = dot_xy_px.mean() / dot_xy_xy.mean()
    for _ in range(10):  # re-weight by the inverse reprojection distance
        weight: Float32[Tensor, " hw"] = (pixels - focal * xy_over_z).norm(dim=-1).clip(min=1e-8).reciprocal()
        focal = (weight * dot_xy_px).mean() / (weight * dot_xy_xy).mean()
    return float(focal.clip(min=0.0))


def save_kf_to_nerfstudio(ns_save_path: Path, keyframes: SharedKeyframes) -> None:
    """Write every keyframe's image and OpenGL pose, the merged confident points (voxel-downsampled) and ``transforms.json``.

    Poses drop the Sim3 scale; the intrinsics come from the last keyframe's point map (no distortion).
    """
    images_dir: Path = ns_save_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    frames: list[NSFrame] = []
    positions: list[Float32[ndarray, "n 3"]] = []
    colors: list[UInt8[ndarray, "n 3"]] = []
    keyframe: Frame | None = None
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        assert keyframe.X_canon is not None and keyframe.C is not None
        rgb: UInt8[ndarray, "h w 3"] = (keyframe.rgb * 255).numpy().astype(np.uint8)
        cv2.imwrite(str(images_dir / f"frame_{i + 1:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        sim3: Float32[ndarray, " 8"] = keyframe.world_sim3_cam.data.reshape(-1).cpu().numpy()
        world_T_cam: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
        world_T_cam[:3, :3], world_T_cam[:3, 3] = Rotation.from_quat(sim3[3:7]).as_matrix(), sim3[:3]
        gl_pose: Float32[ndarray, "4 4"] = conventions.convert_pose(world_T_cam, src_convention=conventions.CC.CV, dst_convention=conventions.CC.GL)
        frames.append(NSFrame(file_path=f"images/frame_{i + 1:05d}.png", transform_matrix=gl_pose, colmap_im_id=i))
        keep: Bool[ndarray, " hw"] = keyframe.C.cpu().numpy().reshape(-1) > CONFIDENCE_THRESH
        positions.append(keyframe.X_canon.cpu().numpy()[keep] @ world_T_cam[:3, :3].T + world_T_cam[:3, 3])
        colors.append(rgb.reshape(-1, 3)[keep])
    if keyframe is None:
        raise ValueError("no keyframes to export")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(positions).astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack(colors).astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(ns_save_path / "sparse_pc.ply"), cloud.voxel_down_sample(voxel_size=VOXEL_M))
    h, w = (int(v) for v in keyframe.img_shape.reshape(-1))
    assert keyframe.X_canon is not None
    focal: float = estimate_focal(keyframe.X_canon.cpu().reshape(h, w, 3))
    data = NerfstudioData(w, h, focal, focal, w / 2, h / 2, 0.0, 0.0, 0.0, 0.0, "OPENCV", frames, np.eye(3, 4, dtype=np.float32), "sparse_pc.ply")
    (ns_save_path / "transforms.json").write_text(to_json(data))


class NerfstudioSink:
    """Export the keyframes for Nerfstudio.

    Exports the final map at flush (after the backend's final ``keyframes``); reads the images and point maps from the
    ``shared`` resource, so the graph adds it with ``needs=("shared",)``."""

    inputs: Mapping[str, PortType] = {"keyframes": TrackedFrame}
    outputs: Mapping[str, PortType] = {}

    def __init__(self, output_dir: Path) -> None:
        self.output_dir: Path = output_dir
        self.exported: bool = False

    def reset(self, ctx: RunContext) -> None:
        self.shared: SharedSlam = ctx.resources["shared"]  # type: ignore[assignment]
        self.exported = False

    def step(self, tick: Tick, out: Emit) -> None:
        pass

    def flush(self, out: Emit) -> None:
        if not self.exported:
            save_kf_to_nerfstudio(self.output_dir, self.shared.keyframes)
            self.exported = True
