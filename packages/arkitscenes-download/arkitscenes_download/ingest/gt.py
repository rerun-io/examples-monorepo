"""Ground-truth mesh and oriented box logging."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import trimesh
from jaxtyping import Float64
from simplecv.rerun_log_utils import mesh_bounding_geometry

from arkitscenes_download.ingest.paths import GT_MESH, gt_box


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    """One world-frame oriented annotation box."""

    center_xyz: Float64[np.ndarray, "3"]
    """World-frame center in metres."""
    half_sizes_xyz: Float64[np.ndarray, "3"]
    """Half extents along the box axes."""
    axes_33: Float64[np.ndarray, "3 3"]
    """Box-to-world rotation matrix."""
    label: str
    """Semantic category label."""
    uid: str
    """Stable annotation identifier."""


@dataclass(frozen=True, slots=True)
class GroundTruthSummary:
    """Box count and mesh framing geometry from the ground-truth layer."""

    box_count: int
    """Number of annotated oriented boxes."""
    mesh_center_xyz: Float64[np.ndarray, "3"]
    """World-frame center of the mesh's axis-aligned bounds."""
    bounding_radius_m: float
    """Radius of the vertex bounding sphere around that center."""


def stable_label_color(label: str) -> tuple[int, int, int]:
    """Return a bright deterministic RGB color shared by 2D and 3D boxes."""
    digest: bytes = hashlib.sha256(label.encode()).digest()
    return tuple(80 + value % 176 for value in digest[:3])  # type: ignore[return-value]


def load_ground_truth_boxes(sequence_dir: Path, video_id: str) -> list[GroundTruthBox]:
    """Load oriented box annotations in world coordinates."""
    annotations: dict[str, Any] = json.loads((sequence_dir / f"{video_id}_3dod_annotation.json").read_text())
    boxes: list[GroundTruthBox] = []
    for item in annotations["data"]:
        box: dict[str, Any] = item["segments"]["obbAligned"]
        boxes.append(
            GroundTruthBox(
                center_xyz=np.asarray(box["centroid"], dtype=np.float64),
                half_sizes_xyz=np.asarray(box["axesLengths"], dtype=np.float64) / 2.0,
                axes_33=np.asarray(box["normalizedAxes"], dtype=np.float64).reshape(3, 3),
                label=str(item["label"]),
                uid=str(item["uid"]),
            )
        )
    return boxes


def log_ground_truth(sequence_dir: Path, video_id: str, recording: rr.RecordingStream) -> GroundTruthSummary:
    """Log the reconstructed mesh and each annotated oriented box."""
    boxes: list[GroundTruthBox] = load_ground_truth_boxes(sequence_dir, video_id)
    loaded = trimesh.load(sequence_dir / f"{video_id}_3dod_mesh.ply", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"expected a single triangle mesh in {video_id}_3dod_mesh.ply, got {type(loaded).__name__}")
    vertices: Float64[np.ndarray, "n 3"] = np.asarray(loaded.vertices, dtype=np.float64)
    mesh_center_xyz, bounding_radius_m = mesh_bounding_geometry(vertices)
    recording.log(
        GT_MESH,
        rr.Mesh3D(
            vertex_positions=vertices,
            vertex_colors=np.asarray(loaded.visual.vertex_colors),  # pyrefly: ignore  # missing-attribute
            triangle_indices=np.asarray(loaded.faces),
            face_rendering=rr.components.MeshFaceRendering.Front,
        ),
        static=True,
    )
    for box in boxes:
        entity_path: str = gt_box(f"box-{box.uid}-{box.label}")
        recording.log(
            entity_path,
            rr.Boxes3D(half_sizes=box.half_sizes_xyz, labels=box.label, colors=stable_label_color(box.label)),
            static=True,
        )
        # normalizedAxes rows are the box axes in world coordinates: world = axes.T @ local + center
        # (Apple's compute_box_3d applies np.transpose(rotmat)); InstancePoses3D applies mat3x3 untransposed.
        # Center goes in InstancePoses3D translations, NOT Boxes3D centers: instance poses compose
        # scale->rotate->translate, whereas Boxes3D centers + separate rotation rotates the center
        # around the entity origin (misplacing rotated boxes).
        recording.log(entity_path, rr.InstancePoses3D(translations=box.center_xyz, mat3x3=box.axes_33.T), static=True)
    return GroundTruthSummary(box_count=len(boxes), mesh_center_xyz=mesh_center_xyz, bounding_radius_m=bounding_radius_m)
