"""Calibrated world placement and compact Rerun layers for EgoHand output."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64, Int64, UInt32
from numpy import ndarray
from rerun.chunk import Chunk, DeriveLens, IndexColumn, LazyChunkStream, OptimizationProfile, Selector
from rerun.experimental import ParquetReader
from scipy.spatial.transform import Rotation, Slerp
from serde import serde

from posekit.apis.egohand_comparison import EGO_POSE_MAX_GAP_S, EgoCamera
from posekit.skeletons import SOMA_HAND


def camera_world_poses(camera: EgoCamera, times_s: Float64[ndarray, "n"]) -> tuple[Float64[ndarray, "n 4 4"], Bool[ndarray, "n"]]:
    """Interpolate the calibrated rig track and compose its camera extrinsic.

    Matches selfcap-curate project_ego's translation lerp, rotation slerp and
    100 ms pose-gap policy. Invalid times are marked, never extrapolated.
    """
    right: Int64[ndarray, "n"] = np.clip(np.searchsorted(camera.times_s, times_s), 1, len(camera.times_s) - 1)
    left: Int64[ndarray, "n"] = right - 1
    gap: Float64[ndarray, "n"] = camera.times_s[right] - camera.times_s[left]
    valid: Bool[ndarray, "n"] = (times_s >= camera.times_s[0]) & (times_s <= camera.times_s[-1]) & (gap <= EGO_POSE_MAX_GAP_S)
    clipped: Float64[ndarray, "n"] = np.clip(times_s, camera.times_s[0], camera.times_s[-1])
    alpha: Float64[ndarray, "n 1"] = ((clipped - camera.times_s[left]) / np.maximum(gap, 1e-12))[:, None]
    poses: Float64[ndarray, "n 4 4"] = np.repeat(np.eye(4)[None], len(times_s), axis=0)
    poses[:, :3, :3] = Slerp(camera.times_s, Rotation.from_matrix(camera.world_T_rig[:, :3, :3]))(clipped).as_matrix()
    poses[:, :3, 3] = (1.0 - alpha) * camera.world_T_rig[left, :3, 3] + alpha * camera.world_T_rig[right, :3, 3]
    return poses @ camera.rig_T_cam, valid


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class MeshLod:
    """Fixed connected vertex clusters shared by all posed display meshes."""

    groups: Int64[ndarray, "v"]
    """Source vertex to display vertex assignment."""
    faces: UInt32[ndarray, "f 3"]
    """Nondegenerate triangles in the display topology."""

    def apply(self, vertices: Float32[ndarray, "v 3"]) -> Float32[ndarray, "m 3"]:
        """Average connected source vertices; this commutes with rigid motion."""
        count: Int64[ndarray, "m"] = np.bincount(self.groups)
        return np.stack([np.bincount(self.groups, weights=vertices[:, axis]) / count for axis in range(3)], axis=1).astype(np.float32)


def make_mesh_lod(vertices: Float32[ndarray, "v 3"], faces: UInt32[ndarray, "f 3"], max_vertices: int = 1000) -> MeshLod:
    """Collapse short topological edges, merging at most four source vertices.

    Clusters cannot jump between nearby fingers. A fixed source assignment
    avoids frame-dependent topology and allows triangles to remain static.
    """
    parents: Int64[ndarray, "v"] = np.arange(len(vertices), dtype=np.int64)
    sizes: Int64[ndarray, "v"] = np.ones(len(vertices), dtype=np.int64)
    edges: UInt32[ndarray, "e 2"] = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths: Float32[ndarray, "e"] = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    remaining: int = len(vertices)
    for edge in edges[np.argsort(lengths)]:
        if remaining <= max_vertices:
            break
        a, b = int(edge[0]), int(edge[1])
        while parents[a] != a:
            a = int(parents[a])
        while parents[b] != b:
            b = int(parents[b])
        if a != b and sizes[a] + sizes[b] <= 4:
            parents[b] = a
            sizes[a] += sizes[b]
            remaining -= 1
    for index in range(len(parents)):
        root: int = index
        while parents[root] != root:
            root = int(parents[root])
        parents[index] = root
    groups: Int64[ndarray, "v"] = np.unique(parents, return_inverse=True)[1]
    reduced: UInt32[ndarray, "f 3"] = groups[faces].astype(np.uint32)
    reduced = reduced[(reduced[:, 0] != reduced[:, 1]) & (reduced[:, 1] != reduced[:, 2]) & (reduced[:, 0] != reduced[:, 2])]
    return MeshLod(groups, reduced)


def mesh_normals(vertices: Float32[ndarray, "v 3"], faces: UInt32[ndarray, "f 3"]) -> Float32[ndarray, "v 3"]:
    """Area-weighted unit vertex normals, with zero for isolated vertices."""
    triangle_normals: Float32[ndarray, "f 3"] = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals: Float32[ndarray, "v 3"] = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], triangle_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), np.float32(1e-12))


def write_world_layer(parts: dict[str, Path], lods: dict[str, MeshLod], output: Path, application_id: str, recording_id: str) -> None:
    """Convert inference Parquet streams through lenses into an optimized layer.

    Only derived entities are emitted. Topology and annotation constants are
    genuine model metadata; they are the only hand-built static chunks.
    """
    # Parquet changes child field names/nullability; Rerun requires canonical vector layouts.
    position_type: pa.DataType = rr.components.Position3DBatch([]).as_arrow_array().type
    normal_type: pa.DataType = rr.components.Vector3DBatch([]).as_arrow_array().type
    pixel_type: pa.DataType = rr.components.Position2DBatch([]).as_arrow_array().type
    streams: list[LazyChunkStream] = []
    metadata: list[Chunk] = []
    color: tuple[int, int, int] = (255, 145, 35)
    context: rr.ClassDescription = rr.ClassDescription(
        info=rr.AnnotationInfo(id=0, label="EgoHand", color=color), keypoint_connections=SOMA_HAND.links
    )
    for side, path in parts.items():
        world: str = f"/world/pred/egohand/{side}"
        lenses: list[DeriveLens] = [
            DeriveLens("data", output_entity=f"{world}/mesh")
            .to_component(rr.Mesh3D.descriptor_vertex_positions(), Selector(".vertices[]").pipe(lambda values: values.cast(position_type)))
            .to_component(rr.Mesh3D.descriptor_vertex_normals(), Selector(".normals[]").pipe(lambda values: values.cast(normal_type))),
            DeriveLens("data", output_entity=f"{world}/joints").to_component(rr.Points3D.descriptor_positions(), Selector(".joints[]").pipe(lambda values: values.cast(position_type))),
        ]
        metadata.extend(
            [
                Chunk.from_columns(world, indexes=[], columns=rr.AnnotationContext.columns(context=rr.components.AnnotationContext([context]))),
                Chunk.from_columns(
                    f"{world}/mesh", indexes=[], columns=rr.Mesh3D.columns(triangle_indices=[lods[side].faces], albedo_factor=[color])
                ),
                Chunk.from_columns(
                    f"{world}/joints",
                    indexes=[],
                    columns=rr.Points3D.columns(keypoint_ids=[np.arange(25)], class_ids=[0], radii=[0.003], colors=[color], show_labels=[False]),
                ),
            ]
        )
        for cam, field in ((0, "rgb"), (1, "mono_left"), (2, "mono_right")):
            image: str = f"/world/rig_04/cam_{cam:02d}/pinhole/pred/egohand/{side}"
            lenses.append(DeriveLens("data", output_entity=image).to_component(rr.Points2D.descriptor_positions(), Selector(f".{field}[]").pipe(lambda values: values.cast(pixel_type))))
            metadata.extend(
                [
                    Chunk.from_columns(image, indexes=[], columns=rr.AnnotationContext.columns(context=rr.components.AnnotationContext([context]))),
                    Chunk.from_columns(
                        image,
                        indexes=[],
                        columns=rr.Points2D.columns(keypoint_ids=[np.arange(25)], class_ids=[0], colors=[color], radii=[2.5], show_labels=[False]),
                    ),
                ]
            )
        stream: LazyChunkStream = ParquetReader(path).stream(index_columns=[IndexColumn.duration("video_time")]).drop(content="/__properties/**")
        streams.append(stream.lenses(lenses, content="/hand", output_mode="drop_unmatched"))
    streams.append(LazyChunkStream.from_iter(metadata))
    LazyChunkStream.merge(*streams).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(
        output, application_id=application_id, recording_id=recording_id
    )
