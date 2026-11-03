"""Write a `soma_xlo` layer: the fitted SOMA body at SOMA-X's xlo LOD, with no refit.

Each xlo vertex takes the per-frame position of its nearest low-LOD vertex in the neutral rest pose, read from the
upper-body crop stored in the segment's soma layer. Xlo vertices whose nearest low vertex lies outside the crop are dropped.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from dataforge.selfcap_schema import SOMA_XLO
from dataforge.writing import atomic_write
from jaxtyping import Float32, Int32, Int64
from numpy import ndarray
from rerun.chunk import Chunk, ChunkStore, LazyChunkStream, OptimizationProfile, RrdReader
from soma.body import SOMALayer, SOMAPoseOutput

from selfcap_soma.assets import soma_assets
from selfcap_soma.batch import selected_segments
from selfcap_soma.overlay import SOMA_MESH, mesh_vertex_rows, read_layer_crop

APPLICATION_ID: str = "selfcap-soma-xlo"
ALBEDO: tuple[float, float, float, float] = (0.25, 0.65, 0.8, 0.5)


@dataclass(frozen=True, slots=True)
class Config:
    soma_dir: Path
    """Batch output root holding each registered soma layer at <dataset>/<segment>/<segment>.rrd."""
    dataset: str
    """Dataset directory under soma_dir."""
    output_dir: Path
    """Destination for <segment>.rrd; an existing file is refused. Nothing is shipped or registered."""
    segment: list[str] = field(default_factory=list)
    """One or more SelfCap segment IDs."""
    segments_file: Path | None = None
    """Optional newline-separated IDs; blank/comment lines are ignored."""
    assets: Path | None = None
    """Local SOMA-X directory; None takes the pinned download."""


@dataclass(frozen=True, slots=True)
class RestMeshes:
    low_vertices: Float32[ndarray, "low 3"]
    """Low-LOD rest vertices."""
    xlo_vertices: Float32[ndarray, "xlo 3"]
    """Xlo-LOD rest vertices, in the same frame as low_vertices."""
    xlo_faces: Int32[ndarray, "f 3"]
    """Xlo-LOD triangles."""


@dataclass(frozen=True, slots=True)
class XloMapping:
    sources: Int64[ndarray, "kept"]
    """Crop-vertex index of each kept xlo vertex, in xlo order."""
    faces: Int32[ndarray, "f 3"]
    """Xlo triangles fully inside the kept vertices, remapped to them."""


def rest_meshes(assets: Path | None) -> RestMeshes:
    """Low and xlo rest meshes of the neutral identity, at zero pose, with no correctives."""
    meshes: dict[str, tuple[Float32[ndarray, "v 3"], Int32[ndarray, "f 3"]]] = {}
    for lod in ("low", "xlo"):
        model: SOMALayer = SOMALayer(
            data_root=soma_assets(assets),
            device="cpu",
            identity_model_type="soma",
            mode="torch",
            lod=lod,
            enable_procedural_transforms=False,
            correctives_model_path=None,
        )
        model.requires_grad_(False)
        with torch.no_grad():
            model.prepare_identity(torch.zeros(1, 128), scale_params=None, global_scale=1.0)
            pose: SOMAPoseOutput = model.pose(torch.zeros(1, 77, 3), apply_correctives=False)
        meshes[lod] = (pose.vertices[0].numpy(), model.faces.numpy())
    return RestMeshes(meshes["low"][0], meshes["xlo"][0], meshes["xlo"][1])


def xlo_mapping(rest: RestMeshes, crop: Int64[ndarray, "v"]) -> XloMapping:
    """Route each xlo vertex through its nearest low vertex into the crop."""
    distances: Float32[ndarray, "xlo low"] = ((rest.xlo_vertices[:, None, :] - rest.low_vertices[None, :, :]) ** 2).sum(axis=-1)
    nearest: Int64[ndarray, "xlo"] = np.argmin(distances, axis=1)
    rank: Int64[ndarray, "low"] = np.full(len(rest.low_vertices), -1, dtype=np.int64)
    rank[crop] = np.arange(len(crop), dtype=np.int64)
    sources: Int64[ndarray, "xlo"] = rank[nearest]
    kept: Int64[ndarray, "kept"] = np.flatnonzero(sources >= 0)
    remap: Int64[ndarray, "xlo"] = np.full(len(sources), -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)
    faces: Int64[ndarray, "f 3"] = remap[rest.xlo_faces]
    return XloMapping(sources[kept], faces[(faces >= 0).all(axis=1)].astype(np.int32))


def vertex_normals(vertices: Float32[ndarray, "n v 3"], faces: Int32[ndarray, "f 3"]) -> Float32[ndarray, "n v 3"]:
    """Area-weighted vertex normals for every frame."""
    corners: Float32[ndarray, "n f 3 3"] = vertices[:, faces]
    face_normals: Float32[ndarray, "n f 3"] = np.cross(corners[:, :, 1] - corners[:, :, 0], corners[:, :, 2] - corners[:, :, 0])
    normals: Float32[ndarray, "n v 3"] = np.zeros_like(vertices)
    for corner in range(3):
        # Swapped views: the frame axis follows the vertex index, and the sum lands in normals.
        np.add.at(normals.swapaxes(0, 1), faces[:, corner], face_normals.swapaxes(0, 1))
    return normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)


def write_segment(config: Config, rest: RestMeshes, segment: str) -> Path:
    """Write one segment's layer from its soma layer's crop and mesh."""
    target: Path = config.output_dir / f"{segment}.rrd"
    if target.exists():
        raise FileExistsError(f"{target} already exists; refusing to overwrite it")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    directory: Path = config.soma_dir / config.dataset / segment
    layer_path: Path = directory / f"{segment}.rrd"
    crop: Int64[ndarray, "v"] = read_layer_crop(RrdReader(layer_path))
    mapping: XloMapping = xlo_mapping(rest, crop)

    def to_xlo(chunk: Chunk) -> Chunk:
        batch: pa.RecordBatch = chunk.to_record_batch()
        stored: Float32[ndarray, "n v 3"] = mesh_vertex_rows(batch.column("Mesh3D:vertex_positions"))
        if stored.shape[1] != len(crop):
            raise ValueError(f"stored mesh has {stored.shape[1]} vertices, the layer's crop has {len(crop)}")
        vertices: Float32[ndarray, "n kept 3"] = stored[:, mapping.sources]
        return Chunk.from_columns(
            SOMA_XLO,
            indexes=[rr.TimeColumn("video_time", duration=batch.column("video_time").cast("int64").to_numpy().astype("timedelta64[ns]"))],
            columns=rr.Mesh3D.columns(
                vertex_positions=vertices.reshape(-1, 3), vertex_normals=vertex_normals(vertices, mapping.faces).reshape(-1, 3)
            ).partition([vertices.shape[1]] * len(vertices)),
        )

    static: Chunk = Chunk.from_columns(
        SOMA_XLO,
        indexes=[],
        columns=[*rr.Mesh3D.columns(triangle_indices=mapping.faces).partition([len(mapping.faces)]), *rr.Mesh3D.columns(albedo_factor=[ALBEDO])],
    )
    frames: LazyChunkStream = (
        RrdReader(layer_path).stream().filter(content=SOMA_MESH, is_static=False, components="Mesh3D:vertex_positions").map(to_xlo)
    )
    layer: LazyChunkStream = LazyChunkStream.merge(LazyChunkStream.from_iter([static]), frames)
    store: ChunkStore = layer.collect(optimize=OptimizationProfile.OBJECT_STORE)
    if not store.stream().filter(is_static=False).to_chunks():
        raise ValueError(f"{layer_path} has no mesh frames on {SOMA_MESH}")
    with atomic_write(target, exclusive=True) as staged:
        store.write_rrd(staged, application_id=APPLICATION_ID, recording_id=segment)
    return target


def main(config: Config) -> None:
    """Write every segment; a failed one is reported and exits nonzero after the others finish."""
    segments: list[str] = selected_segments(config.segment, config.segments_file)
    rest: RestMeshes = rest_meshes(config.assets)
    written: int = 0
    for segment in segments:
        try:
            target: Path = write_segment(config, rest, segment)
        except (ValueError, OSError, RuntimeError) as error:
            print(f"soma_xlo FAIL {segment}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            continue
        written += 1
        print(f"soma_xlo ok {segment}: {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"soma_xlo: {written}/{len(segments)} segments written to {config.output_dir}", flush=True)
    if written != len(segments):
        raise SystemExit(1)
