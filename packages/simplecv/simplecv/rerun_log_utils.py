import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Float32, Float64, Int, Int64
from numpy import ndarray
from pyarrow import RecordBatch

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.rerun_custom_types import (
    PinholeWithDistortion,
)
from simplecv.rerun_custom_types import (
    confidence_scores_to_rgb as confidence_scores_to_rgb,
)

logger: logging.Logger = logging.getLogger(__name__)

def compute_vertex_normals(
    verts: Float32[ndarray, "n_verts 3"],
    faces: Int[ndarray, "n_faces 3"],
    eps: float = 1e-12,
) -> Float32[ndarray, "n_verts 3"]:
    """Compute per-vertex normals for a single mesh.

    Args:
        verts: Float32 array of vertex positions with shape ``(n_verts, 3)``.
        faces: Int array of triangle indices with shape ``(n_faces, 3)``.
        eps: Small epsilon to avoid division by zero when normalizing.

    Returns:
        Float32 array of unit vertex normals with shape ``(n_verts, 3)``; zeros for degenerate vertices.
    """

    # Expand faces to vertex triplets and fetch their positions.
    faces_i: Int[ndarray, "n_faces 3"] = faces.astype(np.int64)
    v0: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 0]]
    v1: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 1]]
    v2: Float32[ndarray, "n_faces 3"] = verts[faces_i[:, 2]]

    # Face normal = cross(edge1, edge2).
    e1: Float32[ndarray, "n_faces 3"] = v1 - v0
    e2: Float32[ndarray, "n_faces 3"] = v2 - v0
    face_normals: Float32[ndarray, "n_faces 3"] = np.cross(e1, e2)

    # Accumulate each face normal into its three vertices with a vectorized scatter-add.
    vertex_normals: Float32[ndarray, "n_verts 3"] = np.zeros_like(verts, dtype=np.float32)
    flat_indices: Int[ndarray, "n_faces3"] = faces_i.reshape(-1)
    face_normals_repeated: Float32[ndarray, "n_faces3 3"] = np.repeat(face_normals, 3, axis=0)
    np.add.at(vertex_normals, flat_indices, face_normals_repeated)

    norms: Float32[ndarray, "n_verts 1"] = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    denom: Float32[ndarray, "n_verts 1"] = np.maximum(norms, eps).astype(np.float32)
    vn_unit: Float32[ndarray, "n_verts 3"] = (vertex_normals / denom).astype(np.float32)
    mask: ndarray = norms > eps
    vn_unit = np.where(mask, vn_unit, np.float32(0.0))
    return vn_unit


def get_safe_application_id() -> str:
    """Get application ID safely, with fallback if __main__.__file__ doesn't exist"""
    try:
        main = sys.modules.get("__main__")
        if main:
            file_attr = getattr(main, "__file__", None)
            if isinstance(file_attr, str):
                return Path(file_attr).stem
    except Exception:
        pass
    return "rerun-application"  # Default fallback


FILE_FLUSH_TICK_S: float = 30.0
"""Batcher tick of a recording that only goes to a file (`file_batcher_config`), seconds."""


def file_batcher_config() -> rr.ChunkBatcherConfig:
    """The chunk batcher of a recording that only goes to a file.

    A file is read back whole, so batch for the reader, not for latency: the default 200 ms tick cuts a per-frame log
    into chunks of a few rows, and the viewer walks every chunk of a file opened by path on every frame (117k chunks
    cost ~55 ms/frame on an M5). The byte and row limits keep their defaults, and a `RERUN_FLUSH_TICK_SECS` in the
    environment keeps its value.
    """
    return rr.ChunkBatcherConfig(flush_tick=None if "RERUN_FLUSH_TICK_SECS" in os.environ else FILE_FLUSH_TICK_S)


@dataclass
class RerunTyroConfig:
    application_id: str = field(default_factory=get_safe_application_id)
    """Name of the application"""
    recording_id: str | None = None
    """Recording ID; pin one (any stable string) so separate processes/restarts
    land in the same recording instead of forking new ones (tyro cannot parse a
    UUID union member, which made this field non-configurable from the CLI)."""
    connect: bool = False
    """Whether to connect to an existing rerun instance or not"""
    save: Path | None = None
    """Path to save the rerun data, this will make it so no data is visualized but saved"""
    serve: bool = False
    """Serve the rerun data"""
    headless: bool = False
    """Run rerun in headless mode"""
    live: bool = False
    """When combined with ``save``, stream to a spawned viewer AND write the .rrd
    file simultaneously (via ``set_sinks``), instead of ``save`` being file-only.
    Ignored when ``headless`` (no viewer), or when ``serve``/``connect`` is set."""
    port: int = 9876
    """Port of the viewer's gRPC proxy (used by ``spawn``, ``live`` + ``save``, and ``connect``)."""
    server_memory_limit: str = "4GB"
    """Memory budget for the spawned viewer's gRPC proxy buffer. The SDK default
    (~1 GiB) silently drops the oldest messages on long multi-camera recordings
    (e.g. a 46-min EPFL session), so the viewer shows truncated history. Accepts a
    size (``"4GB"``) or a percentage of RAM (``"25%"``)."""
    executable_name: str = "rerun"
    """Executable name passed to ``rerun.spawn`` when launching the viewer."""
    executable_path: str | None = None
    """Optional absolute or relative path to the Rerun executable."""

    def __post_init__(self):
        rr.init(
            application_id=self.application_id,
            recording_id=self.recording_id,
            default_enabled=True,
            strict=True,
        )
        file_only = self.save is not None and not self.serve and not self.connect and (not self.live or self.headless)
        if file_only:
            # rr.init takes no batcher config, so reopen its recording id (the pinned one, else the
            # multiprocessing-shared default) with the file's (`file_batcher_config`).
            self.rec_stream: rr.RecordingStream = rr.RecordingStream(
                self.application_id,
                recording_id=self.recording_id or rr.get_recording_id(),
                make_default=True,
                default_enabled=True,
                batcher_config=file_batcher_config(),
            )
        else:
            self.rec_stream = rr.get_global_data_recording()  # type: ignore[assignment]

        if self.serve:
            rr.serve_grpc(server_memory_limit=self.server_memory_limit)
            rr.serve_web_viewer(open_browser=not self.headless)
        elif self.connect:
            # Send logging data to a separate, already-running `rerun` process
            # (honors ``port`` so multiple viewers can coexist on one machine).
            rr.connect_grpc(f"rerun+http://127.0.0.1:{self.port}/proxy")
        elif self.save is not None and self.live and not self.headless:
            # Stream to a spawned viewer AND save to a .rrd at the same time by
            # fanning out through explicit sinks. ``spawn``/``save`` each install a
            # single sink that would replace the other, so we spawn the viewer
            # process without auto-connecting and wire both sinks ourselves.
            rr.spawn(
                port=self.port,
                connect=False,
                server_memory_limit=self.server_memory_limit,
                executable_name=self.executable_name,
                executable_path=self.executable_path,
            )
            rr.set_sinks(
                rr.GrpcSink(f"rerun+http://127.0.0.1:{self.port}/proxy"),
                rr.FileSink(str(self.save)),
            )
        elif self.save is not None:
            # This config's stream by name: inside another active recording, a bare rr.save would attach the file to that one.
            rr.save(self.save, recording=self.rec_stream)
        elif not self.headless:
            rr.spawn(
                port=self.port,
                server_memory_limit=self.server_memory_limit,
                executable_name=self.executable_name,
                executable_path=self.executable_path,
            )


def log_pinhole(
    camera: PinholeParameters | Fisheye62Parameters,
    cam_log_path: Path,
    image_plane_distance: int | float = 0.5,
    static: bool = False,
    *,
    recording: rr.RecordingStream | None = None,
    include_distortion: bool = True,
) -> None:
    """
    Logs the pinhole camera parameters and transformation data.

    Parameters:
    camera (PinholeParameters): The pinhole camera parameters including intrinsics and extrinsics.
    cam_log_path (Path): The path where the camera log will be saved.
    image_plane_distance (float, optional): The distance of the image plane from the camera. Defaults to 0.5.
    static (bool, optional): If True, the log data will be marked as static. Defaults to False.

    Returns:
    None
    """
    # camera intrinsics
    rr.log(
        f"{cam_log_path}/pinhole",
        PinholeWithDistortion.from_camera(
            camera,
            image_plane_distance=image_plane_distance,
            include_distortion=include_distortion,
        ),
        static=static,
        recording=recording,
    )
    # camera extrinsics
    rr.log(
        f"{cam_log_path}",
        rr.Transform3D(
            translation=camera.extrinsics.cam_t_world,
            mat3x3=camera.extrinsics.cam_R_world,
            relation=rr.TransformRelation.ChildFromParent,
        ),
        static=static,
        recording=recording,
    )


def log_video(
    video_source: Path,
    video_log_path: Path,
    timeline: str = "video_time",
    *,
    recording: rr.RecordingStream | None = None,
    output_codec: rr.VideoCodec | None = None,
) -> Int[ndarray, "num_frames"]:
    """
    Logs a video and its frame timestamps.

    Args:
        video_source: Path to an MP4 file.
        video_log_path: The entity path where the video log will be saved.
        timeline: Timeline name for frame timestamps.
        recording: Optional specific recording stream to log to.
        output_codec: Output codec override; ``None`` keeps the source codec.

    Returns:
        Frame timestamps in nanoseconds, sorted ascending.

    Raises:
        RuntimeError: When Rerun cannot ingest or transcode the MP4.
    """
    # get_data_recording also sees a context-local stream (``with rec:``, gradio-rerun's
    # per-request recordings); the global one is unset there.
    target_recording: rr.RecordingStream | None = rr.get_data_recording(recording)
    if target_recording is None:
        raise RuntimeError("No active Rerun recording. Call rr.init() or pass recording= before logging video.")
    # The VideoStream chunks carry the sample times on ``timeline``; they are the frame
    # timestamps, so no separate frame-index probe (which finds no frames in QuickTime-brand .MOV files).
    times: list[Int[ndarray, "rows"]] = []

    def send(try_gpu: bool) -> None:
        reader: rr.experimental.Mp4Reader = rr.experimental.Mp4Reader(
            video_source,
            mode="stream",
            entity_path=str(video_log_path),
            timeline_name=timeline,
            transcode=rr.experimental.Mp4TranscodeOptions(output_codec=output_codec, try_gpu=try_gpu),
        )

        def _chunks_recording_times() -> Iterator[rrc.Chunk]:
            for chunk in reader.stream():
                if not chunk.is_static:
                    batch: RecordBatch = chunk.to_record_batch()
                    if "VideoStream:sample" in batch.schema.names:
                        times.append(batch.column(timeline).to_numpy().astype("timedelta64[ns]").astype(np.int64))
                yield chunk

        target_recording.send_chunks(_chunks_recording_times())

    # A source Rerun must re-encode (B-frames, a codec override) goes to NVENC; when that fails before any sample was
    # sent (no libnvidia-encode, as on ZeroGPU), libx264 encodes it instead.
    try:
        try:
            send(try_gpu=True)
        except RuntimeError as gpu_error:
            if times:
                raise
            logger.warning("NVENC cannot encode %s, the CPU encodes it (slower): %s", video_source, gpu_error)
            send(try_gpu=False)
    except RuntimeError as exc:
        raise RuntimeError(f"Mp4Reader failed for {video_source}: {exc}") from exc

    frame_timestamps_ns: Int[ndarray, "num_frames"] = np.sort(np.concatenate(times)) if times else np.empty(0, dtype=np.int64)
    return frame_timestamps_ns


def mp4_frame_timestamps_ns(video_path: Path) -> Int64[ndarray, "num_frames"]:
    """Frame timestamps of an MP4 file, sorted ascending: the times :func:`log_video` puts on its timeline.

    They come from ``Mp4Reader``'s asset-mode frame index, which parses the container and decodes nothing; stream mode
    would re-encode a B-frame source through FFmpeg only to read its times. The asset chunk is dropped, never logged.

    Raises:
        RuntimeError: When Rerun cannot parse the MP4 or finds no frames in it (a QuickTime-brand .MOV).
    """
    reader: rr.experimental.Mp4Reader = rr.experimental.Mp4Reader(video_path, mode="asset", entity_path="/video", timeline_name="video")
    times: list[Int64[ndarray, "rows"]] = [
        chunk.to_record_batch().column("video").to_numpy().astype("timedelta64[ns]").astype(np.int64) for chunk in reader.stream() if not chunk.is_static
    ]
    if not times:
        raise RuntimeError(f"Mp4Reader found no frames in {video_path}")
    return np.sort(np.concatenate(times))


def mesh_bounding_geometry(vertices: Float64[ndarray, "n 3"]) -> tuple[Float64[ndarray, "3"], float]:
    """Return the AABB center and the radius of the vertex bounding sphere around it."""
    center: Float64[ndarray, "3"] = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    radius: float = float(np.linalg.norm(vertices - center, axis=1).max())
    return center, radius


def orbit_eye_position(
    look_target_xyz: Float64[ndarray, "3"],
    bounding_radius_m: float,
    distance_factor: float,
    direction_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Float64[ndarray, "3"]:
    """Place a 3D-view eye along ``direction_xyz`` at ``distance_factor x bounding_radius_m`` from the target.

    Pairs with ``rrb.archetypes.EyeControls3D``: use the returned position together
    with ``look_target=look_target_xyz`` so the whole bounding sphere stays in frame
    (a ``distance_factor`` around 2.2 tightly fits a default-FOV view).
    """
    direction: Float64[ndarray, "3"] = np.asarray(direction_xyz, dtype=np.float64)
    unit: Float64[ndarray, "3"] = direction / np.linalg.norm(direction)
    return look_target_xyz + distance_factor * bounding_radius_m * unit


def log_open3d_mesh(entity_path: str, mesh: Any, *, static: bool = False, recording: rr.RecordingStream | None = None) -> None:
    """Log an Open3D ``TriangleMesh`` (e.g. from :class:`simplecv.ops.tsdf_depth_fuser.Open3DFuser`) as a ``Mesh3D``.

    Vertex normals are computed if missing. Front-face rendering culls the outward side of TSDF walls (their normals
    point into the free space), so a fused room reads from outside — the same treatment as the ARKit / PromptDA meshes.

    Args:
        entity_path: Where to log the mesh.
        mesh: ``open3d.geometry.TriangleMesh``.
        static: Log as static data (one final mesh) or at the current time (an incrementally growing mesh).
        recording: Target recording; the global one when None.
    """
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
            face_rendering=rr.components.MeshFaceRendering.Front,
        ),
        static=static,
        recording=recording,
    )
