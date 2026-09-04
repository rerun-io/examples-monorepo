"""Generate the depth-only ultrawide_depth layer for a whole ARKitScenes corpus.

One process owns a shard of the queue, connects to the catalog once, and writes
``<output_dir>/<video_id>.rrd`` per segment. Registration is a separate stage:
this tool never touches the catalog's layer set.
"""

import json
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pyarrow as pa
import rerun as rr
import torch
from arkitscenes_download.ingest.depth import encode_depth_png
from arkitscenes_download.ingest.paths import (
    DEPTH_ULTRAWIDE_RECT,
    PINHOLE_ULTRAWIDE,
    PINHOLE_ULTRAWIDE_RECT,
    PROMPTDA_MESH,
    RGB_ULTRAWIDE_RECT,
    TIMELINE,
    VIDEO_ULTRAWIDE,
)
from arkitscenes_download.ingest.recording import atomic_recording
from arkitscenes_download.schema import DEPTH_RANGE_MM
from beartype.roar import BeartypeException
from einops import rearrange
from jaxtyping import Float32, UInt8, UInt32
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView
from torch import Tensor

from gauss_surf.catalog import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    SegmentReader,
    TimedeltaNs,
    _matrix_from_cell,
    _resolution_from_cell,
    _single_instance,
    connect_catalog,
    table_timestamps,
)
from gauss_surf.contracts import (
    DISTORTION_COEFFICIENTS_COLUMN,
    FRAME_SELECTION_LAYER,
    PROMPTDA_LAYER,
    ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
    ULTRAWIDE_FPS,
    ULTRAWIDE_QUATERNION_COLUMN,
    ULTRAWIDE_TRANSLATION_COLUMN,
)
from gauss_surf.render_io import encode_rgb_jpeg
from gauss_surf.uw_geometry import (
    CameraPoses,
    CameraPoseTrack,
    Undistortion,
    build_brown_conrady_undistortion,
    depth_meters_to_uint16_mm,
    load_camera_pose_track,
    raycast_batch_z_depth,
    resolve_camera_poses,
    undistort_rgb,
    world_from_pose,
)

UW_K_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:Pinhole:image_from_camera"
UW_RESOLUTION_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:Pinhole:resolution"
DISTORTION_CENTER_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:distortion_center_xy"
DISTORTION_REFERENCE_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:reference_dimensions_wh"
MESH_VERTICES_COLUMN: str = f"/{PROMPTDA_MESH}:Mesh3D:vertex_positions"
MESH_TRIANGLES_COLUMN: str = f"/{PROMPTDA_MESH}:Mesh3D:triangle_indices"


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for one shard of the corpus-wide ultrawide_depth pass."""

    output_dir: Path = Path("data/ultrawide_depth")
    """Layer-first destination directory that receives ``<video_id>.rrd``."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the catalog server that holds the upstream layers."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset to read."""
    video_ids_file: Path | None = None
    """Queue file with one segment identifier per line; the eligible corpus when unset."""
    shard_index: int = 0
    """Zero-based index of this process within the shard set."""
    shard_count: int = 1
    """Number of processes that split the queue."""
    force: bool = False
    """Regenerate segments whose destination RRD already exists."""
    batch_size: int = 8
    """Frames decoded, raycast, and encoded per inner iteration."""
    encoder_workers: int = 8
    """Threads that run JPEG and PNG encoding for each batch."""
    log_path: Path | None = None
    """JSONL run log; ``<output_dir>/../ultrawide_depth_log.jsonl`` when unset."""


@dataclass(frozen=True, slots=True)
class SegmentStats:
    """Per-segment outcome recorded in the run log."""

    frame_count: int
    """Frames written to the layer."""
    image_wh: tuple[int, int]
    """Rectified image width and height read from the calibration."""
    mesh_triangle_count: int
    """Triangles in the PromptDA TSDF mesh that was raycast."""
    coverage_mean: float
    """Mean fraction of pixels with nonzero mesh z-depth."""
    coverage_min: float
    """Smallest per-frame fraction of pixels with nonzero mesh z-depth."""


def shard_video_ids(video_ids: list[str], shard_index: int, shard_count: int) -> list[str]:
    """Deterministically select one shard's segments from a queue.

    Sorting first makes the split independent of catalog row order, so the same
    shard index always owns the same segments across restarts.

    Args:
        video_ids: Every segment identifier in the queue.
        shard_index: Zero-based index of the requested shard.
        shard_count: Number of shards the queue is split into.

    Returns:
        The requested shard's identifiers in sorted order.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} is outside [0, {shard_count})")
    return sorted(video_ids)[shard_index::shard_count]


def generate_ultrawide_depth(reader: SegmentReader, rrd_path: Path, *, batch_size: int, encoder_workers: int) -> SegmentStats:
    """Write one segment's rectified pinhole, rectified RGB, and mesh depth layer.

    Static components are read with ``index=None``: the timeline-indexed
    latest-at form returns no PromptDA mesh under Rerun 0.37 and is far slower.

    Args:
        reader: Reader bound to a segment carrying the frame_selection and promptda layers.
        rrd_path: Destination recording, written through a temporary file and renamed.
        batch_size: Frames processed per inner iteration.
        encoder_workers: Threads used for JPEG and PNG encoding.

    Returns:
        Frame count, resolution, mesh size, and valid-depth coverage.
    """
    segment_view: DatasetView = reader.segment_view()
    chosen_table: pa.Table = reader.chosen_table(ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN, (UW_K_COLUMN, UW_RESOLUTION_COLUMN))
    timestamps_n: TimedeltaNs = table_timestamps(chosen_table)
    K_uw_33: Float32[ndarray, "3 3"] = _matrix_from_cell(chosen_table.column(UW_K_COLUMN).to_pylist()[0], UW_K_COLUMN)
    image_wh: tuple[int, int] = _resolution_from_cell(chosen_table.column(UW_RESOLUTION_COLUMN).to_pylist()[0], UW_RESOLUTION_COLUMN)

    static_columns: tuple[str, ...] = (
        DISTORTION_COEFFICIENTS_COLUMN,
        DISTORTION_CENTER_COLUMN,
        DISTORTION_REFERENCE_COLUMN,
        ULTRAWIDE_TRANSLATION_COLUMN,
        ULTRAWIDE_QUATERNION_COLUMN,
    )
    static_table: pa.Table = segment_view.reader(index=None).select(*static_columns).to_arrow_table()
    if static_table.num_rows < 1:
        raise RuntimeError(f"segment {reader.video_id} has no static ultrawide calibration row")
    static_row: dict[str, Any] = static_table.to_pylist()[0]
    undistortion: Undistortion = build_brown_conrady_undistortion(
        K_uw_33,
        np.asarray(_single_instance(static_row[DISTORTION_COEFFICIENTS_COLUMN], DISTORTION_COEFFICIENTS_COLUMN), dtype=np.float32),
        np.asarray(static_row[DISTORTION_CENTER_COLUMN], dtype=np.float32),
        (int(static_row[DISTORTION_REFERENCE_COLUMN][0]), int(static_row[DISTORTION_REFERENCE_COLUMN][1])),
        image_wh,
    )
    wide_from_ultrawide_44: Float32[ndarray, "4 4"] = world_from_pose(
        np.asarray(_single_instance(static_row[ULTRAWIDE_TRANSLATION_COLUMN], ULTRAWIDE_TRANSLATION_COLUMN), dtype=np.float32),
        np.asarray(_single_instance(static_row[ULTRAWIDE_QUATERNION_COLUMN], ULTRAWIDE_QUATERNION_COLUMN), dtype=np.float32),
    )
    pose_track: CameraPoseTrack = load_camera_pose_track(reader, wide_from_ultrawide_44=wide_from_ultrawide_44)
    no_wide_timestamps_n: TimedeltaNs = np.asarray([], dtype="timedelta64[ns]")
    poses: CameraPoses = resolve_camera_poses(pose_track, no_wide_timestamps_n, timestamps_n, exact_wide=False)

    mesh_table: pa.Table = segment_view.reader(index=None).select(MESH_VERTICES_COLUMN, MESH_TRIANGLES_COLUMN).to_arrow_table()
    if mesh_table.num_rows < 1:
        raise RuntimeError(f"segment {reader.video_id} has no static PromptDA mesh row")
    mesh_row: dict[str, Any] = mesh_table.to_pylist()[0]
    mesh_vertices: Float32[ndarray, "n_vertices 3"] = np.asarray(mesh_row[MESH_VERTICES_COLUMN], dtype=np.float32)
    mesh_triangles: UInt32[ndarray, "n_triangles 3"] = np.asarray(mesh_row[MESH_TRIANGLES_COLUMN], dtype=np.uint32)
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3 or mesh_triangles.ndim != 2 or mesh_triangles.shape[1] != 3:
        raise RuntimeError(f"segment {reader.video_id} PromptDA mesh has invalid vertex or triangle shapes")
    scene: o3d.t.geometry.RaycastingScene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(mesh_vertices), o3d.core.Tensor(mesh_triangles)))

    frame_count: int = len(timestamps_n)
    coverage_fractions: list[float] = []
    processed_frames: int = 0
    with (
        atomic_recording(rrd_path, reader.video_id, send_properties=False) as recording,
        ThreadPoolExecutor(encoder_workers) as encoders,
    ):
        rr.send_columns(
            PINHOLE_ULTRAWIDE_RECT,
            indexes=[rr.TimeColumn(TIMELINE, duration=timestamps_n)],
            columns=rr.Pinhole.columns(
                image_from_camera=np.repeat(undistortion.K_rect_33[None], frame_count, axis=0),
                resolution=np.tile(np.asarray([[image_wh[0], image_wh[1]]], dtype=np.float32), (frame_count, 1)),
                camera_xyz=[rr.ViewCoordinates.RDF] * frame_count,
            ),
            recording=recording,
        )
        decoded_frames: Iterator[UInt8[Tensor, "3 h w"]] = reader.decode_frames(
            VIDEO_ULTRAWIDE,
            timestamps_n,
            fps=ULTRAWIDE_FPS,
            device=torch.device("cpu"),
        )
        for start in range(0, frame_count, batch_size):
            batch_timestamps_n: TimedeltaNs = timestamps_n[start : start + batch_size]
            batch_count: int = len(batch_timestamps_n)
            rectified_frames: list[UInt8[ndarray, "h w 3"]] = []
            for _timestamp in batch_timestamps_n:
                frame_chw: UInt8[Tensor, "3 h w"] = next(decoded_frames)
                rgb_hw3: UInt8[ndarray, "h w 3"] = rearrange(frame_chw, "c h w -> h w c").cpu().numpy().astype(np.uint8, copy=False)
                rectified_frames.append(undistort_rgb(rgb_hw3, undistortion))
            depth_m_bhw: Float32[ndarray, "batch h w"] = raycast_batch_z_depth(
                scene,
                undistortion.K_rect_33,
                poses.world_from_ultrawide_n44[start : start + batch_count],
                image_wh,
            )
            coverage_fractions.extend(float(np.count_nonzero(depth_m_hw) / depth_m_hw.size) for depth_m_hw in depth_m_bhw)
            rgb_jpeg_blobs: list[bytes] = list(encoders.map(encode_rgb_jpeg, rectified_frames))
            depth_png_blobs: list[bytes] = list(
                encoders.map(lambda depth_m_hw: encode_depth_png(depth_meters_to_uint16_mm(depth_m_hw), level=1), depth_m_bhw)
            )
            rr.send_columns(
                RGB_ULTRAWIDE_RECT,
                indexes=[rr.TimeColumn(TIMELINE, duration=batch_timestamps_n)],
                columns=rr.EncodedImage.columns(blob=rgb_jpeg_blobs, media_type=["image/jpeg"] * batch_count),
                recording=recording,
            )
            rr.send_columns(
                DEPTH_ULTRAWIDE_RECT,
                indexes=[rr.TimeColumn(TIMELINE, duration=batch_timestamps_n)],
                columns=rr.EncodedDepthImage.columns(
                    blob=depth_png_blobs,
                    media_type=["image/png"] * batch_count,
                    meter=[1000.0] * batch_count,
                    depth_range=[DEPTH_RANGE_MM] * batch_count,
                ),
                recording=recording,
            )
            processed_frames += batch_count
    if processed_frames != frame_count:
        raise RuntimeError(f"segment {reader.video_id} wrote {processed_frames} frames for {frame_count} chosen timestamps")
    coverage_n: Float32[ndarray, "n"] = np.asarray(coverage_fractions, dtype=np.float32)
    return SegmentStats(
        frame_count=processed_frames,
        image_wh=image_wh,
        mesh_triangle_count=len(mesh_triangles),
        coverage_mean=float(np.mean(coverage_n)),
        coverage_min=float(np.min(coverage_n)),
    )


def main(config: Config) -> None:
    """Process this shard's segments, logging one JSON line per attempt."""
    client: CatalogClient = connect_catalog(config.catalog_url, config.dataset_name)
    dataset: DatasetEntry = client.get_dataset(config.dataset_name)
    segment_table: pa.Table = pa.Table.from_batches(dataset.segment_table().collect())
    rows_by_video_id: dict[str, dict[str, Any]] = {str(row["rerun_segment_id"]): row for row in segment_table.to_pylist()}

    if config.video_ids_file is not None:
        queued_video_ids: list[str] = [line.strip() for line in config.video_ids_file.read_text().splitlines() if line.strip()]
        unknown_video_ids: list[str] = sorted(set(queued_video_ids) - set(rows_by_video_id))
        if unknown_video_ids:
            raise SystemExit(f"queue file lists segments absent from {config.dataset_name}: {unknown_video_ids}")
    else:
        required_layers: set[str] = {FRAME_SELECTION_LAYER, PROMPTDA_LAYER}
        queued_video_ids = [
            video_id for video_id, row in rows_by_video_id.items() if required_layers <= {str(name) for name in row["rerun_layer_names"]}
        ]
    shard_ids: list[str] = shard_video_ids(queued_video_ids, config.shard_index, config.shard_count)

    output_dir: Path = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path: Path = config.log_path if config.log_path is not None else output_dir.parent / "ultrawide_depth_log.jsonl"
    shard_label: str = f"shard {config.shard_index}/{config.shard_count}"
    print(f"{shard_label}: {len(shard_ids)} of {len(queued_video_ids)} queued segments -> {output_dir}", flush=True)

    for position, video_id in enumerate(shard_ids, start=1):
        rrd_path: Path = output_dir / f"{video_id}.rrd"
        if not config.force and rrd_path.is_file() and rrd_path.stat().st_size > 0:
            print(f"{shard_label} [{position}/{len(shard_ids)}] {video_id}: skipped, destination exists", flush=True)
            continue
        started_at: float = time.perf_counter()
        record: dict[str, Any] = {"video_id": video_id, "shard_index": config.shard_index}
        try:
            reader: SegmentReader = SegmentReader.bind(dataset, rows_by_video_id[video_id])
            reader.require_layers((FRAME_SELECTION_LAYER, PROMPTDA_LAYER))
            stats: SegmentStats = generate_ultrawide_depth(
                reader,
                rrd_path,
                batch_size=config.batch_size,
                encoder_workers=config.encoder_workers,
            )
        except BeartypeException:
            raise
        except (Exception, SystemExit) as error:  # upstream catalog helpers raise SystemExit; one bad segment must not stop the shard
            record |= {"status": "failed", "wall_s": round(time.perf_counter() - started_at, 2), "error": f"{type(error).__name__}: {error}"}
            print(f"{shard_label} [{position}/{len(shard_ids)}] {video_id}: FAILED {record['error']}", flush=True)
        else:
            rrd_path.chmod(0o644)  # NFS uid squash lands new files at mode 0000
            record |= {
                "status": "ok",
                "frames": stats.frame_count,
                "image_wh": list(stats.image_wh),
                "mesh_triangles": stats.mesh_triangle_count,
                "coverage_mean": round(stats.coverage_mean, 4),
                "coverage_min": round(stats.coverage_min, 4),
                "wall_s": round(time.perf_counter() - started_at, 2),
                "rrd_bytes": rrd_path.stat().st_size,
            }
            print(
                f"{shard_label} [{position}/{len(shard_ids)}] {video_id}: {record['frames']} frames "
                f"in {record['wall_s']}s, {record['rrd_bytes'] / 1e6:.1f} MB, coverage {record['coverage_mean']}",
                flush=True,
            )
        with log_path.open("a") as log_file:
            log_file.write(json.dumps(record) + "\n")
    print(f"{shard_label}: finished {len(shard_ids)} segments", flush=True)
