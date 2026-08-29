"""Generate rectified ultrawide RGB, mesh depth, and MoGe normal catalog layers."""

import time
from collections.abc import Iterator
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
    NORMALS_ULTRAWIDE_RECT,
    PINHOLE_ULTRAWIDE,
    PINHOLE_ULTRAWIDE_RECT,
    PROMPTDA_MESH,
    RGB_ULTRAWIDE_RECT,
    TIMELINE,
    VIDEO_ULTRAWIDE,
)
from arkitscenes_download.ingest.recording import atomic_recording
from arkitscenes_download.schema import DEPTH_RANGE_MM
from einops import rearrange
from jaxtyping import Float32, UInt8, UInt32
from monopriors.models.moge_v2 import MoGeV2NormalOutput, MoGeV2TrtPredictor
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetView
from torch import Tensor

from gauss_surf.catalog import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    SegmentReader,
    TimedeltaNs,
    _matrix_from_cell,
    _resolution_from_cell,
    _single_instance,
    register_layer,
    table_timestamps,
)
from gauss_surf.contracts import (
    DISTORTION_COEFFICIENTS_COLUMN,
    DISTORTION_MODEL_COLUMN,
    FRAME_SELECTION_LAYER,
    MOGE_INFERENCE_BATCH_SIZE,
    PROMPTDA_LAYER,
    RIG_QUATERNION_COLUMN,
    RIG_TRANSLATION_COLUMN,
    ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
    ULTRAWIDE_DEPTH_LAYER,
    ULTRAWIDE_FPS,
    ULTRAWIDE_NORMALS_LAYER,
    ULTRAWIDE_QUATERNION_COLUMN,
    ULTRAWIDE_TRANSLATION_COLUMN,
)
from gauss_surf.normals_encoding import encode_normals_png, to_away_from_camera
from gauss_surf.render_io import encode_rgb_jpeg
from gauss_surf.uw_geometry import (
    CameraPoses,
    CameraPoseTrack,
    Undistortion,
    build_brown_conrady_undistortion,
    depth_meters_to_uint16_mm,
    load_camera_pose_track,
    raycast_z_depth,
    resolve_camera_poses,
    undistort_rgb,
    world_from_pose,
)

ULTRAWIDE_IMAGE_HW: tuple[int, int] = (480, 640)
"""Native and network ultrawide height and width."""

UW_K_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:Pinhole:image_from_camera"
UW_RESOLUTION_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:Pinhole:resolution"
DISTORTION_CENTER_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:distortion_center_xy"
DISTORTION_REFERENCE_COLUMN: str = f"/{PINHOLE_ULTRAWIDE}:reference_dimensions_wh"
MESH_VERTICES_COLUMN: str = f"/{PROMPTDA_MESH}:Mesh3D:vertex_positions"
MESH_TRIANGLES_COLUMN: str = f"/{PROMPTDA_MESH}:Mesh3D:triangle_indices"


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for one segment's paired ultrawide signal layers."""

    video_id: str
    """ARKitScenes segment identifier whose chosen frames are processed."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the gauss-surf development catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset to read and update."""
    output_dir: Path = Path("data/ultrawide_signals")
    """Directory for generated per-segment layer RRDs."""


@dataclass(frozen=True, slots=True)
class UltrawideInputs:
    """Catalog inputs resolved onto the chosen ultrawide timeline."""

    timestamps: TimedeltaNs
    """Chosen ultrawide packet times as ``timedelta64[ns]``."""
    undistortion: Undistortion
    """Brown-Conrady remap and minimally cropped rectified pinhole at 640×480."""
    world_from_ultrawide_n44: Float32[ndarray, "n 4 4"]
    """Chosen camera-to-world transforms using causal rig poses."""
    pose_staleness_ms_n: Float32[ndarray, "n"]
    """Age of each selected rig pose in milliseconds."""
    mesh_vertices_n3: Float32[ndarray, "n_vertices 3"]
    """PromptDA TSDF mesh vertices in world coordinates."""
    mesh_triangles_m3: UInt32[ndarray, "n_triangles 3"]
    """PromptDA TSDF mesh triangle vertex indices."""


@dataclass(frozen=True, slots=True)
class SignalStats:
    """Counts and valid-depth coverage from the shared processing pass."""

    frame_count: int
    """Number of frames written to each layer."""
    coverage_fraction_n: Float32[ndarray, "n"]
    """Per-frame fraction of pixels with nonzero mesh z-depth."""


def _single_instance_or_exit(value: Any, component_name: str) -> Any:
    """Preserve this CLI stage's clean-exit behavior for malformed cells."""
    try:
        return _single_instance(value, component_name)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def brown_conrady_coefficients(static_row: dict[str, Any], *, video_id: str) -> Float32[ndarray, "14"]:
    """Read one standard Brown-Conrady model and coefficient vector.

    Args:
        static_row: Catalog row containing the generic model and coefficients.
        video_id: Segment identifier included in migration errors.

    Returns:
        Standard float32 Brown-Conrady coefficients shaped ``14``.
    """
    model_value: Any = _single_instance_or_exit(static_row[DISTORTION_MODEL_COLUMN], DISTORTION_MODEL_COLUMN)
    if not isinstance(model_value, str):
        raise SystemExit(f"component {DISTORTION_MODEL_COLUMN!r} is not a string")
    model: str = model_value
    migration_instruction: str = (
        f"re-ingest segment {video_id}'s calibration with arkitscenes-download-ingest "
        f"(`tools/apps/ingest_sequence.py`), then reseed it with gauss-surf-register-segment"
    )
    if model != "brown_conrady":
        raise SystemExit(
            f"segment {video_id} distortion model is {model!r}, not 'brown_conrady'; {migration_instruction}"
        )
    coefficients_n: Float32[ndarray, "n"] = np.asarray(
        _single_instance_or_exit(static_row[DISTORTION_COEFFICIENTS_COLUMN], DISTORTION_COEFFICIENTS_COLUMN), dtype=np.float32
    )
    if coefficients_n.shape != (14,):
        raise SystemExit(
            f"segment {video_id} Brown-Conrady coefficients have shape {coefficients_n.shape}, expected (14,); {migration_instruction}"
        )
    return coefficients_n


def _load_inputs(reader: SegmentReader) -> UltrawideInputs:
    """Read chosen times, calibration, causal poses, and the static PromptDA mesh."""
    segment_view: DatasetView = reader.segment_view()
    available_columns: set[str] = set(segment_view.arrow_schema().names)
    missing_distortion_columns: list[str] = [
        name for name in (DISTORTION_MODEL_COLUMN, DISTORTION_COEFFICIENTS_COLUMN) if name not in available_columns
    ]
    if missing_distortion_columns:
        raise SystemExit(
            f"segment {reader.video_id} lacks Brown-Conrady calibration columns {missing_distortion_columns}; "
            f"re-ingest that segment's calibration with arkitscenes-download-ingest (`tools/apps/ingest_sequence.py`), "
            f"then reseed it with gauss-surf-register-segment"
        )
    required_columns: tuple[str, ...] = (
        ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
        UW_K_COLUMN,
        UW_RESOLUTION_COLUMN,
        DISTORTION_MODEL_COLUMN,
        DISTORTION_COEFFICIENTS_COLUMN,
        DISTORTION_CENTER_COLUMN,
        DISTORTION_REFERENCE_COLUMN,
        ULTRAWIDE_TRANSLATION_COLUMN,
        ULTRAWIDE_QUATERNION_COLUMN,
        RIG_TRANSLATION_COLUMN,
        RIG_QUATERNION_COLUMN,
        MESH_VERTICES_COLUMN,
        MESH_TRIANGLES_COLUMN,
    )
    missing_columns: list[str] = [name for name in required_columns if name not in available_columns]
    if missing_columns:
        raise SystemExit(f"segment {reader.video_id} is missing required ultrawide columns: {missing_columns}")

    chosen_table: pa.Table = reader.chosen_table(ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN, (UW_K_COLUMN, UW_RESOLUTION_COLUMN))
    timestamps_n: TimedeltaNs = table_timestamps(chosen_table)

    K_values: list[Any] = chosen_table.column(UW_K_COLUMN).to_pylist()
    try:
        K_uw_n33: Float32[ndarray, "n 3 3"] = np.stack([_matrix_from_cell(value, UW_K_COLUMN) for value in K_values])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    K_uw_33: Float32[ndarray, "3 3"] = K_uw_n33[0]
    if not np.allclose(K_uw_n33, K_uw_33[None], atol=1e-5, rtol=0.0):
        raise SystemExit(f"segment {reader.video_id} has varying ultrawide intrinsics; v1 requires one rectified pinhole")
    resolution_values: list[Any] = chosen_table.column(UW_RESOLUTION_COLUMN).to_pylist()
    try:
        image_wh: tuple[int, int] = _resolution_from_cell(resolution_values[0], UW_RESOLUTION_COLUMN)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if image_wh != (ULTRAWIDE_IMAGE_HW[1], ULTRAWIDE_IMAGE_HW[0]):
        raise SystemExit(f"segment {reader.video_id} ultrawide resolution is {image_wh}, expected (640, 480)")

    static_columns: tuple[str, ...] = (
        DISTORTION_MODEL_COLUMN,
        DISTORTION_COEFFICIENTS_COLUMN,
        DISTORTION_CENTER_COLUMN,
        DISTORTION_REFERENCE_COLUMN,
        ULTRAWIDE_TRANSLATION_COLUMN,
        ULTRAWIDE_QUATERNION_COLUMN,
    )
    static_table: pa.Table = segment_view.reader(index=TIMELINE, fill_latest_at=True).select(TIMELINE, *static_columns).limit(1).to_arrow_table()
    if static_table.num_rows != 1:
        raise SystemExit(f"segment {reader.video_id} has no ultrawide static calibration row")
    static_row: dict[str, Any] = static_table.to_pylist()[0]
    distortion_coefficients_14: Float32[ndarray, "14"] = brown_conrady_coefficients(static_row, video_id=reader.video_id)
    distortion_center_reference_xy: Float32[ndarray, "2"] = np.asarray(static_row[DISTORTION_CENTER_COLUMN], dtype=np.float32)
    reference_values_wh: list[int] = [int(value) for value in static_row[DISTORTION_REFERENCE_COLUMN]]
    reference_dimensions_wh: tuple[int, int] = (reference_values_wh[0], reference_values_wh[1])
    wide_from_ultrawide_44: Float32[ndarray, "4 4"] = world_from_pose(
        np.asarray(_single_instance_or_exit(static_row[ULTRAWIDE_TRANSLATION_COLUMN], ULTRAWIDE_TRANSLATION_COLUMN), dtype=np.float32),
        np.asarray(_single_instance_or_exit(static_row[ULTRAWIDE_QUATERNION_COLUMN], ULTRAWIDE_QUATERNION_COLUMN), dtype=np.float32),
    )
    undistortion: Undistortion = build_brown_conrady_undistortion(
        K_uw_33,
        distortion_coefficients_14,
        distortion_center_reference_xy,
        reference_dimensions_wh,
        image_wh,
    )

    try:
        pose_track: CameraPoseTrack = load_camera_pose_track(
            reader,
            wide_from_ultrawide_44=wide_from_ultrawide_44,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    no_wide_timestamps_n: TimedeltaNs = np.asarray([], dtype="timedelta64[ns]")
    poses: CameraPoses = resolve_camera_poses(
        pose_track,
        no_wide_timestamps_n,
        timestamps_n,
        exact_wide=False,
    )

    mesh_table: pa.Table = (
        segment_view.reader(index=TIMELINE, fill_latest_at=True)
        .select(TIMELINE, MESH_VERTICES_COLUMN, MESH_TRIANGLES_COLUMN)
        .limit(1)
        .to_arrow_table()
    )
    if mesh_table.num_rows != 1:
        raise SystemExit(f"segment {reader.video_id} has no static PromptDA mesh row")
    mesh_row: dict[str, Any] = mesh_table.to_pylist()[0]
    mesh_vertices_n3: Float32[ndarray, "n_vertices 3"] = np.asarray(mesh_row[MESH_VERTICES_COLUMN], dtype=np.float32)
    mesh_triangles_m3: UInt32[ndarray, "n_triangles 3"] = np.asarray(mesh_row[MESH_TRIANGLES_COLUMN], dtype=np.uint32)
    if mesh_vertices_n3.ndim != 2 or mesh_vertices_n3.shape[1] != 3 or mesh_triangles_m3.ndim != 2 or mesh_triangles_m3.shape[1] != 3:
        raise SystemExit(f"segment {reader.video_id} PromptDA mesh has invalid vertex or triangle shapes")

    return UltrawideInputs(
        timestamps=timestamps_n,
        undistortion=undistortion,
        world_from_ultrawide_n44=poses.world_from_ultrawide_n44,
        pose_staleness_ms_n=poses.ultrawide_staleness_ms_n,
        mesh_vertices_n3=mesh_vertices_n3,
        mesh_triangles_m3=mesh_triangles_m3,
    )


def _send_rectified_pinhole(recording: rr.RecordingStream, inputs: UltrawideInputs) -> None:
    """Write the rectified pinhole at every chosen timestamp."""
    frame_count: int = len(inputs.timestamps)
    intrinsics_n33: Float32[ndarray, "n 3 3"] = np.repeat(inputs.undistortion.K_rect_33[None], frame_count, axis=0)
    resolutions_n2: Float32[ndarray, "n 2"] = np.tile(np.array([[640.0, 480.0]], dtype=np.float32), (frame_count, 1))
    rr.send_columns(
        PINHOLE_ULTRAWIDE_RECT,
        indexes=[rr.TimeColumn(TIMELINE, duration=inputs.timestamps)],
        columns=rr.Pinhole.columns(
            image_from_camera=intrinsics_n33,
            resolution=resolutions_n2,
            camera_xyz=[rr.ViewCoordinates.RDF] * frame_count,
        ),
        recording=recording,
    )


def _process_and_write(
    reader: SegmentReader,
    inputs: UltrawideInputs,
    predictor: MoGeV2TrtPredictor,
    scene: o3d.t.geometry.RaycastingScene,
    depth_recording: rr.RecordingStream,
    normals_recording: rr.RecordingStream,
) -> SignalStats:
    """Decode and rectify once, then fan each batch into both output layers."""
    decoded_frames: Iterator[UInt8[Tensor, "3 h w"]] = reader.decode_frames(
        VIDEO_ULTRAWIDE,
        inputs.timestamps,
        fps=ULTRAWIDE_FPS,
        device=torch.device("cuda"),
    )
    coverage_fractions: list[float] = []
    processed_frames: int = 0
    for start in range(0, len(inputs.timestamps), MOGE_INFERENCE_BATCH_SIZE):
        batch_timestamps_n: TimedeltaNs = inputs.timestamps[start : start + MOGE_INFERENCE_BATCH_SIZE]
        rectified_frames: list[UInt8[ndarray, "h=480 w=640 3"]] = []
        rgb_jpeg_blobs: list[bytes] = []
        depth_png_blobs: list[bytes] = []
        for batch_offset, _timestamp in enumerate(batch_timestamps_n):
            frame_chw: UInt8[Tensor, "3 h w"] = next(decoded_frames)
            frame_hwc: UInt8[Tensor, "h w 3"] = rearrange(frame_chw, "c h w -> h w c")  # pyrefly: ignore  # bad-argument-type — einops stub
            rgb_hw3: UInt8[ndarray, "h w 3"] = frame_hwc.cpu().numpy().astype(np.uint8, copy=False)
            if rgb_hw3.shape != (ULTRAWIDE_IMAGE_HW[0], ULTRAWIDE_IMAGE_HW[1], 3):
                raise RuntimeError(f"decoded ultrawide frame has shape {rgb_hw3.shape}, expected (480, 640, 3)")
            rectified_hw3: UInt8[ndarray, "h=480 w=640 3"] = undistort_rgb(
                rgb_hw3,
                inputs.undistortion,
            )
            rectified_frames.append(rectified_hw3)
            rgb_jpeg_blobs.append(encode_rgb_jpeg(rectified_hw3))

            frame_index: int = start + batch_offset
            depth_m_hw: Float32[ndarray, "h=480 w=640"] = raycast_z_depth(
                scene,
                inputs.undistortion.K_rect_33,
                inputs.world_from_ultrawide_n44[frame_index],
                image_wh=(ULTRAWIDE_IMAGE_HW[1], ULTRAWIDE_IMAGE_HW[0]),
            )
            coverage_fractions.append(float(np.count_nonzero(depth_m_hw) / depth_m_hw.size))
            depth_mm_hw: ndarray = depth_meters_to_uint16_mm(depth_m_hw)
            depth_png_blobs.append(encode_depth_png(depth_mm_hw, level=1))

        frames_bhw3: UInt8[ndarray, "batch h=480 w=640 3"] = np.stack(rectified_frames)
        frames_cuda_bhw3: UInt8[Tensor, "batch h=480 w=640 3"] = torch.from_numpy(frames_bhw3).to(device="cuda", dtype=torch.uint8)
        prediction: MoGeV2NormalOutput = predictor.predict_normals(frames_cuda_bhw3)
        normals_toward_bhw3: Float32[ndarray, "batch h=480 w=640 3"] = (
            prediction.normals_bhw3.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        normals_away_bhw3: Float32[ndarray, "batch h=480 w=640 3"] = np.stack(
            [to_away_from_camera(normals_toward_hw3) for normals_toward_hw3 in normals_toward_bhw3]
        )
        normals_png_blobs: list[bytes] = [encode_normals_png(normals_hw3) for normals_hw3 in normals_away_bhw3]
        batch_count: int = len(batch_timestamps_n)
        rr.send_columns(
            RGB_ULTRAWIDE_RECT,
            indexes=[rr.TimeColumn(TIMELINE, duration=batch_timestamps_n)],
            columns=rr.EncodedImage.columns(blob=rgb_jpeg_blobs, media_type=["image/jpeg"] * batch_count),
            recording=depth_recording,
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
            recording=depth_recording,
        )
        rr.send_columns(
            NORMALS_ULTRAWIDE_RECT,
            indexes=[rr.TimeColumn(TIMELINE, duration=batch_timestamps_n)],
            columns=rr.EncodedImage.columns(blob=normals_png_blobs, media_type=["image/png"] * batch_count),
            recording=normals_recording,
        )
        processed_frames += batch_count
        print(f"processed {processed_frames}/{len(inputs.timestamps)} ultrawide frames")

    coverage_fraction_n: Float32[ndarray, "n"] = np.asarray(coverage_fractions, dtype=np.float32)
    return SignalStats(frame_count=processed_frames, coverage_fraction_n=coverage_fraction_n)


def main(config: Config) -> None:
    """Generate, atomically publish, and replace both ultrawide signal layers."""
    started_at: float = time.perf_counter()
    if not torch.cuda.is_available():
        raise SystemExit("ultrawide signal generation requires CUDA for NVDEC and TensorRT")
    reader: SegmentReader = SegmentReader.open(config.catalog_url, config.dataset_name, config.video_id)
    reader.require_layers((FRAME_SELECTION_LAYER, PROMPTDA_LAYER))
    reader.require_zero_orientation("ultrawide rectification")
    dataset: DatasetEntry = reader.dataset
    inputs: UltrawideInputs = _load_inputs(reader)
    predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(
        heads="normal",
        network_hw_options=(ULTRAWIDE_IMAGE_HW,),
        batch_size=MOGE_INFERENCE_BATCH_SIZE,
    )
    mesh: o3d.t.geometry.TriangleMesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(inputs.mesh_vertices_n3),
        o3d.core.Tensor(inputs.mesh_triangles_m3),
    )
    scene: o3d.t.geometry.RaycastingScene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    segment_output_dir: Path = config.output_dir / config.video_id
    segment_output_dir.mkdir(parents=True, exist_ok=True)
    depth_rrd_path: Path = segment_output_dir / f"{ULTRAWIDE_DEPTH_LAYER}.rrd"
    normals_rrd_path: Path = segment_output_dir / f"{ULTRAWIDE_NORMALS_LAYER}.rrd"
    with (
        atomic_recording(depth_rrd_path, config.video_id, send_properties=False) as depth_recording,
        atomic_recording(normals_rrd_path, config.video_id, send_properties=False) as normals_recording,
    ):
        _send_rectified_pinhole(depth_recording, inputs)
        stats: SignalStats = _process_and_write(
            reader,
            inputs,
            predictor,
            scene,
            depth_recording,
            normals_recording,
        )
    if stats.frame_count != len(inputs.timestamps):
        raise RuntimeError(f"processed {stats.frame_count} frames for {len(inputs.timestamps)} chosen timestamps")
    register_layer(dataset, depth_rrd_path, ULTRAWIDE_DEPTH_LAYER)
    register_layer(dataset, normals_rrd_path, ULTRAWIDE_NORMALS_LAYER)

    coverage_mean: float = float(np.mean(stats.coverage_fraction_n))
    coverage_min: float = float(np.min(stats.coverage_fraction_n))
    coverage_max: float = float(np.max(stats.coverage_fraction_n))
    wall_seconds: float = time.perf_counter() - started_at
    print(f"K_rect: {inputs.undistortion.K_rect_33.tolist()}")
    print(
        "causal pose staleness: "
        f"mean={float(np.mean(inputs.pose_staleness_ms_n)):.3f}ms "
        f"min={float(np.min(inputs.pose_staleness_ms_n)):.3f}ms "
        f"max={float(np.max(inputs.pose_staleness_ms_n)):.3f}ms (nearest 60 Hz sample at or before each frame; no interpolation)"
    )
    print(
        f"{ULTRAWIDE_DEPTH_LAYER}: {stats.frame_count} frames registered from {depth_rrd_path}; "
        f"valid-depth coverage mean={coverage_mean:.6f} min={coverage_min:.6f} max={coverage_max:.6f}"
    )
    print(
        f"{ULTRAWIDE_NORMALS_LAYER}: {stats.frame_count} frames registered from {normals_rrd_path}; "
        f"shared-pass depth coverage mean={coverage_mean:.6f}"
    )
    print(f"completed both ultrawide layers in {wall_seconds:.2f}s ({stats.frame_count / wall_seconds:.2f} frames/s)")
