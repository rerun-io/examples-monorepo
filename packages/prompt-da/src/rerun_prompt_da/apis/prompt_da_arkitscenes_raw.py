"""Run Prompt Depth Anything over raw ARKitScenes files.

This sibling of the catalog-dataloader pipeline decodes the original wide MOV
with torchcodec, pairs raw low-resolution depth and confidence images, fuses
depth in the original landscape camera frame, and writes a ``promptda_raw``
layer back to the ARKitScenes catalog.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import pyarrow as pa
import rerun as rr
import torch
from arkitscenes_download.ingest.blueprint import DEPTH_RANGE_MM
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL
from arkitscenes_download.ingest.clock import clamped_to_epoch, nearest_index, path_timestamps, recover_clock_from_packets, shared_epoch
from arkitscenes_download.ingest.depth import encode_depth_png, sorted_timestamped_paths
from arkitscenes_download.ingest.metadata import PoseSelection, select_pose_source
from arkitscenes_download.ingest.mov import MetadataPacket, demux_metadata_streams, track_packet_times
from arkitscenes_download.ingest.paths import PINHOLE_WIDE
from arkitscenes_download.ingest.rig import (
    Intrinsics as TimedIntrinsics,
)
from arkitscenes_download.ingest.rig import (
    Trajectory,
    load_intrinsics,
    load_trajectory,
    measured_orientation_quarter_turns,
    resample_trajectory,
    scale_intrinsics,
    sky_angles,
)
from jaxtyping import Float, Float32, Float64, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, RegistrationHandle
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from rerun_prompt_da.apis.arkitscenes_shared import (
    ARKITSCENES_DATASET,
    NATIVE_FPS,
    connect_catalog,
    filter_depth_for_fusion,
    run_promptda_batch,
    segments_to_process,
    stride_for,
    world_t_cam_from_pose,
)
from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.mesh_logging import log_fused_mesh
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

PROMPTDA_RAW_LAYER = "promptda_raw"
PAIRING_TOLERANCE_S = 0.002
WIDE_HEIGHT = 1440
WIDE_WIDTH = 1920
DEPTH_PROMPTDA_RAW = f"{PINHOLE_WIDE}/depth_promptda_raw"
PROMPTDA_RAW_MESH = "world/promptda_raw/mesh"


@dataclass
class PDAArkitScenesRawConfig:
    """Runtime configuration for raw ARKitScenes PromptDA inference."""

    video_id: str | None = None
    """One catalog segment to process and replace, when present."""
    process_all: bool = False
    """Process every unfinished segment for which raw data exists locally."""
    raw_dir: Path = Path("../arkitscenes-download/data/raw")
    """Root containing the Training and Validation raw sequence directories."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    output_dir: Path = Path("data/promptda_raw")
    """Root for generated per-segment raw PromptDA RRD files."""
    target_fps: float = 10.0
    """Inference rate selected by stride over the native 60 Hz stream."""
    max_image_size: int = 1008
    """Largest image dimension accepted by the PromptDA predictor."""
    batch_size: int = 8
    """Frames per torchcodec fetch and TensorRT inference batch."""
    decode_device: str = "cuda"
    """Device on which torchcodec returns decoded video frames."""
    max_depth_range_meter: float = 4.0
    """Maximum predicted depth retained for TSDF fusion, in metres."""
    depth_fusion_resolution: float = 0.015
    """TSDF voxel resolution in metres."""
    register: bool = True
    """Register each completed RRD as a replacement raw PromptDA layer."""


@dataclass(slots=True)
class RawSegmentResult:
    """Output and wall-time breakdown for one raw segment."""

    rrd_path: Path
    """Written raw PromptDA layer RRD."""
    inferred_frames: int
    """Number of frames sent through PromptDA and fused."""
    skipped_frames: int
    """Selected frames skipped because depth or confidence did not pair."""
    pose_source: str
    """Stable label for the selected stream-4 or trajectory pose source."""
    prepare_s: float = 0.0
    """Clock, calibration, orientation, and pose preparation wall time."""
    decode_s: float = 0.0
    """Accumulated torchcodec indexed-decode wall time."""
    inference_s: float = 0.0
    """Accumulated PromptDA preprocess, runtime, and postprocess wall time."""
    fusion_s: float = 0.0
    """Accumulated depth filtering and TSDF fusion wall time."""
    log_s: float = 0.0
    """Accumulated Rerun depth and mesh logging wall time."""
    register_s: float = 0.0
    """Catalog registration and completion wait wall time."""
    wall_s: float = 0.0
    """Total segment wall time."""


def nearest_timestamped_path(
    paths: list[Path], timestamp: float, tolerance_s: float | None, timestamps: Float64[ndarray, "n"] | None = None
) -> Path | None:
    """Find the path nearest an uptime timestamp.

    Args:
        paths: Timestamp-sorted paths whose final stem component is seconds.
        timestamp: Query ARKit uptime in seconds.
        tolerance_s: Maximum accepted absolute difference, or None for no gate.
        timestamps: Precomputed ``path_timestamps(paths)``; parsing thousands of
            filename stems per query costs ~14% of segment wall time otherwise.

    Returns:
        Closest path, or None when there are no paths or the tolerance fails.
    """
    if not paths:
        return None
    if timestamps is None:
        timestamps = path_timestamps(paths)
    nearest: int = nearest_index(timestamps, timestamp)
    if tolerance_s is not None and abs(float(timestamps[nearest]) - timestamp) > tolerance_s:
        return None
    return paths[nearest]


def rotate_depth_for_catalog(depth_hw: UInt16[ndarray, "h w"], quarter_turns: int) -> UInt16[ndarray, "h2 w2"]:
    """Rotate an unbaked landscape depth image into catalog orientation.

    Args:
        depth_hw: Raw-orientation uint16 depth with shape ``(H, W)``.
        quarter_turns: Counter-clockwise quarter turns used by ingest.

    Returns:
        Contiguous baked-orientation uint16 depth with shape ``(H2, W2)``.
    """
    return np.ascontiguousarray(np.rot90(depth_hw, quarter_turns))


def raw_sequence_dir(raw_dir: Path, video_id: str) -> Path | None:
    """Locate a raw sequence by scanning both dataset folds."""
    return next((candidate for fold in ("Training", "Validation") if (candidate := raw_dir / fold / video_id).is_dir()), None)


def _read_raw_png(path: Path, dtype: type[np.uint8] | type[np.uint16]) -> ndarray:
    """Read one raw PNG without changing its stored dtype or dimensions."""
    image_hw: ndarray | None = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_hw is None:
        raise ValueError(f"failed to decode {path}")
    if image_hw.dtype != dtype:
        raise ValueError(f"expected {dtype.__name__} PNG at {path}, got {image_hw.dtype}")
    return image_hw


def _print_timing(result: RawSegmentResult) -> None:
    """Print a readable stage-time report for one completed segment."""
    print(f"raw PromptDA timing for {result.rrd_path.parent.name} (pose={result.pose_source}):")
    for name, seconds in (
        ("prepare", result.prepare_s),
        ("decode", result.decode_s),
        ("inference", result.inference_s),
        ("fusion", result.fusion_s),
        ("log", result.log_s),
        ("register", result.register_s),
    ):
        percentage: float = 100.0 * seconds / result.wall_s if result.wall_s > 0.0 else 0.0
        print(f"  {name:10s} {seconds:9.3f}s  {percentage:6.2f}%")
    effective_fps: float = result.inferred_frames / result.wall_s if result.wall_s > 0.0 else 0.0
    print(
        f"  wall       {result.wall_s:9.3f}s  100.00% | inferred={result.inferred_frames} "
        f"skipped={result.skipped_frames} effective_fps={effective_fps:.3f}"
    )


def process_segment_raw(
    dataset_entry: DatasetEntry,
    segment_id: str,
    config: PDAArkitScenesRawConfig,
    predictor: PromptDATrtPredictor,
) -> RawSegmentResult:
    """Decode, infer, fuse, log, and optionally register one raw segment."""
    wall_started: float = time.perf_counter()
    prepare_started: float = time.perf_counter()
    sequence_dir: Path | None = raw_sequence_dir(config.raw_dir, segment_id)
    if sequence_dir is None:
        raise FileNotFoundError(f"raw sequence {segment_id!r} not found under {config.raw_dir}")
    mov_path: Path = sequence_dir / f"{segment_id}.mov"
    lowres_depth_paths: list[Path] = sorted_timestamped_paths(sequence_dir / "lowres_depth")
    confidence_paths: list[Path] = sorted_timestamped_paths(sequence_dir / "confidence")
    pincam_paths: list[Path] = sorted_timestamped_paths(sequence_dir / "lowres_wide_intrinsics", ".pincam")
    if not lowres_depth_paths or not confidence_paths or not pincam_paths:
        raise FileNotFoundError(f"raw sequence {segment_id!r} is missing depth, confidence, or wide intrinsics")
    metadata_packets: dict[int, list[MetadataPacket]] = demux_metadata_streams(mov_path, (1, 3, 4))
    alignment = recover_clock_from_packets(metadata_packets[1], metadata_packets[3], lowres_depth_paths)
    trajectory_sparse: Trajectory = load_trajectory(sequence_dir / "lowres_wide.traj")
    pose_selection: PoseSelection = select_pose_source(mov_path, trajectory_sparse, metadata_packets[4])
    wide_source_times: Float64[ndarray, "n"] = np.asarray(track_packet_times(mov_path, 0) + alignment.offset_seconds, dtype=np.float64)
    ultrawide_source_times: Float64[ndarray, "m"] = np.asarray(
        track_packet_times(mov_path, 2) + alignment.ultrawide_offset_seconds, dtype=np.float64
    )
    epoch, wide_drop, _ = shared_epoch(wide_source_times, ultrawide_source_times)
    retained_indices: list[int] = list(range(wide_drop, len(wide_source_times)))
    retained_uptimes: Float64[ndarray, "r"] = wide_source_times[wide_drop:]
    video_times: Float64[ndarray, "r"] = clamped_to_epoch(retained_uptimes, epoch) - epoch
    lowres_intrinsics: TimedIntrinsics = load_intrinsics(pincam_paths)
    trajectory_at_intrinsics: Trajectory = resample_trajectory(trajectory_sparse, lowres_intrinsics.timestamps)
    quarter_turns: int = measured_orientation_quarter_turns(sky_angles(trajectory_at_intrinsics.quaternion_xyzw))
    wide_intrinsics: TimedIntrinsics = scale_intrinsics(lowres_intrinsics, 7.5)
    retained_poses: Trajectory = resample_trajectory(pose_selection.trajectory, retained_uptimes)
    prepare_s: float = time.perf_counter() - prepare_started

    decoder: VideoDecoder = VideoDecoder(
        mov_path,
        device=config.decode_device,
        seek_mode="exact",
        dimension_order="NHWC",
        num_ffmpeg_threads=0,
    )
    decoder_num_frames: int | None = decoder.metadata.num_frames
    if decoder_num_frames != len(wide_source_times):
        raise AssertionError(
            f"torchcodec reports {decoder_num_frames} frames but packet timestamps contain {len(wide_source_times)} for {segment_id}"
        )
    stride: int = stride_for(NATIVE_FPS, config.target_fps)
    selected_indices: list[int] = retained_indices[::stride]
    chunks: list[list[int]] = [selected_indices[index : index + config.batch_size] for index in range(0, len(selected_indices), config.batch_size)]
    rrd_path: Path = config.output_dir / segment_id / "promptda_raw.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording: rr.RecordingStream = rr.RecordingStream(
        application_id=ARKITSCENES_DATASET,
        recording_id=segment_id,
        send_properties=False,
    )
    recording.save(rrd_path)
    fuser: Open3DFuser = Open3DFuser(
        fusion_resolution=config.depth_fusion_resolution,
        max_fusion_depth=config.max_depth_range_meter,
    )
    decode_s: float = 0.0
    inference_s: float = 0.0
    fusion_s: float = 0.0
    log_s: float = 0.0
    inferred_frames: int = 0
    skipped_frames: int = 0
    lowres_depth_times: Float64[ndarray, "d"] = path_timestamps(lowres_depth_paths)
    confidence_times: Float64[ndarray, "c"] = path_timestamps(confidence_paths)

    for chunk in tqdm(chunks, desc=f"PromptDA raw {segment_id}", unit="batch"):
        decode_started: float = time.perf_counter()
        decoded_batch = decoder.get_frames_at(chunk)
        decoded_bhw3: UInt8[Tensor, "b 1440 1920 3"] = decoded_batch.data
        decode_s += time.perf_counter() - decode_started
        valid_rows: list[int] = []
        depth_arrays: list[UInt16[ndarray, "192 256"]] = []
        confidence_arrays: list[UInt8[ndarray, "192 256"]] = []
        for row, frame_index in enumerate(chunk):
            uptime: float = float(wide_source_times[frame_index])
            depth_path: Path | None = nearest_timestamped_path(lowres_depth_paths, uptime, PAIRING_TOLERANCE_S, timestamps=lowres_depth_times)
            confidence_path: Path | None = nearest_timestamped_path(confidence_paths, uptime, PAIRING_TOLERANCE_S, timestamps=confidence_times)
            if depth_path is None or confidence_path is None:
                skipped_frames += 1
                continue
            depth_hw: UInt16[ndarray, "192 256"] = np.asarray(_read_raw_png(depth_path, np.uint16), dtype=np.uint16)
            confidence_hw: UInt8[ndarray, "192 256"] = np.asarray(_read_raw_png(confidence_path, np.uint8), dtype=np.uint8)
            if depth_hw.shape != (192, 256) or confidence_hw.shape != (192, 256):
                raise ValueError(f"unexpected low-resolution shape for frame {frame_index}: depth={depth_hw.shape}, confidence={confidence_hw.shape}")
            valid_rows.append(row)
            depth_arrays.append(depth_hw)
            confidence_arrays.append(confidence_hw)
        if not valid_rows:
            continue

        rgb_bhw3: UInt8[Tensor, "b 1440 1920 3"] = torch.stack([decoded_bhw3[row] for row in valid_rows])
        prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(np.stack(depth_arrays).astype(np.float32) / 1000.0)
        inference_started: float = time.perf_counter()
        depth_mm_bhw: UInt16[ndarray, "b 1440 1920"]
        depth_model_mm_bhw: UInt16[ndarray, "b nh nw"]
        depth_mm_bhw, depth_model_mm_bhw = run_promptda_batch(predictor, rgb_bhw3, prompt_bhw, (WIDE_HEIGHT, WIDE_WIDTH))
        inference_s += time.perf_counter() - inference_started

        for valid_index, (row, depth_confidence_hw) in enumerate(zip(valid_rows, confidence_arrays, strict=True)):
            frame_index: int = chunk[row]
            uptime: float = float(wide_source_times[frame_index])
            predicted_depth_hw: UInt16[ndarray, "1440 1920"] = depth_mm_bhw[valid_index]
            fusion_depth_hw: UInt16[ndarray, "fh fw"] = depth_model_mm_bhw[valid_index]
            rgb_hw3: UInt8[ndarray, "1440 1920 3"] = np.asarray(rgb_bhw3[valid_index].cpu().numpy(), dtype=np.uint8)

            fusion_started: float = time.perf_counter()
            filtered_depth_hw: UInt16[ndarray, "fh fw"] = filter_depth_for_fusion(
                fusion_depth_hw, depth_confidence_hw, config.max_depth_range_meter
            )
            rgb_fusion_hw3: UInt8[ndarray, "fh fw 3"] = np.asarray(
                cv2.resize(rgb_hw3, (fusion_depth_hw.shape[1], fusion_depth_hw.shape[0]), interpolation=cv2.INTER_AREA), dtype=np.uint8
            )
            intrinsics_index: int = nearest_index(wide_intrinsics.timestamps, uptime)
            k_33: Float64[ndarray, "3 3"] = wide_intrinsics.matrices[intrinsics_index]
            full_intrinsics: Intrinsics = Intrinsics.from_k_matrix(
                camera_conventions="RDF", k_matrix=k_33, height=WIDE_HEIGHT, width=WIDE_WIDTH
            )
            fusion_intrinsics: Intrinsics = rescale_intri(
                full_intrinsics, target_height=fusion_depth_hw.shape[0], target_width=fusion_depth_hw.shape[1]
            )
            if fusion_intrinsics.k_matrix is None:
                raise RuntimeError("rescaled fusion intrinsics have no K matrix")
            pose_index: int = frame_index - wide_drop
            world_t_cam_44: Float[ndarray, "4 4"] = world_t_cam_from_pose(
                retained_poses.translation[pose_index], retained_poses.quaternion_xyzw[pose_index]
            )
            fuser.fuse_frames(
                depth_hw=filtered_depth_hw,
                K_33=fusion_intrinsics.k_matrix,
                cam_T_world_44=np.linalg.inv(world_t_cam_44),
                rgb_hw3=rgb_fusion_hw3,
            )
            fusion_s += time.perf_counter() - fusion_started

            log_started: float = time.perf_counter()
            depth_baked_hw: UInt16[ndarray, "bh bw"] = rotate_depth_for_catalog(predicted_depth_hw, quarter_turns)
            rr.set_time("video_time", duration=float(video_times[pose_index]), recording=recording)
            rr.log(
                DEPTH_PROMPTDA_RAW,
                rr.EncodedDepthImage(
                    # Deflate level 1 trades ~12% blob size for half the encode wall time.
                    blob=encode_depth_png(depth_baked_hw, level=1),
                    media_type="image/png",
                    meter=1000.0,
                    # TODO(rerun#upstream): drop depth_range once the viewer auto-ranges encoded depth (see DEPTH_RANGE_MM in ingest/blueprint.py).
                    depth_range=DEPTH_RANGE_MM,
                ),
                recording=recording,
            )
            log_s += time.perf_counter() - log_started
            inferred_frames += 1

    if inferred_frames == 0:
        raise RuntimeError(f"segment {segment_id!r} produced zero inferred frames")
    fusion_started = time.perf_counter()
    mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
    mesh.compute_vertex_normals()
    fusion_s += time.perf_counter() - fusion_started
    log_started = time.perf_counter()
    log_fused_mesh(recording, PROMPTDA_RAW_MESH, mesh)
    recording.flush()
    log_s += time.perf_counter() - log_started

    register_s: float = 0.0
    if config.register:
        register_started: float = time.perf_counter()
        registration_handle: RegistrationHandle = dataset_entry.register(
            [rrd_path.resolve().as_uri()],
            layer_name=PROMPTDA_RAW_LAYER,
            on_duplicate=OnDuplicateSegmentLayer.REPLACE,
        )
        registration_handle.wait()
        register_s = time.perf_counter() - register_started
    result: RawSegmentResult = RawSegmentResult(
        rrd_path=rrd_path,
        inferred_frames=inferred_frames,
        skipped_frames=skipped_frames,
        pose_source=pose_selection.source,
        prepare_s=prepare_s,
        decode_s=decode_s,
        inference_s=inference_s,
        fusion_s=fusion_s,
        log_s=log_s,
        register_s=register_s,
        wall_s=time.perf_counter() - wall_started,
    )
    _print_timing(result)
    return result


def main(config: PDAArkitScenesRawConfig) -> None:
    """Run raw PromptDA for selected catalog segments."""
    client: CatalogClient = connect_catalog(config.catalog_url, ARKITSCENES_DATASET)
    dataset_entry: DatasetEntry = client.get_dataset(ARKITSCENES_DATASET)
    segment_table: pa.Table = pa.Table.from_batches(dataset_entry.segment_table().collect())
    rows: list[dict[str, Any]] = segment_table.to_pylist()

    def raw_data_available(segment_id: str) -> bool:
        if raw_sequence_dir(config.raw_dir, segment_id) is None:
            print(f"skipping {segment_id}: raw data not found under {config.raw_dir}")
            return False
        return True

    segment_ids: list[str] = segments_to_process(rows, config.video_id, config.process_all, PROMPTDA_RAW_LAYER, raw_data_available)
    if not segment_ids:
        print("nothing to process: no unfinished catalog segments have local raw data")
        return
    predictor: PromptDATrtPredictor = PromptDATrtPredictor(
        model_type="large",
        image_hw=network_image_hw((WIDE_HEIGHT, WIDE_WIDTH), config.max_image_size),
        batch_size=config.batch_size,
    )
    results: list[RawSegmentResult] = []
    for segment_id in segment_ids:
        results.append(process_segment_raw(dataset_entry, segment_id, config, predictor))
    print(f"segments processed: {len(results)}")
    print(f"frames inferred: {sum(result.inferred_frames for result in results)}")
    print(f"frames skipped: {sum(result.skipped_frames for result in results)}")
    for result in results:
        print(f"rrd: {result.rrd_path}")
