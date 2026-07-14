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
import pyarrow as pa
import rerun as rr
import torch
from arkitscenes_download.ingest.blueprint import DEPTH_RANGE_MM
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL
from arkitscenes_download.ingest.clock import path_timestamps, recover_clock_from_packets, timestamp_from_path
from arkitscenes_download.ingest.depth import encode_depth_png
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
from beartype.roar import BeartypeException
from einops import rearrange
from jaxtyping import Float, Float32, Float64, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, RegistrationHandle
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from torch import Tensor
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor, postprocess_depth, preprocess_batch

ARKITSCENES_DATASET = "arkitscenes"
PROMPTDA_RAW_LAYER = "promptda_raw"
NATIVE_FPS = 60.0
ARKIT_CONFIDENCE_MEDIUM = 1
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


def stride_for(native_fps: float, target_fps: float) -> int:
    """Return the closest positive whole-frame stride for a target rate."""
    if target_fps <= 0.0:
        raise ValueError("target_fps must be greater than zero")
    return max(1, round(native_fps / target_fps))


def nearest_timestamped_path(paths: list[Path], timestamp: float, tolerance_s: float | None) -> Path | None:
    """Find the path nearest an uptime timestamp.

    Args:
        paths: Timestamp-sorted paths whose final stem component is seconds.
        timestamp: Query ARKit uptime in seconds.
        tolerance_s: Maximum accepted absolute difference, or None for no gate.

    Returns:
        Closest path, or None when there are no paths or the tolerance fails.
    """
    if not paths:
        return None
    timestamps: Float64[ndarray, "n"] = path_timestamps(paths)
    insertion: int = int(np.searchsorted(timestamps, timestamp))
    candidate_indices: list[int] = [index for index in (insertion - 1, insertion) if 0 <= index < len(paths)]
    nearest_index: int = min(candidate_indices, key=lambda index: abs(float(timestamps[index]) - timestamp))
    if tolerance_s is not None and abs(float(timestamps[nearest_index]) - timestamp) > tolerance_s:
        return None
    return paths[nearest_index]


def retained_video_times(
    wide_source_times: Float64[ndarray, "n"], epoch: float, wide_drop: int
) -> Float64[ndarray, "m"]:
    """Compute retained catalog video times with ingest's first-sample snap.

    Args:
        wide_source_times: Presentation-ordered raw frame uptimes with shape ``(N,)``.
        epoch: Shared first-wide-and-ultrawide uptime epoch.
        wide_drop: Count of wide frames preceding the epoch.

    Returns:
        Catalog-relative retained frame times with shape ``(M,)``.
    """
    video_times: Float64[ndarray, "m"] = np.asarray(wide_source_times[wide_drop:] - epoch, dtype=np.float64)
    if len(video_times) > 0 and video_times[0] < 0.25:
        video_times[0] = 0.0
    return video_times


def rotate_depth_for_catalog(depth_hw: UInt16[ndarray, "h w"], quarter_turns: int) -> UInt16[ndarray, "h2 w2"]:
    """Rotate an unbaked landscape depth image into catalog orientation.

    Args:
        depth_hw: Raw-orientation uint16 depth with shape ``(H, W)``.
        quarter_turns: Counter-clockwise quarter turns used by ingest.

    Returns:
        Contiguous baked-orientation uint16 depth with shape ``(H2, W2)``.
    """
    return np.ascontiguousarray(np.rot90(depth_hw, quarter_turns))


def world_t_cam_from_pose(
    translation: Float[ndarray, "3"], quaternion_xyzw: Float[ndarray, "4"]
) -> Float[ndarray, "4 4"]:
    """Build a camera-to-world transform from an unbaked ARKit pose.

    Args:
        translation: World-space camera translation with shape ``(3,)``.
        quaternion_xyzw: Camera rotation as an xyzw quaternion with shape ``(4,)``.

    Returns:
        Homogeneous ``world_T_cam`` transform with shape ``(4, 4)``.
    """
    world_t_cam_44: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    world_t_cam_44[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    world_t_cam_44[:3, 3] = translation
    return world_t_cam_44


def filter_depth_for_fusion(
    depth_mm: UInt16[ndarray, "h w"], confidence: UInt8[ndarray, "h2 w2"], max_depth_meter: float
) -> UInt16[ndarray, "h w"]:
    """Mask low-confidence and over-range predictions before fusion.

    Args:
        depth_mm: Predicted uint16 depth in millimetres with shape ``(H, W)``.
        confidence: Raw uint8 confidence values with shape ``(H2, W2)``.
        max_depth_meter: Furthest depth retained for fusion, in metres.

    Returns:
        Filtered uint16 depth with the same shape as ``depth_mm``.
    """
    confidence_hw: UInt8[ndarray, "h w"] = np.asarray(
        cv2.resize(confidence, (depth_mm.shape[1], depth_mm.shape[0]), interpolation=cv2.INTER_NEAREST), dtype=np.uint8
    )
    filtered_depth_mm: UInt16[ndarray, "h w"] = depth_mm.copy()
    filtered_depth_mm[confidence_hw < ARKIT_CONFIDENCE_MEDIUM] = 0
    filtered_depth_mm[depth_mm > max_depth_meter * 1000.0] = 0
    return filtered_depth_mm


def raw_sequence_dir(raw_dir: Path, video_id: str) -> Path | None:
    """Locate a raw sequence by scanning both dataset folds."""
    return next((candidate for fold in ("Training", "Validation") if (candidate := raw_dir / fold / video_id).is_dir()), None)


def segments_to_process(rows: list[dict[str, Any]], video_id: str | None, process_all: bool, raw_dir: Path) -> list[str]:
    """Select catalog segments according to mode, layer state, and local data."""
    if (video_id is None) == (not process_all):
        raise SystemExit("give exactly one of --video-id or --process-all")
    available_ids: list[str] = [str(row["rerun_segment_id"]) for row in rows]
    if video_id is not None:
        if video_id not in available_ids:
            raise SystemExit(f"video id {video_id!r} is absent; available ids: {', '.join(available_ids)}")
        return [video_id]
    selected: list[str] = []
    for row in rows:
        segment_id: str = str(row["rerun_segment_id"])
        if PROMPTDA_RAW_LAYER in (row.get("rerun_layer_names") or []):
            continue
        if raw_sequence_dir(raw_dir, segment_id) is None:
            print(f"skipping {segment_id}: raw data not found under {raw_dir}")
            continue
        selected.append(segment_id)
    return selected


def connect_catalog(catalog_url: str, dataset_name: str) -> CatalogClient:
    """Connect to the local ARKitScenes catalog or terminate with guidance."""
    try:
        client: CatalogClient = CatalogClient(catalog_url)
        dataset_names: list[str] = client.dataset_names()
    except BeartypeException:
        raise
    except Exception as error:
        raise SystemExit(f"catalog not reachable at {catalog_url} — start it with `pixi run arkitscenes-download-serve`") from error
    if dataset_name not in dataset_names:
        raise SystemExit(f"dataset {dataset_name!r} is absent — create it with `pixi run arkitscenes-download-register`")
    return client


def _sorted_timestamped_paths(directory: Path, suffix: str) -> list[Path]:
    """Return timestamp-named files in numeric uptime order."""
    return sorted(directory.glob(f"*{suffix}"), key=timestamp_from_path)


def _read_raw_png(path: Path, dtype: type[np.uint8] | type[np.uint16]) -> ndarray:
    """Read one raw PNG without changing its stored dtype or dimensions."""
    image_hw: ndarray | None = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image_hw is None:
        raise ValueError(f"failed to decode {path}")
    if image_hw.dtype != dtype:
        raise ValueError(f"expected {dtype.__name__} PNG at {path}, got {image_hw.dtype}")
    return image_hw


def _nearest_timestamp_index(timestamps: Float64[ndarray, "n"], timestamp: float) -> int:
    """Return the index nearest a timestamp in a sorted array."""
    insertion: int = int(np.searchsorted(timestamps, timestamp))
    candidates: list[int] = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    if not candidates:
        raise ValueError("cannot select from an empty timestamp sequence")
    return min(candidates, key=lambda index: abs(float(timestamps[index]) - timestamp))


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
    lowres_depth_paths: list[Path] = _sorted_timestamped_paths(sequence_dir / "lowres_depth", ".png")
    confidence_paths: list[Path] = _sorted_timestamped_paths(sequence_dir / "confidence", ".png")
    pincam_paths: list[Path] = _sorted_timestamped_paths(sequence_dir / "lowres_wide_intrinsics", ".pincam")
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
    epoch: float = max(float(wide_source_times[0]), float(ultrawide_source_times[0]))
    wide_drop: int = int(np.count_nonzero(wide_source_times < epoch))
    retained_indices: list[int] = list(range(wide_drop, len(wide_source_times)))
    video_times: Float64[ndarray, "r"] = retained_video_times(wide_source_times, epoch, wide_drop)
    lowres_intrinsics: TimedIntrinsics = load_intrinsics(pincam_paths)
    trajectory_at_intrinsics: Trajectory = resample_trajectory(trajectory_sparse, lowres_intrinsics.timestamps)
    quarter_turns: int = measured_orientation_quarter_turns(sky_angles(trajectory_at_intrinsics.quaternion_xyzw))
    wide_intrinsics: TimedIntrinsics = scale_intrinsics(lowres_intrinsics, 7.5)
    retained_uptimes: Float64[ndarray, "r"] = wide_source_times[wide_drop:]
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
            depth_path: Path | None = nearest_timestamped_path(lowres_depth_paths, uptime, PAIRING_TOLERANCE_S)
            confidence_path: Path | None = nearest_timestamped_path(confidence_paths, uptime, PAIRING_TOLERANCE_S)
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
        image_b3hw: Float32[Tensor, "b 3 nh nw"]
        prompt_b1hw: Float32[Tensor, "b 1 192 256"]
        image_b3hw, prompt_b1hw = preprocess_batch(rgb_bhw3, prompt_bhw, predictor.image_hw)
        depth_model_b1hw: Float32[Tensor, "b 1 nh nw"] = predictor.runtime(image_b3hw, prompt_b1hw)
        depth_bhw: Float32[Tensor, "b 1440 1920"] = postprocess_depth(depth_model_b1hw, (WIDE_HEIGHT, WIDE_WIDTH))
        depth_mm_bhw: UInt16[ndarray, "b 1440 1920"] = (depth_bhw.cpu().numpy() * 1000.0).astype(np.uint16)
        depth_model_mm_bhw: UInt16[ndarray, "b nh nw"] = (
            rearrange(depth_model_b1hw, "b 1 h w -> b h w").cpu().numpy() * 1000.0
        ).astype(np.uint16)
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
            intrinsics_index: int = _nearest_timestamp_index(wide_intrinsics.timestamps, uptime)
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
                    blob=encode_depth_png(depth_baked_hw),
                    media_type="image/png",
                    meter=1000.0,
                    depth_range=DEPTH_RANGE_MM,
                ),
                recording=recording,
            )
            log_s += time.perf_counter() - log_started
            inferred_frames += 1

    if inferred_frames == 0:
        raise RuntimeError(f"segment {segment_id!r} produced zero inferred frames")
    fusion_started = time.perf_counter()
    mesh: Any = fuser.get_mesh()
    mesh.compute_vertex_normals()
    fusion_s += time.perf_counter() - fusion_started
    log_started = time.perf_counter()
    rr.log(
        PROMPTDA_RAW_MESH,
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=np.asarray(mesh.vertex_colors),
            face_rendering=rr.components.MeshFaceRendering.Front,
        ),
        static=True,
        recording=recording,
    )
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
    segment_ids: list[str] = segments_to_process(rows, config.video_id, config.process_all, config.raw_dir)
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
