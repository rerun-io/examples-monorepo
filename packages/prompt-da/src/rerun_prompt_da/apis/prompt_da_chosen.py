"""Run PromptDA only at frame-selection's chosen wide timestamps.

Unlike :mod:`prompt_da_arkitscenes`, which infers on a fixed ~10 Hz sampling
grid, this module reads the segment's registered ``frame_selection`` layer and
completes depth exactly at the chosen (sharpest-per-window) wide packet
timestamps. The layer therefore aligns one-to-one with the frames every
downstream gauss-surf stage trains on, and its timeline carries real packet
timestamps instead of synthesized grid slots.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import pyarrow as pa
import rerun as rr
import torch
from arkitscenes_download.ingest.blueprint import make_blueprint
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL
from arkitscenes_download.ingest.paths import (
    CONFIDENCE,
    DEPTH,
    FRAME_SELECTION_WIDE,
    PINHOLE_WIDE,
    PROMPTDA_MESH,
    RIG,
    TIMELINE,
    VIDEO_WIDE,
)
from beartype.roar import BeartypeException
from datafusion import col
from jaxtyping import Float, Float32, Float64, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_dataloader import open_segment_decoder
from simplecv.rerun_log_utils import mesh_bounding_geometry
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from rerun_prompt_da.apis.arkitscenes_shared import (
    ARKITSCENES_DATASET,
    NATIVE_FPS,
    connect_catalog,
    log_fused_mesh,
    run_promptda_batch,
    world_t_cam_from_pose,
)
from rerun_prompt_da.apis.prompt_da_arkitscenes import (
    PROMPTDA_LAYER,
    CompletedPromptDABatch,
    PDAArkitScenesConfig,
    SegmentResult,
    fuse_and_log_batch,
    orientation_quarter_turns_from_segment_row,
    portrait_from_segment_row,
)
from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

CHOSEN_SHARPNESS_COLUMN: str = f"/{FRAME_SELECTION_WIDE}:sharpness"
"""Sparse chosen-wide-frame component written by gauss-surf frame selection."""
DEPTH_BLOB_COLUMN: str = f"/{DEPTH}:EncodedDepthImage:blob"
CONFIDENCE_COLUMN: str = f"/{CONFIDENCE}:SegmentationImage:buffer"
INTRINSICS_COLUMN: str = f"/{PINHOLE_WIDE}:Pinhole:image_from_camera"
POSE_TRANSLATION_COLUMN: str = f"/{RIG}:Transform3D:translation"
POSE_QUATERNION_COLUMN: str = f"/{RIG}:Transform3D:quaternion"


@dataclass(frozen=True, slots=True)
class PDAChosenConfig:
    """Configuration for chosen-frame PromptDA over one segment or a batch queue."""

    video_id: str | None = None
    """Single catalog segment to process."""
    video_ids_file: Path | None = None
    """Newline-separated segment identifiers for batch mode; mutually exclusive with ``video_id``."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = "arkitscenes-v2"
    """Catalog dataset to read segments from and register layers on."""
    output_dir: Path = Path("data/promptda")
    """Directory for generated layer RRDs (``<output_dir>/<video_id>.rrd``, corpus layer-major naming)."""
    force: bool = False
    """Regenerate and re-register segments whose output RRD already exists."""
    fs_layer_dir: Path | None = None
    """When set, wait for ``<fs_layer_dir>/<video_id>.rrd`` before querying chosen frames (lane ordering)."""
    fs_wait_timeout_s: float = 3600.0
    """Longest wait for the frame-selection RRD before the segment fails."""
    batch_size: int = 8
    """Frames per TensorRT batch (the engine profile's max/opt batch)."""
    max_image_size: int = 1008
    """Largest image dimension accepted by the PromptDA predictor."""
    max_depth_range_meter: float = 4.0
    """Maximum predicted depth retained for TSDF fusion, in metres."""
    depth_fusion_resolution: float = 0.015
    """TSDF voxel resolution in metres."""
    register: bool = True
    """Register each completed RRD as a replacement PromptDA layer."""


def _instance(value: Any, column_name: str) -> Any:
    """Unwrap the single component instance from one reader cell."""
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"column {column_name!r} has an empty cell")
        if len(value) == 1 and isinstance(value[0], (list, np.ndarray, bytes)):
            return value[0]
    return value


def _chosen_inputs_table(dataset: DatasetEntry, video_id: str) -> pa.Table:
    """Read chosen timestamps with latest-at prompt, confidence, intrinsics, and pose."""
    view = dataset.filter_segments(video_id)
    if CHOSEN_SHARPNESS_COLUMN not in set(view.arrow_schema().names):
        raise RuntimeError(f"segment {video_id} has no registered frame_selection layer (column {CHOSEN_SHARPNESS_COLUMN!r} absent)")
    table: pa.Table = (
        view.reader(index=TIMELINE, fill_latest_at=True)
        .filter(col(f'"{CHOSEN_SHARPNESS_COLUMN}"').is_not_null())
        .select(TIMELINE, DEPTH_BLOB_COLUMN, CONFIDENCE_COLUMN, INTRINSICS_COLUMN, POSE_TRANSLATION_COLUMN, POSE_QUATERNION_COLUMN)
        .sort(TIMELINE)
        .to_arrow_table()
    )
    if table.num_rows == 0:
        raise RuntimeError(f"segment {video_id} frame_selection layer contains no chosen wide rows")
    return table


def process_segment_chosen(
    dataset: DatasetEntry,
    video_id: str,
    portrait: bool,
    orientation_quarter_turns: int,
    config: PDAChosenConfig,
    predictor: PromptDATrtPredictor,
) -> SegmentResult:
    """Infer and fuse one segment's chosen wide frames into a PromptDA RRD."""
    device: torch.device = torch.device("cuda")
    table: pa.Table = _chosen_inputs_table(dataset, video_id)
    rows: list[dict[str, Any]] = table.to_pylist()

    packet_times_n: ndarray
    decoder: VideoDecoder
    packet_times_n, _samples, _keyframes, decoder = open_segment_decoder(dataset, video_id, VIDEO_WIDE, TIMELINE, device, int(NATIVE_FPS))
    chosen_times_n: ndarray = np.array([row[TIMELINE].value for row in rows], dtype="timedelta64[ns]")
    packet_indices_n: ndarray = np.searchsorted(packet_times_n, chosen_times_n, side="left")
    in_bounds = packet_indices_n < len(packet_times_n)
    if not (np.all(in_bounds) and np.array_equal(packet_times_n[packet_indices_n], chosen_times_n)):
        raise RuntimeError(f"segment {video_id}: chosen timestamps do not map one-to-one onto wide video packets")

    model_quarter_turns: int = (-orientation_quarter_turns) % 4 if portrait else 0
    rrd_path: Path = config.output_dir / f"{video_id}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    fusion_config: PDAArkitScenesConfig = PDAArkitScenesConfig(
        output_dir=config.output_dir,
        max_depth_range_meter=config.max_depth_range_meter,
        depth_fusion_resolution=config.depth_fusion_resolution,
        register=False,
        update_blueprint=False,
    )
    fuser: Open3DFuser = Open3DFuser(fusion_resolution=config.depth_fusion_resolution, max_fusion_depth=config.max_depth_range_meter)
    recording: rr.RecordingStream = rr.RecordingStream(application_id=ARKITSCENES_DATASET, recording_id=video_id, send_properties=False)
    recording.save(rrd_path)

    inferred_frames: int = 0
    pending_batch: Future[int] | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="promptda-rrd") as executor:
        for batch_start in range(0, len(rows), config.batch_size):
            batch_rows: list[dict[str, Any]] = rows[batch_start : batch_start + config.batch_size]
            batch_packet_indices: list[int] = [int(i) for i in packet_indices_n[batch_start : batch_start + config.batch_size]]

            frames_3hw: list[UInt8[Tensor, "3 h w"]] = [decoder.get_frame_at(i).data for i in batch_packet_indices]
            prompts_hw: list[UInt16[ndarray, "ph pw"]] = []
            confidences_hw: list[UInt8[ndarray, "ph pw"]] = []
            k_matrices_33: list[Float32[ndarray, "3 3"]] = []
            world_T_cam_44: list[Float64[ndarray, "4 4"]] = []
            timestamps_ns: list[int] = []
            for row in batch_rows:
                blob_u8: UInt8[ndarray, "n"] = np.asarray(_instance(row[DEPTH_BLOB_COLUMN], DEPTH_BLOB_COLUMN), dtype=np.uint8)
                prompt_hw: UInt16[ndarray, "ph pw"] = cv2.imdecode(blob_u8, cv2.IMREAD_UNCHANGED)
                if prompt_hw is None or prompt_hw.dtype != np.uint16:
                    raise RuntimeError(f"segment {video_id}: prompt depth blob failed to decode as uint16 PNG")
                confidence_flat: UInt8[ndarray, "n"] = np.asarray(_instance(row[CONFIDENCE_COLUMN], CONFIDENCE_COLUMN), dtype=np.uint8)
                k_flat: Float32[ndarray, "9"] = np.asarray(_instance(row[INTRINSICS_COLUMN], INTRINSICS_COLUMN), dtype=np.float32)
                pose_t: Float[ndarray, "3"] = np.asarray(_instance(row[POSE_TRANSLATION_COLUMN], POSE_TRANSLATION_COLUMN), dtype=np.float64)
                pose_q: Float[ndarray, "4"] = np.asarray(_instance(row[POSE_QUATERNION_COLUMN], POSE_QUATERNION_COLUMN), dtype=np.float64)
                prompts_hw.append(prompt_hw)
                confidences_hw.append(confidence_flat.reshape(prompt_hw.shape))
                k_matrices_33.append(k_flat.reshape(3, 3).T)
                world_T_cam_44.append(world_t_cam_from_pose(pose_t, pose_q))
                timestamps_ns.append(int(row[TIMELINE].value))

            stored_hw: tuple[int, int] = (int(frames_3hw[0].shape[1]), int(frames_3hw[0].shape[2]))
            rgb_b3hw: UInt8[Tensor, "b 3 h w"] = torch.rot90(torch.stack(frames_3hw), model_quarter_turns, dims=(2, 3))
            rgb_bhw3: UInt8[Tensor, "b h w 3"] = rgb_b3hw.permute(0, 2, 3, 1).contiguous()
            prompt_stored_bhw: UInt16[Tensor, "b ph pw"] = torch.from_numpy(np.stack(prompts_hw)).to(device, non_blocking=True)
            prompt_bhw: Float32[Tensor, "b 192 256"] = torch.rot90(prompt_stored_bhw, model_quarter_turns, dims=(1, 2)).float() / 1000.0

            inference_result: tuple[UInt16[ndarray, "b h w"], UInt16[ndarray, "b nh nw"]] = run_promptda_batch(
                predictor,
                rgb_bhw3,
                prompt_bhw,
                (rgb_bhw3.shape[1], rgb_bhw3.shape[2]),
            )
            rgb_stored_bhw3: UInt8[ndarray, "b sh sw 3"] = np.asarray(
                torch.rot90(rgb_bhw3, -model_quarter_turns, dims=(1, 2)).cpu().numpy(), dtype=np.uint8
            )
            if pending_batch is not None:
                inferred_frames += pending_batch.result()
            completed: CompletedPromptDABatch = CompletedPromptDABatch(
                timestamps_ns=timestamps_ns,
                quarter_turns=model_quarter_turns,
                depth_mm_bhw=inference_result[0],
                depth_model_mm_bhw=inference_result[1],
                rgb_stored_bhw3=rgb_stored_bhw3,
                confidence_bhw=np.stack(confidences_hw),
                K_native_b33=np.stack(k_matrices_33),
                world_T_cam_b44=np.stack(world_T_cam_44),
                stored_hw=stored_hw,
            )
            pending_batch = executor.submit(fuse_and_log_batch, completed, recording, fuser, fusion_config)
        if pending_batch is not None:
            inferred_frames += pending_batch.result()

    if inferred_frames == 0:
        raise RuntimeError(f"segment {video_id!r} produced zero inferred frames")
    mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
    mesh.compute_vertex_normals()
    vertices: Float[ndarray, "n 3"] = np.asarray(mesh.vertices)
    log_fused_mesh(recording, PROMPTDA_MESH, mesh)
    if len(vertices) > 0:
        rr.send_blueprint(
            make_blueprint(portrait=portrait, framing=mesh_bounding_geometry(vertices), include_promptda=True),
            recording=recording,
        )
    recording.flush()
    return SegmentResult(rrd_path=rrd_path, inferred_frames=inferred_frames, skipped_decodes=0)


def main(config: PDAChosenConfig) -> None:
    """Run chosen-frame PromptDA for one segment or a batch queue with skip-if-exists semantics."""
    if (config.video_id is None) == (config.video_ids_file is None):
        raise SystemExit("provide exactly one of --video-id or --video-ids-file")
    if config.video_id is not None:
        video_ids: list[str] = [config.video_id]
    else:
        assert config.video_ids_file is not None  # enforced by the exclusive-arg check above
        video_ids = [line.strip() for line in config.video_ids_file.read_text().splitlines() if line.strip()]

    client: CatalogClient = connect_catalog(config.catalog_url, config.dataset_name)
    dataset: DatasetEntry = client.get_dataset(config.dataset_name)
    rows: list[dict[str, Any]] = pa.Table.from_batches(dataset.segment_table().collect()).to_pylist()
    row_by_id: dict[str, dict[str, Any]] = {str(row["rerun_segment_id"]): row for row in rows}
    predictor: PromptDATrtPredictor = PromptDATrtPredictor(
        model_type="large",
        image_hw=network_image_hw((1440, 1920), config.max_image_size),
        batch_size=config.batch_size,
    )

    completed: int = 0
    skipped: int = 0
    failed: list[str] = []
    for video_id in video_ids:
        rrd_path: Path = config.output_dir / f"{video_id}.rrd"
        if rrd_path.is_file() and not config.force:
            skipped += 1
            print(f"SKIP {video_id}: {rrd_path} exists (use --force to regenerate)", flush=True)
            continue
        if config.fs_layer_dir is not None:
            fs_rrd: Path = config.fs_layer_dir / f"{video_id}.rrd"
            wait_start: float = time.monotonic()
            while not fs_rrd.is_file() and time.monotonic() - wait_start < config.fs_wait_timeout_s:
                time.sleep(5.0)
            if not fs_rrd.is_file():
                failed.append(video_id)
                print(f"FAIL {video_id}: frame_selection RRD never appeared at {fs_rrd}", flush=True)
                continue
        start: float = time.perf_counter()
        try:
            result: SegmentResult = process_segment_chosen(
                dataset,
                video_id,
                portrait_from_segment_row(row_by_id[video_id]),
                orientation_quarter_turns_from_segment_row(row_by_id[video_id]),
                config,
                predictor,
            )
            if config.register:
                dataset.register(
                    [result.rrd_path.resolve().as_uri()],
                    layer_name=PROMPTDA_LAYER,
                    on_duplicate=OnDuplicateSegmentLayer.REPLACE,
                ).wait()
        except BeartypeException:
            raise
        except Exception as error:
            failed.append(video_id)
            print(f"FAIL {video_id}: {type(error).__name__}: {error}", flush=True)
            if config.video_id is not None:
                raise
            continue
        completed += 1
        print(
            f"DONE {video_id}: {result.inferred_frames} frames in {time.perf_counter() - start:.1f}s "
            f"({completed} done, {skipped} skipped, {len(failed)} failed)",
            flush=True,
        )

    print(f"promptda-chosen batch: {completed} done, {skipped} skipped, {len(failed)} failed of {len(video_ids)}")
    if failed:
        print("failed segments: " + ", ".join(failed))
