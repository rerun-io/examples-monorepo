"""Run Prompt Depth Anything over ARKitScenes catalog segments.

This module reads RGB, prompt depth, confidence, calibration, and poses from a
local Rerun catalog, completes depth frame-by-frame, fuses a final TSDF mesh,
and writes a replaceable ``promptda`` layer back to the dataset.
"""

import ctypes
import gc
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import pyarrow as pa
import rerun as rr
import torch
import torch.nn.functional as F
from arkitscenes_download.ingest.blueprint import DEPTH_RANGE_MM, make_blueprint
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL
from arkitscenes_download.ingest.cells import landscape_quarter_turns, portrait_from_segment_row
from arkitscenes_download.ingest.depth import encode_depth_png
from arkitscenes_download.ingest.paths import DEPTH_PROMPTDA, PROMPTDA_MESH, TIMELINE
from arkitscenes_download.ingest.recording import atomic_recording
from einops import rearrange
from jaxtyping import Float, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, RegistrationHandle
from rerun.experimental.dataloader import RerunIterableDataset
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.ops.tsdf_depth_fuser import Open3DFuser, log_fused_mesh
from simplecv.rerun_log_utils import mesh_bounding_geometry
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rerun_prompt_da.apis.arkitscenes_shared import (
    ARKITSCENES_DATASET,
    NATIVE_FPS,
    connect_catalog,
    filter_depth_for_fusion,
    run_promptda_batch,
    segments_to_process,
    stride_for,
)
from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.promptda_stream import PromptDABatch, PromptDACollate, promptda_dataset
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

PROMPTDA_LAYER = "promptda"

_MALLOC_TRIM = ctypes.CDLL(None).malloc_trim
"""glibc ``malloc_trim``; Linux-only, like the rest of this pipeline (NVDEC, TensorRT)."""
_MALLOC_TRIM.argtypes = [ctypes.c_size_t]
_MALLOC_TRIM.restype = ctypes.c_int


@dataclass
class PDAArkitScenesConfig:
    """Runtime configuration for ARKitScenes PromptDA inference."""

    video_id: str | None = None
    """One catalog segment to process and replace, when present."""
    process_all: bool = False
    """Process every segment that does not yet have a PromptDA layer."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = "arkitscenes-v2"
    """Catalog dataset to read segments from and register layers on. Distinct from
    ``ARKITSCENES_DATASET``, which is the recording application id shared by every
    layer RRD regardless of which catalog dataset serves them."""
    output_dir: Path = Path("data/promptda")
    """Directory for generated layer RRDs (``<output_dir>/<segment_id>.rrd``, corpus layer-major naming)."""
    target_fps: float = 10.0
    """Inference rate used to build the catalog sampling grid."""
    max_image_size: int = 1008
    """Largest image dimension accepted by the PromptDA predictor."""
    batch_size: int = 8
    """Frames per TensorRT batch (the engine profile's max/opt batch)."""
    max_depth_range_meter: float = 4.0
    """Maximum predicted depth retained for TSDF fusion, in metres."""
    depth_fusion_resolution: float = 0.015
    """TSDF voxel resolution in metres (matches the ARKit mesh's ~1.5cm edge length;
    fusion is decode-bound so finer voxels are effectively free in wall time)."""
    register: bool = True
    """Register each completed RRD as a replacement PromptDA layer."""
    update_blueprint: bool = True
    """Register portrait and landscape blueprints containing PromptDA views."""


@dataclass(slots=True)
class SegmentResult:
    """Outcome of one segment's inference + fusion pipeline."""

    rrd_path: Path
    """Written PromptDA layer RRD."""
    inferred_frames: int
    """Number of frames sent through PromptDA."""
    skipped_decodes: int
    """Sampling-grid rows skipped because one or more required fields were absent."""


@dataclass(frozen=True, slots=True)
class CompletedPromptDABatch:
    """CPU-resident PromptDA results ready for ordered RRD logging and fusion."""

    timestamps_ns: list[int]
    """Sampling-grid timestamps in row order, in nanoseconds."""
    quarter_turns: int
    """Counter-clockwise quarter turns previously applied for inference."""
    depth_mm_bhw: UInt16[ndarray, "b h w"]
    """Full-resolution predicted depth in millimetres, in inference orientation."""
    depth_model_mm_bhw: UInt16[ndarray, "b nh nw"]
    """Network-resolution predicted depth in millimetres, in inference orientation."""
    rgb_stored_bhw3: UInt8[ndarray, "b fh fw 3"]
    """RGB frames at the fusion (network) resolution, restored to catalog orientation."""
    confidence_bhw: UInt8[ndarray, "b stored_prompt_h stored_prompt_w"]
    """ARKit confidence in catalog orientation."""
    K_native_b33: Float[ndarray, "b 3 3"]
    """Native intrinsics in catalog orientation."""
    world_T_cam_b44: Float[ndarray, "b 4 4"]
    """Camera-to-world transforms."""
    stored_hw: tuple[int, int]
    """Native frame height and width in catalog orientation."""


def log_promptda_frame(
    recording: rr.RecordingStream, timestamp_ns: int, predicted_depth_hw: UInt16[ndarray, "h w"]
) -> None:
    """Log one encoded PromptDA depth frame on the layer's duration timeline.

    Args:
        recording: PromptDA layer recording stream.
        timestamp_ns: Frame timestamp on ``video_time``, in nanoseconds.
        predicted_depth_hw: Full-resolution uint16 depth in millimetres with shape ``(H, W)``.
    """
    rr.set_time(TIMELINE, duration=np.timedelta64(timestamp_ns, "ns"), recording=recording)
    rr.log(
        DEPTH_PROMPTDA,
        # TODO(rerun#upstream): drop depth_range once the viewer auto-ranges encoded depth (see DEPTH_RANGE_MM in ingest/blueprint.py).
        rr.EncodedDepthImage(blob=encode_depth_png(predicted_depth_hw), media_type="image/png", meter=1000.0, depth_range=DEPTH_RANGE_MM),
        recording=recording,
    )


def fuse_and_log_batch(
    batch: CompletedPromptDABatch,
    recording: rr.RecordingStream,
    fuser: Open3DFuser,
    max_depth_meter: float,
) -> int:
    """Log and fuse one completed batch in row order.

    Args:
        batch: CPU-resident inference results and matching sensor metadata.
        recording: PromptDA layer recording stream.
        fuser: Segment TSDF accumulator.
        max_depth_meter: Furthest predicted depth retained for fusion, in metres.

    Returns:
        Number of rows logged and fused.
    """
    row: int
    timestamp_ns: int
    for row, timestamp_ns in enumerate(batch.timestamps_ns):
        predicted_depth_hw: UInt16[ndarray, "h w"] = np.ascontiguousarray(np.rot90(batch.depth_mm_bhw[row], -batch.quarter_turns))
        fusion_depth_hw: UInt16[ndarray, "fh fw"] = np.ascontiguousarray(np.rot90(batch.depth_model_mm_bhw[row], -batch.quarter_turns))
        confidence_hw: UInt8[ndarray, "h2 w2"] = batch.confidence_bhw[row]
        log_promptda_frame(recording, timestamp_ns, predicted_depth_hw)
        filtered_depth_hw: UInt16[ndarray, "h w"] = filter_depth_for_fusion(
            fusion_depth_hw,
            confidence_hw,
            max_depth_meter,
        )
        # Already at the fusion resolution and catalog orientation — downscaled on the GPU.
        rgb_fusion_hw3: UInt8[ndarray, "fh fw 3"] = batch.rgb_stored_bhw3[row]
        stored_k_33: Float[ndarray, "3 3"] = batch.K_native_b33[row]
        stored_intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=stored_k_33,
            height=batch.stored_hw[0],
            width=batch.stored_hw[1],
        )
        fusion_intrinsics: Intrinsics = rescale_intri(
            stored_intrinsics,
            target_height=fusion_depth_hw.shape[0],
            target_width=fusion_depth_hw.shape[1],
        )
        assert fusion_intrinsics.k_matrix is not None
        world_t_cam_44: Float[ndarray, "4 4"] = batch.world_T_cam_b44[row]
        fuser.fuse_frames(
            depth_hw=filtered_depth_hw,
            K_33=fusion_intrinsics.k_matrix,
            cam_T_world_44=np.linalg.inv(world_t_cam_44),
            rgb_hw3=rgb_fusion_hw3,
        )
    return len(batch.timestamps_ns)


def grid_batches(
    dataset_entry: DatasetEntry,
    segment_id: str,
    quarter_turns: int,
    config: PDAArkitScenesConfig,
) -> tuple[Iterator[PromptDABatch], int]:
    """Yield fixed-rate sampling-grid batches for one segment, plus the grid slot count."""
    device: torch.device = torch.device("cuda")
    stride: int = stride_for(NATIVE_FPS, config.target_fps)
    samples: RerunIterableDataset = promptda_dataset(dataset_entry, segment_id, NATIVE_FPS / stride, device)[0]
    # NVDEC and the collate's sampling-grid position are stateful, so iteration
    # must remain in natural order in this process.
    loader: DataLoader = DataLoader(
        samples,
        batch_size=config.batch_size,
        num_workers=0,
        collate_fn=PromptDACollate(
            samples,
            device,
            quarter_turns=quarter_turns,
            timestamp_step_ns=round(1_000_000_000 / NATIVE_FPS) * stride,
        ),
    )
    return (batch for batch in loader if batch is not None), samples.sample_index.total_samples


def run_segment(
    batches: Iterable[PromptDABatch],
    segment_id: str,
    portrait: bool,
    rrd_path: Path,
    predictor: PromptDATrtPredictor,
    *,
    max_depth_range_meter: float,
    depth_fusion_resolution: float,
    expected_frames: int | None = None,
) -> SegmentResult:
    """Infer, fuse, and atomically publish one segment's PromptDA layer from any batch source.

    Args:
        batches: Landscape-oriented inference batches in timeline order.
        segment_id: Catalog segment identifier (the recording id).
        portrait: Whether the segment's embedded blueprint uses the portrait layout.
        rrd_path: Layer RRD destination; written atomically.
        predictor: Cached TensorRT PromptDA predictor.
        max_depth_range_meter: Furthest predicted depth retained for TSDF fusion.
        depth_fusion_resolution: TSDF voxel resolution in metres.
        expected_frames: Frames the source could have produced, for the skipped count.
    """
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    fuser: Open3DFuser = Open3DFuser(fusion_resolution=depth_fusion_resolution, max_fusion_depth=max_depth_range_meter)
    inferred_frames: int = 0
    pending_batch: Future[int] | None = None
    with (
        atomic_recording(rrd_path, segment_id, send_properties=False) as recording,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="promptda-rrd") as executor,
    ):
        for batch in tqdm(batches, desc=f"PromptDA {segment_id}", unit="batch"):
            inference_result: tuple[UInt16[ndarray, "b h w"], UInt16[ndarray, "b nh nw"]] = run_promptda_batch(
                predictor,
                batch.rgb_bhw3,
                batch.prompt_bhw,
                (batch.rgb_bhw3.shape[1], batch.rgb_bhw3.shape[2]),
            )
            # Fusion only needs colour at the network resolution, so downscale on the
            # GPU and keep the full-resolution frame off the bus entirely.
            fusion_hw: tuple[int, int] = (inference_result[1].shape[1], inference_result[1].shape[2])
            rgb_fusion_bchw: Float[Tensor, "b 3 fh fw"] = F.interpolate(
                rearrange(batch.rgb_bhw3, "b h w c -> b c h w").float(),
                size=fusion_hw,
                mode="bilinear",
                antialias=True,
            )
            rgb_stored_bhw3: UInt8[ndarray, "b fh fw 3"] = np.ascontiguousarray(
                rearrange(torch.rot90(rgb_fusion_bchw, -batch.quarter_turns, dims=(2, 3)), "b c h w -> b h w c")
                .round()
                .clamp(0.0, 255.0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            if pending_batch is not None:
                inferred_frames += pending_batch.result()
            completed_batch: CompletedPromptDABatch = CompletedPromptDABatch(
                timestamps_ns=batch.timestamps_ns,
                quarter_turns=batch.quarter_turns,
                depth_mm_bhw=inference_result[0],
                depth_model_mm_bhw=inference_result[1],
                rgb_stored_bhw3=rgb_stored_bhw3,
                confidence_bhw=batch.confidence_bhw,
                K_native_b33=batch.K_native_b33,
                world_T_cam_b44=batch.world_T_cam_b44,
                stored_hw=batch.stored_hw,
            )
            pending_batch = executor.submit(fuse_and_log_batch, completed_batch, recording, fuser, max_depth_range_meter)
        if pending_batch is not None:
            inferred_frames += pending_batch.result()
        if inferred_frames == 0:
            raise RuntimeError(f"segment {segment_id!r} produced zero inferred frames")
        mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
        mesh.compute_vertex_normals()
        vertices: Float[ndarray, "n 3"] = np.asarray(mesh.vertices)
        log_fused_mesh(PROMPTDA_MESH, mesh, recording=recording)
        # Embed the PromptDA layout in this layer, framed on the fused mesh (the
        # same per-sequence treatment the base layer gives the ARKit mesh). Sent at
        # the end of the stream so it wins blueprint activation when the segment's
        # layers load together in the viewer.
        if len(vertices) > 0:
            rr.send_blueprint(
                make_blueprint(portrait=portrait, framing=mesh_bounding_geometry(vertices), include_promptda=True),
                recording=recording,
            )
    # Open3D's TSDF churn leaves ~0.6 GB of freed native pages per segment resident in
    # glibc arenas (measured: OOM at 142 GB after 234 segments). The fuser, executor and
    # recording are out of scope here, so return those pages to the OS at the boundary.
    gc.collect()
    _MALLOC_TRIM(0)
    skipped_decodes: int = 0 if expected_frames is None else expected_frames - inferred_frames
    return SegmentResult(rrd_path=rrd_path, inferred_frames=inferred_frames, skipped_decodes=skipped_decodes)


def process_segment(
    dataset_entry: DatasetEntry,
    segment_id: str,
    portrait: bool,
    quarter_turns: int,
    config: PDAArkitScenesConfig,
    predictor: PromptDATrtPredictor,
) -> SegmentResult:
    """Infer and fuse one catalog segment's sampling grid into a PromptDA RRD."""
    batches, grid_slots = grid_batches(dataset_entry, segment_id, quarter_turns, config)
    return run_segment(
        batches,
        segment_id,
        portrait,
        config.output_dir / f"{segment_id}.rrd",
        predictor,
        max_depth_range_meter=config.max_depth_range_meter,
        depth_fusion_resolution=config.depth_fusion_resolution,
        expected_frames=grid_slots,
    )


def main(config: PDAArkitScenesConfig) -> None:
    """Run PromptDA for selected segments and optionally update the catalog."""
    client: CatalogClient = connect_catalog(config.catalog_url, config.dataset_name)
    dataset_entry: DatasetEntry = client.get_dataset(config.dataset_name)
    segment_table: pa.Table = pa.Table.from_batches(dataset_entry.segment_table().collect())
    rows: list[dict] = segment_table.to_pylist()
    segment_ids: list[str] = segments_to_process(rows, config.video_id, config.process_all, PROMPTDA_LAYER)
    if not segment_ids:
        print("nothing to process: every segment already has the promptda layer")
        return
    # ARKitScenes wide video is always 1920x1440; the engine wants a static,
    # 14-aligned network resolution shared across segments.
    predictor: PromptDATrtPredictor = PromptDATrtPredictor(
        model_type="large",
        image_hw=network_image_hw((1440, 1920), config.max_image_size),
        batch_size=config.batch_size,
    )
    portrait_by_id: dict[str, bool] = {str(row["rerun_segment_id"]): portrait_from_segment_row(row) for row in rows}
    quarter_turns_by_id: dict[str, int] = {str(row["rerun_segment_id"]): landscape_quarter_turns(row) for row in rows}
    results: list[SegmentResult] = []
    registration_handles: list[RegistrationHandle] = []
    for segment_id in segment_ids:
        result: SegmentResult = process_segment(
            dataset_entry,
            segment_id,
            portrait_by_id[segment_id],
            quarter_turns_by_id[segment_id],
            config,
            predictor,
        )
        results.append(result)
        if config.register:
            registration_handles.append(
                dataset_entry.register(
                    [result.rrd_path.resolve().as_uri()],
                    layer_name=PROMPTDA_LAYER,
                    on_duplicate=OnDuplicateSegmentLayer.REPLACE,
                )
            )
    for registration_handle in registration_handles:
        registration_handle.wait()

    if config.register and config.update_blueprint:
        # blueprints() returns opaque rec_* ids with no name column, so stale entries from
        # previous runs cannot be matched by name. The dataset's blueprints are wholly owned
        # by ingest and this tool (always the same portrait/landscape pair), so drop them all
        # and register a fresh pair to keep the viewer's blueprint picker clean.
        stale_blueprints: list[str] = dataset_entry.blueprints()
        if stale_blueprints:
            blueprint_dataset: DatasetEntry | None = dataset_entry.blueprint_dataset()
            if blueprint_dataset is None:
                raise RuntimeError(f"dataset reports blueprints {stale_blueprints!r} but has no blueprint dataset")
            blueprint_dataset.unregister(segments_to_drop=stale_blueprints, layers_to_drop=[])
        for orientation, is_portrait in (("landscape", False), ("portrait", True)):
            blueprint_path: Path = config.output_dir / f"{ARKITSCENES_DATASET}-{orientation}.rbl"
            blueprint_path.parent.mkdir(parents=True, exist_ok=True)
            make_blueprint(portrait=is_portrait, include_promptda=True).save(f"arkitscenes-{orientation}", blueprint_path)
            dataset_entry.register_blueprint(blueprint_path.resolve().as_uri(), set_default=is_portrait)

    print(f"segments processed: {len(results)}")
    print(f"frames inferred: {sum(r.inferred_frames for r in results)}")
    print(f"skipped decodes: {sum(r.skipped_decodes for r in results)}")
    for result in results:
        print(f"rrd: {result.rrd_path}")
