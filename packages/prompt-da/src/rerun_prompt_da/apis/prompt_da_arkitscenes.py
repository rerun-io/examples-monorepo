"""Run Prompt Depth Anything over ARKitScenes catalog segments.

This module reads RGB, prompt depth, confidence, calibration, and poses from a
local Rerun catalog, completes depth frame-by-frame, fuses a final TSDF mesh,
and writes a replaceable ``promptda`` layer back to the dataset.
"""

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import open3d as o3d
import pyarrow as pa
import rerun as rr
import torch
from arkitscenes_download.ingest.blueprint import DEPTH_RANGE_MM, make_blueprint
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL
from arkitscenes_download.ingest.depth import encode_depth_png
from arkitscenes_download.ingest.paths import CONFIDENCE, DEPTH, DEPTH_PROMPTDA, PINHOLE_WIDE, PROMPTDA_MESH, RIG, VIDEO_WIDE
from jaxtyping import Float, Float32, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, RegistrationHandle
from rerun.experimental.dataloader import (
    DataSource,
    Field,
    FixedRateSampling,
    ImageDecoder,
    NumericDecoder,
    RerunMapDataset,
    VideoFrameDecoder,
)
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import mesh_bounding_geometry
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rerun_prompt_da.apis.arkitscenes_shared import (
    ARKITSCENES_DATASET,
    NATIVE_FPS,
    connect_catalog,
    filter_depth_for_fusion,
    log_fused_mesh,
    run_promptda_batch,
    segments_to_process,
    stride_for,
    world_t_cam_from_pose,
)
from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

PROMPTDA_LAYER = "promptda"


@dataclass
class PDAArkitScenesConfig:
    """Runtime configuration for ARKitScenes PromptDA inference."""

    video_id: str | None = None
    """One catalog segment to process and replace, when present."""
    process_all: bool = False
    """Process every segment that does not yet have a PromptDA layer."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    output_dir: Path = Path("data/promptda")
    """Root for generated per-segment PromptDA RRD files."""
    target_fps: float = 10.0
    """Inference rate selected by client-side stride over the native stream."""
    max_image_size: int = 1008
    """Largest image dimension accepted by the PromptDA predictor."""
    batch_size: int = 8
    """Frames per TensorRT batch (the engine profile's max/opt batch)."""
    num_workers: int = 4
    """Spawned DataLoader worker processes fetching/decoding batches in parallel."""
    max_depth_range_meter: float = 4.0
    """Maximum predicted depth retained for TSDF fusion, in metres."""
    depth_fusion_resolution: float = 0.015
    """TSDF voxel resolution in metres (matches the GT mesh's ~1.5cm edge length;
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
    """Frames skipped because a field was absent (includes recovered AV1 decoder errors)."""


class ResilientVideoFrameDecoder(VideoFrameDecoder):
    """A ``VideoFrameDecoder`` that survives per-sample AV1 decode failures.

    The dataloader assembles decode windows with ``fill_latest_at`` over a
    fixed-rate sampling grid, which silently drops a stored packet whenever
    timestamp jitter puts two packets in one grid slot (upstream, RR-5087).
    Every window spanning the dropped reference frame then fails
    deterministically with ``InvalidDataError`` until the next keyframe —
    on this dataset, a ~0.5 s dead zone per segment. Each sample is decoded
    with a fresh codec context, so the failure is recoverable at sample
    granularity: map it onto the decoder's existing ``None``-frame skip path
    instead of killing the whole iteration. The caught exceptions are kept
    alive on purpose — releasing a failed decode's traceback frees the
    errored codec context un-drained, whose teardown at interpreter shutdown
    can deadlock the process in ``dav1d_flush``.
    """

    def __init__(self, *, keyframe_interval: int = 30, fps_estimate: float = 30.0, codec: str = "h264") -> None:
        super().__init__(keyframe_interval=keyframe_interval, fps_estimate=fps_estimate, codec=codec)
        self.decode_failures: list[BaseException] = []

    def decode(
        self, raw: pa.ChunkedArray, index_value: int | np.datetime64 | np.timedelta64, segment_id: str
    ) -> Tensor | None:
        """Decode one sample, degrading decoder errors to a skippable ``None``."""
        try:
            return super().decode(raw, index_value, segment_id)
        except av.error.InvalidDataError as error:
            self.decode_failures.append(error)
            return None


def portrait_from_segment_row(row: dict[str, Any]) -> bool:
    """Parse the required catalog portrait property from one segment row."""
    property_name: str = "property:portrait:value"
    if property_name not in row or row[property_name] is None:
        raise KeyError(f"segment {row.get('rerun_segment_id', '<unknown>')!r} is missing the portrait property")
    value: Any = row[property_name]
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"segment {row.get('rerun_segment_id', '<unknown>')!r} has an empty portrait property")
        value = value[0]
    return str(value) == "True"


def orientation_quarter_turns_from_segment_row(row: dict[str, Any]) -> int:
    """Parse the ingest rotation needed to undo a stored portrait segment."""
    property_name: str = "property:orientation_quarter_turns_ccw:value"
    if property_name not in row or row[property_name] is None:
        raise KeyError(f"segment {row.get('rerun_segment_id', '<unknown>')!r} is missing the orientation quarter-turn property")
    value: Any = row[property_name]
    if isinstance(value, (list, np.ndarray)):
        if len(value) == 0:
            raise ValueError(f"segment {row.get('rerun_segment_id', '<unknown>')!r} has an empty orientation quarter-turn property")
        value = value[0]
    return int(value) % 4


def rotate_portrait_to_landscape(array_hw: ndarray, quarter_turns: int) -> ndarray:
    """Undo ingest's counter-clockwise portrait bake before inference."""
    return np.ascontiguousarray(np.rot90(array_hw, -quarter_turns))


def rotate_landscape_to_portrait(array_hw: ndarray, quarter_turns: int) -> ndarray:
    """Restore ingest's counter-clockwise portrait bake after inference."""
    return np.ascontiguousarray(np.rot90(array_hw, quarter_turns))


def _list_collate(batch: list[dict[str, Tensor | None]]) -> list[dict[str, Tensor | None]]:
    """Identity collate: DataLoader's default would stack per-field tensors and crash on ``None`` (skipped-decode) samples.

    A module-level function rather than a lambda so spawn workers can pickle it.
    """
    return batch


def process_segment(
    dataset_entry: DatasetEntry,
    segment_id: str,
    portrait: bool,
    orientation_quarter_turns: int,
    config: PDAArkitScenesConfig,
    predictor: PromptDATrtPredictor,
) -> SegmentResult:
    """Infer and fuse one catalog segment into a PromptDA RRD."""
    source: DataSource = DataSource(dataset=dataset_entry, segments=[segment_id])
    video_decoder: ResilientVideoFrameDecoder = ResilientVideoFrameDecoder(codec="av1", keyframe_interval=300, fps_estimate=60.0)
    fields: dict[str, Field] = {
        "video": Field(f"/{VIDEO_WIDE}:VideoStream:sample", decode=video_decoder),
        "depth": Field(f"/{DEPTH}:EncodedDepthImage:blob", decode=ImageDecoder()),
        "conf": Field(f"/{CONFIDENCE}:SegmentationImage:buffer", decode=NumericDecoder()),
        "pose_t": Field(f"/{RIG}:Transform3D:translation", decode=NumericDecoder()),
        "pose_q": Field(f"/{RIG}:Transform3D:quaternion", decode=NumericDecoder()),
        "k": Field(f"/{PINHOLE_WIDE}:Pinhole:image_from_camera", decode=NumericDecoder()),
    }
    # A map dataset fetches only the strided samples. In released rerun-sdk
    # 0.34.1, the iterable dataset would decode every sample on the 60 fps grid
    # (each decode replays the whole GOP window; TODO RR-4751), i.e. `stride`×
    # the work for frames we throw away. RR-5167 / PR #2683 fixes this upstream.
    rerun_dataset: RerunMapDataset = RerunMapDataset(
        source,
        index="video_time",
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=NATIVE_FPS),
    )
    stride: int = stride_for(NATIVE_FPS, config.target_fps)
    rrd_path: Path = config.output_dir / segment_id / "promptda.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording: rr.RecordingStream | None = None
    fuser: Open3DFuser = Open3DFuser(
        fusion_resolution=config.depth_fusion_resolution,
        max_fusion_depth=config.max_depth_range_meter,
    )
    inferred_frames: int = 0
    skipped_decodes: int = 0
    strided_indices: list[int] = list(range(0, rerun_dataset.sample_index.total_samples, stride))
    chunks: list[list[int]] = [strided_indices[i : i + config.batch_size] for i in range(0, len(strided_indices), config.batch_size)]
    # Fetch whole chunks, in order, on spawn-based worker processes: in released
    # rerun-sdk 0.34.1 the batch fetch (server query + per-sample GOP decode back
    # to the previous keyframe; TODO RR-4751) is the pipeline's slowest stage and
    # GIL-bound in-process, so worker processes are the only way to parallelize
    # it (incremental decoding lands upstream in RR-5167 / PR #2683). The Rerun
    # dataloader requires ``spawn`` (forked workers deadlock on their first
    # catalog call), and torch rejects the worker kwargs when num_workers == 0.
    loader: DataLoader = (
        DataLoader(rerun_dataset, batch_sampler=chunks, collate_fn=_list_collate)
        if config.num_workers == 0
        else DataLoader(
            rerun_dataset,
            batch_sampler=chunks,
            collate_fn=_list_collate,
            num_workers=config.num_workers,
            multiprocessing_context=multiprocessing.get_context("spawn"),
            prefetch_factor=2,
        )
    )
    for chunk, samples in tqdm(zip(chunks, loader, strict=True), total=len(chunks), desc=f"PromptDA {segment_id}", unit="batch"):
        valid: list[tuple[int, dict[str, Tensor]]] = [
            (sample_idx, sample)
            for sample_idx, sample in zip(chunk, samples, strict=True)
            if all(sample.get(field_name) is not None for field_name in fields)
        ]
        skipped_decodes += len(chunk) - len(valid)
        if not valid:
            continue
        rgb_arrays: list[UInt8[ndarray, "h w 3"]] = [np.asarray(sample["video"].permute(1, 2, 0).numpy(), dtype=np.uint8) for _, sample in valid]
        prompt_arrays: list[ndarray] = [sample["depth"].numpy().squeeze(0) for _, sample in valid]
        confidence_shape_hw: tuple[int, int] = (256, 192) if portrait else (192, 256)
        confidence_arrays: list[UInt8[ndarray, "h w"]] = [
            np.asarray(sample["conf"].numpy().reshape(confidence_shape_hw), dtype=np.uint8) for _, sample in valid
        ]
        if portrait:
            rgb_arrays = [rotate_portrait_to_landscape(rgb_hw3, orientation_quarter_turns) for rgb_hw3 in rgb_arrays]
            prompt_arrays = [rotate_portrait_to_landscape(prompt_hw, orientation_quarter_turns) for prompt_hw in prompt_arrays]
            confidence_arrays = [rotate_portrait_to_landscape(confidence_hw, orientation_quarter_turns) for confidence_hw in confidence_arrays]
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = torch.from_numpy(np.stack(rgb_arrays))
        prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(np.stack(prompt_arrays).astype(np.float32) / 1000.0)
        depth_mm_bhw: UInt16[ndarray, "b h w"]
        depth_model_mm_bhw: UInt16[ndarray, "b nh nw"]
        depth_mm_bhw, depth_model_mm_bhw = run_promptda_batch(predictor, rgb_bhw3, prompt_bhw, (rgb_bhw3.shape[1], rgb_bhw3.shape[2]))
        if recording is None:
            recording = rr.RecordingStream(
                application_id=ARKITSCENES_DATASET,
                recording_id=segment_id,
                send_properties=False,
            )
            recording.save(rrd_path)
        for row, (sample_idx, sample) in enumerate(valid):
            predicted_depth_hw: UInt16[ndarray, "h w"] = depth_mm_bhw[row]
            fusion_depth_hw: UInt16[ndarray, "fh fw"] = depth_model_mm_bhw[row]
            confidence_hw: UInt8[ndarray, "h2 w2"] = confidence_arrays[row]
            rgb_hw3: UInt8[ndarray, "h w 3"] = np.asarray(rgb_bhw3[row].numpy(), dtype=np.uint8)
            if portrait:
                predicted_depth_hw = rotate_landscape_to_portrait(predicted_depth_hw, orientation_quarter_turns)
                fusion_depth_hw = rotate_landscape_to_portrait(fusion_depth_hw, orientation_quarter_turns)
                confidence_hw = rotate_landscape_to_portrait(confidence_hw, orientation_quarter_turns)
                rgb_hw3 = rotate_landscape_to_portrait(rgb_hw3, orientation_quarter_turns)
            timestamp: np.timedelta64 = rerun_dataset.sample_index.global_to_local(sample_idx)[1]  # pyrefly: ignore  # bad-assignment — duration timeline
            timestamp_seconds: float = float(timestamp / np.timedelta64(1, "s"))
            rr.set_time("video_time", duration=timestamp_seconds, recording=recording)
            rr.log(
                DEPTH_PROMPTDA,
                # TODO(rerun#upstream): drop depth_range once the viewer auto-ranges encoded depth (see DEPTH_RANGE_MM in ingest/blueprint.py).
                rr.EncodedDepthImage(blob=encode_depth_png(predicted_depth_hw), media_type="image/png", meter=1000.0, depth_range=DEPTH_RANGE_MM),
                recording=recording,
            )
            filtered_depth_hw: UInt16[ndarray, "h w"] = filter_depth_for_fusion(
                fusion_depth_hw,
                confidence_hw,
                config.max_depth_range_meter,
            )
            rgb_fusion_hw3: UInt8[ndarray, "fh fw 3"] = np.asarray(
                cv2.resize(rgb_hw3, (fusion_depth_hw.shape[1], fusion_depth_hw.shape[0]), interpolation=cv2.INTER_AREA), dtype=np.uint8
            )
            # Rerun stores Pinhole image_from_camera flattened column-major.
            stored_k_33: Float[ndarray, "3 3"] = np.asarray(sample["k"].numpy()).reshape(3, 3, order="F")
            stored_intrinsics: Intrinsics = Intrinsics.from_k_matrix(
                camera_conventions="RDF",
                k_matrix=stored_k_33,
                height=rgb_hw3.shape[0],
                width=rgb_hw3.shape[1],
            )
            fusion_intrinsics: Intrinsics = rescale_intri(
                stored_intrinsics,
                target_height=fusion_depth_hw.shape[0],
                target_width=fusion_depth_hw.shape[1],
            )
            assert fusion_intrinsics.k_matrix is not None
            world_t_cam_44: Float[ndarray, "4 4"] = world_t_cam_from_pose(
                np.asarray(sample["pose_t"].numpy()), np.asarray(sample["pose_q"].numpy())
            )
            fuser.fuse_frames(
                depth_hw=filtered_depth_hw,
                K_33=fusion_intrinsics.k_matrix,
                cam_T_world_44=np.linalg.inv(world_t_cam_44),
                rgb_hw3=rgb_fusion_hw3,
            )
            inferred_frames += 1

    if inferred_frames == 0 or recording is None:
        raise RuntimeError(f"segment {segment_id!r} produced zero inferred frames")
    mesh: o3d.geometry.TriangleMesh = fuser.get_mesh()
    mesh.compute_vertex_normals()
    vertices: Float[ndarray, "n 3"] = np.asarray(mesh.vertices)
    log_fused_mesh(recording, PROMPTDA_MESH, mesh)
    # Embed the PromptDA layout in this layer, framed on the fused mesh (the
    # same per-sequence treatment the base layer gives the GT mesh). Sent at
    # the end of the stream so it wins blueprint activation when the segment's
    # layers load together in the viewer.
    if len(vertices) > 0:
        mesh_geometry: tuple[Float[ndarray, "3"], float] = mesh_bounding_geometry(vertices)
        rr.send_blueprint(
            make_blueprint(portrait=portrait, framing=mesh_geometry, include_promptda=True),
            recording=recording,
        )
    recording.flush()
    return SegmentResult(rrd_path=rrd_path, inferred_frames=inferred_frames, skipped_decodes=skipped_decodes)


def main(config: PDAArkitScenesConfig) -> None:
    """Run PromptDA for selected segments and optionally update the catalog."""
    client: CatalogClient = connect_catalog(config.catalog_url, ARKITSCENES_DATASET)
    dataset_entry: DatasetEntry = client.get_dataset(ARKITSCENES_DATASET)
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
    orientation_quarter_turns_by_id: dict[str, int] = {
        str(row["rerun_segment_id"]): orientation_quarter_turns_from_segment_row(row) for row in rows
    }
    results: list[SegmentResult] = []
    registration_handles: list[RegistrationHandle] = []
    for segment_id in segment_ids:
        result: SegmentResult = process_segment(
            dataset_entry,
            segment_id,
            portrait_by_id[segment_id],
            orientation_quarter_turns_by_id[segment_id],
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
