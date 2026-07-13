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
import pyarrow as pa
import rerun as rr
import torch
from arkitscenes_download.ingest.blueprint import make_blueprint
from arkitscenes_download.ingest.depth import encode_depth_png
from arkitscenes_download.ingest.paths import CONFIDENCE, DEPTH, DEPTH_PROMPTDA, PINHOLE_WIDE, PROMPTDA_MESH, RIG, VIDEO_WIDE
from beartype.roar import BeartypeException
from jaxtyping import Float, Float32, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer
from rerun.experimental.dataloader import (
    DataSource,
    Field,
    FixedRateSampling,
    ImageDecoder,
    NumericDecoder,
    RerunMapDataset,
    VideoFrameDecoder,
)
from scipy.spatial.transform import Rotation
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.trt_predictor import PromptDATrtPredictor

ARKITSCENES_DATASET = "arkitscenes"
PROMPTDA_LAYER = "promptda"
NATIVE_FPS = 60.0
ARKIT_CONFIDENCE_MEDIUM = 1


@dataclass
class PDAArkitScenesConfig:
    """Runtime configuration for ARKitScenes PromptDA inference."""

    video_id: str | None = None
    """One catalog segment to process and replace, when present."""
    process_all: bool = False
    """Process every segment that does not yet have a PromptDA layer."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
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
    depth_fusion_resolution: float = 0.04
    """TSDF voxel resolution in metres."""
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

    dav1d occasionally rejects a decode window outright (upstream,
    order-dependent — see the mv-api catalog dataloader docs). Each sample is
    decoded with a fresh codec context, so the failure is recoverable at
    sample granularity: map it onto the decoder's existing ``None``-frame
    skip path instead of killing the whole iteration. The caught exceptions
    are kept alive on purpose — releasing a failed decode's traceback frees
    the poisoned codec context, whose dav1d teardown can deadlock the
    process in ``dav1d_flush``.
    """

    def __init__(self, *, keyframe_interval: int = 30, fps_estimate: float = 30.0, codec: str = "h264") -> None:
        super().__init__(keyframe_interval=keyframe_interval, fps_estimate=fps_estimate, codec=codec)
        self.decode_failures: list[BaseException] = []

    def decode(self, raw: Any, index_value: Any, segment_id: str) -> Any:
        """Decode one sample, degrading decoder errors to a skippable ``None``."""
        try:
            return super().decode(raw, index_value, segment_id)
        except av.error.InvalidDataError as error:
            self.decode_failures.append(error)
            return None


def stride_for(native_fps: float, target_fps: float) -> int:
    """Return the closest positive whole-frame stride for a target rate."""
    if target_fps <= 0.0:
        raise ValueError("target_fps must be greater than zero")
    return max(1, round(native_fps / target_fps))


def k_matrix_from_flat(k_flat: Float[ndarray, "9"]) -> Float[ndarray, "3 3"]:
    """Decode a flattened column-major camera intrinsic matrix.

    Args:
        k_flat: Nine floating-point matrix elements in column-major order.

    Returns:
        Floating-point ``3x3`` image-from-camera matrix.
    """
    return np.asarray(k_flat).reshape(3, 3, order="F")


def world_t_cam_from_pose(
    translation: Float[ndarray, "3"], quaternion_xyzw: Float[ndarray, "4"]
) -> Float[ndarray, "4 4"]:
    """Build a camera-to-world transform from an ARKit rig pose.

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
    """Mask low-confidence and out-of-range predictions before fusion.

    Args:
        depth_mm: Predicted uint16 depth in millimetres with shape ``(H, W)``.
        confidence: ARKit uint8 confidence values with shape ``(H2, W2)``.
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


def segments_to_process(rows: list[dict], video_id: str | None, process_all: bool) -> list[str]:
    """Select catalog segment ids according to the requested execution mode."""
    if (video_id is None) == (not process_all):
        raise SystemExit("give exactly one of --video-id or --process-all")
    available_ids: list[str] = [str(row["rerun_segment_id"]) for row in rows]
    if video_id is not None:
        if video_id not in available_ids:
            raise SystemExit(f"video id {video_id!r} is absent; available ids: {', '.join(available_ids)}")
        return [video_id]
    return [
        str(row["rerun_segment_id"])
        for row in rows
        if PROMPTDA_LAYER not in (row.get("rerun_layer_names") or [])
    ]


def _list_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a fetched chunk as a plain list of sample dicts (no tensor stacking)."""
    return batch


def _batched_loader(rerun_dataset: RerunMapDataset, chunks: list[list[int]], num_workers: int) -> DataLoader:
    """Build a DataLoader that fetches whole chunks, in order, on worker processes.

    The batch fetch (server query + per-sample GOP decode back to the previous
    keyframe) is the pipeline's slowest stage and is GIL-bound in-process, so
    spawn-based worker processes are the only way to parallelize it. The
    Rerun dataloader requires the ``spawn`` start method (forked workers
    deadlock on their first catalog call).
    """
    if num_workers == 0:
        return DataLoader(rerun_dataset, batch_sampler=chunks, collate_fn=_list_collate)
    return DataLoader(
        rerun_dataset,
        batch_sampler=chunks,
        collate_fn=_list_collate,
        num_workers=num_workers,
        multiprocessing_context=multiprocessing.get_context("spawn"),
        prefetch_factor=2,
    )


def connect_catalog(catalog_url: str) -> CatalogClient:
    """Connect to the local ARKitScenes catalog or terminate with setup guidance."""
    try:
        client: CatalogClient = CatalogClient(catalog_url)
        client.dataset_names()
    except BeartypeException:
        raise
    except Exception as error:
        raise SystemExit(f"catalog not reachable at {catalog_url} — start it with `pixi run arkitscenes-download-serve`") from error
    return client


def process_segment(
    dataset_entry: DatasetEntry,
    segment_id: str,
    portrait: bool,
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
    # A map dataset fetches only the strided samples. The iterable dataset
    # would decode every sample on the 60 fps grid (each decode replays the
    # whole GOP window), i.e. `stride`× the work for frames we throw away.
    rerun_dataset: RerunMapDataset = RerunMapDataset(
        source,
        index="video_time",
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=NATIVE_FPS),
    )
    stride: int = stride_for(NATIVE_FPS, config.target_fps)
    rrd_path: Path = config.output_dir / segment_id / "promptda.rrd"
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
    inferred_frames: int = 0
    skipped_decodes: int = 0
    strided_indices: list[int] = list(range(0, rerun_dataset.sample_index.total_samples, stride))
    chunks: list[list[int]] = [strided_indices[i : i + config.batch_size] for i in range(0, len(strided_indices), config.batch_size)]
    loader: DataLoader = _batched_loader(rerun_dataset, chunks, config.num_workers)
    for chunk, samples in tqdm(zip(chunks, loader, strict=True), total=len(chunks), desc=f"PromptDA {segment_id}", unit="batch"):
        valid: list[tuple[int, dict[str, Any]]] = [
            (sample_idx, sample)
            for sample_idx, sample in zip(chunk, samples, strict=True)
            if all(sample.get(field_name) is not None for field_name in fields)
        ]
        skipped_decodes += len(chunk) - len(valid)
        if not valid:
            continue
        rgb_bhw3: UInt8[Tensor, "b h w 3"] = torch.stack([sample["video"].permute(1, 2, 0) for _, sample in valid])
        prompt_bhw: Float32[Tensor, "b 192 256"] = torch.from_numpy(
            np.stack([sample["depth"].numpy().squeeze(0) for _, sample in valid]).astype(np.float32) / 1000.0
        )
        depth_bhw: Float32[Tensor, "b h w"] = predictor(rgb_bhw3, prompt_bhw)
        depth_mm_bhw: UInt16[ndarray, "b h w"] = (depth_bhw.cpu().numpy() * 1000.0).astype(np.uint16)
        for row, (sample_idx, sample) in enumerate(valid):
            predicted_depth_hw: UInt16[ndarray, "h w"] = depth_mm_bhw[row]
            confidence_hw: UInt8[ndarray, "h2 w2"] = np.asarray(sample["conf"].numpy().reshape(192, 256), dtype=np.uint8)
            timestamp: Any = rerun_dataset.sample_index.global_to_local(sample_idx)[1]
            timestamp_seconds: float = float(timestamp / np.timedelta64(1, "s"))
            rr.set_time("video_time", duration=timestamp_seconds, recording=recording)
            rr.log(
                DEPTH_PROMPTDA,
                rr.EncodedDepthImage(blob=encode_depth_png(predicted_depth_hw), media_type="image/png", meter=1000.0),
                recording=recording,
            )
            filtered_depth_hw: UInt16[ndarray, "h w"] = filter_depth_for_fusion(
                predicted_depth_hw,
                confidence_hw,
                config.max_depth_range_meter,
            )
            world_t_cam_44: Float[ndarray, "4 4"] = world_t_cam_from_pose(
                np.asarray(sample["pose_t"].numpy()), np.asarray(sample["pose_q"].numpy())
            )
            fuser.fuse_frames(
                depth_hw=filtered_depth_hw,
                K_33=k_matrix_from_flat(np.asarray(sample["k"].numpy())),
                cam_T_world_44=np.linalg.inv(world_t_cam_44),
                rgb_hw3=np.asarray(rgb_bhw3[row].numpy()),
            )
            inferred_frames += 1

    mesh: Any = fuser.get_mesh()
    mesh.compute_vertex_normals()
    vertices: Float[ndarray, "n 3"] = np.asarray(mesh.vertices)
    rr.log(
        PROMPTDA_MESH,
        rr.Mesh3D(
            vertex_positions=vertices,
            triangle_indices=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=np.asarray(mesh.vertex_colors),
            # Same see-through treatment as the GT mesh: cull back-facing walls
            # so the 3D view looks into the room from outside.
            face_rendering=rr.components.MeshFaceRendering.Front,
        ),
        static=True,
        recording=recording,
    )
    # Embed the PromptDA layout in this layer, framed on the fused mesh (the
    # same per-sequence treatment the base layer gives the GT mesh). Sent at
    # the end of the stream so it wins blueprint activation when the segment's
    # layers load together in the viewer.
    if len(vertices) > 0:
        mesh_center_xyz: Float[ndarray, "3"] = vertices.mean(axis=0)
        bounding_radius_m: float = float(np.linalg.norm(vertices - mesh_center_xyz, axis=1).max())
        rr.send_blueprint(
            make_blueprint(portrait=portrait, framing=(mesh_center_xyz, bounding_radius_m), include_promptda=True),
            recording=recording,
        )
    recording.flush()
    return SegmentResult(rrd_path=rrd_path, inferred_frames=inferred_frames, skipped_decodes=skipped_decodes)


def run_prompt_da_arkitscenes(config: PDAArkitScenesConfig) -> None:
    """Run PromptDA for selected segments and optionally update the catalog."""
    if (config.video_id is None) == (not config.process_all):
        raise SystemExit("give exactly one of --video-id or --process-all")
    client: CatalogClient = connect_catalog(config.catalog_url)
    if ARKITSCENES_DATASET not in client.dataset_names():
        raise SystemExit(f"dataset {ARKITSCENES_DATASET!r} is absent — create it with `pixi run arkitscenes-download-register`")
    dataset_entry: DatasetEntry = client.get_dataset(ARKITSCENES_DATASET)
    segment_table: pa.Table = pa.Table.from_batches(dataset_entry.segment_table().collect())
    rows: list[dict] = segment_table.to_pylist()
    segment_ids: list[str] = segments_to_process(rows, config.video_id, config.process_all)
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
    portrait_by_id: dict[str, bool] = {str(row["rerun_segment_id"]): bool(row.get("property:portrait:value")) for row in rows}
    results: list[SegmentResult] = []
    for segment_id in segment_ids:
        result: SegmentResult = process_segment(dataset_entry, segment_id, portrait_by_id[segment_id], config, predictor)
        results.append(result)
        if config.register:
            dataset_entry.register(
                [result.rrd_path.resolve().as_uri()],
                layer_name=PROMPTDA_LAYER,
                on_duplicate=OnDuplicateSegmentLayer.REPLACE,
            ).wait()

    if config.register and config.update_blueprint:
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


def main(config: PDAArkitScenesConfig) -> None:
    """Run the ARKitScenes PromptDA command-line workflow."""
    run_prompt_da_arkitscenes(config)
