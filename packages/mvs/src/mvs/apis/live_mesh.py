"""A TSDF mesh grown from the depth model over one ARKitScenes catalog segment, logged as it grows.

Frames come through the Rerun dataloader, decoded on the GPU (:func:`mvs.depth_stream.depth_dataset`); each depth
frame is fused into a sparse colour TSDF (``packages/tsdf-rs``) at the catalog pose. The surface is logged with
:class:`simplecv.rerun_live_mesh.LiveMeshLog`: the changed super-blocks on the timeline while the segment plays, one
mesh from its last depth frame. The recording carries the original SimpleRecon live-meshing layout
(:func:`mvs.blueprint.create_blueprint`, ``native``) and logs what its views show (:mod:`mvs.frame_views`): the green
current frustum, the seven source frusta and images of each depth query, the depth, its normals, the lowest matching
cost, and the per-stage timings in ms. The source video is relayed once as a VideoStream, never re-encoded per frame.

One run meshes one segment (:func:`single_segment`): every segment restarts its frame index at 0, so a second segment
in the same recording would mix its mesh history with the first one's.
"""

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from jaxtyping import Float, Float32
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.rerun_live_mesh import BlockKey, LiveMeshLog, MeshArrays
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from trtkit import OnnxBackendConfig, OnnxOrTrtBackendConfig
from tsdf_rs import TsdfVolume

from mvs.blueprint import SOURCE_CAMS, create_blueprint
from mvs.catalog import FRAME_TIMELINE, PATHS, TIMELINE, CatalogDataConfig, depth_at_lowres, log_camera, log_depth, log_source_video
from mvs.depth_engine import DepthEngine, DepthInputs, DepthOutputs
from mvs.depth_stream import DepthCollate, DepthSample, depth_dataset
from mvs.frame_views import LogConfig, log_current_cam_style, log_lowest_cost, log_normals, log_sources, rgb_at_depth_raster
from mvs.log_worker import LogWorker


@dataclass(frozen=True)
class FusionConfig:
    """TSDF fusion of the predicted depth, in metres."""

    voxel_size_m: float = 0.04
    """Voxel edge."""
    truncation_voxels: float = 3.0
    """Signed-distance truncation in voxels (Open3D's ``3 * voxel`` convention)."""
    max_depth_m: float = 5.0
    """Farthest depth fused; the model's far predictions are the least reliable."""
    log_every: int = 10
    """Depth frames between two logs of the changed blocks; at least 1."""

    def __post_init__(self) -> None:
        if self.log_every < 1:
            raise ValueError(f"log_every must be at least 1, got {self.log_every}")


@dataclass
class Config:
    """Configuration for the live meshing tool."""

    rr_config: RerunTyroConfig
    """Rerun viewer/save/connect behavior."""
    data: CatalogDataConfig = field(default_factory=CatalogDataConfig)
    """Where the catalog segments come from."""
    model_path: Path = Path("data/models/depth_model_dot.onnx")
    """Portable ONNX depth model artifact."""
    backend: OnnxOrTrtBackendConfig = field(default_factory=OnnxBackendConfig)
    """CUDA ONNX Runtime or cached TensorRT execution backend."""
    fusion: FusionConfig = field(default_factory=FusionConfig)
    """TSDF fusion and mesh logging."""
    log: LogConfig = field(default_factory=LogConfig)
    """Which groups of the layout are logged; the defaults fill every view."""


@contextmanager
def timed(times_ms: dict[str, float], stage: str) -> Iterator[None]:
    """Record the wall time of the block in ``times_ms[stage]``, ms, synchronising CUDA on both sides (as the
    original's StageTimer does) so queued GPU work lands in the stage that queued it."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start: float = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times_ms[stage] = (time.perf_counter() - start) * 1000.0


def single_segment(segment_ids: Sequence[str]) -> str:
    """The one segment a run meshes. Segments restart their frame index at 0, so two in one recording would mix
    their mesh histories on the same entity paths (a clear cannot hide rows logged before it): one segment per run."""
    if len(segment_ids) != 1:
        raise ValueError(f"mvs-live-mesh meshes one segment per recording, got {len(segment_ids)}: pass --data.segments <id>")
    return segment_ids[0]


def log_depth_frame(
    log: LogConfig,
    mesh_log: LiveMeshLog,
    depth_hw: Float32[ndarray, "h=192 w=256"],
    K_depth_33: Float32[ndarray, "3 3"],
    inputs: DepthInputs,
    depth_b1hw: Float[Tensor, "1 1 h=192 w=256"],
    lowest_cost_bhw: Float[Tensor, "1 h=96 w=128"],
    changed_blocks: list[tuple[BlockKey, MeshArrays]] | None,
) -> None:
    """One depth frame's views, logged on the log worker at the frame's time point: the depth, the source frusta and
    images, the normals, the lowest-cost depth and, when extracted, the changed mesh blocks."""
    log_depth(depth_hw, K_depth_33)
    if log.source_cams:
        log_sources(PATHS, inputs, log.jpeg_quality)
    if log.normals:
        log_normals(PATHS, depth_b1hw, K_depth_33, log.jpeg_quality)
    if log.lowest_cost:
        log_lowest_cost(PATHS, lowest_cost_bhw)
    if changed_blocks is not None:
        mesh_log.log_changed(changed_blocks)


def main(config: Config) -> None:
    """Predict depth for one catalog segment, fuse it, and log the growing mesh with the views of the layout."""
    started: float = time.perf_counter()
    client: CatalogClient = CatalogClient(config.data.catalog_url)
    dataset: DatasetEntry = client.get_dataset(name=config.data.dataset_name)
    segment_id: str = single_segment(list(config.data.segments) or dataset.segment_ids())
    engine: DepthEngine = DepthEngine(config.model_path, config.backend)
    fusion: FusionConfig = config.fusion
    log: LogConfig = config.log
    rr.send_blueprint(create_blueprint(PATHS, variant="native", source_cams=SOURCE_CAMS if log.source_cams else 0))
    rr.log(PATHS.world, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_current_cam_style(PATHS)
    device: torch.device = torch.device("cuda")

    samples, video_decoder = depth_dataset(dataset, segment_id, device)
    # num_workers stays 0: the collate is causal (keyframe buffer) and the NVDEC decoder is CUDA-stateful.
    loader: DataLoader = DataLoader(samples, batch_size=1, num_workers=0, collate_fn=DepthCollate(samples, device))
    volume: TsdfVolume = TsdfVolume(voxel_size=fusion.voxel_size_m, sdf_trunc=fusion.truncation_voxels * fusion.voxel_size_m, fuse_color=True)
    mesh_log: LiveMeshLog = LiveMeshLog(PATHS.mesh)
    frames: int = 0
    depth_frames: int = 0
    last_depth_time: tuple[int, int] | None = None
    times_ms: dict[str, float] = {}
    waited_from: float = time.perf_counter()
    sample: DepthSample | None
    # The log worker encodes and logs each depth frame's views off the main loop; leaving the block, on an exception
    # too, waits for its queued jobs (raising theirs) and joins it.
    with LogWorker() as log_worker:
        for sample in tqdm(loader, desc=segment_id, unit="frame", total=samples.sample_index.total_samples):
            # data_wait: the loader's decode, preprocessing and query assembly since the last frame's work ended.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times_ms["data_wait"] = (time.perf_counter() - waited_from) * 1000.0
            if sample is None:
                waited_from = time.perf_counter()
                continue
            frames += 1
            rr.set_time(TIMELINE, duration=np.timedelta64(sample.timestamp_ns, "ns"))
            rr.set_time(FRAME_TIMELINE, sequence=sample.frame_index)
            K_native_33: Float32[ndarray, "3 3"] = sample.K_native_33.numpy()
            world_T_cam_44: Float32[ndarray, "4 4"] = sample.world_T_cam_44.numpy()
            log_camera(world_T_cam_44, K_native_33, sample.resolution_wh)
            if frames == 1 and log.camera:
                log_source_video(video_decoder)
            if sample.inputs is None:
                waited_from = time.perf_counter()
                continue

            with timed(times_ms, "model_forward"):
                depth_outputs: DepthOutputs = engine(sample.inputs)
            depth_hw: Float32[ndarray, "h=192 w=256"]
            K_depth_33: Float32[ndarray, "3 3"]
            depth_hw, K_depth_33 = depth_at_lowres(depth_outputs, K_native_33, sample.resolution_wh)
            with timed(times_ms, "tsdf_fuse"):
                volume.integrate(
                    depth_hw,
                    K_depth_33,
                    np.linalg.inv(world_T_cam_44.astype(np.float64)).astype(np.float32),
                    rgb_at_depth_raster(sample.inputs.cur_image_b3hw)[0],
                    max_depth=fusion.max_depth_m,
                )
            depth_frames += 1
            last_depth_time = (sample.timestamp_ns, sample.frame_index)
            extract_mesh: bool = depth_frames % fusion.log_every == 0
            changed_blocks: list[tuple[BlockKey, MeshArrays]] | None = None
            if extract_mesh:
                with timed(times_ms, "mesh_extract"):
                    changed_blocks = volume.extract_dirty_block_meshes()
            # rerun_log measures only the back-pressure wait for the worker, as in the original.
            with timed(times_ms, "rerun_log"):
                # The engine may reuse its output buffers, so the worker gets its own copies.
                log_worker.submit(
                    sample.timestamp_ns,
                    sample.frame_index,
                    log_depth_frame,
                    log,
                    mesh_log,
                    depth_hw,
                    K_depth_33,
                    sample.inputs,
                    depth_outputs.depth_b1hw.clone(),
                    depth_outputs.lowest_cost_bhw.clone(),
                    changed_blocks,
                )
            if log.timings:
                stage: str
                for stage in ("data_wait", "model_forward", "tsdf_fuse", "mesh_extract", "rerun_log"):
                    if stage == "mesh_extract" and not extract_mesh:
                        continue
                    rr.log(PATHS.timing(stage), rr.Scalars(times_ms[stage]))
            waited_from = time.perf_counter()

        if last_depth_time is not None:
            # The final mesh goes through the worker too, so it and its Clear follow every block log.
            log_worker.submit(last_depth_time[0], last_depth_time[1], mesh_log.finish, volume.extract_mesh())
    print(f"segment {segment_id}: {frames} frames, {depth_frames} fused depth frames in {time.perf_counter() - started:.2f}s")
