"""A TSDF mesh grown from the depth model over one ARKitScenes catalog segment, logged as it grows.

Frames come through the Rerun dataloader, decoded on the GPU (:func:`mvs.depth_stream.depth_dataset`); each depth
frame is fused into a sparse colour TSDF (``packages/tsdf-rs``) at the catalog pose. The surface is logged with
:class:`simplecv.rerun_live_mesh.LiveMeshLog`: the changed super-blocks on the timeline while the segment plays, one
mesh from its last depth frame. Every depth frame is logged as mvs-catalog-depth logs it, and the recording carries a
default layout (:func:`mvs.catalog.blueprint`). The source video is relayed once as a VideoStream, never re-encoded per
frame.

One run meshes one segment (:func:`single_segment`): every segment restarts its frame index at 0, so a second segment
in the same recording would mix its mesh history with the first one's.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
from einops import rearrange, reduce
from jaxtyping import Float32, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.rerun_live_mesh import LiveMeshLog
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from trtkit import OnnxBackendConfig, OnnxOrTrtBackendConfig
from tsdf_rs import TsdfVolume

from mvs.catalog import FRAME_TIMELINE, TIMELINE, CatalogDataConfig, blueprint, depth_at_lowres, log_camera, log_depth, log_source_video
from mvs.depth_engine import IMAGENET_MEAN, IMAGENET_STD, DepthEngine, DepthOutputs
from mvs.depth_stream import DepthCollate, DepthSample, depth_dataset

MESH_PATH: str = "world/mesh"


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


def depth_raster_rgb(cur_image_b3hw: Float32[Tensor, "1 3 h=384 w=512"]) -> UInt8[ndarray, "h=192 w=256 3"]:
    """The model's current image, un-normalised and averaged 2x2 down to the depth raster, as 8-bit RGB."""
    mean_311: Float32[Tensor, "3 1 1"] = rearrange(cur_image_b3hw.new_tensor(IMAGENET_MEAN), "c -> c 1 1")
    std_311: Float32[Tensor, "3 1 1"] = rearrange(cur_image_b3hw.new_tensor(IMAGENET_STD), "c -> c 1 1")
    rgb_3hw: Float32[Tensor, "3 h w"] = cur_image_b3hw[0] * std_311 + mean_311
    rgb_hw3: Float32[Tensor, "h2 w2 3"] = reduce(rgb_3hw, "c (h 2) (w 2) -> h w c", "mean")
    return (rgb_hw3.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()


def single_segment(segment_ids: Sequence[str]) -> str:
    """The one segment a run meshes. Segments restart their frame index at 0, so two in one recording would mix
    their mesh histories on the same entity paths (a clear cannot hide rows logged before it): one segment per run."""
    if len(segment_ids) != 1:
        raise ValueError(f"mvs-live-mesh meshes one segment per recording, got {len(segment_ids)}: pass --data.segments <id>")
    return segment_ids[0]


def main(config: Config) -> None:
    """Predict depth for one catalog segment, fuse it, and log the growing mesh."""
    started: float = time.perf_counter()
    client: CatalogClient = CatalogClient(config.data.catalog_url)
    dataset: DatasetEntry = client.get_dataset(name=config.data.dataset_name)
    segment_id: str = single_segment(list(config.data.segments) or dataset.segment_ids())
    engine: DepthEngine = DepthEngine(config.model_path, config.backend)
    fusion: FusionConfig = config.fusion
    rr.send_blueprint(blueprint())
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    device: torch.device = torch.device("cuda")

    samples, video_decoder = depth_dataset(dataset, segment_id, device)
    # num_workers stays 0: the collate is causal (keyframe buffer) and the NVDEC decoder is CUDA-stateful.
    loader: DataLoader = DataLoader(samples, batch_size=1, num_workers=0, collate_fn=DepthCollate(samples, device))
    volume: TsdfVolume = TsdfVolume(voxel_size=fusion.voxel_size_m, sdf_trunc=fusion.truncation_voxels * fusion.voxel_size_m, fuse_color=True)
    mesh_log: LiveMeshLog = LiveMeshLog(MESH_PATH)
    frames: int = 0
    depth_frames: int = 0
    last_depth_time: tuple[int, int] | None = None
    sample: DepthSample | None
    for sample in tqdm(loader, desc=segment_id, unit="frame", total=samples.sample_index.total_samples):
        if sample is None:
            continue
        frames += 1
        rr.set_time(TIMELINE, duration=np.timedelta64(sample.timestamp_ns, "ns"))
        rr.set_time(FRAME_TIMELINE, sequence=sample.frame_index)
        K_native_33: Float32[ndarray, "3 3"] = sample.K_native_33.numpy()
        world_T_cam_44: Float32[ndarray, "4 4"] = sample.world_T_cam_44.numpy()
        log_camera(world_T_cam_44, K_native_33, sample.resolution_wh)
        if frames == 1:
            log_source_video(video_decoder)
        if sample.inputs is None:
            continue

        depth_outputs: DepthOutputs = engine(sample.inputs)
        depth_hw: Float32[ndarray, "h=192 w=256"]
        K_depth_33: Float32[ndarray, "3 3"]
        depth_hw, K_depth_33 = depth_at_lowres(depth_outputs, K_native_33, sample.resolution_wh)
        log_depth(depth_hw, K_depth_33)
        volume.integrate(
            depth_hw,
            K_depth_33,
            np.linalg.inv(world_T_cam_44.astype(np.float64)).astype(np.float32),
            depth_raster_rgb(sample.inputs.cur_image_b3hw),
            max_depth=fusion.max_depth_m,
        )
        depth_frames += 1
        last_depth_time = (sample.timestamp_ns, sample.frame_index)
        if depth_frames % fusion.log_every == 0:
            mesh_log.log_changed(volume.extract_dirty_block_meshes())

    if last_depth_time is not None:
        rr.set_time(TIMELINE, duration=np.timedelta64(last_depth_time[0], "ns"))
        rr.set_time(FRAME_TIMELINE, sequence=last_depth_time[1])
        mesh_log.finish(volume.extract_mesh())
    print(f"segment {segment_id}: {frames} frames, {depth_frames} fused depth frames in {time.perf_counter() - started:.2f}s")
