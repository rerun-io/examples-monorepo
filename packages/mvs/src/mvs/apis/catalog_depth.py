"""Run the multiview depth model over catalog video segments."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
from einops import rearrange
from jaxtyping import Float32, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import device as torch_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from trtkit import OnnxBackendConfig, OnnxOrTrtBackendConfig

from mvs.apis.live_mesh import TIMELINE, CatalogDataConfig
from mvs.depth_engine import DepthEngine, DepthOutputs
from mvs.depth_stream import DepthCollate, DepthSample, depth_dataset

CAMERA_PATH: str = "world/rig_00/cam_00"
PINHOLE_PATH: str = f"{CAMERA_PATH}/pinhole"
VIDEO_PATH: str = f"{PINHOLE_PATH}/video"
# The viewer back-projects a DepthImage with its parent pinhole's intrinsics, so the
# 256x192 depth needs its own depth-resolution pinhole beside the native one. Path
# names mirror the arkitscenes-download rig schema (ingest/paths.py).
PINHOLE_LOWRES_PATH: str = f"{CAMERA_PATH}/pinhole_lowres"


@dataclass
class Config:
    """Catalog depth demo configuration."""

    rr_config: RerunTyroConfig
    """Rerun viewer, connection, and save behavior."""
    data: CatalogDataConfig = field(default_factory=CatalogDataConfig)
    """Catalog server, dataset, and segment selection."""
    model_path: Path = Path("data/models/depth_model_dot.onnx")
    """Portable ONNX depth model artifact."""
    backend: OnnxOrTrtBackendConfig = field(default_factory=OnnxBackendConfig)
    """CUDA ONNX Runtime or cached TensorRT execution backend."""


def main(config: Config) -> None:
    """Predict and log metric depth for every requested catalog segment.

    Args:
        config: Catalog, model, backend, and Rerun options.
    """
    started: float = time.perf_counter()

    client: CatalogClient = CatalogClient(config.data.catalog_url)
    dataset: DatasetEntry = client.get_dataset(name=config.data.dataset_name)
    segment_ids: list[str] = list(config.data.segments) or dataset.segment_ids()
    engine: DepthEngine = DepthEngine(config.model_path, config.backend)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    device: torch_device = torch_device("cuda")
    total_frames: int = 0
    total_depth_frames: int = 0

    segment_id: str
    for segment_id in segment_ids:
        samples, video_decoder = depth_dataset(dataset, segment_id, device)
        # num_workers stays 0: the collate is causal (keyframe buffer) and the NVDEC
        # decoder is CUDA-stateful; workers shard the sample space. Training over
        # precomputed tuples is where workers and real batches apply.
        loader: DataLoader = DataLoader(samples, batch_size=1, num_workers=0, collate_fn=DepthCollate(samples, device))
        grid_slots: int = samples.sample_index.total_samples
        segment_frames: int = 0
        segment_depth_frames: int = 0
        sample: DepthSample | None
        for sample in tqdm(loader, desc=segment_id, unit="frame", total=grid_slots):
            if sample is None:
                continue
            native_wh: tuple[int, int] = sample.resolution_wh
            segment_frames += 1
            total_frames += 1
            rr.set_time(TIMELINE, duration=np.timedelta64(sample.timestamp_ns, "ns"))
            rr.set_time("frame", sequence=sample.frame_index)

            # numpy appears only at the rr.log boundary, as zero-copy views of CPU tensors.
            K_native_33: Float32[ndarray, "3 3"] = sample.K_native_33.numpy()
            rr.log(
                CAMERA_PATH,
                rr.Transform3D(
                    mat3x3=sample.world_T_cam_44[:3, :3].numpy(),
                    translation=sample.world_T_cam_44[:3, 3].numpy(),
                ),
            )
            rr.log(
                PINHOLE_PATH,
                rr.Pinhole(
                    image_from_camera=K_native_33,
                    width=native_wh[0],
                    height=native_wh[1],
                    camera_xyz=rr.ViewCoordinates.RDF,
                ),
            )
            # The decoder already fetched the segment's raw AV1 samples: relog them once
            # as a VideoStream layer (the source catalog's own schema) instead of
            # re-encoding every decoded frame to JPEG.
            if segment_frames == 1:
                assert video_decoder.samples is not None and video_decoder.keyframes is not None
                rr.log(VIDEO_PATH, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
                rr.send_columns(
                    VIDEO_PATH,
                    indexes=[rr.TimeColumn(TIMELINE, duration=video_decoder.times)],
                    columns=rr.VideoStream.columns(sample=video_decoder.samples, is_keyframe=video_decoder.keyframes),
                )

            if sample.inputs is not None:
                depth_outputs: DepthOutputs = engine(sample.inputs)
                depth_hw: Float32[ndarray, "h=192 w=256"] = (
                    rearrange(depth_outputs.depth_b1hw, "1 1 h w -> h w").cpu().numpy()
                )
                depth_mm_hw: UInt16[ndarray, "h=192 w=256"] = np.rint(depth_hw * 1000.0).astype(np.uint16)
                K_depth_33: Float32[ndarray, "3 3"] = K_native_33.copy()
                K_depth_33[0] *= 256.0 / native_wh[0]
                K_depth_33[1] *= 192.0 / native_wh[1]
                rr.log(
                    PINHOLE_LOWRES_PATH,
                    rr.Pinhole(
                        image_from_camera=K_depth_33,
                        width=256,
                        height=192,
                        camera_xyz=rr.ViewCoordinates.RDF,
                    ),
                )
                rr.log(f"{PINHOLE_LOWRES_PATH}/depth", rr.DepthImage(depth_mm_hw, meter=1000.0).compress())
                segment_depth_frames += 1
                total_depth_frames += 1

        if segment_frames != grid_slots:
            raise RuntimeError(
                f"Segment {segment_id} decoded {segment_frames} frames but the grid has {grid_slots} slots."
            )
        print(f"segment {segment_id}: processed {segment_frames} frames, produced {segment_depth_frames} depth frames")

    wall_time: float = time.perf_counter() - started
    print(
        f"summary: processed {total_frames} frames, produced {total_depth_frames} depth frames in {wall_time:.2f}s "
        f"across {len(segment_ids)} segment(s)"
    )
