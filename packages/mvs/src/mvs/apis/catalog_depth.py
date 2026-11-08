"""Run the multiview depth model over catalog video segments."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import depth_png
import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import device as torch_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from trtkit import OnnxBackendConfig, OnnxOrTrtBackendConfig

from mvs.catalog import PINHOLE_LOWRES_PATH, TIMELINE, CatalogDataConfig, depth_at_lowres, log_camera, log_source_video
from mvs.depth_engine import DepthEngine, DepthOutputs
from mvs.depth_stream import DepthCollate, DepthSample, depth_dataset


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
            log_camera(sample.world_T_cam_44.numpy(), K_native_33, native_wh)
            if segment_frames == 1:
                log_source_video(video_decoder)

            if sample.inputs is not None:
                depth_outputs: DepthOutputs = engine(sample.inputs)
                depth_hw: Float32[ndarray, "h=192 w=256"]
                depth_hw, _ = depth_at_lowres(depth_outputs, K_native_33, native_wh)
                depth_mm_hw: UInt16[ndarray, "h=192 w=256"] = np.rint(depth_hw * 1000.0).astype(np.uint16)
                rr.log(
                    f"{PINHOLE_LOWRES_PATH}/depth",
                    rr.EncodedDepthImage(blob=depth_png.encode(depth_mm_hw), media_type=rr.components.MediaType.PNG, meter=1000.0),
                )
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
