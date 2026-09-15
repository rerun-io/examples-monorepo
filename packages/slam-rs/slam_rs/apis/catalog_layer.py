"""Run one catalog segment and replace its single slam_rs results layer."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import torch
from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer

from slam_rs import _core
from slam_rs.catalog_feed import CatalogSegment, RigProfile, SegmentFeed, open_segment
from slam_rs.catalog_layer import write_layer
from slam_rs.config import ImuParameters, SlamConfig, config_text_sha256, load_slam_config
from slam_rs.tracking import Lockstep, SegmentRun, _drive, check_calibration_matches_recording, robocap_estimator_files


@dataclass(slots=True)
class Config:
    """Read catalog cameras and IMU, run VIO, then replace one pose layer."""

    segment: str
    """Full segment ID, including the known dataset prefix (robocap, msd-index, msd-odyssey or msd-g2)."""
    output_dir: Path
    """This dataset's results directory, visible at the same path to this host and the catalog server."""
    catalog: str | None = None
    """Catalog URL; defaults to slam.toml."""
    profile: Literal["fast", "reference"] = "fast"
    """Estimator configuration profile."""
    backend: Literal["auto", "gpu", "cpu"] = "auto"
    """Auto uses the GPU build when a driver is available; GPU requires it; CPU disables it."""
    decode_device: Literal["auto", "cuda", "cpu"] = "auto"
    """CUDA/NVDEC when available; CPU retains the reference PyAV grayscale conversion."""


def main(config: Config) -> None:
    """Track the complete segment before publishing any replacement."""
    if Path(config.segment).name != config.segment:
        raise ValueError("segment ID must not contain path separators")
    settings: SlamConfig = load_slam_config()
    dataset_name: str = config.segment.split("__", 1)[0]
    is_robocap: bool = dataset_name == "robocap"
    parameters: ImuParameters = settings.robocap.imu if is_robocap else settings.dataset(dataset_name).imu
    rig_profile: RigProfile = RigProfile.from_robocap(settings.robocap) if is_robocap else RigProfile()
    catalog_url: str = config.catalog or settings.catalog_url
    dataset: DatasetEntry = CatalogClient(catalog_url).get_dataset(dataset_name)
    output: Path = config.output_dir.resolve() / f"{config.segment}.rrd"
    decode_device: Literal["cpu", "cuda"] = (
        "cuda" if config.decode_device == "cuda" or (config.decode_device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if decode_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA decoding requires an available NVIDIA GPU and the slam-rs-cuda environment")
    print(f"Loading {config.segment}: one bulk video query, decode={decode_device}", flush=True)
    started: float = perf_counter()
    feed: SegmentFeed
    with open_segment(CatalogSegment(catalog_url, dataset_name, config.segment), parameters, profile=rig_profile,
                      cache_video=True, decode_device=decode_device, include_ground_truth=False) as feed:
        load_s: float = perf_counter() - started
        calibration: _core.Calibration
        flow: _core.VioConfig
        config_text: str
        if is_robocap:
            calibration, flow, config_text = robocap_estimator_files(settings, config.profile)
            check_calibration_matches_recording(calibration, feed.cameras, parameters, rig_profile.downscale)
        else:
            calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)
            config_text = settings.vio_config_text(dataset_name, config.profile)
            flow = _core.VioConfig.from_json(config_text)
        use_gpu: bool = config.backend == "gpu" or (config.backend == "auto" and _core.gpu_backend is not None)
        vio: _core.Vio
        try:
            vio = _core.Vio(calibration, flow, gpu=use_gpu)
        except ValueError as error:
            # Only an absent adapter permits auto fallback. Invalid calibration,
            # config, client panics and failures during tracking must still fail.
            if config.backend != "auto" or "wgpu found no" not in str(error):
                raise
            print(f"GPU unavailable: {error}; using CPU", flush=True)
            vio = _core.Vio(calibration, flow, gpu=False)
        backend: str = str(_core.gpu_backend) if vio.gpu else "cpu"
        print(f"{config.segment}: profile={config.profile}, backend={backend}, decode={decode_device}, load={load_s:.1f}s, "
              f"{len(feed.frame_t_ns)} framesets, cameras={feed.camera_positions}", flush=True)
        run: SegmentRun = _drive(feed, Lockstep(vio), config_sha256=config_text_sha256(config_text))
        write_layer(output, config.segment, run, clock_offset_ns=feed.export_offset_ns + parameters.cam_time_offset_ns,
                    profile=config.profile, backend=backend, decoder=decode_device)
    dataset.register([output.as_uri()], layer_name="slam_rs", on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    print(f"Registered {len(run.estimate)} poses; {run.wall_s:.1f} s, {run.framesets / run.wall_s:.1f} framesets/s; {output}")
    print(dataset.segment_url(config.segment))
