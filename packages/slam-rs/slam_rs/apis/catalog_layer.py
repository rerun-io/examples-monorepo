"""Run one catalog segment and replace its single slam_rs results layer."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer

from slam_rs import BackendRequest, _core
from slam_rs.catalog_feed import (
    ROBOCAP_RIG,
    CatalogSegment,
    DecodeDevice,
    DecodeRequest,
    RigProfile,
    SegmentFeed,
    dataset_rig,
    open_segment,
    resolve_catalog_segments,
    select_decode_device,
)
from slam_rs.catalog_layer import POSE_SOURCE, write_layer
from slam_rs.config import SlamConfig, config_text_sha256, load_slam_config
from slam_rs.tracking import Lockstep, SegmentRun, drive


@dataclass(slots=True)
class Config:
    """Read catalog cameras and IMU, run VIO, then replace one pose layer."""

    segment: str
    """Full segment ID, including the known dataset prefix (robocap or a slam.toml ``[[dataset]]``, e.g. msd-g2 or vslam-lab-euroc)."""
    output_dir: Path
    """This dataset's results directory, visible at the same path to this host and the catalog server."""
    catalog: str | None = None
    """Catalog URL; defaults to slam.toml."""
    profile: Literal["fast", "reference"] = "fast"
    """Estimator configuration profile."""
    backend: BackendRequest = "auto"
    """Frontend: auto runs the GPU when this host can, otherwise the CPU with one warning; gpu fails instead; cpu forces it."""
    decode_device: DecodeRequest = "auto"
    """CUDA/NVDEC when available; CPU retains the reference PyAV grayscale conversion."""


def main(config: Config) -> None:
    """Track the complete segment before publishing any replacement."""
    if Path(config.segment).name != config.segment:
        raise ValueError("segment ID must not contain path separators")
    settings: SlamConfig = load_slam_config()
    dataset_name: str = config.segment.split("__", 1)[0]
    is_robocap: bool = dataset_name == "robocap"
    rig_profile: RigProfile = ROBOCAP_RIG if is_robocap else dataset_rig(settings.dataset(dataset_name))
    catalog_url: str = config.catalog or settings.catalog_url
    # One manifest query resolves the dataset handle that open_segment and the final register share.
    segment: CatalogSegment = resolve_catalog_segments((CatalogSegment(catalog_url, dataset_name, config.segment),))[0]
    dataset: DatasetEntry | None = segment.dataset
    if dataset is None:
        raise ValueError(f"{config.segment}: catalog resolution returned no dataset handle")
    output: Path = config.output_dir.resolve() / f"{config.segment}.rrd"
    decode_device: DecodeDevice = select_decode_device(config.decode_device)
    print(f"Loading {config.segment}: one bulk video query, decode={decode_device}", flush=True)
    started: float = perf_counter()
    feed: SegmentFeed
    with open_segment(segment, profile=rig_profile, cache_video=True, decode_device=decode_device, include_ground_truth=False) as feed:
        load_s: float = perf_counter() - started
        config_text: str = settings.robocap_config_text(config.profile) if is_robocap else settings.vio_config_text(dataset_name, config.profile)
        calibration: _core.Calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)
        flow: _core.VioConfig = _core.VioConfig.from_json(config_text)
        # The shared selector decides; only a failed device probe lets auto fall back to the CPU.
        vio: _core.Vio = _core.Vio(calibration, flow, backend=config.backend)
        backend: str = str(_core.gpu_backend) if vio.gpu else "cpu"
        print(f"{config.segment}: profile={config.profile}, backend={backend}, decode={decode_device}, load={load_s:.1f}s, "
              f"{len(feed.frame_t_ns)} framesets, cameras={feed.camera_positions}", flush=True)
        run: SegmentRun = drive(feed, Lockstep(vio), config_sha256=config_text_sha256(config_text), calibration=calibration, run_gba=True)
        if run.gba_failure is not None:
            print(f"global BA: failed: {run.gba_failure}; the final run is re-anchored on the map after loops and PGO", flush=True)
        for drop in run.dropped:
            print(f"Dropped frameset {drop.t_ns}: {drop.reason}", flush=True)
        write_layer(output, config.segment, run, clock_offset_ns=feed.video_time_offset_ns,
                    profile=config.profile, backend=backend, decoder=decode_device, animate_base_rig=not segment.has_ground_truth)
    dataset.register([output.as_uri()], layer_name=POSE_SOURCE, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    print(f"Registered {len(run.estimate)} poses / {run.framesets} framesets, {len(run.dropped)} explained drops; "
          f"{run.wall_s:.1f} s, {run.framesets / run.wall_s:.1f} framesets/s; {output}")
    print(dataset.segment_url(config.segment))
