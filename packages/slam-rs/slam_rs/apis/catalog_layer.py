"""Run one catalog segment and replace its single slam_rs results layer."""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from dataforge.writing import atomic_recording
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from simplecv.catalog_video_reader import DecodeDevice, DecodeRequest, select_decode_device
from simplecv.run_record import RunStart, start_run, write_run_record

from slam_rs import BackendRequest, _core
from slam_rs.catalog_feed import (
    ROBOCAP_RIG,
    CatalogSegment,
    RigProfile,
    SegmentFeed,
    dataset_rig,
    open_segment,
    resolve_catalog_segments,
)
from slam_rs.catalog_layer import POSE_SOURCE, LayerDetail, LayerRun, track_layer
from slam_rs.config import SlamConfig, load_slam_config
from slam_rs.tracking import SegmentRun


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
    detail: LayerDetail = "full"
    """``full``: the layer also holds what the run saw frameset by frameset (keypoints, window, landmarks, map, counters),
    from the same single run; ``poses``: both runs' paths alone."""
    mask_entity: str | None = None
    """Segmentation masks, read from a layer of the segment: the name of each camera's mask entity under its ``pinhole``
    (e.g. ``people_mask`` reads ``<camera>/pinhole/people_mask``). The frontend neither detects nor keeps tracks where a
    camera's latest mask at or before its frame is nonzero (``slam_rs.frame_masks``). None tracks every pixel."""


def main(config: Config) -> None:
    """Track the complete segment before publishing any replacement."""
    if Path(config.segment).name != config.segment:
        raise ValueError("segment ID must not contain path separators")
    settings: SlamConfig = load_slam_config()
    dataset_name: str = config.segment.split("__", 1)[0]
    is_robocap: bool = dataset_name == "robocap"
    rig_profile: RigProfile = ROBOCAP_RIG if is_robocap else dataset_rig(settings.dataset(dataset_name))
    catalog_url: str = config.catalog or settings.catalog_url
    run: RunStart = start_run(dataclasses.replace(config, catalog=catalog_url))
    # One manifest query resolves the dataset handle that open_segment and the final register share.
    segment: CatalogSegment = resolve_catalog_segments((CatalogSegment(catalog_url, dataset_name, config.segment),))[0]
    dataset: DatasetEntry | None = segment.dataset
    if dataset is None:
        raise ValueError(f"{config.segment}: catalog resolution returned no dataset handle")
    output: Path = config.output_dir.resolve() / f"{config.segment}.rrd"
    decode_device: DecodeDevice = select_decode_device(config.decode_device)
    print(f"Loading {config.segment}: video index, then chunked reads, decode={decode_device}", flush=True)
    started: float = perf_counter()
    feed: SegmentFeed
    with open_segment(segment, profile=rig_profile, decode_device=decode_device, include_ground_truth=False, mask_entity=config.mask_entity) as feed:
        load_s: float = perf_counter() - started
        config_text: str = settings.robocap_config_text(config.profile) if is_robocap else settings.vio_config_text(dataset_name, config.profile)
        calibration: _core.Calibration = _core.Calibration.from_catalog(feed.cameras, feed.imu)
        flow: _core.VioConfig = _core.VioConfig.from_json(config_text)
        # The shared selector decides; only a failed device probe lets auto fall back to the CPU.
        vio: _core.Vio = _core.Vio(calibration, flow, backend=config.backend)
        backend: str = str(_core.gpu_backend) if vio.gpu else "cpu"
        print(f"{config.segment}: profile={config.profile}, backend={backend}, decode={decode_device}, load={load_s:.1f}s, "
              f"{len(feed.frame_t_ns)} framesets, cameras={feed.camera_positions}", flush=True)
        # Every row goes into the temporary file; a refused run leaves the registered layer as it was.
        with atomic_recording(output, recording_id=config.segment, send_properties=False) as rec:
            layer: LayerRun = track_layer(rec, feed, vio, config_text=config_text, calibration=calibration, run_gba=True,
                                          detail=config.detail, profile=config.profile,
                                          animate_base_rig=not segment.has_ground_truth, strict=True)
            write_run_record(rec, run.finish(POSE_SOURCE, config.segment))
        run: SegmentRun = layer.run
        if run.gba_failure is not None:
            print(f"global BA: failed: {run.gba_failure}; the final run is re-anchored on the map after loops and PGO", flush=True)
        for drop in run.dropped:
            print(f"Dropped frameset {drop.t_ns}: {drop.reason}", flush=True)
    dataset.register([output.as_uri()], layer_name=POSE_SOURCE, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
    print(f"Registered {len(run.estimate)} poses / {run.framesets} framesets, {len(run.dropped)} explained drops; "
          f"{run.wall_s:.1f} s, {run.framesets / run.wall_s:.1f} framesets/s; {output}")
    print(dataset.segment_url(config.segment))
