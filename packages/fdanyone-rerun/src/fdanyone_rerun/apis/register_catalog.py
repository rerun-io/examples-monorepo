"""``fdanyone-register-catalog``: put a finished 4DAnyone run on a Rerun catalog in the shared rig layout (no GPU).

The run is generated once (``fdanyone-run --rr-config.save run.rrd``); this tool moves its recording into the
``exoego:v2`` rig layout the MAMMA datasets use (``fdanyone_rerun.catalog``), writes it as one segment ``.rrd`` with its
run record (``simplecv.run_record``) and registers that file as the segment's ``base`` layer.
"""

from dataclasses import dataclass
from pathlib import Path

import rerun.catalog as rrcat
import rerun.chunk as rrc
from simplecv.catalog_layers import atomic_recording
from simplecv.run_record import RunStart, start_run, write_run_record

from fdanyone_rerun.catalog import (
    DATASET,
    RunCamera,
    SegmentProperties,
    VideoClock,
    ViewPose,
    read_run_cameras,
    read_video_clock,
    segment_id,
    segment_stream,
    view_pose,
)
from fdanyone_rerun.outputs import read_camera_rig, read_run_metadata


@dataclass
class Config:
    """Convert one run recording into a catalog segment and register it."""

    rrd: Path
    """The run recording (``fdanyone-run ... --rr-config.save``)."""
    output_dir: Path = Path("/mnt/nas/ssd-8tb/open4dgs/fdanyone/catalog")
    """Where the segment ``.rrd`` goes (the catalog server must read this path)."""
    run_dir: Path | None = None
    """The run directory; when given, the seed and the source video name come from its ``metadata.json``."""
    seed: int | None = None
    """Noise seed of the run; required without ``run_dir``."""
    clip: str = "pexels-2785536"
    """Short name of the source clip, the first part of the segment id."""
    source_video: str = "2785536-uhd_2160_3840_25fps.mp4"
    """Source video file name (overridden by ``run_dir``'s metadata)."""
    catalog_url: str | None = "rerun+http://pablo-dl-server:51235"
    """Catalog to register on; None writes the segment file only."""
    dataset: str = DATASET
    """Catalog dataset (created when missing; a segment of the same id is replaced)."""


def main(config: Config) -> Path:
    """Write the segment and register it; returns the segment file."""
    run: RunStart = start_run(config, source=Path(__file__))
    seed: int | None = config.seed
    source_video: str = config.source_video
    if config.run_dir is not None:
        metadata = read_run_metadata(config.run_dir)
        if seed is not None and seed != metadata.generation.seed:
            raise ValueError(f"--seed {seed} disagrees with {config.run_dir}/metadata.json seed {metadata.generation.seed}")
        seed, source_video = metadata.generation.seed, metadata.input.filename
    store: rrc.LazyStore = rrc.RrdReader(config.rrd).store()
    cameras: list[RunCamera] = read_run_cameras(store)
    poses: list[ViewPose] = [view_pose(camera.world_T_cam) for camera in cameras]
    if config.run_dir is not None:
        # The rig as upstream wrote it must be the rig the recording carries (same views, same yaw / pitch).
        rig = read_camera_rig(config.run_dir)
        for camera, pose, record in zip(cameras, poses, rig.cameras, strict=True):
            if abs((pose.yaw_deg - record.yaw + 180.0) % 360.0 - 180.0) > 0.01 or abs(pose.pitch_deg - record.pitch) > 0.01:
                raise ValueError(f"view {camera.view}: recording yaw/pitch {pose} != cameras.json {record.yaw}/{record.pitch}")
    if seed is None:
        raise ValueError("pass --seed or --run-dir: the run recording does not carry its seed")

    clock: VideoClock = read_video_clock(store)
    segment: str = segment_id(config.clip, len(cameras), seed)
    properties = SegmentProperties(
        num_cameras=len(cameras),
        num_frames=clock.num_frames,
        fps=clock.fps,
        width=cameras[0].width,
        height=cameras[0].height,
        seed=seed,
        clip=config.clip,
        source_video=source_video,
        yaws_deg=tuple(pose.yaw_deg for pose in poses),
        pitches_deg=tuple(pose.pitch_deg for pose in poses),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path: Path = config.output_dir / f"{segment}.rrd"
    # The segment stream has Python callbacks: run it with the native collect, then send the finished chunks. A flow runner
    # checks the base's run record to tell this run's layer from an older one, so the record rides in the same atomic file.
    segment_chunks: rrc.ChunkStore = segment_stream(store, cameras, clock, properties).collect(optimize=rrc.OptimizationProfile.OBJECT_STORE)
    with atomic_recording(path, application_id=config.dataset, recording_id=segment, send_properties=False) as recording:
        recording.send_chunks(segment_chunks)
        write_run_record(recording, run.finish("base", segment))
    path.chmod(0o644)
    print(f"{segment}: {len(cameras)} views, {clock.num_frames} frames at {clock.fps} fps ({clock.codec}) -> {path}")
    if config.catalog_url is not None:
        dataset: rrcat.DatasetEntry = rrcat.CatalogClient(config.catalog_url).create_dataset(config.dataset, exist_ok=True)
        dataset.register([path.resolve().as_uri()], layer_name="base", on_duplicate=rrcat.OnDuplicateSegmentLayer.REPLACE).wait()
        print(f"registered {segment} on {config.catalog_url} / {config.dataset}")
    return path
