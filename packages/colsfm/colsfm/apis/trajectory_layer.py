"""Write a cuSFM `frames_meta.json` trajectory as a `/world/runs/<source>` layer of a catalog segment, and register it.

This is how a starting trajectory made outside the catalog reaches it — for KITTI 06, the cuVSLAM SLAM poses the cuSFM
binary computes, which the fork's 0.928 m run started from. `colsfm catalog --source.trajectory-entity
/world/runs/<source>/rig` then reads it back like any other trajectory layer.

A slam-rs trajectory CSV (`--slam-csv`, the `slam_rs.csv` of `slam-rs-replay`) is the third source: world-from-IMU
poses on the absolute device clock, moved onto the base recording's `video_time` by the base's `capture:start_time_ns`
(`--base-rrd`; slam-rs's export offset). The tool reports how many poses land on a video sample of the base's first
camera, which proves the clock (a camera time offset the converter did not apply would show up as zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient
from scipy.spatial.transform import Rotation

from colsfm.catalog import DEFAULT_CATALOG_URL, DEFAULT_TIMELINE
from colsfm.frames_meta import read_frames_meta
from colsfm.rerun_layer import log_rig_trajectory, open_layer, read_rrd_trajectory, rig_poses


@dataclass
class Config:
    """Which trajectory, which segment, where the layer goes."""

    frames_meta: Path | None = None
    """A cuSFM `frames_meta.json` whose `camera_to_world` poses are the trajectory."""
    rrd: Path | None = None
    """Or an `.rrd` holding the trajectory as a temporal `Transform3D` of `entity` on `timeline`."""
    entity: str = "/world/runs/slam_rs/rig"
    """The entity to read from `rrd`."""
    slam_csv: Path | None = None
    """Or a slam-rs trajectory CSV (`t_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z`, absolute device clock); needs `base_rrd`."""
    base_rrd: Path | None = None
    """The segment's base recording, for `slam_csv`: its `capture:start_time_ns` maps the CSV clock onto `timeline`,
    and its segment id is the layer's recording id when `segment` is empty."""
    segment: str = ""
    """Catalog segment (recording id) the layer belongs to."""
    output: Path = Path("trajectory_layer.rrd")
    """Where the `.rrd` goes. It must be readable by the catalog server at the same path when registering."""
    source: str = "cuvslam"
    """Run source: the layer lands under `/world/runs/<source>`."""
    timeline: str = DEFAULT_TIMELINE
    """The segment's video timeline."""
    dataset: str | None = None
    """Register the layer on this catalog dataset (layer name = `source`); `None` only writes the file."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """Catalog server."""


def read_slam_csv(path: Path, clock_offset_ns: int) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 4 4"]]:
    """Read a slam-rs trajectory CSV as `world_T_rig` on the base recording's video timeline.

    Args:
        path: `t_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z` rows (header lines start with `#`), quaternion w first.
        clock_offset_ns: Subtracted from the CSV stamps (the capture start on the device clock).

    Returns:
        Timestamps (ns on the video timeline) and poses, in file order.
    """
    stamps: Int64[ndarray, " n"] = np.loadtxt(path, delimiter=",", comments="#", ndmin=1, usecols=0, dtype=np.int64)
    rows: Float64[ndarray, "n 7"] = np.loadtxt(path, delimiter=",", comments="#", ndmin=2, usecols=range(1, 8), dtype=np.float64)
    world_T_rig: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (len(stamps), 1, 1))
    world_T_rig[:, :3, :3] = Rotation.from_quat(rows[:, [4, 5, 6, 3]]).as_matrix()  # scipy takes x y z w
    world_T_rig[:, :3, 3] = rows[:, :3]
    return stamps - clock_offset_ns, world_T_rig


def _base_clock(base_rrd: Path, timeline: str) -> tuple[str, int, Int64[ndarray, " m"]]:
    """The base recording's segment id, its `capture:start_time_ns`, and its first camera's video stamps."""
    with rr.server.Server(host="127.0.0.1", port=0, datasets={"base": [str(base_rrd)]}) as server:
        dataset = server.client().get_dataset("base")
        segment: str = dataset.segment_ids()[0]
        properties = dataset.filter_contents(["/__properties/**"]).reader(index=None).to_arrow_table()
        start_ns: int = int(np.asarray(properties["property:capture:start_time_ns"].to_pylist()[0]).reshape(-1)[0])
        video = dataset.filter_contents(["/world/rig_00/cam_00/pinhole/video"]).reader(index=timeline).to_arrow_table()
        stamps: Int64[ndarray, " m"] = video[timeline].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
    return segment, start_ns, stamps


def main(config: Config) -> None:
    """Write the layer and, when a dataset is given, register it.

    Args:
        config: Trajectory, segment and destination.
    """
    if sum(source is not None for source in (config.frames_meta, config.rrd, config.slam_csv)) != 1:
        raise ValueError("give exactly one of --frames-meta, --rrd and --slam-csv")
    segment: str = config.segment
    if config.slam_csv is not None:
        if config.base_rrd is None:
            raise ValueError("--slam-csv needs --base-rrd (the clock offset and the segment id)")
        base_segment, start_ns, video_ns = _base_clock(config.base_rrd, config.timeline)
        segment = segment or base_segment
        times_ns, world_T_rig = read_slam_csv(config.slam_csv, start_ns)
        on_video: int = int(np.isin(times_ns, video_ns).sum())
        print(f"[colsfm] {config.slam_csv}: clock offset {start_ns} ns, {on_video} of {len(times_ns)} poses on a video sample of cam_00")
        if on_video == 0:
            raise ValueError(f"{config.slam_csv}: no pose lands on a video stamp of {config.base_rrd}; the clock offset is wrong")
    elif config.frames_meta is not None:
        times_ns, world_T_rig = rig_poses(read_frames_meta(config.frames_meta))
    else:
        assert config.rrd is not None
        times_ns, world_T_rig = read_rrd_trajectory(config.rrd, config.entity, config.timeline)
    if not segment:
        raise ValueError("--segment is required: it is the layer's recording id")
    recording = open_layer(config.output, segment)
    log_rig_trajectory(recording, f"/world/runs/{config.source}", times_ns, world_T_rig, config.timeline, color=(60, 160, 255))
    recording.flush()
    print(f"[colsfm] {config.output}: {len(times_ns)} rig poses under /world/runs/{config.source}")
    if config.dataset is not None:
        dataset = CatalogClient(config.catalog_url).get_dataset(name=config.dataset)
        dataset.register([config.output.resolve().as_uri()], layer_name=config.source).wait()
        print(f"[colsfm] registered as layer {config.source!r} of {segment} on {config.dataset}")
