"""Write a cuSFM `frames_meta.json` trajectory as a `/world/runs/<source>` layer of a catalog segment, and register it.

This is how a starting trajectory made outside the catalog reaches it — for KITTI 06, the cuVSLAM SLAM poses the cuSFM
binary computes, which the fork's 0.928 m run started from. `colsfm catalog --source.trajectory-entity
/world/runs/<source>/rig` then reads it back like any other trajectory layer.

A slam-rs trajectory CSV (`--slam-csv`, the `slam_rs.csv` of `slam-rs-replay`) is the third source: world-from-IMU
poses on the absolute device clock, moved onto the segment's `video_time` by the offset the CSV states in its header
(`#video_time_offset_ns=`, slam-rs's `SegmentFeed.video_time_offset_ns`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient
from scipy.spatial.transform import Rotation

from colsfm.catalog import DEFAULT_CATALOG_URL, DEFAULT_TIMELINE
from colsfm.frames_meta import read_frames_meta
from colsfm.rerun_layer import log_rig_trajectory, open_layer, read_rrd_trajectory, rig_poses

VIDEO_TIME_OFFSET_PREFIX: str = "#video_time_offset_ns="
"""The slam-rs export header line (`slam_rs.trajectory.VIDEO_TIME_OFFSET_PREFIX`) whose value maps `t_ns` onto `video_time`."""


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
    """Or a slam-rs trajectory CSV (`t_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z`, absolute device clock, its `video_time`
    offset in the header)."""
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


def read_slam_csv(path: Path) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 4 4"]]:
    """Read a slam-rs trajectory CSV as `world_T_rig` on the segment's video timeline.

    Args:
        path: `t_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z` rows, quaternion w first, after `#` header lines, one of which
            is `#video_time_offset_ns=<N>`: `t_ns - N` is the segment's `video_time`.

    Returns:
        Timestamps (ns on the video timeline) and poses, in file order.

    Raises:
        ValueError: If the header states no `video_time` offset (an export older than slam-rs writing it).
    """
    header: list[str] = [line for line in path.read_text().splitlines() if line.startswith(VIDEO_TIME_OFFSET_PREFIX)]
    if not header:
        raise ValueError(f"{path}: no {VIDEO_TIME_OFFSET_PREFIX}<N> header line; export it again with slam-rs-replay")
    clock_offset_ns: int = int(header[0][len(VIDEO_TIME_OFFSET_PREFIX) :])
    stamps: Int64[ndarray, " n"] = np.loadtxt(path, delimiter=",", comments="#", ndmin=1, usecols=0, dtype=np.int64)
    rows: Float64[ndarray, "n 7"] = np.loadtxt(path, delimiter=",", comments="#", ndmin=2, usecols=range(1, 8), dtype=np.float64)
    world_T_rig: Float64[ndarray, "n 4 4"] = np.tile(np.eye(4), (len(stamps), 1, 1))
    world_T_rig[:, :3, :3] = Rotation.from_quat(rows[:, [4, 5, 6, 3]]).as_matrix()  # scipy takes x y z w
    world_T_rig[:, :3, 3] = rows[:, :3]
    return stamps - clock_offset_ns, world_T_rig


def main(config: Config) -> None:
    """Write the layer and, when a dataset is given, register it.

    Args:
        config: Trajectory, segment and destination.
    """
    if sum(source is not None for source in (config.frames_meta, config.rrd, config.slam_csv)) != 1:
        raise ValueError("give exactly one of --frames-meta, --rrd and --slam-csv")
    segment: str = config.segment
    if config.slam_csv is not None:
        times_ns, world_T_rig = read_slam_csv(config.slam_csv)
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
