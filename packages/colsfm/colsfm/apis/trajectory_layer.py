"""Write a cuSFM `frames_meta.json` trajectory as a `/world/runs/<source>` layer of a catalog segment, and register it.

This is how a starting trajectory made outside the catalog reaches it — for KITTI 06, the cuVSLAM SLAM poses the cuSFM
binary computes, which the fork's 0.928 m run started from. `colsfm catalog --source.trajectory-entity
/world/runs/<source>/rig` then reads it back like any other trajectory layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient

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


def main(config: Config) -> None:
    """Write the layer and, when a dataset is given, register it.

    Args:
        config: Trajectory, segment and destination.
    """
    if (config.frames_meta is None) == (config.rrd is None):
        raise ValueError("give exactly one of --frames-meta and --rrd")
    if not config.segment:
        raise ValueError("--segment is required: it is the layer's recording id")
    if config.frames_meta is not None:
        times_ns, world_T_rig = rig_poses(read_frames_meta(config.frames_meta))
    else:
        assert config.rrd is not None
        times_ns, world_T_rig = read_rrd_trajectory(config.rrd, config.entity, config.timeline)
    recording = open_layer(config.output, config.segment)
    log_rig_trajectory(recording, f"/world/runs/{config.source}", times_ns, world_T_rig, config.timeline, color=(60, 160, 255))
    recording.flush()
    print(f"[colsfm] {config.output}: {len(times_ns)} rig poses under /world/runs/{config.source}")
    if config.dataset is not None:
        dataset = CatalogClient(config.catalog_url).get_dataset(name=config.dataset)
        dataset.register([config.output.resolve().as_uri()], layer_name=config.source).wait()
        print(f"[colsfm] registered as layer {config.source!r} of {config.segment} on {config.dataset}")
