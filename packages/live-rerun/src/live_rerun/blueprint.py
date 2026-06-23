"""Rerun blueprint: a 3D view of the camera rig above a row of video panels."""

from __future__ import annotations

import rerun.blueprint as rrb


def create_blueprint(video_paths: dict[str, str], *, world_path: str = "world") -> rrb.Blueprint:
    """Build the default layout: 3D rig frusta on top, video streams side by side below.

    ``video_paths`` maps each camera label to its ``.../pinhole/video`` entity path
    (as produced by :class:`live_rerun.rerun_video_logger.RerunVideoLogger`).
    """
    video_views: list[rrb.Spatial2DView] = [rrb.Spatial2DView(name=label, origin=path) for label, path in video_paths.items()]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="rig", origin=world_path),
            rrb.Horizontal(*video_views),
            row_shares=[2, 1],
        ),
        collapse_panels=True,
    )
