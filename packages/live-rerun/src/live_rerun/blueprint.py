"""Rerun blueprint: a 3D view of the camera rig above a row of video panels."""

from __future__ import annotations

import rerun.blueprint as rrb


def create_blueprint(panels: dict[str, str], *, world_path: str = "world") -> rrb.Blueprint:
    """Build the default layout: 3D rig frusta on top, video streams side by side below.

    ``panels`` maps each panel's display name to the entity it shows. Use the
    canonical ``cam_<NN>`` ids as keys (e.g. :attr:`RerunVideoLogger.pinhole_paths`)
    so the 2D panel titles match the 3D entity tree; the human role label
    (``left``/``rgb``/``right``) is available as per-sensor metadata.
    """
    video_views: list[rrb.Spatial2DView] = [rrb.Spatial2DView(name=name, origin=origin) for name, origin in panels.items()]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="rig", origin=world_path),
            rrb.Horizontal(*video_views),
            row_shares=[2, 1],
        ),
        collapse_panels=True,
    )
