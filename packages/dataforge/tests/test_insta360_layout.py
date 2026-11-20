"""The one Insta360 layout: an overview and a rig-rooted follow view over the SLAM and colsfm layers, and both lenses."""

from __future__ import annotations

import rerun.blueprint as rrb

from dataforge import blueprints, schema, writing
from dataforge.datasets.insta360 import COLSFM_SOURCE, FOLLOW_PATH_WINDOW_S, RIG, VIO_SOURCE, segment_layout


def test_the_layout_has_an_overview_and_a_follow_view_that_rides_the_rig() -> None:
    overview, follow, lens0, lens1 = writing.blueprint_views(segment_layout(panorama=False))[:4]
    assert isinstance(overview, rrb.Spatial3DView) and overview.origin == "/"
    assert isinstance(follow, rrb.Spatial3DView) and follow.origin == schema.rig_path(RIG), "the follow view is rooted at the rig, not tracked"
    # One rule per item: a newline-joined string reaches the viewer as one entity path and both views drew nothing.
    assert list(overview.contents)[0] == "$origin/**" and f"- {schema.pinhole_path(RIG, 0)}/image/**" in list(overview.contents)  # type: ignore[arg-type]
    assert list(follow.contents)[0] == "/**" and f"- {schema.pinhole_path(RIG, 0)}/image/**" in list(follow.contents)  # type: ignore[arg-type]
    for cam, pane in enumerate((lens0, lens1)):
        assert isinstance(pane, rrb.Spatial2DView) and pane.origin == schema.pinhole_path(RIG, cam)
        assert f"{schema.pinhole_path(RIG, cam)}/keypoints" in list(pane.contents)  # type: ignore[arg-type]
        assert f"{schema.pinhole_path(RIG, cam)}/image" not in list(pane.contents)  # type: ignore[arg-type]


def test_paths_and_the_colsfm_cloud_accumulate_through_cursor_windows() -> None:
    overview, follow = writing.blueprint_views(segment_layout(panorama=False))[:2]
    vio_path: str = schema.trajectory_path(VIO_SOURCE)
    points: str = f"{schema.run_path(COLSFM_SOURCE)}/points"
    overview_overrides = overview.visualizer_overrides or {}
    follow_overrides = follow.visualizer_overrides or {}
    assert overview_overrides[vio_path] == blueprints.cursor_window(None)
    assert overview_overrides[points] == blueprints.cursor_window(None), "the cloud grows from the start of the walk"
    assert follow_overrides[points] == blueprints.cursor_window(None)
    follow_path = follow_overrides[vio_path]
    assert isinstance(follow_path, list) and blueprints.cursor_window(FOLLOW_PATH_WINDOW_S) in follow_path, "the follow view draws the recent path only"
    assert overview_overrides[schema.trail_path(VIO_SOURCE)] == rrb.EntityBehavior(visible=False)


def test_the_panorama_pane_comes_only_with_the_panorama_layer() -> None:
    names = [view.name for view in writing.blueprint_views(segment_layout(panorama=True))]
    assert "panorama" in names and "panorama" not in [view.name for view in writing.blueprint_views(segment_layout(panorama=False))]
