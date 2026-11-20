"""The one Insta360 layout: an overview and a rig-rooted follow view over the SLAM and colsfm layers, and both lenses."""

from __future__ import annotations

import rerun.blueprint as rrb
from conftest import content_rules  # pyrefly: ignore[missing-import]

from dataforge import blueprints, schema, writing
from dataforge.datasets.insta360 import COLSFM_SOURCE, FOLLOW_PATH_WINDOW_S, OVERVIEW_PATH_WINDOW_S, RIG, VIO_SOURCE, segment_layout


def test_the_layout_has_an_overview_and_a_follow_view_that_rides_the_rig() -> None:
    overview, follow, lens0, lens1 = writing.blueprint_views(segment_layout(panorama=False))[:4]
    assert isinstance(overview, rrb.Spatial3DView) and overview.origin == "/"
    assert isinstance(follow, rrb.Spatial3DView) and follow.origin == schema.rig_path(RIG), "the follow view is rooted at the rig, not tracked"
    # One rule per item: a newline-joined string reaches the viewer as one entity path and both views drew nothing.
    assert content_rules(overview)[0] == "$origin/**" and f"- {schema.pinhole_path(RIG, 0)}/image/**" in content_rules(overview)
    assert content_rules(follow)[0] == "/**" and f"- {schema.pinhole_path(RIG, 0)}/image/**" in content_rules(follow)
    for cam, pane in enumerate((lens0, lens1)):
        assert isinstance(pane, rrb.Spatial2DView) and pane.origin == schema.pinhole_path(RIG, cam)
        assert f"{schema.pinhole_path(RIG, cam)}/keypoints" in content_rules(pane)
        assert f"{schema.pinhole_path(RIG, cam)}/image" not in content_rules(pane)


def test_paths_and_the_colsfm_cloud_accumulate_through_cursor_windows() -> None:
    overview, follow = writing.blueprint_views(segment_layout(panorama=False))[:2]
    vio_path: str = schema.trajectory_path(VIO_SOURCE)
    points: str = f"{schema.run_path(COLSFM_SOURCE)}/points_growing"
    overview_overrides = overview.visualizer_overrides or {}
    follow_overrides = follow.visualizer_overrides or {}
    assert overview_overrides[vio_path] == blueprints.cursor_window(OVERVIEW_PATH_WINDOW_S), "a bounded window keeps a long walk fast"
    assert overview_overrides[points] == blueprints.cursor_window(None), "the cloud grows from the start of the walk"
    assert follow_overrides[points] == blueprints.cursor_window(None)
    follow_path = follow_overrides[vio_path]
    assert isinstance(follow_path, list) and blueprints.cursor_window(FOLLOW_PATH_WINDOW_S) in follow_path, (
        "the follow view draws the recent path only"
    )
    for view in (overview, follow):
        assert f"- {schema.run_path(COLSFM_SOURCE)}/points/**" in content_rules(view), "the static copy would hide the growth"
    assert overview_overrides[schema.trail_path(VIO_SOURCE)] == rrb.EntityBehavior(visible=False)


def test_the_panorama_pane_comes_only_with_the_panorama_layer() -> None:
    names = [view.name for view in writing.blueprint_views(segment_layout(panorama=True))]
    assert "panorama" in names and "panorama" not in [view.name for view in writing.blueprint_views(segment_layout(panorama=False))]


def test_the_run_sources_name_the_entities_their_producers_write() -> None:
    """slam-rs and colsfm write these paths; an override on a renamed entity is inert, so pin the names here."""
    assert schema.trajectory_path(schema.SLAM_RS_VIO_SOURCE) == "/world/runs/slam_rs_vio/trajectory"
    assert schema.run_path(schema.SLAM_RS_SOURCE) == "/world/runs/slam_rs"
    # colsfm.rerun_layer.write_run_layer (colsfm does not depend on dataforge).
    assert f"{schema.run_path(schema.COLSFM_SOURCE)}/points_growing" == "/world/runs/colsfm/points_growing"
