"""The Insta360 layout: SLAM, colsfm and Comparison tabs with one skeleton, one colour per source and a level follow view."""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from conftest import content_rules  # pyrefly: ignore[missing-import]
from jaxtyping import Float64
from numpy import ndarray

from dataforge import blueprints, schema, writing
from dataforge.datasets.insta360_blueprint import (
    COLSFM_COLOR,
    FOLLOW_FLOOR_WINDOW_S,
    FOLLOW_PATH_WINDOW_S,
    OVERVIEW_PATH_WINDOW_S,
    PANO_ENTITY,
    RIG,
    SLAM_COLOR,
    segment_layout,
    walk_eye,
)

SLAM: str = schema.run_path(schema.SLAM_RS_SOURCE)
VIO: str = schema.run_path(schema.SLAM_RS_VIO_SOURCE)
COLSFM: str = schema.run_path(schema.COLSFM_SOURCE)


def tabs(blueprint: rrb.Blueprint) -> dict[str, list[rrb.View]]:
    """Each tab's views, in layout order, by tab name."""
    root = blueprint.root_container
    assert isinstance(root, rrb.Tabs)
    return {str(tab.name): [tab] if isinstance(tab, rrb.View) else writing.blueprint_views(rrb.Blueprint(tab)) for tab in root.contents}


def override(view: rrb.View, entity: str) -> list[object]:
    """One entity's overrides in a view, as a list."""
    found = (view.visualizer_overrides or {})[entity]
    return list(found) if isinstance(found, list) else [found]


def stroke_color(view: rrb.View, entity: str) -> tuple[int, ...]:
    """The colour a view repaints a line entity in."""
    strips: rr.LineStrips3D = next(item for item in override(view, entity) if isinstance(item, rr.LineStrips3D))
    assert strips.colors is not None
    return tuple(strips.colors.as_arrow_array().to_pylist())


def packed(color: tuple[int, int, int]) -> tuple[int, ...]:
    """A colour as the viewer packs it: one RGBA integer, opaque."""
    return ((color[0] << 24) | (color[1] << 16) | (color[2] << 8) | 255,)


def test_the_tabs_are_slam_colsfm_and_comparison_and_the_panorama_only_with_its_layer() -> None:
    assert list(tabs(segment_layout(panorama=False))) == ["SLAM", "colsfm", "Comparison"]
    with_panorama: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=True))
    assert list(with_panorama) == ["SLAM", "colsfm", "Comparison", "Panorama"]
    assert [str(view.origin) for view in with_panorama["Panorama"]] == [PANO_ENTITY]


def test_every_tab_starts_with_a_world_overview_and_a_level_follow_view() -> None:
    """The same two 3D views lead every tab: the overview in the world, the follow view in the yaw-only follow frame
    (a view rooted at the rig rolls and pitches with it), which the layer that moves the rig writes."""
    assert schema.follow_path(RIG) == "/world/rig_00_follow"
    for name, views in tabs(segment_layout(panorama=False)).items():
        overview, follow = views[:2]
        assert isinstance(overview, rrb.Spatial3DView) and overview.origin == "/world", name
        assert isinstance(follow, rrb.Spatial3DView) and follow.origin == schema.follow_path(RIG), name
        for view in (overview, follow):
            rules: list[str] = content_rules(view)
            assert f"- {schema.cam_path(RIG, 0)}/**" in rules or "/world/**" not in rules, "no picture is drawn in 3D"
            assert all("\n" not in rule for rule in content_rules(view)), "one rule per item: the viewer reads a joined string as one path"


def test_the_slam_tab_shows_both_lenses_with_keypoints_and_masks_over_a_short_imu_strip() -> None:
    views: list[rrb.View] = tabs(segment_layout(panorama=False))["SLAM"]
    lenses: list[rrb.View] = [view for view in views if isinstance(view, rrb.Spatial2DView)]
    assert [str(view.origin) for view in lenses] == [schema.pinhole_path(RIG, 0), schema.pinhole_path(RIG, 1)]
    for cam, pane in enumerate(lenses):
        pinhole: str = schema.pinhole_path(RIG, cam)
        assert content_rules(pane) == [schema.video_path(RIG, cam), f"{pinhole}/keypoints", f"{pinhole}/people_mask"]
    plots: list[rrb.View] = [view for view in views if isinstance(view, rrb.TimeSeriesView)]
    assert [content_rules(view) for view in plots] == [[schema.gyro_path(RIG, 0)], [schema.accel_path(RIG, 0)]]
    slam = list(segment_layout(panorama=False).root_container.contents)[0]
    assert isinstance(slam, rrb.Vertical) and slam.row_shares is not None
    shares: Float64[ndarray, " 2"] = np.asarray(slam.row_shares, dtype=np.float64)
    assert shares[0] >= 5 * shares[1], "the IMU is a strip"


def test_the_colsfm_tab_shows_the_six_cube_faces_with_their_masks() -> None:
    faces: list[rrb.View] = [view for view in tabs(segment_layout(panorama=False))["colsfm"] if isinstance(view, rrb.Spatial2DView)]
    assert [str(view.origin) for view in faces] == [schema.pinhole_path(RIG, cam) for cam in range(2, 8)]
    assert [str(view.name) for view in faces] == ["Face front 0", "Face front 1", "Face left", "Face right", "Face up", "Face down"]
    for cam, pane in zip(range(2, 8), faces, strict=True):
        assert content_rules(pane) == [schema.video_path(RIG, cam), f"{schema.pinhole_path(RIG, cam)}/people_mask"]


def test_paths_and_the_colsfm_cloud_accumulate_through_cursor_windows() -> None:
    layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False))
    slam_overview, slam_follow = layout["SLAM"][:2]
    assert blueprints.cursor_window(OVERVIEW_PATH_WINDOW_S) in override(slam_overview, f"{VIO}/trajectory"), "a bounded window keeps a long walk fast"
    assert blueprints.cursor_window(FOLLOW_PATH_WINDOW_S) in override(slam_follow, f"{VIO}/trajectory"), "the follow view draws the recent path only"
    for view in layout["colsfm"][:2]:
        assert blueprints.cursor_window(None) in override(view, f"{COLSFM}/points_growing"), "the cloud grows from the start of the walk"
        assert f"- {COLSFM}/points/**" in content_rules(view), "the static copy would hide the growth"
    for view in layout["Comparison"][:2]:
        assert f"- {COLSFM}/points_growing/**" in content_rules(view), "the comparison shows both whole clouds"


def test_the_follow_views_show_the_floor_around_the_cursor_and_the_overviews_all_of_it() -> None:
    """slam-rs stamps each floor edge with the time the walk passed it. A follow view shows the floor of the seconds
    around the cursor, so on a walk over several storeys it draws the floor under the rig and not the one above; an
    overview shows every floor, whatever the cursor."""
    around = rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(
            schema.TIMELINE,
            start=rrb.TimeRangeBoundary.cursor_relative(seconds=-FOLLOW_FLOOR_WINDOW_S),
            end=rrb.TimeRangeBoundary.cursor_relative(seconds=FOLLOW_FLOOR_WINDOW_S),
        )
    )
    whole = rrb.VisibleTimeRanges(rrb.VisibleTimeRange(schema.TIMELINE, start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite()))
    for name, views in tabs(segment_layout(panorama=False)).items():
        overview, follow = views[:2]
        assert override(overview, schema.FLOOR_GRID_PATH) == [whole], name
        assert override(follow, schema.FLOOR_GRID_PATH) == [around], name


def test_each_source_keeps_one_colour_in_every_tab() -> None:
    """slam-rs is orange and colsfm blue wherever a path of theirs is drawn; in Comparison their clouds take the tint too."""
    layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False))
    for view in layout["SLAM"][:2]:
        assert stroke_color(view, f"{VIO}/trajectory") == packed(SLAM_COLOR)
    assert stroke_color(layout["colsfm"][1], f"{COLSFM}/trajectory") == packed(COLSFM_COLOR)
    for view in layout["Comparison"][:2]:
        assert stroke_color(view, f"{VIO}/path") == packed(SLAM_COLOR)
        assert stroke_color(view, f"{COLSFM}/trajectory") == packed(COLSFM_COLOR)
        tints: dict[str, tuple[int, ...]] = {}
        for entity in (f"{SLAM}/map/points", f"{COLSFM}/points"):
            points: rr.Points3D = next(item for item in override(view, entity) if isinstance(item, rr.Points3D))
            assert points.colors is not None, "a cloud in its own image colours cannot be told from the other"
            tints[entity] = tuple(points.colors.as_arrow_array().to_pylist())
        assert tints[f"{SLAM}/map/points"] != tints[f"{COLSFM}/points"]


def test_the_comparison_plots_the_position_difference_the_colsfm_layer_writes() -> None:
    plot: rrb.View = next(view for view in tabs(segment_layout(panorama=False))["Comparison"] if isinstance(view, rrb.TimeSeriesView))
    assert plot.origin == "/world/runs/colsfm/position_difference", "colsfm.rerun_layer.write_run_layer writes it (colsfm does not depend on dataforge)"


def test_the_overview_frames_the_walk_when_the_caller_knows_it() -> None:
    """Without the walk the viewer frames the view's bounds; with it the eye looks at the walk's middle from above its
    long side, far enough to hold all of it."""
    assert all(view.properties.get("EyeControls3D") is None for view in writing.blueprint_views(segment_layout(panorama=False)) if view.origin == "/world")
    # A walk 40 m along y and 10 m along x, 1.5 m up.
    along: Float64[ndarray, " n"] = np.linspace(0.0, 40.0, 200)
    walk: Float64[ndarray, "n 3"] = np.column_stack([10.0 * np.sin(along / 40.0 * np.pi), along, np.full(200, 1.5)])
    eye: rrb.EyeControls3D = walk_eye(walk, aspect=2.0)
    assert eye.look_target is not None and eye.position is not None and eye.eye_up is not None
    target: Float64[ndarray, " 3"] = np.asarray(eye.look_target.as_arrow_array().to_pylist()[0], dtype=np.float64)
    position: Float64[ndarray, " 3"] = np.asarray(eye.position.as_arrow_array().to_pylist()[0], dtype=np.float64)
    np.testing.assert_allclose(target, [5.0, 20.0, 1.5], atol=0.2)
    back: Float64[ndarray, " 3"] = position - target
    assert abs(back[1]) < 0.15 * np.linalg.norm(back), "the eye looks across the long axis, so the walk runs across the pane"
    assert back[2] > 0.5 * np.linalg.norm(back), "from above"
    assert 20.0 < np.linalg.norm(back) < 80.0, "far enough to hold a 40 m walk, near enough to read it"
    assert eye.eye_up.as_arrow_array().to_pylist()[0] == [0.0, 0.0, 1.0], "gravity is down"
    framed: list[rrb.View] = [view for view in writing.blueprint_views(segment_layout(panorama=False, walk=walk)) if view.origin == "/world"]
    assert len(framed) == 3 and all(view.properties.get("EyeControls3D") is not None for view in framed)


def test_the_run_sources_name_the_entities_their_producers_write() -> None:
    """slam-rs and colsfm write these paths; an override on a renamed entity is inert, so pin the names here."""
    assert schema.trajectory_path(schema.SLAM_RS_VIO_SOURCE) == "/world/runs/slam_rs_vio/trajectory"
    assert schema.run_path(schema.SLAM_RS_SOURCE) == "/world/runs/slam_rs"
    # slam_rs.follow_frame (a slam-rs test pins its two names to these).
    assert schema.follow_path(0) == "/world/rig_00_follow" and schema.FLOOR_GRID_PATH == "/world/floor_grid"
    # colsfm.rerun_layer.write_run_layer (colsfm does not depend on dataforge).
    assert f"{COLSFM}/points_growing" == "/world/runs/colsfm/points_growing"
