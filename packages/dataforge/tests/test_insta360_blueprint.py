"""The Insta360 layout: SLAM, colsfm and Comparison tabs with one skeleton, one colour per source and a level follow view."""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from conftest import content_rules  # pyrefly: ignore[missing-import]
from jaxtyping import Float64
from numpy import ndarray
from simplecv.rerun_walk_path import ENDPOINTS, TRAIL, TRAJECTORY

from dataforge import blueprints, schema, writing
from dataforge.datasets.insta360_blueprint import (
    COMPARISON_SLAM_COLOR,
    FACE_CAMS,
    OVERVIEW_SPIN_RAD_S,
    PANO_ENTITY,
    RIG,
    TRAIL_S,
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


def stroke_color(view: rrb.View, entity: str) -> tuple[int, ...] | None:
    """The colour a view repaints a line entity in; None when it keeps the entity's own (the walk gradient)."""
    strips: rr.LineStrips3D = next(item for item in override(view, entity) if isinstance(item, rr.LineStrips3D))
    return None if strips.colors is None else tuple(strips.colors.as_arrow_array().to_pylist())


def packed(color: tuple[int, ...]) -> tuple[int, ...]:
    """A colour as the viewer packs it: one RGBA integer, opaque unless it names its alpha."""
    return ((color[0] << 24) | (color[1] << 16) | (color[2] << 8) | (color[3] if len(color) == 4 else 255),)


def test_the_tabs_are_slam_colsfm_and_comparison_and_the_panorama_only_with_its_layer() -> None:
    assert list(tabs(segment_layout(panorama=False))) == ["SLAM", "colsfm", "Comparison"]
    with_panorama: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=True))
    assert list(with_panorama) == ["SLAM", "colsfm", "Comparison", "Panorama"]
    assert [str(view.origin) for view in with_panorama["Panorama"]] == [PANO_ENTITY]


FOLLOWED: dict[str, str] = {"SLAM": schema.SLAM_RS_VIO_SOURCE, "colsfm": schema.SLAM_RS_SOURCE, "Comparison": schema.SLAM_RS_SOURCE}
"""The run each tab's follow view centres on: the SLAM tab shows the causal VIO (its window, landmarks and lens frusta
live in that run's frame), colsfm and Comparison the final path colsfm started from (clip 007: 16 m from the VIO at 65 s)."""


def test_every_tab_starts_with_a_world_overview_and_a_follow_view_on_the_path_it_shows() -> None:
    """The same two 3D views lead every tab: the overview in the world, the follow view in the level follow frame of
    the run the tab shows (a view rooted at a rig rolls and pitches with it)."""
    assert schema.follow_path(schema.SLAM_RS_SOURCE) == "/world/runs/slam_rs/follow"
    for name, views in tabs(segment_layout(panorama=False)).items():
        overview, follow = views[:2]
        assert isinstance(overview, rrb.Spatial3DView) and overview.origin == "/world", name
        assert isinstance(follow, rrb.Spatial3DView) and follow.origin == schema.follow_path(FOLLOWED[name]), name
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
    assert [str(view.origin) for view in faces] == [schema.pinhole_path(RIG, cam) for cam in FACE_CAMS]
    assert [str(view.name) for view in faces] == ["Face front 0", "Face front 1", "Face left", "Face right", "Face up", "Face down"]
    for cam, pane in zip(FACE_CAMS, faces, strict=True):
        assert content_rules(pane) == [schema.video_path(RIG, cam), schema.people_mask_path(RIG, cam)]


def test_paths_are_trails_and_the_colsfm_cloud_accumulates_through_cursor_windows() -> None:
    """A path's segments (``schema.trail_path``) show over the last ``TRAIL_S`` of the cursor in every view; its whole
    path (``schema.trajectory_path``, logged once at the last frame) takes no window, so it shows from the end on only."""
    layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False))
    sources: dict[str, list[str]] = {
        "SLAM": [schema.SLAM_RS_VIO_SOURCE, schema.SLAM_RS_SOURCE],
        "colsfm": [schema.COLSFM_SOURCE],
        "Comparison": [schema.SLAM_RS_SOURCE, schema.COLSFM_SOURCE],
    }
    for name, runs in sources.items():
        for view in layout[name][:2]:
            for run in runs:
                assert blueprints.cursor_window(TRAIL_S) in override(view, schema.trail_path(run)), (name, run)
                assert not any(isinstance(item, rrb.VisibleTimeRanges) for item in override(view, schema.trajectory_path(run))), (name, run)
    for view in layout["colsfm"][:2]:
        assert blueprints.cursor_window(None) in override(view, f"{COLSFM}/points_growing"), "the cloud grows from the start of the walk"
        assert f"- {COLSFM}/points/**" in content_rules(view), "the static copy would hide the growth"
    for view in layout["Comparison"][:2]:
        assert f"- {COLSFM}/points_growing/**" in content_rules(view), "the comparison shows both whole clouds"


def test_no_view_draws_a_grid_and_the_overviews_turn_slowly_about_the_walk() -> None:
    walk: Float64[ndarray, "n 3"] = np.column_stack([np.linspace(0.0, 10.0, 50), np.zeros(50), np.full(50, 1.5)])
    for name, views in tabs(segment_layout(panorama=False, walk=walk)).items():
        overview, follow = views[:2]
        for view in (overview, follow):
            assert view.properties["LineGrid3D"] is not None, name
            assert not any("floor" in str(entity) for entity in (view.visualizer_overrides or {})), name
        eye: rrb.EyeControls3D = overview.properties["EyeControls3D"]
        assert eye.spin_speed is not None and eye.spin_speed.as_arrow_array().to_pylist() == [OVERVIEW_SPIN_RAD_S], name
        assert follow.properties["EyeControls3D"].spin_speed.as_arrow_array().to_pylist() == [0.0], name


def test_the_marginalized_frames_are_hidden_in_every_slam_view() -> None:
    """slam-rs logs the frames a marginalization removed (``marginalized/NN``); they stay in the views, hidden."""
    for view in tabs(segment_layout(panorama=False))["SLAM"][:2]:
        assert rrb.EntityBehavior(visible=False) in override(view, f"{VIO}/marginalized")


def test_the_run_frusta_are_restyled_where_their_layers_put_the_pinholes() -> None:
    """slam-rs and colsfm draw their rigs camera, then ``pinhole``, then the picture (the dataset's own layout): the
    layout colours and sizes the ``pinhole`` entities, both lenses of the live rig and of every window frame, and the
    six faces. The Comparison tab leaves the pictures out: there the frusta are the two sources' colours."""
    layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False))
    slam_follow: rrb.View = layout["SLAM"][1]
    for entity in (f"{VIO}/rig/cam_00/pinhole", f"{VIO}/rig/cam_01/pinhole", f"{VIO}/window/00/cam_00/pinhole", f"{VIO}/window/08/cam_01/pinhole"):
        assert any(isinstance(item, rr.Pinhole) for item in override(slam_follow, entity)), entity
    window: rr.Pinhole = next(item for item in override(slam_follow, f"{VIO}/window/03/cam_00/pinhole") if isinstance(item, rr.Pinhole))
    assert window.color is None, "a window frame keeps the role colour slam-rs logs"
    rules: list[str] = content_rules(slam_follow)
    assert f"- {VIO}/window/03/cam_01/pinhole/video" in rules, "a window keyframe's pictures would decode two frames per slot and step"
    assert not any(rule.startswith(f"- {VIO}/rig") for rule in rules), "the SLAM follow view shows the lens pictures"
    for view in layout["colsfm"][:2]:
        assert any(isinstance(item, rr.Pinhole) for item in override(view, f"{COLSFM}/rig/cube6_up/pinhole"))
        assert not any("pinhole/video" in rule for rule in content_rules(view)), "the colsfm views show the face pictures"
    for view in layout["Comparison"][:2]:
        for picture in (f"{COLSFM}/rig/cube6_down/pinhole/video",):
            assert f"- {picture}/**" in content_rules(view), picture


def test_paths_keep_their_gradient_and_comparison_tells_the_runs_apart() -> None:
    """The raw slam-rs run and colsfm keep the walk gradient their layers log; in Comparison slam-rs is flat grey beside
    colsfm's gradient, colsfm's frusta keep its colour and the two clouds take their sources' tints."""
    layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False))
    for view in layout["SLAM"][:2]:
        assert stroke_color(view, schema.trail_path(schema.SLAM_RS_VIO_SOURCE)) is None, "the raw run carries its walk gradient"
    assert stroke_color(layout["colsfm"][1], schema.trail_path(schema.COLSFM_SOURCE)) is None, "colsfm carries its walk gradient"
    for view in layout["Comparison"][:2]:
        assert stroke_color(view, schema.trail_path(schema.SLAM_RS_SOURCE)) == packed(COMPARISON_SLAM_COLOR), "one gradient per view"
        assert stroke_color(view, schema.trail_path(schema.COLSFM_SOURCE)) is None
        assert next(item for item in override(view, f"{COLSFM}/rig/cube6_up/pinhole") if isinstance(item, rr.Pinhole)).color is not None
        tints: dict[str, tuple[int, ...]] = {}
        for entity in (f"{SLAM}/map/points", f"{COLSFM}/points"):
            points: rr.Points3D = next(item for item in override(view, entity) if isinstance(item, rr.Points3D))
            assert points.colors is not None, "a cloud in its own image colours cannot be told from the other"
            tints[entity] = tuple(points.colors.as_arrow_array().to_pylist())
        assert tints[f"{SLAM}/map/points"] != tints[f"{COLSFM}/points"]


def test_colsfm_and_comparison_show_the_final_path_and_no_causal_vio() -> None:
    """colsfm started from the final slam-rs path; the causal VIO (and its rig, window and pictures) would stand metres
    off early in a long walk. The rig mark of each overview sits on the followed run."""
    for name in ("colsfm", "Comparison"):
        for view in tabs(segment_layout(panorama=False))[name][:2]:
            rules: list[str] = content_rules(view)
            assert f"- {VIO}/**" in rules or ("/world/**" not in rules and not any(rule.startswith(VIO) for rule in rules)), name
            assert f"- {schema.rig_path(RIG)}" in rules or schema.rig_path(RIG) not in rules, (name, "the base rig rides the VIO in a replay")
    for name, views in tabs(segment_layout(panorama=False)).items():
        mark = override(views[0], schema.follow_path(FOLLOWED[name]))
        assert any(isinstance(item, rr.Points3D) for item in mark), name


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
    assert schema.follow_path(schema.SLAM_RS_VIO_SOURCE) == "/world/runs/slam_rs_vio/follow"
    # colsfm.rerun_layer.write_run_layer (colsfm does not depend on dataforge).
    assert f"{COLSFM}/points_growing" == "/world/runs/colsfm/points_growing"


def test_with_the_viewing_faces_the_colsfm_views_play_them_on_the_viewing_rig() -> None:
    """With ``cube_view``: the face panes play ``video_view`` (every frame) and the colsfm and Comparison views draw
    ``rig_view`` (a pose at every viewing frame) and drop ``rig``; without it they draw ``rig`` and drop ``rig_view``."""
    for view_faces, drawn, dropped, video in ((True, "rig_view", "rig", schema.video_view_path), (False, "rig", "rig_view", schema.video_path)):
        layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False, view_faces=view_faces))
        faces: list[rrb.View] = [view for view in layout["colsfm"] if isinstance(view, rrb.Spatial2DView)]
        assert [content_rules(pane)[0] for pane in faces] == [video(RIG, cam) for cam in FACE_CAMS]
        for name in ("colsfm", "Comparison"):
            for view in layout[name][:2]:
                assert f"- {COLSFM}/{dropped}/**" in content_rules(view), (name, view_faces)
                assert any(isinstance(item, rr.Pinhole) for item in override(view, f"{COLSFM}/{drawn}/cube6_front0/pinhole")), (name, view_faces)


def test_a_colsfm_layer_without_the_viewing_rig_keeps_its_rig_beside_the_viewing_faces() -> None:
    """``rig_view`` is the colsfm layer's own (``colsfm run`` with viewing faces writes it; ``align_to_prior`` and
    ``--no-view-rig`` do not): without it the views keep ``rig``, and the face panes still play the viewing faces."""
    layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False, view_faces=True, view_rig=False))
    faces: list[rrb.View] = [view for view in layout["colsfm"] if isinstance(view, rrb.Spatial2DView)]
    assert [content_rules(pane)[0] for pane in faces] == [schema.video_view_path(RIG, cam) for cam in FACE_CAMS]
    for name in ("colsfm", "Comparison"):
        for view in layout[name][:2]:
            assert f"- {COLSFM}/rig_view/**" in content_rules(view), name
            assert f"- {COLSFM}/rig/**" not in content_rules(view), name
            assert any(isinstance(item, rr.Pinhole) for item in override(view, f"{COLSFM}/rig/cube6_front0/pinhole")), name

WRITTEN: dict[str, set[str]] = {
    schema.SLAM_RS_SOURCE: {"rig", TRAIL, TRAJECTORY, ENDPOINTS, "follow", "map", "keyframes", "loops", "gba", "pgo", "pose_graph"},
    schema.SLAM_RS_VIO_SOURCE: {"rig", TRAIL, TRAJECTORY, ENDPOINTS, "follow", "landmarks", "window", "marginalized"},
    schema.COLSFM_SOURCE: {"rig", "rig_view", "points", "points_growing", TRAIL, TRAJECTORY, ENDPOINTS, "position_difference"},
}
"""The first entity under each run that its writers produce (slam-rs ``vio_log``, ``catalog_layer``, ``map_loops``,
``gba_log``; ``colsfm.rerun_layer``); the path leaves are :mod:`simplecv.rerun_walk_path`'s."""


def test_every_run_override_targets_an_entity_a_writer_produces() -> None:
    """An override on an entity no writer produces is inert, so a renamed or removed entity (a slam-rs ``path``, a
    ``floor_grid``) would leave a view silently unstyled: each overridden run entity is one the writers produce."""
    for name, views in tabs(segment_layout(panorama=True)).items():
        for view in views:
            for entity in view.visualizer_overrides or {}:
                parts: list[str] = str(entity).removeprefix("/world/runs/").split("/")
                if str(entity).startswith("/world/runs/"):
                    assert parts[1] in WRITTEN[parts[0]], (name, view.name, str(entity))


def test_every_run_trail_a_tab_shows_is_drawn_through_a_cursor_window() -> None:
    """A trail is a segment a pose: without a cursor window a view draws only the latest one, and the whole path comes
    only at the last frame, so a run would show no path through the walk."""
    shown: dict[str, tuple[str, ...]] = {
        "SLAM": (schema.SLAM_RS_VIO_SOURCE, schema.SLAM_RS_SOURCE),
        "colsfm": (schema.COLSFM_SOURCE,),
        "Comparison": (schema.SLAM_RS_SOURCE, schema.COLSFM_SOURCE),
    }
    layout: dict[str, list[rrb.View]] = tabs(segment_layout(panorama=False))
    for name, sources in shown.items():
        for view in layout[name][:2]:
            for source in sources:
                assert any(isinstance(item, rrb.VisibleTimeRanges) for item in override(view, schema.trail_path(source))), (name, view.name, source)
