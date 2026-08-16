"""Behavior contracts for the per-source-tab ARKitScenes blueprint."""

from typing import cast

import rerun.blueprint as rrb
from rerun.blueprint.api import Container, View

from arkitscenes_download.ingest.blueprint import (
    make_blueprint,
    make_capture_page,
    make_fit_triage_page,
    make_gt_page,
    make_promptda_page,
    make_splat_page,
    make_table_blueprint,
)
from arkitscenes_download.ingest.paths import (
    ARKIT_MESH,
    CONFIDENCE,
    DEPTH,
    DEPTH_PROMPTDA,
    DEPTH_SPLAT_ULTRAWIDE,
    DEPTH_SPLAT_WIDE,
    DEPTH_ULTRAWIDE_RECT,
    ERROR_DEPTH_ULTRAWIDE_RECT,
    ERROR_NORMAL_ULTRAWIDE_RECT,
    ERROR_RGB_ULTRAWIDE_RECT,
    GT,
    GT_BOXES,
    GT_DEPTH,
    GT_PINHOLE_WIDE,
    IMU_ACCEL,
    IMU_GYRO,
    NORMALS_MOGE,
    NORMALS_SPLAT_ULTRAWIDE_RECT,
    NORMALS_SPLAT_WIDE,
    NORMALS_ULTRAWIDE_RECT,
    PINHOLE_MOGE,
    PINHOLE_ULTRAWIDE,
    PINHOLE_ULTRAWIDE_RECT,
    PINHOLE_WIDE,
    PINHOLE_WIDE_LOWRES,
    PROMPTDA,
    PROMPTDA_MESH,
    RGB_SPLAT_ULTRAWIDE_RECT,
    RGB_ULTRAWIDE_RECT,
    RIG,
    SHARPNESS_ULTRAWIDE,
    SHARPNESS_WIDE,
    SPLATS,
    VIDEO_ULTRAWIDE,
    VIDEO_WIDE,
)


def iter_views(node: rrb.Blueprint | Container | View) -> list[View]:
    """Flatten a blueprint container tree into its leaf views."""
    if isinstance(node, rrb.Blueprint):
        return iter_views(node.root_container)
    if isinstance(node, Container):
        views: list[View] = []
        for child in node.contents:
            views.extend(iter_views(child))
        return views
    return [node]


def find_view(blueprint: rrb.Blueprint, name: str) -> View:
    """Return the unique view with a given name."""
    matches: list[View] = [view for view in iter_views(blueprint) if str(view.name) == name]
    assert len(matches) == 1, f"expected exactly one view named {name!r}, got {len(matches)}"
    return matches[0]


def top_pages(blueprint: rrb.Blueprint) -> tuple[rrb.Tabs, tuple[Container, ...]]:
    """Return the named top-level tabs and their container pages."""
    root: Container = blueprint.root_container
    assert isinstance(root, rrb.Tabs)
    pages: tuple[Container, ...] = cast(tuple[Container, ...], root.contents)
    assert all(isinstance(page, Container) for page in pages)
    return root, pages


def children(container: Container) -> tuple[Container | View, ...]:
    """Materialize a container's stub-typed iterable for structural indexing."""
    return tuple(container.contents)


def assert_depth_view(view: View, entity_path: str) -> None:
    """Check one explicit depth entity and its encoded-depth range override."""
    assert cast(list[str], view.contents) == [f"/{entity_path}"]
    assert view.visualizer_overrides is not None and f"/{entity_path}" in view.visualizer_overrides


def test_page_builders_return_the_five_named_pages() -> None:
    pages: tuple[Container, ...] = (
        make_capture_page(),
        make_gt_page(),
        make_promptda_page(),
        make_splat_page(),
        make_fit_triage_page(),
    )
    assert [str(page.name) for page in pages] == ["Capture", "GT", "PromptDA", "Splat", "Fit triage"]
    assert [type(page) for page in pages] == [rrb.Vertical, rrb.Horizontal, rrb.Horizontal, rrb.Vertical, rrb.Horizontal]


def test_top_level_pages_are_feature_gated_and_splat_is_active() -> None:
    expected_variants: tuple[tuple[bool, bool, list[str], int], ...] = (
        (False, False, ["Capture", "GT"], 0),
        (True, False, ["Capture", "GT", "PromptDA"], 0),
        (False, True, ["Capture", "GT", "Splat", "Fit triage"], 2),
        (True, True, ["Capture", "GT", "PromptDA", "Splat", "Fit triage"], 3),
    )
    for include_promptda, include_gauss_surf, expected_names, expected_active_tab in expected_variants:
        root, pages = top_pages(
            make_blueprint(include_promptda=include_promptda, include_gauss_surf=include_gauss_surf)
        )
        assert root.name == "ARKitScenes"
        assert [str(page.name) for page in pages] == expected_names
        assert root.active_tab == expected_active_tab


def test_capture_and_gt_pages_separate_sources_and_keep_imu_on_capture() -> None:
    _, pages = top_pages(make_blueprint())
    capture: Container = pages[0]
    gt: Container = pages[1]

    assert isinstance(capture, rrb.Vertical)
    assert capture.row_shares == [3, 1]
    capture_main: Container = cast(Container, children(capture)[0])
    signals: Container = cast(Container, children(capture)[1])
    assert isinstance(capture_main, rrb.Horizontal)
    assert capture_main.column_shares == [2.0, 1.0, 1.0]
    world: View = cast(View, children(capture_main)[0])
    cameras: Container = cast(Container, children(capture_main)[1])
    sensors: Container = cast(Container, children(capture_main)[2])
    assert cast(list[str], world.contents) == ["$origin/**", f"- /{GT}/**", f"- /{SPLATS}/**", f"- /{PROMPTDA}/**"]
    assert isinstance(cameras, rrb.Vertical)
    wide: View = cast(View, children(cameras)[0])
    ultrawide: View = cast(View, children(cameras)[1])
    assert str(wide.origin) == PINHOLE_WIDE
    assert cast(list[str], wide.contents) == ["$origin/video", f"/{GT_BOXES}/**"]
    assert str(ultrawide.origin) == PINHOLE_ULTRAWIDE
    assert cast(list[str], ultrawide.contents) == ["$origin/video", f"/{GT_BOXES}/**"]
    assert isinstance(sensors, rrb.Vertical)
    depth_arkit: View = cast(View, children(sensors)[0])
    confidence: View = cast(View, children(sensors)[1])
    assert str(depth_arkit.origin) == PINHOLE_WIDE_LOWRES
    assert_depth_view(depth_arkit, DEPTH)
    assert str(confidence.origin) == PINHOLE_WIDE_LOWRES
    assert cast(list[str], confidence.contents) == ["$origin/confidence"]
    assert isinstance(signals, rrb.Horizontal)
    assert [str(cast(View, view).origin) for view in signals.contents] == [IMU_GYRO, IMU_ACCEL]

    assert isinstance(gt, rrb.Horizontal)
    assert gt.column_shares == [2.0, 1.0]
    world_gt: View = cast(View, children(gt)[0])
    gt_cameras: Container = cast(Container, children(gt)[1])
    assert cast(list[str], world_gt.contents) == [
        "$origin/**",
        f"- /{ARKIT_MESH}/**",
        f"- /{GT_BOXES}/**",
        f"- /{DEPTH}/**",
        f"- /{CONFIDENCE}/**",
    ]
    assert isinstance(gt_cameras, rrb.Vertical)
    gt_wide: View = cast(View, children(gt_cameras)[0])
    gt_depth: View = cast(View, children(gt_cameras)[1])
    gt_ultrawide: View = cast(View, children(gt_cameras)[2])
    assert cast(list[str], gt_wide.contents) == ["$origin/video", f"/{GT_BOXES}/**"]
    assert str(gt_depth.origin) == GT_PINHOLE_WIDE
    assert_depth_view(gt_depth, GT_DEPTH)
    assert cast(list[str], gt_ultrawide.contents) == ["$origin/video", f"/{GT_BOXES}/**"]
    assert all(str(view.name) not in {"gyro", "accel"} for view in iter_views(gt))


def test_portrait_uses_wider_world_column_shares() -> None:
    _, pages = top_pages(make_blueprint(portrait=True, include_promptda=True, include_gauss_surf=True))
    capture: Container = pages[0]
    gt: Container = pages[1]
    promptda: Container = pages[2]
    splat: Container = pages[3]
    assert cast(Container, children(capture)[0]).column_shares == [2.6, 1.0, 1.0]
    assert gt.column_shares == [2.6, 1.0]
    assert promptda.column_shares == [2.6, 1.0]
    assert cast(Container, children(splat)[0]).column_shares == [3.4, 2.0, 1.0]


def test_promptda_page_has_its_world_video_and_two_depth_sources() -> None:
    _, pages = top_pages(make_blueprint(include_promptda=True))
    promptda: Container = pages[2]
    assert isinstance(promptda, rrb.Horizontal)
    assert promptda.column_shares == [2.0, 1.0]
    world_promptda: View = cast(View, children(promptda)[0])
    source_column: Container = cast(Container, children(promptda)[1])
    assert cast(list[str], world_promptda.contents) == [
        "$origin/**",
        f"- /{ARKIT_MESH}/**",
        f"- /{GT_DEPTH}/**",
        f"- /{PINHOLE_WIDE_LOWRES}/**",
    ]
    assert isinstance(source_column, rrb.Vertical)
    wide_video: View = cast(View, children(source_column)[0])
    promptda_depth: View = cast(View, children(source_column)[1])
    arkit_depth: View = cast(View, children(source_column)[2])
    assert str(wide_video.origin) == PINHOLE_WIDE
    assert cast(list[str], wide_video.contents) == ["$origin/video"]
    assert_depth_view(promptda_depth, DEPTH_PROMPTDA)
    assert_depth_view(arkit_depth, DEPTH)


def test_splat_page_puts_videos_and_predictions_beside_a_dominant_world_view() -> None:
    _, pages = top_pages(make_blueprint(include_promptda=True, include_gauss_surf=True))
    splat: Container = pages[3]
    assert isinstance(splat, rrb.Vertical)
    assert splat.row_shares == [6, 1]
    main: Container = cast(Container, children(splat)[0])
    sharpness: View = cast(View, children(splat)[1])
    assert isinstance(main, rrb.Horizontal)
    assert main.column_shares == [4.8, 2.2, 1.0]
    world_splat: View = cast(View, children(main)[0])
    videos: Container = cast(Container, children(main)[1])
    predictions: Container = cast(Container, children(main)[2])
    assert cast(list[str], world_splat.contents) == [
        f"/{SPLATS}/**",
        f"/{PROMPTDA_MESH}",
        f"/{RIG}",
        f"/{PINHOLE_WIDE}",
        f"/{VIDEO_WIDE}",
    ]
    mesh_override = cast(list[object], (world_splat.visualizer_overrides or {})[f"/{PROMPTDA_MESH}"])
    assert any(isinstance(o, rrb.EntityBehavior) for o in mesh_override), "ghost mesh must default to hidden in the Splat tab"

    assert isinstance(videos, rrb.Vertical)
    wide_video: View = cast(View, children(videos)[0])
    rgb_tabs: Container = cast(Container, children(videos)[1])
    assert str(wide_video.name) == "wide video"
    assert str(wide_video.origin) == PINHOLE_WIDE
    assert cast(list[str], wide_video.contents) == ["$origin/video"]

    assert isinstance(predictions, rrb.Vertical)
    wide_depth_tabs: Container = cast(Container, children(predictions)[0])
    wide_normal_tabs: Container = cast(Container, children(predictions)[1])
    depth_tabs: Container = cast(Container, children(predictions)[2])
    normal_tabs: Container = cast(Container, children(predictions)[3])
    assert isinstance(wide_depth_tabs, rrb.Tabs) and wide_depth_tabs.active_tab == 0
    wide_splat_depth: View = cast(View, children(wide_depth_tabs)[0])
    wide_promptda_depth: View = cast(View, children(wide_depth_tabs)[1])
    assert [str(cast(View, view).name) for view in wide_depth_tabs.contents] == ["splat depth", "PromptDA depth"]
    assert_depth_view(wide_splat_depth, DEPTH_SPLAT_WIDE)
    assert_depth_view(wide_promptda_depth, DEPTH_PROMPTDA)
    assert isinstance(wide_normal_tabs, rrb.Tabs) and wide_normal_tabs.active_tab == 0
    wide_splat_normals: View = cast(View, children(wide_normal_tabs)[0])
    wide_moge_normals: View = cast(View, children(wide_normal_tabs)[1])
    assert [str(cast(View, view).name) for view in wide_normal_tabs.contents] == ["splat normals", "MoGe normals"]
    assert str(wide_splat_normals.origin) == PINHOLE_WIDE
    assert cast(list[str], wide_splat_normals.contents) == [f"/{NORMALS_SPLAT_WIDE}"]
    assert str(wide_moge_normals.origin) == PINHOLE_MOGE
    assert cast(list[str], wide_moge_normals.contents) == [f"/{NORMALS_MOGE}"]
    assert isinstance(rgb_tabs, rrb.Tabs) and rgb_tabs.active_tab == 0
    assert [str(cast(View, view).name) for view in rgb_tabs.contents] == ["video", "rect RGB", "splat RGB"]
    rgb_video: View = cast(View, children(rgb_tabs)[0])
    rect_rgb: View = cast(View, children(rgb_tabs)[1])
    splat_rgb: View = cast(View, children(rgb_tabs)[2])
    assert str(rgb_video.origin) == PINHOLE_ULTRAWIDE
    assert cast(list[str], rgb_video.contents) == ["$origin/video"]
    assert str(rect_rgb.origin) == PINHOLE_ULTRAWIDE_RECT
    assert str(splat_rgb.origin) == PINHOLE_ULTRAWIDE_RECT
    assert cast(list[str], rect_rgb.contents) == [f"/{RGB_ULTRAWIDE_RECT}"]
    assert cast(list[str], splat_rgb.contents) == [f"/{RGB_SPLAT_ULTRAWIDE_RECT}"]
    assert isinstance(depth_tabs, rrb.Tabs) and depth_tabs.active_tab == 0
    assert [str(cast(View, view).name) for view in depth_tabs.contents] == ["splat depth", "raycast depth"]
    assert_depth_view(cast(View, children(depth_tabs)[0]), DEPTH_SPLAT_ULTRAWIDE)
    assert_depth_view(cast(View, children(depth_tabs)[1]), DEPTH_ULTRAWIDE_RECT)
    assert isinstance(normal_tabs, rrb.Tabs) and normal_tabs.active_tab == 0
    assert [str(cast(View, view).name) for view in normal_tabs.contents] == ["splat normals", "MoGe rect normals"]
    assert cast(list[str], cast(View, children(normal_tabs)[0]).contents) == [f"/{NORMALS_SPLAT_ULTRAWIDE_RECT}"]
    assert cast(list[str], cast(View, children(normal_tabs)[1]).contents) == [f"/{NORMALS_ULTRAWIDE_RECT}"]

    assert str(sharpness.name) == "sharpness"
    assert cast(list[str], sharpness.contents) == [f"/{SHARPNESS_WIDE}", f"/{SHARPNESS_ULTRAWIDE}"]


def test_fit_triage_page_keeps_the_three_by_three_diagnostic_grid() -> None:
    _, pages = top_pages(make_blueprint(include_gauss_surf=True))
    triage: Container = pages[3]
    assert isinstance(triage, rrb.Horizontal)
    assert triage.column_shares == [1.0, 2.0]
    triage_world: View = cast(View, children(triage)[0])
    grid: Container = cast(Container, children(triage)[1])
    assert cast(list[str], triage_world.contents) == [f"/{SPLATS}/**", f"/{PROMPTDA_MESH}"]
    assert isinstance(grid, rrb.Vertical)
    assert grid.row_shares == [1.0, 1.0, 1.0]
    triage_entities: dict[str, str] = {
        "Source RGB": RGB_ULTRAWIDE_RECT,
        "Fitted RGB": RGB_SPLAT_ULTRAWIDE_RECT,
        "RGB error [0, 0.5]": ERROR_RGB_ULTRAWIDE_RECT,
        "Source depth": DEPTH_ULTRAWIDE_RECT,
        "Fitted depth": DEPTH_SPLAT_ULTRAWIDE,
        "Depth error [0, 0.5 m]": ERROR_DEPTH_ULTRAWIDE_RECT,
        "Source normal": NORMALS_ULTRAWIDE_RECT,
        "Fitted normal": NORMALS_SPLAT_ULTRAWIDE_RECT,
        "Normal error [0, 45 deg]": ERROR_NORMAL_ULTRAWIDE_RECT,
    }
    grid_views: dict[str, View] = {str(view.name): view for view in iter_views(grid)}
    for view_name, entity_path in triage_entities.items():
        assert cast(list[str], grid_views[view_name].contents) == [f"/{entity_path}"]


def test_table_blueprint_hides_video_depth_and_confidence() -> None:
    """The segment-table preview cards must not pay AV1 decode or depth/confidence uploads."""
    contents: list[str] = cast(list[str], find_view(make_table_blueprint(), "world").contents)
    for excluded in (VIDEO_WIDE, VIDEO_ULTRAWIDE, GT_DEPTH, DEPTH, CONFIDENCE):
        assert f"- /{excluded}/**" in contents
    assert "$origin/**" in contents
