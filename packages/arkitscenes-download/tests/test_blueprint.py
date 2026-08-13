"""Blueprint composition: the PromptDA variant extends the stock layout."""

from typing import cast

import rerun.blueprint as rrb
from rerun.blueprint.api import Container, View

from arkitscenes_download.ingest.blueprint import make_blueprint, make_table_blueprint
from arkitscenes_download.ingest.paths import (
    ARKIT_MESH,
    CONFIDENCE,
    DEPTH,
    DEPTH_PROMPTDA,
    GT,
    GT_BOXES,
    GT_DEPTH,
    GT_PINHOLE_WIDE,
    PINHOLE_WIDE_LOWRES,
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


def view_names(blueprint: rrb.Blueprint) -> list[str]:
    return [str(v.name) for v in iter_views(blueprint)]


def find_view(blueprint: rrb.Blueprint, name: str):
    matches = [v for v in iter_views(blueprint) if str(v.name) == name]
    assert len(matches) == 1, f"expected exactly one view named {name!r}, got {len(matches)}"
    return matches[0]


def test_default_blueprint_has_no_promptda_views() -> None:
    names = view_names(make_blueprint(portrait=True))
    assert "depth PromptDA" not in names
    assert "world PromptDA" not in names
    assert "world" in names
    assert "world GT" in names


def test_blueprint_variants_construct_with_world_gt_tab() -> None:
    """Portrait, landscape, and PromptDA layouts always expose the deliberate GT marker tab."""
    for portrait, include_promptda in ((False, False), (True, False), (False, True), (True, True)):
        names: list[str] = view_names(make_blueprint(portrait=portrait, include_promptda=include_promptda))
        assert "world GT" in names
        assert ("world PromptDA" in names) is include_promptda


def test_world_tabs_separate_arkit_and_laser_ground_truth() -> None:
    """The stock world hides laser GT while the GT tab hides ARKit geometry and low-res depth."""
    blueprint = make_blueprint()
    assert cast(list[str], find_view(blueprint, "world").contents) == ["$origin/**", f"- /{GT}/**"]
    assert cast(list[str], find_view(blueprint, "world GT").contents) == [
        "$origin/**",
        f"- /{ARKIT_MESH}/**",
        f"- /{GT_BOXES}/**",
        f"- /{DEPTH}/**",
        f"- /{CONFIDENCE}/**",
    ]

    laser_depth_view = find_view(blueprint, "depth GT (laser)")
    assert str(laser_depth_view.origin) == GT_PINHOLE_WIDE
    assert cast(list[str], laser_depth_view.contents) == ["$origin/depth"]


def test_promptda_blueprint_adds_depth_tab_and_world_tab() -> None:
    blueprint = make_blueprint(portrait=True, include_promptda=True)
    names = view_names(blueprint)
    assert "depth PromptDA" in names
    assert "world PromptDA" in names
    # The stock views survive untouched.
    for kept in ("world", "wide", "ultrawide", "depth ARKit", "depth GT (laser)", "confidence", "gyro", "accel"):
        assert kept in names

    depth_view = find_view(blueprint, "depth PromptDA")
    assert cast(list[str], depth_view.contents) == ["$origin/depth_promptda"]


def test_depth_views_override_colormap_range() -> None:
    """Encoded uint16 depth defaults to a 0-65535 colormap range in the viewer; the views must pin a sane one."""
    blueprint = make_blueprint(include_promptda=True)
    for view_name, entity_path in (("depth ARKit", DEPTH), ("depth GT (laser)", GT_DEPTH), ("depth PromptDA", DEPTH_PROMPTDA)):
        overrides = find_view(blueprint, view_name).visualizer_overrides
        assert overrides is not None and f"/{entity_path}" in overrides, f"{view_name} is missing its depth_range override"


def test_promptda_world_tab_hides_arkit_mesh_and_other_depths() -> None:
    blueprint = make_blueprint(include_promptda=True)
    world_promptda = find_view(blueprint, "world PromptDA")
    contents = cast(list[str], world_promptda.contents)
    assert f"- /{ARKIT_MESH}/**" in contents
    assert f"- /{GT_DEPTH}/**" in contents
    assert f"- /{PINHOLE_WIDE_LOWRES}/**" in contents
    # Everything else under /world stays visible (boxes, frustum, promptda mesh).
    assert "$origin/**" in contents


def test_camera_views_overlay_gt_boxes() -> None:
    """Both RGB views project the new top-level GT-box subtree."""
    blueprint = make_blueprint(include_promptda=True)
    for view_name in ("wide", "ultrawide"):
        assert f"/{GT_BOXES}/**" in cast(list[str], find_view(blueprint, view_name).contents)


def test_table_blueprint_hides_video_depth_and_confidence() -> None:
    """The segment-table preview cards must not pay AV1 decode or depth/confidence uploads."""
    contents = cast(list[str], find_view(make_table_blueprint(), "world").contents)
    for excluded in (VIDEO_WIDE, VIDEO_ULTRAWIDE, GT_DEPTH, DEPTH, CONFIDENCE):
        assert f"- /{excluded}/**" in contents
    # Mesh, boxes, and frusta stay: everything else under /world remains included.
    assert "$origin/**" in contents
