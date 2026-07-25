"""Blueprint composition: the PromptDA variant extends the stock layout."""

from typing import cast

import rerun.blueprint as rrb
from rerun.blueprint.api import Container, View

from arkitscenes_download.ingest.blueprint import make_blueprint, make_table_blueprint
from arkitscenes_download.ingest.paths import (
    CONFIDENCE,
    DEPTH,
    DEPTH_GT,
    DEPTH_PROMPTDA,
    GT_MESH,
    PINHOLE_WIDE_LOWRES,
    PROMPTDA_MESH,
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
    for view_name, entity_path in (("depth ARKit", DEPTH), ("depth GT (laser)", DEPTH_GT), ("depth PromptDA", DEPTH_PROMPTDA)):
        overrides = find_view(blueprint, view_name).visualizer_overrides
        assert overrides is not None and f"/{entity_path}" in overrides, f"{view_name} is missing its depth_range override"


def test_promptda_world_tab_hides_gt_mesh_and_other_depths() -> None:
    blueprint = make_blueprint(include_promptda=True)
    world_promptda = find_view(blueprint, "world PromptDA")
    contents = cast(list[str], world_promptda.contents)
    assert f"- /{GT_MESH}/**" in contents
    assert f"- /{DEPTH_GT}/**" in contents
    assert f"- /{PINHOLE_WIDE_LOWRES}/**" in contents
    # Everything else under /world stays visible (boxes, frustum, promptda mesh).
    assert "$origin/**" in contents


def test_stock_world_tab_hides_promptda_geometry_when_comparison_is_enabled() -> None:
    """Keep PromptDA geometry exclusive to its dedicated world tab."""
    blueprint = make_blueprint(include_promptda=True)
    contents = cast(list[str], find_view(blueprint, "world").contents)
    assert f"- /{PROMPTDA_MESH}/**" in contents
    assert f"- /{DEPTH_PROMPTDA}/**" in contents


def test_table_blueprint_hides_video_depth_and_confidence() -> None:
    """The segment-table preview cards must not pay AV1 decode or depth/confidence uploads."""
    contents = cast(list[str], find_view(make_table_blueprint(), "world").contents)
    for excluded in (VIDEO_WIDE, VIDEO_ULTRAWIDE, DEPTH_GT, DEPTH, CONFIDENCE):
        assert f"- /{excluded}/**" in contents
    # Mesh, boxes, and frusta stay: everything else under /world remains included.
    assert "$origin/**" in contents
