"""The shared exo/ego layout every multi-camera dataset's default blueprint is built from."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pytest
import rerun.blueprint as rrb

from dataforge import blueprints, schema


def panes(rig: int, count: int) -> list[rrb.Spatial2DView]:
    return [blueprints.camera_view(schema.cam_path(rig, cam).removeprefix("/world/"), rig, cam) for cam in range(count)]


def world_view() -> rrb.Spatial3DView:
    return rrb.Spatial3DView(name="3D", origin="/world")


def children(node: object, kind: type[rrb.Container]) -> list[rrb.Container | rrb.View]:
    """A container's children, after checking it is the expected container kind."""
    assert isinstance(node, kind), node
    return list(cast("Iterable[rrb.Container | rrb.View]", node.contents))


def test_exo_strip_under_the_3d_view_and_ego_column() -> None:
    world: rrb.Spatial3DView = world_view()
    ego: list[rrb.Spatial2DView] = panes(8, 4)
    exo: list[rrb.Spatial2DView] = panes(0, 8)
    root = blueprints.exoego_blueprint(world, ego_panes=ego, exo_panes=exo).root_container
    assert isinstance(root, rrb.Vertical) and root.row_shares == list(blueprints.EXOEGO_SHARES)
    top, strip = children(root, rrb.Vertical)
    assert isinstance(top, rrb.Horizontal) and top.column_shares == list(blueprints.EXOEGO_SHARES)
    scene, column = children(top, rrb.Horizontal)
    assert scene is world
    assert children(column, rrb.Vertical) == ego
    assert children(strip, rrb.Horizontal) == exo


def test_instruction_sits_under_the_3d_view() -> None:
    world: rrb.Spatial3DView = world_view()
    instruction = rrb.TextDocumentView(name="Instruction", origin=schema.instruction_path())
    root = blueprints.exoego_blueprint(world, ego_panes=panes(1, 2), exo_panes=panes(0, 8), instruction=instruction).root_container
    top_left = children(children(root, rrb.Vertical)[0], rrb.Horizontal)[0]
    assert isinstance(top_left, rrb.Vertical) and top_left.row_shares == list(blueprints.INSTRUCTION_SHARES)
    assert children(top_left, rrb.Vertical) == [world, instruction]


@pytest.mark.parametrize(("count", "tabs"), [(9, None), (10, [9, 1]), (32, [9, 9, 9, 5])])
def test_exo_overflow_fills_tabs_in_order(count: int, tabs: list[int] | None) -> None:
    exo: list[rrb.Spatial2DView] = panes(0, count)
    strip = children(blueprints.exoego_blueprint(world_view(), ego_panes=[], exo_panes=exo).root_container, rrb.Vertical)[1]
    if tabs is None:
        assert children(strip, rrb.Horizontal) == exo
        return
    pages: list[list[rrb.Container | rrb.View]] = [children(page, rrb.Horizontal) for page in children(strip, rrb.Tabs)]
    assert [len(page) for page in pages] == tabs
    assert [pane for page in pages for pane in page] == exo
    second = children(strip, rrb.Tabs)[1]
    assert isinstance(second, rrb.Horizontal)
    assert second.name == (exo[9].name if tabs[1] == 1 else f"{exo[9].name} – {exo[17].name}")


def test_ego_only_rig_has_no_exo_strip() -> None:
    world: rrb.Spatial3DView = world_view()
    ego: list[rrb.Spatial2DView] = panes(0, 4)
    scene, column = children(blueprints.exoego_blueprint(world, ego_panes=ego, exo_panes=[]).root_container, rrb.Horizontal)
    assert scene is world
    assert children(column, rrb.Vertical) == ego
