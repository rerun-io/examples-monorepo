"""The RoboCap result layout: four tabs, the right cameras per tab, the follow views on the VIO run (no data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun.blueprint as rrb

from dataforge import schema
from dataforge.datasets import robocap_blueprint as layout
from dataforge.writing import blueprint_views


def tab(blueprint: rrb.Blueprint, name: str) -> rrb.Container:
    tabs = blueprint.root_container
    assert isinstance(tabs, rrb.Tabs)
    found = [child for child in tabs.contents if isinstance(child, rrb.Container) and getattr(child, "name", None) == name]
    assert len(found) == 1, f"no tab {name}"
    return found[0]


def views_of(node: rrb.Container) -> list[rrb.View]:
    out: list[rrb.View] = []

    def walk(item: rrb.View | rrb.Container) -> None:
        if isinstance(item, rrb.View):
            out.append(item)
        else:
            for child in item.contents:
                walk(child)

    walk(node)
    return out


def test_result_layout_has_four_tabs_and_serializes(tmp_path: Path) -> None:
    blueprint: rrb.Blueprint = layout.result_layout(np.array([[0.0, 0.0, 0.0], [120.0, 30.0, 1.0], [250.0, -20.0, 2.0]]))
    assert [child.name for child in blueprint.root_container.contents] == ["SLAM", "colsfm", "Comparison", "Map"]
    path: Path = tmp_path / "robocap-result.rbl"
    blueprint.save("robocap", str(path))
    assert path.stat().st_size > 0
    assert len(blueprint_views(blueprint)) > 0


def test_slam_tab_shows_the_four_fed_cameras_with_keypoints() -> None:
    views: list[rrb.View] = views_of(tab(layout.result_layout(), "SLAM"))
    panes: list[rrb.View] = [view for view in views if isinstance(view, rrb.Spatial2DView)]
    assert [pane.name for pane in panes] == ["left_front", "right_front", "left", "right"]
    assert [layout.SLAM_CAMERAS[index] for index in range(4)] == [0, 1, 4, 5]


def test_colsfm_tab_shows_all_six_cameras() -> None:
    panes = [view for view in views_of(tab(layout.result_layout(), "colsfm")) if isinstance(view, rrb.Spatial2DView)]
    assert [pane.name for pane in panes] == list(layout.CAMERAS)


def test_follow_views_ride_the_vio_run_and_the_map_tab_has_a_map() -> None:
    blueprint: rrb.Blueprint = layout.result_layout()
    follows = [view for view in blueprint_views(blueprint) if isinstance(view, rrb.Spatial3DView) and str(view.name).startswith("Follow")]
    assert len(follows) == 3
    assert {str(view.origin) for view in follows} == {schema.follow_path(schema.SLAM_RS_VIO_SOURCE)}
    assert any(isinstance(view, rrb.MapView) for view in views_of(tab(blueprint, "Map")))
