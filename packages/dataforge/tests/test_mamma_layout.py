"""MAMMA layouts: the catalog blueprints of a family cover every camera any of its rigs has."""

import pytest
import rerun.blueprint as rrb

from dataforge import schema
from dataforge.datasets.mamma import MammaConfig, MammaDataset
from dataforge.datasets.mamma_source import Family


def views[V: rrb.Container | rrb.View](blueprint: rrb.Blueprint, kind: type[V]) -> list[V]:
    """The blueprint's views or containers of ``kind``, in layout order."""
    found: list[V] = []
    stack: list[object] = [blueprint.root_container]
    while stack:
        node = stack.pop()
        if isinstance(node, kind):
            found.append(node)
        if isinstance(node, rrb.Container):
            stack.extend(reversed(node.contents))  # type: ignore[arg-type]
    return found


def scene(blueprint: rrb.Blueprint) -> rrb.Spatial3DView:
    """The blueprint's one 3D view, wherever the layout put it."""
    (view,) = views(blueprint, rrb.Spatial3DView)
    return view


def test_the_eval_card_leaves_out_every_dance_video() -> None:
    # Eval Dance rigs have 32 cameras; a card decodes only its one pane.
    dataset = MammaConfig(family="eval").setup()
    assert isinstance(dataset, MammaDataset)
    contents: list[str] = list(scene(dataset.table_blueprint()).contents)  # type: ignore[arg-type]
    assert f"- {schema.video_path(32, 0)}/**" in contents
    assert f"- {schema.pinhole_path(32, 0)}/masks/**" in contents


def test_the_eval_scene_hides_every_dance_video_and_keeps_masks_in_the_panes() -> None:
    dataset = MammaConfig(family="eval").setup()
    assert isinstance(dataset, MammaDataset)
    view: rrb.Spatial3DView = scene(dataset.default_blueprint())
    contents: list[str] = list(view.contents)  # type: ignore[arg-type]
    assert f"- {schema.pinhole_path(32, 0)}/masks/**" in contents
    assert f"- {schema.coco133_uv_projected_path(32, 0)}/**" in contents
    hidden = view.visualizer_overrides or {}
    assert schema.video_path(32, 0) in {str(path) for path in hidden}


def test_the_eval_scene_keeps_the_visibility_landmarks_in_the_camera_panes() -> None:
    # The landmarks are pixels of one camera's image: in 3D they would sit on that camera's frustum image.
    dataset = MammaConfig(family="eval").setup()
    for blueprint in (dataset.default_blueprint(), dataset.table_blueprint()):
        contents: list[str] = list(scene(blueprint).contents)  # type: ignore[arg-type]
        assert f"- {schema.pinhole_path(32, 0)}/visibility/**" in contents


@pytest.mark.parametrize(
    ("family", "names", "tabs"),
    [
        # Eval rigs have 16 (Singles, Extra) or 32 (Dance) cameras: every take fills tabs 1-2, Dance also 3-4.
        ("eval", [f"IOI_{n:02d}" for n in range(1, 33)], 4),
        ("markerless-dance", [f"IOI_{n:02d}" for n in range(1, 33, 4)], 0),
        ("markerless-iphones", ["A001", "B001", "C001", "D001"], 0),
    ],
)
def test_catalog_panes_carry_the_release_camera_names_and_hide_the_landmarks(family: Family, names: list[str], tabs: int) -> None:
    blueprint: rrb.Blueprint = MammaConfig(family=family).setup().default_blueprint()
    panes: list[rrb.Spatial2DView] = views(blueprint, rrb.Spatial2DView)
    assert [pane.name for pane in panes] == names
    assert [len(list(tab.contents)) for tab in views(blueprint, rrb.Tabs)] == ([tabs] if tabs else [])
    # On the small panes the landmark dots cover the people: they start hidden (the blueprint panel shows them).
    assert all({str(path) for path in pane.visualizer_overrides or {}} == {f"{pane.origin}/visibility"} for pane in panes)

