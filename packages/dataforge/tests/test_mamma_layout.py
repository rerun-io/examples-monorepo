"""MAMMA layouts: the catalog blueprints of a family cover every camera any of its rigs has."""

import rerun.blueprint as rrb

from dataforge import schema
from dataforge.datasets.mamma import MammaConfig, MammaDataset


def scene(blueprint: rrb.Blueprint) -> rrb.Spatial3DView:
    """The blueprint's one 3D view, wherever the layout put it."""
    found: list[rrb.Spatial3DView] = []
    stack: list[object] = [blueprint.root_container]
    while stack:
        node = stack.pop()
        if isinstance(node, rrb.Spatial3DView):
            found.append(node)
        elif isinstance(node, rrb.Container):
            stack.extend(node.contents)  # type: ignore[arg-type]
    (view,) = found
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
