"""MetricTrail's catalog layouts on dataforge's hook (no catalog server, no GPU)."""

from pathlib import Path

import pytest
import rerun.blueprint as rrb
from dataforge.datasets.lamaria import LamariaConfig
from dataforge.datasets.vslam_lab import Prediction, VslamLabConfig

from metrictrail.catalog.blueprint import (
    NOT_RUN,
    PATH_COLORS,
    LayoutName,
    MetricTrailLamaria,
    MetricTrailVslamLab,
    content_rules,
    iter_views,
    metrictrail_dataset,
    publish_layout,
    shown_predictions,
)
from metrictrail.palette import GT_RGB, OURS_RGB, PAPER_PORT_RGB, UPSTREAM_RGB

OURS: Prediction = Prediction("metrictrail", depth=("/world/runs/metrictrail/cam/pinhole/depth",), images=("/world/runs/metrictrail/cam/pinhole/normals",))
PAPER: Prediction = Prediction("metrictrail_paper", depth=("/world/runs/metrictrail_paper/cam/pinhole/depth",))
UPSTREAM: Prediction = Prediction("metrictrail_upstream", depth=("/world/runs/metrictrail_upstream/cam/pinhole/depth",))


def tum(tmp_path: Path, layout: LayoutName) -> MetricTrailVslamLab:
    return MetricTrailVslamLab(VslamLabConfig(benchmark="rgbdtum", root=tmp_path), layout)


def names(blueprint: rrb.Blueprint) -> list[str]:
    return [str(view.name) for view in iter_views(blueprint)]


def view3d(blueprint: rrb.Blueprint, origin: str) -> list[str]:
    return next(content_rules(view.contents) for view in iter_views(blueprint) if isinstance(view, rrb.Spatial3DView) and str(view.origin) == origin)


def test_legend_is_the_explainer_palette() -> None:
    assert PATH_COLORS == {"gt": GT_RGB, "metrictrail": OURS_RGB, "metrictrail_paper": PAPER_PORT_RGB, "metrictrail_upstream": UPSTREAM_RGB}
    assert len(set(PATH_COLORS.values())) == 4


def test_default_shows_ground_truth_and_ours_only(tmp_path: Path) -> None:
    segment, card = tum(tmp_path, "default").layouts((OURS, PAPER, UPSTREAM), {"metrictrail"})
    assert names(segment) == [
        "Rig",
        "Follow",
        "rgb_0",
        "depth_0",
        "ours: keyframe depth",
        "ours: keyframe normals",
    ]
    for blueprint, origin in ((segment, "/"), (segment, "/world/rig_00"), (card, "/world/rig_00")):
        rules: list[str] = view3d(blueprint, origin)
        assert {"- /world/runs/metrictrail_paper/**", "- /world/runs/metrictrail_upstream/**"} <= set(rules), (origin, rules)
        assert not any(rule.startswith("- /world/runs/metrictrail/**") for rule in rules)
    assert "- /world/runs/metrictrail/points/**" in view3d(segment, "/")  # the overview shows the normals-coloured cloud
    assert "- /world/runs/metrictrail/points_normals/**" in view3d(segment, "/world/rig_00")


def test_comparison_shows_every_run_with_a_depth_pane(tmp_path: Path) -> None:
    segment, _ = tum(tmp_path, "comparison").layouts((OURS, PAPER, UPSTREAM), {"metrictrail"})
    assert names(segment)[4:] == [
        "ours: keyframe depth",
        "ours: keyframe normals",
        "torch port: keyframe depth",
        "upstream ViPE: keyframe depth",
    ]
    for origin in ("/", "/world/rig_00"):
        assert not {f"- /world/runs/{source}/**" for source in ("metrictrail", "metrictrail_paper", "metrictrail_upstream")} & set(view3d(segment, origin))


def test_a_dataset_without_the_shipped_path_says_so_and_hides_the_rest(tmp_path: Path) -> None:
    dataset = MetricTrailVslamLab(VslamLabConfig(benchmark="kitti", root=tmp_path), "default")
    segment, _ = dataset.layouts((PAPER, UPSTREAM), set())
    assert names(segment) == ["Rig", "Follow", "rgb_0", "rgb_1", f"ours: {NOT_RUN}"]
    assert {"- /world/runs/metrictrail_paper/**", "- /world/runs/metrictrail_upstream/**"} <= set(view3d(segment, "/"))


def test_shown_predictions_keep_layout_order_and_fill_missing_runs() -> None:
    assert [prediction.source for prediction in shown_predictions((UPSTREAM, PAPER), "comparison")] == [
        "metrictrail",
        "metrictrail_paper",
        "metrictrail_upstream",
    ]
    assert shown_predictions((), "default") == (Prediction("metrictrail", depth=("/world/runs/metrictrail/cam/pinhole/depth",)),)


def test_lamaria_adds_the_run_panes_to_its_own_layout(tmp_path: Path) -> None:
    segment, card = MetricTrailLamaria(LamariaConfig(root=tmp_path), "comparison").layouts((PAPER,), set())
    titles: list[str] = names(segment)
    assert "torch port: keyframe depth" in titles
    assert f"ours: {NOT_RUN}" in titles
    assert f"upstream ViPE: {NOT_RUN}" in titles
    default, _ = MetricTrailLamaria(LamariaConfig(root=tmp_path), "default").layouts((PAPER,), set())
    assert "- /world/runs/metrictrail_paper/**" in view3d(default, "/")
    assert not any("torch port" in title for title in names(default))


def test_unknown_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="layouts exist for"):
        metrictrail_dataset("dataforge-epfl", "default")


def test_lamaria_default_layout_stays_dataforges(tmp_path: Path) -> None:
    """dataforge-lamaria is dataforge's dataset: our default is refused before any catalog access; comparison stays."""
    with pytest.raises(ValueError, match="keeps dataforge's default layout"):
        publish_layout("rerun+http://127.0.0.1:1", "dataforge-lamaria", "default", tmp_path)
    assert not (tmp_path / "blueprints").exists()
    assert isinstance(metrictrail_dataset("dataforge-lamaria", "comparison"), MetricTrailLamaria)
    assert isinstance(metrictrail_dataset("dataforge-vslam-lab-rgbdtum", "default"), MetricTrailVslamLab)
