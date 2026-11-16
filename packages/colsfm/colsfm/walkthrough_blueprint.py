"""The blueprint that lays the recording out as a tab per stage.

Separate from the panels because it is written from the stage *vocabulary* and
one run-dependent input — the sample panes stage 2 picked — rather than from any
stage's data. It is sent once, at the end, since which keyframes get sampled is
not known before the run.
"""


from __future__ import annotations

from collections.abc import Sequence

import rerun.blueprint as rrb

from colsfm.walkthrough_stages import (
    STAGE_TIMELINE,
    SamplePane,
    WalkthroughStage,
)

# ══════════════════════════════════════════════════════════════════════════════════════
# the blueprint
# ══════════════════════════════════════════════════════════════════════════════════════


def _notes_view(stage: WalkthroughStage) -> rrb.TextDocumentView:
    """The notes pane every tab ends with.

    Args:
        stage: The stage the pane documents.

    Returns:
        A text view over `<stage>/notes`.
    """
    return rrb.TextDocumentView(origin=f"{stage.entity_root}/notes", name="Notes")


def _scene_view(stage: WalkthroughStage, name: str) -> rrb.Spatial3DView:
    """A 3D view over one stage's whole subtree, minus its text panes.

    Args:
        stage: The stage to show.
        name: Pane title.

    Returns:
        The view.
    """
    return rrb.Spatial3DView(
        origin=stage.entity_root,
        name=name,
        contents=["+ $origin/**", "- $origin/notes", "- $origin/report"],
    )


def stage_tab(stage: WalkthroughStage, samples: Sequence[SamplePane]) -> rrb.Container:
    """Lay out one stage's tab: the views that make that stage legible.

    Args:
        stage: The stage to lay out.
        samples: Stage 2's sample panes; ignored by every other stage.

    Returns:
        A container holding the stage's spatial, tabular and textual panes.
    """
    root: str = stage.entity_root
    name: str = stage.card.title
    if stage.name == "feature_extraction":
        # One view per sample: several images under one 2D origin all sit at the
        # same pixel coordinates, so only the last one logged would be visible.
        sample_views: list[rrb.View] = [
            rrb.Spatial2DView(origin=pane.entity_path, name=pane.name) for pane in samples
        ] or [rrb.Spatial2DView(origin=f"{root}/samples", name="Keyframes + ALIKED keypoints")]
        return rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(*sample_views),
                rrb.BarChartView(origin=f"{root}/keypoint_counts", name="Keypoints per image"),
                row_shares=[3.0, 1.0],
            ),
            _notes_view(stage),
            column_shares=[3.0, 2.0],
            name=name,
        )
    if stage.name == "matching":
        return rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{root}/pair/stacked", name="Verified matches"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{root}/pair/left", name="Left"),
                    rrb.Spatial2DView(origin=f"{root}/pair/right", name="Right"),
                ),
                rrb.Horizontal(
                    rrb.BarChartView(origin=f"{root}/inliers_per_pair", name="Inliers per pair"),
                    rrb.BarChartView(origin=f"{root}/inlier_histogram", name="Inlier histogram"),
                ),
                row_shares=[2.0, 2.0, 1.5],
            ),
            _notes_view(stage),
            column_shares=[3.0, 2.0],
            name=name,
        )
    if stage.name == "reconstruction":
        return rrb.Horizontal(
            _scene_view(stage, "Mapped cloud"),
            rrb.Vertical(
                rrb.BarChartView(origin=f"{root}/round_chart/mean_reprojection_error_px", name="Reprojection px per round"),
                rrb.BarChartView(origin=f"{root}/round_chart/num_observations", name="Observations per round"),
                rrb.TimeSeriesView(origin=f"{root}/rounds", name="Rounds (switch to the `round` timeline)"),
            ),
            _notes_view(stage),
            column_shares=[3.0, 2.0, 2.0],
            name=name,
        )
    if stage.name == "export":
        return rrb.Horizontal(
            _scene_view(stage, "Cloud, rigs and trajectories"),
            rrb.TextDocumentView(origin=f"{root}/report", name="Benchmark report"),
            _notes_view(stage),
            column_shares=[3.0, 2.0, 2.0],
            name=name,
        )
    return rrb.Horizontal(_scene_view(stage, name), _notes_view(stage), column_shares=[3.0, 2.0], name=name)


def build_blueprint(stages: Sequence[WalkthroughStage], samples: Sequence[SamplePane] = ()) -> rrb.Blueprint:
    """One tab per stage, with the time panel pinned to the `stage` timeline.

    Args:
        stages: The stages this recording holds, in order.
        samples: Stage 2's sample panes, which only exist once that stage has run.

    Returns:
        The blueprint the recording ships with.
    """
    return rrb.Blueprint(
        rrb.Tabs(*[stage_tab(stage, samples) for stage in stages], active_tab=0),
        rrb.TimePanel(state="expanded", timeline=STAGE_TIMELINE),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
    )


