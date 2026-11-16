"""What a walkthrough stage *is*, and the palette every panel draws it with.

The vocabulary layer of `colsfm.walkthrough`: one `StageCard` per pipeline stage
holding the prose that never changes between runs, the `WalkthroughStage` that
pairs a card with its timeline index and entity root, and the constants — entity
paths, timelines, colours, relative radii — that keep eight panels and a
blueprint drawing the same recording rather than eight of their own.

Every radius here is a *fraction* of the scene's diagonal, because the
walkthrough draws scenes of very different size in one recording: Galileo sweeps
0.66 m and RoboCap walks 124 m, and a metre-fixed radius makes one an unreadable
blob and the other invisible.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

import rerun as rr

from colsfm import DATA_DIR
from colsfm.rerun_log import Rgb
from colsfm.run_lifecycle import StageName, stage_names

# ══════════════════════════════════════════════════════════════════════════════════════
# constants
# ══════════════════════════════════════════════════════════════════════════════════════

WALKTHROUGH_PATH: Final[str] = "walkthrough"
"""Root of the walkthrough's entity tree; every stage hangs one level below it."""

STAGE_TIMELINE: Final[str] = "stage"
"""Sequence timeline carrying one integer per stage, 1-based, in pipeline order."""

ROUND_TIMELINE: Final[str] = "round"
"""Sequence timeline carrying the mapper's bundle-adjustment rounds, 0-based."""

INPUT_COLOR: Final[Rgb] = (135, 135, 145)
"""Grey — the poses that went in, before any stage touched them."""

SELECTED_COLOR: Final[Rgb] = (90, 170, 255)
"""Blue — the colsfm pipeline's own hue, inherited from `colsfm.rerun_log`."""

STEREO_COLOR: Final[Rgb] = (255, 150, 40)
"""Orange — the declared stereo pairs, so they read apart from the consecutive ones."""

LOOP_COLOR: Final[Rgb] = (235, 60, 70)
"""Red — loop constraints, the one edge kind that is not sequential."""

SOLVED_COLOR: Final[Rgb] = (80, 230, 120)
"""Green — the solved side of a before/after pair."""

REFERENCE_COLOR: Final[Rgb] = (255, 150, 40)
"""Orange — the NVIDIA blob reference run, as `colsfm.rerun_log` colours it."""

FRUSTUM_RADIUS_FRACTION: Final[float] = 0.0025
"""Camera-frustum line radius as a fraction of the scene's diagonal."""

EDGE_RADIUS_FRACTION: Final[float] = 0.0035
"""Pose-graph, pair and trajectory line radius as a fraction of the scene's diagonal."""

NODE_RADIUS_FRACTION: Final[float] = 0.008
"""Camera-centre and rig-node marker radius as a fraction of the scene's diagonal.

Every radius here is relative because the walkthrough draws two scenes of very
different size in the same recording: Galileo sweeps 0.66 m, RoboCap walks 124 m,
and a metre-fixed radius makes one an unreadable blob and the other invisible."""

SOLVED_NODE_SHRINK: Final[float] = 0.55
"""How much smaller the solved rig nodes are drawn, so the input ones stay visible
behind them when a solve moved nothing."""

INPUT_FRUSTUM_FRACTION: Final[float] = 0.07
"""Frustum depth of the greyed-out input poses, as a fraction of the scene's diagonal."""

SELECTED_FRUSTUM_FRACTION: Final[float] = 0.11
"""Frustum depth of the poses a stage kept; deliberately longer than the input's, so
the two are told apart even when selection dropped nothing."""

KEYPOINT_RADIUS: Final[float] = 2.5
"""Keypoint marker radius, in pixels."""

MATCH_LINE_RADIUS: Final[float] = 0.6
"""Match-line radius on the concatenated pair image, in pixels."""

STACKED_MAX_WIDTH: Final[int] = 1920
"""Width the concatenated match image is scaled down to, in pixels."""

SAMPLE_JPEG_QUALITY: Final[int] = 90
"""JPEG quality of the sample frames logged into the recording (display only; the run reads the original pixels)."""

MAX_MATCH_LINES: Final[int] = 200
"""Match lines drawn on the concatenated image; more is a solid block of colour."""

INLIER_HISTOGRAM_BINS: Final[int] = 24
"""Bars in the per-pair inlier histogram."""

ARC_POINTS: Final[int] = 16
"""Vertices per loop arc; enough that the bulge reads as a curve."""

ARC_LIFT_FRACTION: Final[float] = 0.25
"""Height of a loop arc's bulge as a fraction of the chord length."""

MAX_LOOP_ARCS: Final[int] = 400
"""Strongest loop arcs drawn; the rest would be an unreadable mat of lines."""

MIN_LOOP_RIG_GAP: Final[int] = 2
"""Rig frames a verified pair must span before it counts as a revisit, not a sequence link."""

DEFAULT_EXTRINSIC_ARROW_SCALE: Final[float] = 200.0
"""Extrinsic shifts are sub-millimetre; the arrows are scaled up to be visible."""

DEFAULT_WORK_DIR: Final[Path] = DATA_DIR / "runs" / "galileo_walkthrough"
"""Where the walkthrough's own cuSFM-layout workspace goes when none is given."""

DEFAULT_OUTPUT: Final[Path] = DATA_DIR / "rrd" / "galileo_walkthrough.rrd"
"""Where the recording goes when none is given."""

APPLICATION_ID: Final[str] = "colsfm_walkthrough"
"""Rerun application id stamped into the recording."""

MarkdownRows: TypeAlias = list[tuple[str, str]]
"""Name/value rows rendered into a stage's "this run" table."""


@dataclass(frozen=True, slots=True)
class SamplePane:
    """One sample keyframe's 2D pane: which entity it lives at and what to call it.

    The blueprint cannot be written before the run, because which keyframes get
    sampled depends on the dataset. `log_walkthrough` therefore collects these
    while stage 2 logs and sends the blueprint once, at the end.
    """

    entity_path: str
    """Entity holding the image and its keypoints."""
    name: str
    """Pane title: the sensor and the keyframe id."""


# ══════════════════════════════════════════════════════════════════════════════════════
# the stage vocabulary
# ══════════════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class StageCard:
    """What a stage is, independent of any run: the text half of its notes document."""

    title: str
    """Human-readable stage name, also the blueprint tab's label."""
    computes: str
    """One sentence on what the stage turns into what."""
    replaces: str
    """The cuSFM binary (or half of one) this stage stands in for."""
    module: str
    """The `colsfm` modules that implement it."""
    figure: str
    """The box of the paper's Figure 2 this stage occupies."""


STAGE_CARDS: Final[dict[StageName, StageCard]] = {
    "keyframe_selection": StageCard(
        title="1 · Keyframe selection",
        computes=(
            "Reads `frames_meta.json` and the config profile, then drops every keyframe whose rig has not moved far "
            "enough or turned far enough since the last kept one, and regroups what is left into rig frames by "
            "`synced_sample_id`."
        ),
        replaces="`feature_extractor_main` (selection half)",
        module="`colsfm.keyframe_selection`, `colsfm.frames_meta`, `colsfm.config`",
        figure="System inputs — initial poses, image sequences and camera parameters (§2).",
    ),
    "feature_extraction": StageCard(
        title="2 · Feature extraction",
        computes=(
            "Creates the COLMAP database carrying cuSFM's own camera, rig, frame and image ids, then runs RaCo-ALIKED (the default `--features-backend raco`; COLMAP's ALIKED on `pycolmap`) over "
            "every kept keyframe and stores keypoints and descriptors in it."
        ),
        replaces="`feature_extractor_main` (per-image half)",
        module="`colsfm.database`, `colsfm.cameras`, `colsfm.features`",
        figure="Feature Extraction, second row (§3.1) — ALIKED, the paper's own default; here RaCo-ALIKED.",
    ),
    "pair_selection": StageCard(
        title="3 · Pair selection",
        computes=(
            "Enumerates the image pairs the matcher will be given from the metadata alone: each camera to itself "
            "across the next `connected_keyframe_num` rig frames, plus the two cameras of every declared stereo pair "
            "inside one rig frame. Loop pairs cannot exist yet."
        ),
        replaces="`feature_matcher_task_builder_main`",
        module="`colsfm.pairs`",
        figure="View-Graph Construction via Pose Graph Priors (§3.4) — the non-redundant data association idea.",
    ),
    "matching": StageCard(
        title="4 · Matching",
        computes=(
            "Runs the selected matcher (FlexGlue by default; LightGlue+ on `--matching-backend raco`) over every selected pair, verifies each match set with COLMAP's two-view geometry "
            "estimator, subsamples the inliers over an image-space grid, and writes both into the database."
        ),
        replaces="`feature_matcher_main`",
        module="`colsfm.matching`",
        figure="Feature Matching, second row (§3.1) — LightGlue in the paper; FlexGlue here by default.",
    ),
    "loop_closure": StageCard(
        title="5 · Loop closure",
        computes=(
            "Builds a retrieval index over the stored descriptors, shortlists revisit candidates by score and by time "
            "gap, matches the ones the earlier stage had not, measures a metric rig-to-rig pose for each, and gates "
            "the result on the config's translation and rotation thresholds."
        ),
        replaces="`generate_bow_vocabulary_main`, `generate_bow_index_main`, `generate_association_main`",
        module="`colsfm.retrieval`, `colsfm.loop_closure`, `colsfm.loop_pose`",
        figure="Dictionary Construction (§3.2) → Loop Closure Detection (§3.3), first row.",
    ),
    "pose_graph": StageCard(
        title="6 · Pose graph",
        computes=(
            "Turns each rig frame into one node carrying `world_T_rig`, adds a sequential constraint per "
            "`connected_keyframe_num` link and the loop constraints stage 5 produced, and solves the graph on Ceres."
        ),
        replaces="`pose_graph_main`",
        module="`colsfm.pose_graph`, `colsfm.ceres_pose`",
        figure="Pose Graph Optimization (§4.2), first row.",
    ),
    "reconstruction": StageCard(
        title="7 · Triangulation + BA",
        computes=(
            "Builds a `pycolmap.Reconstruction` on the pose-graph poses, triangulates every registered image until a "
            "pass adds nothing, then runs merge → complete → filter → global bundle adjustment for each round of a "
            "decaying reprojection-error schedule."
        ),
        replaces="`keypoints_mapper_main`",
        module="`colsfm.reconstruction`, `colsfm.mapping`",
        figure="Camera Projection (§3.5) feeding Iterative Triangulation and Mapping (§4.3), third row.",
    ),
    "extrinsic_refinement": StageCard(
        title="7b · Extrinsic refinement",
        computes=(
            "Alternates a regularised solve over the rig's `sensor_from_rig` extrinsics — carrying cuSFM's absolute "
            "(Eq. 14) and inter-camera relative (Eq. 6) priors — with pycolmap's own bundle adjustment, holding the "
            "gauge camera fixed."
        ),
        replaces="`keypoints_mapper_main --optimize_extrinsics=True`",
        module="`colsfm.extrinsic_refinement`, `colsfm.mapping`",
        figure="Extrinsic Refinement (§4.4), third row.",
    ),
    "export": StageCard(
        title="8 · Export",
        computes=(
            "Colours the points from the source imagery and writes the four artifacts a cuSFM run leaves behind: the "
            "`sparse/` COLMAP model, `kpmap/keyframes/frames_meta.json`, the TUM trajectories and `runtime.csv`."
        ),
        replaces="`kpmap_to_colmap`, `extract_pose_from_map_main`, `update_keyframe_pose_main`",
        module="`colsfm.export`, `colsfm.benchmark`, `colsfm.bench_report`",
        figure="System outputs — refined trajectory, sparse structure and optimised extrinsics (§2).",
    ),
}
"""One card per `colsfm.pipeline.StageName`, so a stage cannot be logged without its prose."""


@dataclass(frozen=True, slots=True)
class WalkthroughStage:
    """One stage of the walkthrough: its position on the timeline and its prose."""

    index: int
    """1-based position on the `stage` timeline, in pipeline order."""
    name: StageName
    """The `colsfm.pipeline` stage this shows."""
    card: StageCard
    """What the stage is, run-independently."""
    also_at_zero: tuple[str, ...] = ()
    """Duration timelines every panel of this stage is also stamped on, at 0 — so a recording that adds a
    `video_time` overview shows the stage tabs on that timeline too, with no timeline switch."""

    @property
    def entity_root(self) -> str:
        """Where this stage's entities live.

        Returns:
            `/walkthrough/stage_NN`, zero-padded so the paths sort in stage order.
        """
        return f"/{WALKTHROUGH_PATH}/stage_{self.index:02d}"


def walkthrough_stages(optimize_extrinsics: bool, also_at_zero: tuple[str, ...] = ()) -> tuple[WalkthroughStage, ...]:
    """The stages this walkthrough will record, numbered in order.

    Args:
        optimize_extrinsics: Whether the run makes the second mapping pass, which
            adds stage 7b and pushes export one index later.
        also_at_zero: Duration timelines the panels are also stamped on at 0 (`WalkthroughStage.also_at_zero`).

    Returns:
        One stage per `colsfm.pipeline.stage_names` entry, 1-based.
    """
    return tuple(
        WalkthroughStage(index=index, name=name, card=STAGE_CARDS[name], also_at_zero=also_at_zero)
        for index, name in enumerate(stage_names(optimize_extrinsics), start=1)
    )


def stamp_stage(stage: WalkthroughStage) -> None:
    """Set the time a stage's panels are logged at: its index on `stage`, and 0 on every `also_at_zero` timeline.

    Args:
        stage: The stage about to be logged.
    """
    rr.set_time(STAGE_TIMELINE, sequence=stage.index)
    for timeline in stage.also_at_zero:
        rr.set_time(timeline, duration=0.0)
