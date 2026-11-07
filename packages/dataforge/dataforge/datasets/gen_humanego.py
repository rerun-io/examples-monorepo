"""Gen-HumanEgo (GenRobot DAS-Ego, hands only): the whole-body converter's sibling, plus Ego-Depth.

The same DAS-Ego rig and MCAP topic family as Gen-HumanWholeBody (``genrobot``), without the body track and with
four new topics: camera2's metric depth (``depth`` layer, ``gen_humanego_depth``), GenRobot's per-hand detection
scores (``handtracking/scores_debug``, kept in the ``pose`` layer), and the unsmoothed hand track
(``handtracking_raw``, sidecar only: its MANO parameters land on none of its bones). Everything else, readers,
lenses, clock, calibration and the MANO check, is ``DasEgoDataset``'s. Differences, solved from the data, are
in ``docs/gen-humanego.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray

from dataforge import blueprints, paths, schema
from dataforge.das_ego.dataset import PANE_ORDER, DasEgoConfig, DasEgoDataset, HubEpisode
from dataforge.das_ego.layers import RIG
from dataforge.das_ego.pose import POSE_LAYER, SIDES, PoseTrack, read_rig_poses, write_pose_layer
from dataforge.das_ego.source import REFERENCE_CAMERA, Episode, EpisodeClock, EpisodeFile, optional_message, struct_field, topic_rows
from dataforge.datasets.gen_humanego_depth import (
    DEPTH_CAM,
    DEPTH_INFO_TOPIC,
    DEPTH_LAYER,
    DEPTH_PINHOLE_LAYER,
    DEPTH_TOPICS,
    DepthInfo,
    depth_path,
    write_depth_layer,
    write_depth_pinhole_layer,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import TRAJECTORY_COLOR, TRAJECTORY_RADIUS_UI_POINTS, frame_index_column, time_column

REPO_ID: str = "genrobot2025/Gen-HumanEgo"
"""Hugging Face dataset repository (gated; the machine's HF login has access)."""
REVISION: str = "3cca54aeba0d9af3ebf4995b035d23b41f9387da"
"""Pinned Hub commit, last modified 2026-09-25: 44,632 MCAPs, 62.0 TB."""
SCORES_TOPIC: str = "/robot0/handtracking/scores_debug"
"""Per frame and hand: the detector's score and its quality estimates."""
HANDS_EYE: rrb.EyeControls3D = rrb.EyeControls3D(
    kind=rrb.Eye3DKind.Orbital,
    position=(-1.0, -1.0, 1.3),
    look_target=(0.25, 0.25, -0.6),
    eye_up=(0.0, 0.0, 1.0),
    tracking_entity="/world/rig_00",
    spin_speed=0.0,
)
"""A steep eye above the head, looking down at the hands and the depth around them, that rides along with the rig.

The VIO world's heading is wherever the wearer first looked, so no fixed eye is "behind" every wearer; from above,
any heading reads.

Without a body there is nothing to frame at the whole-body ``SCENE_EYE``'s distance: the hands are 0.3-0.5 m below
the device and the depth reaches a few metres ahead. The eye tracks the rig's position, so a wearer who walks stays
in view; it stays upright because the view is rooted at the world, not the rig."""
VIO_RUN_SOURCE: str = "vio"
"""``/world/runs/vio``: the device's own VIO, the only trajectory the release ships."""


def scores_path(side: str) -> str:
    """``/world/gt/hands/<side>/scores``, beside the hand's ``quality``."""
    return f"{schema.hands_path(side)}/scores"


def write_hand_scores(recording: rr.RecordingStream, file: EpisodeFile, clock: EpisodeClock) -> None:
    """Each hand's shipped ``pred_score``, ``pred_score_quality`` and ``pred_quality`` (three values), as shipped, on its frames."""
    stamps, values = topic_rows(file, SCORES_TOPIC)
    if not len(values):
        return
    indexes: list[rr.TimeColumn] = [time_column(clock.video_time_ns(stamps)), frame_index_column(clock.frame_indices(stamps))]
    for side in SIDES:
        scores: pa.StructArray = struct_field(values, side)
        rr.send_columns(
            scores_path(side),
            indexes=indexes,
            columns=rr.AnyValues.columns(
                pred_score=struct_field(scores, "pred_score").cast(pa.float64()),
                pred_score_quality=struct_field(scores, "pred_score_quality").cast(pa.float64()),
                pred_quality=struct_field(scores, "pred_quality").cast(pa.list_(pa.float64())),
            ),
            recording=recording,
        )


def write_trajectory(recording: rr.RecordingStream, rig_poses: Float64[ndarray, "f 4 4"]) -> None:
    """The VIO path of the whole episode as one static strip under ``/world/runs/vio``; frames without a pose are left out."""
    posed: Bool[ndarray, "f"] = np.isfinite(rig_poses).all(axis=(1, 2))
    positions: Float32[ndarray, "n 3"] = rig_poses[posed, :3, 3].astype(np.float32)
    rr.log(
        schema.trajectory_path(VIO_RUN_SOURCE),
        rr.LineStrips3D([positions], colors=TRAJECTORY_COLOR, radii=rr.Radius.ui_points(TRAJECTORY_RADIUS_UI_POINTS)),
        static=True,
        recording=recording,
    )


def write_humanego_pose_layer(recording: rr.RecordingStream, file: EpisodeFile, episode: Episode, track: PoseTrack) -> None:
    """The shared ``pose`` layer without a body, plus the hand scores and the VIO trajectory only Gen-HumanEgo's layout shows."""
    write_pose_layer(recording, track, with_body=False)
    write_hand_scores(recording, file, episode.clock)
    write_trajectory(recording, read_rig_poses(file, episode.clock, episode.clock.frame_numbers))


@dataclass
class GenHumanEgoConfig(DasEgoConfig):
    """Gen-HumanEgo episodes, fetched from the Hub one at a time."""

    command: ClassVar[str] = "gen-humanego"
    """Registry key and catalog dataset."""
    _target: type = field(default_factory=lambda: GenHumanEgoDataset)
    """Dataset instantiated by setup."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "gen-humanego")
    """Local scratch: the manifest and the MCAPs being converted. Never the NAS."""
    revision: str = REVISION
    """Pinned Hub commit; an override must be a full commit sha."""


class GenHumanEgoDataset(DasEgoDataset[GenHumanEgoConfig]):
    """Gen-HumanEgo: hands, MANO, actions and camera2's metric depth; no body."""

    repo_id: ClassVar[str] = REPO_ID
    with_body: ClassVar[bool] = False
    mcap_layers: ClassVar[tuple[str, ...]] = (paths.BASE_LAYER, DEPTH_LAYER)
    sidecar_exclusions: ClassVar[tuple[str, ...]] = DEPTH_TOPICS
    layers: tuple[str, ...] = (paths.BASE_LAYER, DEPTH_LAYER, DEPTH_PINHOLE_LAYER, POSE_LAYER, paths.ACTIONS_LAYER, paths.PROJECTIONS_LAYER, paths.HAND_MESH_LAYER)
    """Publication and loading order."""

    def write_mcap_layers(
        self, identity: SequenceIdentity, mcap: Path, targets: dict[str, Path], pending: Sequence[str], *, episode: Episode, source: HubEpisode
    ) -> None:
        """The base layer, then the depth layer when the episode ships ``depth_info``."""
        super().write_mcap_layers(identity, mcap, targets, pending, episode=episode, source=source)
        if DEPTH_LAYER not in pending:
            return
        info: DepthInfo | None = optional_message(EpisodeFile(mcap), DEPTH_INFO_TOPIC, DepthInfo)
        if info is None:
            targets[DEPTH_LAYER].unlink(missing_ok=True)
            print(f"{identity.sequence_key}: no {DEPTH_INFO_TOPIC}, so no depth layer")
            return
        writer = partial(write_depth_layer, mcap=mcap, clock=episode.clock, calibration=episode.calibrations[REFERENCE_CAMERA], info=info)
        self.write_layers(identity, targets, [DEPTH_LAYER], {DEPTH_LAYER: lambda recording: print(f"{identity.sequence_key}: depth {writer(recording)} frames")})

    def extra_writers(
        self, targets: dict[str, Path], pending: Sequence[str], file: EpisodeFile, episode: Episode
    ) -> dict[str, Callable[[rr.RecordingStream], None]]:
        """``depth_pinhole``, from the published depth layer and the sidecar's ``depth_info``; the MCAP is not needed."""
        info: DepthInfo | None = optional_message(file, DEPTH_INFO_TOPIC, DepthInfo)
        if DEPTH_PINHOLE_LAYER not in pending or info is None or not targets[DEPTH_LAYER].is_file():
            return {}
        calibration = episode.calibrations[REFERENCE_CAMERA]
        return {DEPTH_PINHOLE_LAYER: partial(write_depth_pinhole_layer, depth_rrd=targets[DEPTH_LAYER], calibration=calibration, info=info)}

    def pose_writer(self, file: EpisodeFile, episode: Episode, track: PoseTrack) -> Callable[[rr.RecordingStream], None]:
        return partial(write_humanego_pose_layer, file=file, episode=episode, track=track)

    def world_contents(self) -> list[str]:
        """The shared 3D scene with the pinhole-resampled depth, without the fisheye depth image a ``Pinhole`` would bend."""
        return [*super().world_contents(), f"- {depth_path()}"]

    def default_blueprint(self) -> rrb.Blueprint:
        """The wearer in 3D with the VIO path and hand meshes over the task and action lanes; camera2 with its hands and its depth
        on the right, above the other five cameras in yaw order."""
        others: list[int] = [cam for cam in PANE_ORDER if cam != REFERENCE_CAMERA]
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name="DAS-Ego hands", origin="/world", contents=self.world_contents(), eye_controls=HANDS_EYE),
                    rrb.TextDocumentView(name="Task", origin=schema.instruction_path(), contents=[schema.instruction_path()]),
                    rrb.StateTimelineView(name="Actions", origin="/task", contents=[schema.actions_path("coarse"), schema.actions_path("fine")]),
                    row_shares=[5, 0.7, 1.3],
                ),
                rrb.Vertical(
                    rrb.Horizontal(
                        blueprints.camera_view(f"camera{REFERENCE_CAMERA} with hands", RIG, REFERENCE_CAMERA, contents=self.pane_contents(REFERENCE_CAMERA)),
                        rrb.Spatial2DView(name="camera2 depth", origin=schema.pinhole_path(RIG, DEPTH_CAM), contents=[depth_path()]),
                        column_shares=[1, 1],
                    ),
                    rrb.Grid(
                        *(blueprints.camera_view(f"camera{cam}", RIG, cam, contents=self.pane_contents(cam)) for cam in others),
                        grid_columns=3,
                        name="Other cameras, left to right",
                    ),
                    row_shares=[1, 1],
                ),
                column_shares=[1, 1],
            ),
            rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
            collapse_panels=True,
        )
