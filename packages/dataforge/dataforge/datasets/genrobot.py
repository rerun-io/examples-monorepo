"""Gen-HumanWholeBody (GenRobot DAS-Ego): a pinned Hub listing, one MCAP at a time, five layers.

The corpus is 15,943 MCAPs and 8.8 TB, so nothing is kept raw: ``download`` writes the pinned listing
(``manifest.json``), and ``convert`` fetches one episode at a time (the next one in the background),
verifies it against its LFS sha256, writes the sidecar and the base layer from it, and deletes it again.
The derived layers read only the sidecar.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun.blueprint as rrb
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile
from jaxtyping import Float64
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.genrobot_layers import RIG, SIDECAR_NAME, BaseCensus, write_base_layer, write_sidecar
from dataforge.datasets.genrobot_pose import (
    POSE_LAYER,
    ManoCheck,
    PoseTrack,
    read_pose_track,
    read_rig_poses,
    smplx22_uv_projected_path,
    write_actions_layer,
    write_hand_mesh_layer,
    write_pose_layer,
    write_projections_layer,
)
from dataforge.datasets.genrobot_source import (
    NUM_CAMERAS,
    REFERENCE_CAMERA,
    REPO_ID,
    REVISION,
    Episode,
    EpisodeFile,
    parse_hub_path,
    read_episode,
)
from dataforge.identity import SequenceIdentity
from dataforge.writing import TableField, TableFields

MANIFEST_NAME: str = "manifest.json"
"""The pinned listing, under the raw root."""
PANE_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
"""Cameras left to right as the wearer sees them: camera0 looks 42 degrees left, camera5 43 degrees right."""
SCENE_EYE: rrb.EyeControls3D = blueprints.eye_controls_from_pose((-1.7, -1.7, 1.1), (0.0, 0.0, -0.55), (0.0, 0.0, 1.0))
"""An upright three-quarter eye on the VIO world, aimed at the wearer's middle.

The world starts at the device's first pose with +z up, but its heading is wherever the wearer looked
first, so the eye is oblique rather than behind anyone. A view rooted at the rig would tilt with every
nod: wearers look down 45-80 degrees at their hands."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class HubEpisode:
    """One episode file of the pinned listing."""

    path: str
    """Repository path, also the raw-root-relative destination."""
    size_bytes: int
    """Exact size."""
    sha256: str
    """LFS content hash, checked after every fetch."""

    @property
    def domain(self) -> tuple[str, ...]:
        """Domain directories, outermost first."""
        return parse_hub_path(self.path)[0]

    @property
    def uuid(self) -> str:
        """Episode uuid, the file name."""
        return parse_hub_path(self.path)[1]

    @property
    def info(self) -> transports.HfFileInfo:
        """What ``transports.hf_fetch_verified`` checks the fetched bytes against."""
        return transports.HfFileInfo(self.path, self.size_bytes, self.sha256, blob_id="")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Manifest:
    """``manifest.json``: every episode of the pinned commit."""

    revision: str
    """Hub commit the listing was read at."""
    episodes: list[HubEpisode]
    """Every MCAP, in path order."""


def list_hub_episodes(revision: str) -> Manifest:
    """Read the pinned listing from the Hub; every file under ``data/`` must be an episode."""
    entries: list[RepoFile] = [
        entry for entry in HfApi().list_repo_tree(REPO_ID, path_in_repo="data", recursive=True, repo_type="dataset", revision=revision) if isinstance(entry, RepoFile)
    ]
    episodes: list[HubEpisode] = []
    for entry in sorted(entries, key=lambda entry: entry.path):
        if entry.lfs is None:
            raise ValueError(f"{REPO_ID}@{revision} {entry.path} is not an LFS file")
        parse_hub_path(entry.path)
        episodes.append(HubEpisode(path=entry.path, size_bytes=entry.size, sha256=entry.lfs.sha256))
    return Manifest(revision=revision, episodes=episodes)


def pane_contents(cam: int) -> list[str]:
    """One fisheye pane: its video and the 2D projected through its lens.

    The pane rule for a camera with a distortion model: show the lens-projected 2D, leave out
    ``coco133_xyz``, which a Rerun ``Pinhole`` would draw through the wrong (undistorted) projection.
    """
    return [schema.video_path(RIG, cam), schema.coco133_uv_projected_path(RIG, cam), smplx22_uv_projected_path(cam)]


def world_contents() -> list[str]:
    """The 3D scene: rig, frustums with their video, body, hands and IMU node; no 2D projections on the image planes."""
    return [
        "+ /world/**",
        *(f"- {schema.coco133_uv_projected_path(RIG, cam)}" for cam in range(NUM_CAMERAS)),
        *(f"- {smplx22_uv_projected_path(cam)}" for cam in range(NUM_CAMERAS)),
    ]


@dataclass
class GenrobotConfig(DataforgeDatasetConfig):
    """Gen-HumanWholeBody episodes, fetched from the Hub one at a time."""

    command: ClassVar[str] = "genrobot"
    """Registry key and catalog dataset."""
    _target: type = field(default_factory=lambda: GenrobotDataset)
    """Dataset instantiated by setup."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "genrobot")
    """Local scratch: the manifest and the MCAPs being converted. Never the NAS."""
    sequences: tuple[str, ...] | None = None
    """Episode uuids to convert, added to ``per_domain``'s sample."""
    per_domain: int | None = None
    """Sample this many episodes of every top-level domain, in uuid order (uuids are random hex)."""
    keep_raw: bool = False
    """Keep each MCAP after its base layer is published."""
    revision: str = REVISION
    """Pinned Hub commit; an override must be a full commit sha."""

    def __post_init__(self) -> None:
        transports.require_commit_sha(self.revision)


class GenrobotDataset(DataforgeDataset[GenrobotConfig, HubEpisode]):
    """Fetch, verify, convert and delete one MCAP at a time."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, POSE_LAYER, paths.ACTIONS_LAYER, paths.PROJECTIONS_LAYER, paths.HAND_MESH_LAYER)
    """Publication and loading order."""

    def manifest(self) -> Manifest:
        """The pinned listing ``download`` wrote."""
        path: Path = self.config.root / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"{path}: run dataforge-download genrobot first")
        manifest: Manifest = from_json(Manifest, path.read_text())
        if manifest.revision != self.config.revision:
            raise ValueError(f"{path} lists {manifest.revision}, the config pins {self.config.revision}: run dataforge-download genrobot again")
        return manifest

    def download(self) -> None:
        manifest: Manifest = list_hub_episodes(self.config.revision)
        with writing.atomic_write(self.config.root / MANIFEST_NAME) as temp:
            temp.write_text(to_json(manifest))
        domains: Counter[str] = Counter(episode.domain[0] for episode in manifest.episodes)
        total_tb: float = sum(episode.size_bytes for episode in manifest.episodes) / 1e12
        print(f"genrobot: {len(manifest.episodes)} episodes, {total_tb:.2f} TB, {len(domains)} top-level domains")
        print("  convert fetches one MCAP at a time (the next in the background) and deletes it once its base layer is published")

    def discover(self) -> list[tuple[SequenceIdentity, HubEpisode]]:
        episodes: list[HubEpisode] = self.manifest().episodes
        chosen: list[HubEpisode] = episodes
        if self.config.per_domain is not None or self.config.sequences is not None:
            wanted: set[str] = set(self.config.sequences or ())
            unknown: set[str] = wanted - {episode.uuid for episode in episodes}
            if unknown:
                raise ValueError(f"episodes absent from {self.config.revision}: {sorted(unknown)}")
            taken: Counter[str] = Counter()
            chosen = []
            for episode in sorted(episodes, key=lambda episode: (episode.domain[0], episode.uuid)):
                in_sample: bool = self.config.per_domain is not None and taken[episode.domain[0]] < self.config.per_domain
                if in_sample or episode.uuid in wanted:
                    taken[episode.domain[0]] += in_sample
                    chosen.append(episode)
        return [(SequenceIdentity(self.config.name, (episode.uuid,)), episode) for episode in chosen]

    def sidecar(self, identity: SequenceIdentity) -> Path:
        """Where the episode's non-camera topics are kept."""
        return paths.sidecar_path(paths.output_root(), identity, SIDECAR_NAME)

    def needs_mcap(self, identity: SequenceIdentity, *, force: bool) -> bool:
        """Only the base layer and the sidecar read the MCAP."""
        return not self.sidecar(identity).is_file() or not writing.should_skip(self.targets(identity)[paths.BASE_LAYER], force=force)

    def fetch(self, source: HubEpisode) -> Path:
        """The episode's MCAP in the raw root, fetched and verified unless it is already there."""
        transports.hf_fetch_verified(REPO_ID, source.info, local_dir=self.config.root, revision=self.config.revision)
        return self.config.root / source.path

    def prefetch(self, identity: SequenceIdentity, source: HubEpisode, *, force: bool) -> None:
        """Fetch the next MCAP while the current one converts; must not touch self.timer, recordings, or print per-sequence progress."""
        if self.needs_mcap(identity, force=force):
            self.fetch(source)

    def convert(self, identity: SequenceIdentity, source: HubEpisode, *, force: bool) -> Path:
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        sidecar: Path = self.sidecar(identity)
        if self.needs_mcap(identity, force=force):
            with self.timer.stage("fetch"):
                mcap: Path = self.fetch(source)
            episode: Episode = read_episode(EpisodeFile(mcap))
            if force or not sidecar.is_file():
                with self.timer.stage("write:sidecar"):
                    write_sidecar(mcap, sidecar, recording_id=identity.recording_id)
            if paths.BASE_LAYER in pending:
                with self.timer.stage("write:base"):
                    census: BaseCensus = write_base_layer(
                        identity, mcap, targets[paths.BASE_LAYER], episode=episode, domain=source.domain, hf_revision=self.config.revision,
                        default_blueprint=self.default_blueprint(),
                    )  # fmt: skip
                print(f"{identity.sequence_key}: base {census.camera_frames} camera frames, {census.rig_poses} rig poses, {census.imu_samples} IMU samples")
            if not self.config.keep_raw:
                mcap.unlink()
        derived: list[str] = [layer for layer in pending if layer != paths.BASE_LAYER]
        file: EpisodeFile = EpisodeFile(sidecar)
        episode = read_episode(file)
        self.timer.capture_s = float(episode.clock.frame_times_ns[-1] - episode.clock.frame_times_ns[0]) / 1e9
        skipped: list[str] = []
        if paths.ACTIONS_LAYER in derived:
            with self.timer.stage(f"write:{paths.ACTIONS_LAYER}"):
                if not write_actions_layer(identity, episode, targets[paths.ACTIONS_LAYER]):
                    skipped.append(paths.ACTIONS_LAYER)
        if {POSE_LAYER, paths.PROJECTIONS_LAYER, paths.HAND_MESH_LAYER} & set(derived):
            track: PoseTrack = read_pose_track(file, episode.clock)
            if POSE_LAYER in derived:
                with self.timer.stage(f"write:{POSE_LAYER}"):
                    if not write_pose_layer(identity, track, targets[POSE_LAYER]):
                        skipped.append(POSE_LAYER)
            if paths.PROJECTIONS_LAYER in derived:
                with self.timer.stage(f"write:{paths.PROJECTIONS_LAYER}"):
                    rig_poses: Float64[ndarray, "f 4 4"] = read_rig_poses(file, episode.clock, track.frame_indices)
                    write_projections_layer(identity, track, rig_poses, episode.calibrations, targets[paths.PROJECTIONS_LAYER])
            if paths.HAND_MESH_LAYER in derived:
                with self.timer.stage(f"write:{paths.HAND_MESH_LAYER}"):
                    checks: tuple[ManoCheck, ManoCheck] = write_hand_mesh_layer(identity, track, targets[paths.HAND_MESH_LAYER])
                if not any(check.passed for check in checks):
                    skipped.append(paths.HAND_MESH_LAYER)
                errors_mm: list[str] = [f"{side} {check.median_m * 1000:.2f} mm{'' if check.passed else ' (no mesh)'}" for side, check in zip(("left", "right"), checks, strict=True)]
                print(f"{identity.sequence_key}: MANO joints vs shipped bones: {', '.join(errors_mm)}")
        if skipped:
            # A forced rebuild that finds nothing to publish must not leave an older file to be registered.
            for layer in skipped:
                targets[layer].unlink(missing_ok=True)
            print(f"{identity.sequence_key}: nothing to publish for {', '.join(skipped)}")
        print(f"done {identity.sequence_key}: {', '.join(layer for layer in pending if layer not in skipped)}")
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """The wearer in 3D over the task and the coarse and fine action lanes, beside the six cameras.

        Not ``blueprints.exoego_blueprint``: its worn-camera column fits one or two headset cameras, and
        six would shrink to slivers; its exo strip is for static cameras. The six fisheye panes get half
        the width in a two-column grid, read left to right and top to bottom in the cameras' yaw order.
        """
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name="DAS-Ego wearer", origin="/world", contents=world_contents(), eye_controls=SCENE_EYE),
                    rrb.TextDocumentView(name="Task", origin=schema.instruction_path(), contents=[schema.instruction_path()]),
                    rrb.StateTimelineView(name="Actions", origin="/task", contents=[schema.actions_path("coarse"), schema.actions_path("fine")]),
                    row_shares=[5, 0.7, 1.3],
                ),
                rrb.Grid(
                    *(blueprints.camera_view(f"camera{cam}", RIG, cam, contents=pane_contents(cam)) for cam in PANE_ORDER),
                    grid_columns=2,
                    name="Cameras, left to right",
                ),
                column_shares=[1, 1],
            ),
            rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
            collapse_panels=True,
        )

    def table_fields(self) -> TableFields:
        task: TableField = TableField("property:episode:task", "task")
        domain: tuple[TableField, ...] = (TableField("property:episode:domain_0", "domain"), TableField("property:episode:domain_1", "subdomain"))
        success: TableField = TableField("property:episode:is_success", "success")
        duration: TableField = TableField("property:capture:duration_s", "seconds")
        device: TableField = TableField("property:episode:device", "device")
        return TableFields(cards=(task, *domain, duration), table=(*domain, duration, success, device, task))

    def table_blueprint(self) -> rrb.Blueprint:
        """The wearer in 3D without any video, beside the reference camera with its projected skeleton."""
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Wearer",
                origin="/world",
                contents=[*world_contents(), *blueprints.video_exclusions((RIG, cam) for cam in range(NUM_CAMERAS))],
                eye_controls=SCENE_EYE,
            ),
            blueprints.camera_view(f"camera{REFERENCE_CAMERA}", RIG, REFERENCE_CAMERA, contents=pane_contents(REFERENCE_CAMERA)),
        )

