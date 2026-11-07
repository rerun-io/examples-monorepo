"""GenRobot DAS-Ego datasets: a pinned Hub listing, one MCAP at a time, layered rrds.

``DasEgoDataset`` is the shared converter; Gen-HumanWholeBody (``genrobot``, here) and Gen-HumanEgo
(``gen-humanego``, in ``gen_humanego``) are its two releases. Both corpora are too big to keep raw:
``download`` writes the pinned listing (``manifest.json``), and ``convert`` fetches one episode at a time
(the next one in the background), verifies it against its LFS sha256, writes the sidecar and the layers
that need the MCAP, and deletes it again. The derived layers read only the sidecar.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar, TypeVar

import rerun as rr
import rerun.blueprint as rrb
from serde import serde
from serde.json import to_json

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.genrobot_layers import RIG, SIDECAR_NAME, BaseCensus, write_base_layer, write_sidecar
from dataforge.datasets.genrobot_pose import (
    POSE_LAYER,
    ManoCheck,
    PoseTrack,
    check_mano,
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
    Annotation,
    Episode,
    EpisodeFile,
    parse_hub_path,
    read_episode,
)
from dataforge.identity import SequenceIdentity
from dataforge.records import read_json
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

T = TypeVar("T")
"""Item type of a sharded selection."""


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


def list_hub_episodes(repo_id: str, revision: str) -> Manifest:
    """Read the pinned listing from the Hub; every file under ``data/`` must be an episode."""
    episodes: list[HubEpisode] = []
    for file in sorted(transports.hf_list_files(repo_id, revision, "data"), key=lambda file: file.path):
        if file.lfs_sha256 is None:
            raise ValueError(f"{repo_id}@{revision} {file.path} is not an LFS file")
        parse_hub_path(file.path)
        episodes.append(HubEpisode(path=file.path, size_bytes=file.size_bytes, sha256=file.lfs_sha256))
    return Manifest(revision=revision, episodes=episodes)


def stratified_subset(episodes: Sequence[HubEpisode], budget_bytes: int) -> list[HubEpisode]:
    """A task-stratified sample that fits ``budget_bytes``: every task once before any task twice.

    The task is the full domain path (four levels on Gen-HumanEgo). Each round takes the next episode, in
    uuid order (uuids are random hex, so this is a random draw), of every task that has one left; tasks
    within a round go in the order of a hash of their path, so no domain is favoured by its name. The
    sample stops at the first episode that would pass the budget.

    Returns:
        The chosen episodes, in path order.
    """
    by_task: dict[tuple[str, ...], list[HubEpisode]] = defaultdict(list)
    for item in sorted(episodes, key=lambda item: item.uuid):
        by_task[item.domain].append(item)
    order: list[tuple[str, ...]] = sorted(by_task, key=lambda task: hashlib.sha256("/".join(task).encode()).hexdigest())
    chosen: list[HubEpisode] = []
    total: int = 0
    for round_index in range(max(len(items) for items in by_task.values())):
        for task in order:
            if round_index >= len(by_task[task]):
                continue
            item: HubEpisode = by_task[task][round_index]
            if total + item.size_bytes > budget_bytes:
                return sorted(chosen, key=lambda item: item.path)
            chosen.append(item)
            total += item.size_bytes
    return sorted(chosen, key=lambda item: item.path)


def shard(items: Sequence[T], index: int, count: int) -> list[T]:  # noqa: UP047 — beartype requires legacy generics
    """Every ``count``-th item from ``index``: ``count`` processes split a selection without overlap."""
    if not 0 <= index < count:
        raise ValueError(f"shard {index} of {count} does not exist")
    return list(items[index::count])


def require_published(targets: dict[str, Path], layers: Sequence[str]) -> None:
    """Check that every named layer file exists and holds bytes; the MCAP is deleted only after this passes.

    Raises:
        ValueError: A layer file is missing or empty.
    """
    for layer in layers:
        target: Path = targets[layer]
        if not target.is_file():
            raise ValueError(f"{target}: {layer} layer missing after conversion")
        if target.stat().st_size == 0:
            raise ValueError(f"{target}: {layer} layer is empty after conversion")


def pane_contents(cam: int, *, with_body: bool = True) -> list[str]:
    """One fisheye pane: its video and the 2D projected through its lens.

    The pane rule for a camera with a distortion model: show the lens-projected 2D, leave out
    ``coco133_xyz``, which a Rerun ``Pinhole`` would draw through the wrong (undistorted) projection.
    """
    return [schema.video_path(RIG, cam), schema.coco133_uv_projected_path(RIG, cam), *([smplx22_uv_projected_path(cam)] if with_body else [])]


def world_contents(*, with_body: bool = True) -> list[str]:
    """The 3D scene: rig, frustums with their video, body, hands and IMU node; no 2D projections on the image planes."""
    return [
        "+ /world/**",
        *(f"- {schema.coco133_uv_projected_path(RIG, cam)}" for cam in range(NUM_CAMERAS)),
        *(f"- {smplx22_uv_projected_path(cam)}" for cam in range(NUM_CAMERAS) if with_body),
    ]


@dataclass
class DasEgoConfig(DataforgeDatasetConfig):
    """Episodes of a DAS-Ego release, fetched from the Hub one at a time."""

    root: Path = field(default_factory=lambda: paths.raw_root() / "genrobot")
    """Local scratch: the manifest and the MCAPs being converted. Never the NAS."""
    sequences: tuple[str, ...] | None = None
    """Episode uuids to convert, added to the samples below."""
    per_domain: int | None = None
    """Sample this many episodes of every top-level domain, in uuid order (uuids are random hex)."""
    budget_tb: float | None = None
    """Sample every task in turn until the MCAPs reach this many terabytes (``stratified_subset``)."""
    shard_index: int = 0
    """Convert only this shard of the selection, so ``num_shards`` processes split it."""
    num_shards: int = 1
    """Processes the selection is split across."""
    keep_raw: bool = False
    """Keep each MCAP after its layers are published; by default it is deleted once every layer file is checked."""
    revision: str = REVISION
    """Pinned Hub commit; an override must be a full commit sha."""

    def __post_init__(self) -> None:
        transports.require_commit_sha(self.revision)
        shard([], self.shard_index, self.num_shards)


DasEgoConfigT = TypeVar("DasEgoConfigT", bound=DasEgoConfig)
"""Config of one DAS-Ego release."""


class DasEgoDataset(DataforgeDataset[DasEgoConfigT, HubEpisode]):
    """Fetch, verify, convert and delete one MCAP at a time; a release sets the class attributes."""

    repo_id: ClassVar[str] = REPO_ID
    """Hugging Face dataset repository."""
    with_body: ClassVar[bool] = True
    """Whether the release ships the SMPL-X body22 track."""
    mcap_layers: ClassVar[tuple[str, ...]] = (paths.BASE_LAYER,)
    """Layers written from the MCAP itself; every other layer reads the sidecar."""
    sidecar_exclusions: ClassVar[tuple[str, ...]] = ()
    """Non-camera topics kept out of the sidecar because a layer already holds them whole."""

    def manifest(self) -> Manifest:
        """The pinned listing ``download`` wrote."""
        path: Path = self.config.root / MANIFEST_NAME
        command: str = self.config.command
        if not path.is_file():
            raise FileNotFoundError(f"{path}: run dataforge-download {command} first")
        manifest: Manifest = read_json(path, Manifest)
        if manifest.revision != self.config.revision:
            raise ValueError(f"{path} lists {manifest.revision}, the config pins {self.config.revision}: run dataforge-download {command} again")
        return manifest

    def download(self) -> None:
        manifest: Manifest = list_hub_episodes(self.repo_id, self.config.revision)
        with writing.atomic_write(self.config.root / MANIFEST_NAME) as temp:
            temp.write_text(to_json(manifest))
        domains: Counter[str] = Counter(episode.domain[0] for episode in manifest.episodes)
        total_tb: float = sum(episode.size_bytes for episode in manifest.episodes) / 1e12
        print(f"{self.config.command}: {len(manifest.episodes)} episodes, {total_tb:.2f} TB, {len(domains)} top-level domains")
        print("  convert fetches one MCAP at a time (the next in the background) and deletes it once its layers are published")

    def discover(self) -> list[tuple[SequenceIdentity, HubEpisode]]:
        episodes: list[HubEpisode] = self.manifest().episodes
        config: DasEgoConfig = self.config
        chosen: list[HubEpisode] = episodes
        if config.per_domain is not None or config.sequences is not None or config.budget_tb is not None:
            wanted: set[str] = set(config.sequences or ())
            unknown: set[str] = wanted - {episode.uuid for episode in episodes}
            if unknown:
                raise ValueError(f"episodes absent from {config.revision}: {sorted(unknown)}")
            taken: Counter[str] = Counter()
            picked: dict[str, HubEpisode] = {}
            for episode in sorted(episodes, key=lambda episode: (episode.domain[0], episode.uuid)):
                in_sample: bool = config.per_domain is not None and taken[episode.domain[0]] < config.per_domain
                if in_sample or episode.uuid in wanted:
                    taken[episode.domain[0]] += in_sample
                    picked[episode.path] = episode
            if config.budget_tb is not None:
                picked.update((episode.path, episode) for episode in stratified_subset(episodes, int(config.budget_tb * 1e12)))
            chosen = sorted(picked.values(), key=lambda episode: episode.path)
        chosen = shard(chosen, config.shard_index, config.num_shards)
        return [(SequenceIdentity(config.name, (episode.uuid,)), episode) for episode in chosen]

    def sidecar(self, identity: SequenceIdentity) -> Path:
        """Where the episode's non-camera topics are kept."""
        return paths.sidecar_path(paths.output_root(), identity, SIDECAR_NAME)

    def needs_mcap(self, identity: SequenceIdentity, *, force: bool) -> bool:
        """Only the sidecar and ``mcap_layers`` read the MCAP."""
        targets: dict[str, Path] = self.targets(identity)
        return not self.sidecar(identity).is_file() or any(not writing.should_skip(targets[layer], force=force) for layer in self.mcap_layers)

    def fetch(self, source: HubEpisode) -> Path:
        """The episode's MCAP in the raw root, fetched and verified unless it is already there."""
        transports.hf_fetch_verified(self.repo_id, source.info, local_dir=self.config.root, revision=self.config.revision)
        return self.config.root / source.path

    def prefetch(self, identity: SequenceIdentity, source: HubEpisode, *, force: bool) -> None:
        """Fetch the next MCAP while the current one converts; must not touch self.timer, recordings, or print per-sequence progress."""
        if self.needs_mcap(identity, force=force):
            self.fetch(source)

    def write_mcap_layers(
        self, identity: SequenceIdentity, mcap: Path, targets: dict[str, Path], pending: Sequence[str], *, episode: Episode, source: HubEpisode
    ) -> None:
        """Publish the pending ``mcap_layers`` of one episode; the base layer here, a release adds its own."""
        if paths.BASE_LAYER not in pending:
            return
        with self.timer.stage("write:base"):
            census: BaseCensus = write_base_layer(
                identity, mcap, targets[paths.BASE_LAYER], episode=episode, domain=source.domain, hf_revision=self.config.revision,
                default_blueprint=self.default_blueprint(),
            )  # fmt: skip
        print(f"{identity.sequence_key}: base {census.camera_frames} camera frames, {census.rig_poses} rig poses, {census.imu_samples} IMU samples")

    def extra_writers(
        self, targets: dict[str, Path], pending: Sequence[str], file: EpisodeFile, episode: Episode
    ) -> dict[str, Callable[[rr.RecordingStream], None]]:
        """Writers of a release's own derived layers that have something to publish; none by default."""
        return {}

    def pose_writer(self, file: EpisodeFile, episode: Episode, track: PoseTrack) -> Callable[[rr.RecordingStream], None]:
        """The ``pose`` layer's writer; a release may add what only it ships."""
        return partial(write_pose_layer, track=track, with_body=self.with_body)

    def convert(self, identity: SequenceIdentity, source: HubEpisode, *, force: bool) -> Path:
        """Convert one episode; the MCAP is deleted only once every layer is published and checked.

        A failure after the fetch deletes the episode's layers and sidecar, so nothing partial is registered,
        and keeps the MCAP for the retry.
        """
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        sidecar: Path = self.sidecar(identity)
        if not pending and sidecar.is_file():
            return targets[paths.BASE_LAYER]
        if not self.needs_mcap(identity, force=force):
            return self.convert_fetched(identity, source, None, targets, pending, force=force)
        with self.timer.stage("fetch"):
            mcap: Path = self.fetch(source)
        try:
            base: Path = self.convert_fetched(identity, source, mcap, targets, pending, force=force)
        except BaseException:
            for target in (*targets.values(), sidecar):
                target.unlink(missing_ok=True)
            raise
        if not self.config.keep_raw:
            mcap.unlink()
        return base

    def convert_fetched(
        self, identity: SequenceIdentity, source: HubEpisode, mcap: Path | None, targets: dict[str, Path], pending: list[str], *, force: bool
    ) -> Path:
        """Write the sidecar and the MCAP layers when ``mcap`` is given, then the derived layers; check every layer written."""
        sidecar: Path = self.sidecar(identity)
        if mcap is not None and (force or not sidecar.is_file()):
            with self.timer.stage("write:sidecar"):
                write_sidecar(mcap, sidecar, recording_id=identity.recording_id, excluded_topics=self.sidecar_exclusions)
        file: EpisodeFile = EpisodeFile(sidecar)
        episode: Episode = read_episode(file)  # once, from the sidecar: the MCAP layers and every derived one
        self.timer.capture_s = episode.clock.duration_s
        if mcap is not None:
            self.write_mcap_layers(identity, mcap, targets, [layer for layer in pending if layer in self.mcap_layers], episode=episode, source=source)
        # A derived layer gets a writer only when it has something to publish.
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {}
        annotation: Annotation | None = episode.annotation
        if annotation is not None and annotation.segments_info:
            writers[paths.ACTIONS_LAYER] = partial(write_actions_layer, clock=episode.clock, annotation=annotation)
        if {POSE_LAYER, paths.PROJECTIONS_LAYER, paths.HAND_MESH_LAYER} & set(pending):
            track: PoseTrack = read_pose_track(file, episode.clock)
            if track.has_pose:
                writers[POSE_LAYER] = self.pose_writer(file, episode, track)
            writers[paths.PROJECTIONS_LAYER] = lambda recording: write_projections_layer(
                recording, track, read_rig_poses(file, episode.clock, track.frame_indices), episode.calibrations, with_body=self.with_body
            )
            if paths.HAND_MESH_LAYER in pending:
                with self.timer.stage("check:mano"):
                    checks: tuple[ManoCheck, ManoCheck] = (check_mano(track, 0), check_mano(track, 1))
                errors_mm: list[str] = [
                    f"{side} {check.median_m * 1000:.2f} mm{' (root mirrored)' if check.mirror_root else ''}{'' if check.passed else ' (no mesh)'}"
                    for side, check in zip(("left", "right"), checks, strict=True)
                ]
                print(f"{identity.sequence_key}: MANO joints vs shipped bones: {', '.join(errors_mm)}")
                if any(check.passed for check in checks):
                    writers[paths.HAND_MESH_LAYER] = partial(write_hand_mesh_layer, track=track, checks=checks)
        writers.update(self.extra_writers(targets, pending, file, episode))
        derived: list[str] = [layer for layer in pending if layer not in self.mcap_layers]
        skipped: list[str] = [layer for layer in derived if layer not in writers]
        if skipped:
            # A forced rebuild that finds nothing to publish must not leave an older file to be registered.
            for layer in skipped:
                targets[layer].unlink(missing_ok=True)
            print(f"{identity.sequence_key}: nothing to publish for {', '.join(skipped)}")
        self.write_layers(identity, targets, [layer for layer in derived if layer in writers], writers)
        written: list[str] = [layer for layer in pending if layer not in skipped and targets[layer].exists()]
        require_published({**targets, "sidecar": sidecar}, [*dict.fromkeys([paths.BASE_LAYER, *written]), "sidecar"])  # base is never optional
        print(f"done {identity.sequence_key}: {', '.join(written)}")
        return targets[paths.BASE_LAYER]

    def world_contents(self) -> list[str]:
        """What the 3D views show."""
        return world_contents(with_body=self.with_body)

    def pane_contents(self, cam: int) -> list[str]:
        """What one fisheye pane shows."""
        return pane_contents(cam, with_body=self.with_body)

    def default_blueprint(self) -> rrb.Blueprint:
        """The wearer in 3D over the task and the coarse and fine action lanes, beside the six cameras.

        Not ``blueprints.exoego_blueprint``: its worn-camera column fits one or two headset cameras, and
        six would shrink to slivers; its exo strip is for static cameras. The six fisheye panes get half
        the width in a two-column grid, read left to right and top to bottom in the cameras' yaw order.
        """
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name="DAS-Ego wearer", origin="/world", contents=self.world_contents(), eye_controls=SCENE_EYE),
                    rrb.TextDocumentView(name="Task", origin=schema.instruction_path(), contents=[schema.instruction_path()]),
                    rrb.StateTimelineView(name="Actions", origin="/task", contents=[schema.actions_path("coarse"), schema.actions_path("fine")]),
                    row_shares=[5, 0.7, 1.3],
                ),
                rrb.Grid(
                    *(blueprints.camera_view(f"camera{cam}", RIG, cam, contents=self.pane_contents(cam)) for cam in PANE_ORDER),
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
                contents=[*self.world_contents(), *blueprints.video_exclusions((RIG, cam) for cam in range(NUM_CAMERAS))],
                eye_controls=SCENE_EYE,
            ),
            blueprints.camera_view(f"camera{REFERENCE_CAMERA}", RIG, REFERENCE_CAMERA, contents=self.pane_contents(REFERENCE_CAMERA)),
        )


@dataclass
class GenrobotConfig(DasEgoConfig):
    """Gen-HumanWholeBody episodes, fetched from the Hub one at a time."""

    command: ClassVar[str] = "genrobot"
    """Registry key and catalog dataset."""
    _target: type = field(default_factory=lambda: GenrobotDataset)
    """Dataset instantiated by setup."""


class GenrobotDataset(DasEgoDataset[GenrobotConfig]):
    """Gen-HumanWholeBody: body22 skeleton, hands, MANO, actions."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, POSE_LAYER, paths.ACTIONS_LAYER, paths.PROJECTIONS_LAYER, paths.HAND_MESH_LAYER)
    """Publication and loading order."""
