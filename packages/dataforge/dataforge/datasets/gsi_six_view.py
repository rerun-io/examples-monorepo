"""GSI Egocentric Six-View: download through GSI's API, and repack its shipped rrds into the exoego:v2 layout.

GSI ships every episode as two Rerun 0.37 rrds under one recording id (application ``gsi_off_the_shelf``, recording
id = episode id): ``hands.rrd`` holds six H.265 camera streams with their pinholes and rig extrinsics, both hands'
MANO meshes, keypoints and parameters, a calibration blob and the episode properties; ``slam.rrd`` holds the rig
trajectory. Convert decodes no video. Lenses move the cameras, pinholes, videos and the SLAM pose onto
``/world/rig_00`` and turn GSI's FLU camera frames into RDF. The hands need each frame's SLAM pose to reach the world
frame, so they are read column-wise and written through the shared hand helpers. See docs/gsi-six-view.md.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig, RemoteSequence
from dataforge.datasets.gsi_six_view_layers import (
    POSE_LAYER,
    trail_overrides,
    write_base_layer,
    write_hand_mesh_layer,
    write_pose_layer,
    write_projections_layer,
)
from dataforge.datasets.gsi_six_view_remote import API_URL, GsiSession, Manifest, ManifestEpisode, fetch_episodes, fetch_manifest, read_manifest
from dataforge.datasets.gsi_six_view_source import (
    FILES,
    RIG,
    SIDES,
    VIEWS,
    Episode,
    read_episode,
)
from dataforge.identity import SequenceIdentity

PANE_ORDER: tuple[int, ...] = (2, 3, 0, 1, 4, 5)
"""The panes in eye, front and side pairs, left before right."""
FOLLOW_EYE: rrb.EyeControls3D = blueprints.headset_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
"""A chase eye in the rig frame (left_front's RDF frame: forward +z, up -y), aimed down at the hands."""


@dataclass
class GsiSixViewConfig(DataforgeDatasetConfig):
    """GSI's partner edition of Egocentric Six-View, fetched with the partner sign-in in ``GSI_PARTNER_USERNAME``/``GSI_PARTNER_PASSWORD``."""

    command: ClassVar[str] = "gsi-six-view"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: GsiSixViewDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "gsi-six-view")
    """Download root: ``manifest.json`` and ``episodes/<episode_id>/{hands,slam}.rrd`` until each episode is converted."""
    sequences: tuple[str, ...] | None = None
    """Episode ids to download and convert; None selects the whole edition."""
    edition: str = "01a0f166-7994-7b43-a622-a325541c4356"
    """GSI edition id: Egocentric Six-View, 10 h, release 2026-09-v2."""
    api_url: str = API_URL
    """GSI's API."""
    keep_download: bool = False
    """Keep each episode's download once its layers are published, for another reader of the same download root."""
    skip_converted: bool = False
    """Fetch no episode whose layers are all published: a whole-edition top-up after converts deleted their downloads.
    Without it, download fetches every requested episode whose files are not on disk, so ``--force`` can rebuild one."""



class GsiSixViewDataset(DataforgeDataset[GsiSixViewConfig, Path]):
    """One catalog dataset in the exoego:v2 layout; one segment per episode."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, POSE_LAYER, paths.HAND_MESH_LAYER, paths.PROJECTIONS_LAYER)
    optimized_layers: ClassVar[tuple[str, ...]] = (POSE_LAYER, paths.PROJECTIONS_LAYER)

    @property
    def manifest_path(self) -> Path:
        """Where ``download`` keeps the edition manifest."""
        return self.config.root / "manifest.json"

    @cached_property
    def manifest(self) -> Manifest:
        """The edition manifest ``download`` fetched, parsed once per run: convert reads it for every episode."""
        return read_manifest(self.manifest_path)

    @cached_property
    def manifest_episodes(self) -> dict[str, ManifestEpisode]:
        """The manifest's episodes by id."""
        return {episode.episode_id: episode for episode in self.manifest.episodes}

    def download(self) -> None:
        """Fetch the manifest, then every selected episode's files not already on disk (and, with ``skip_converted``, not yet converted)."""
        session: GsiSession = GsiSession.sign_in(self.config.api_url)
        fetch_manifest(session, self.config.edition, self.manifest_path)
        episodes: list[str] = [episode.episode_id for episode in read_manifest(self.manifest_path).episodes]
        selected: list[str] = episodes if self.config.sequences is None else list(dict.fromkeys(self.config.sequences))
        unknown: list[str] = sorted(set(selected) - set(episodes))
        if unknown:
            raise ValueError(f"{self.config.command}: the edition has no episode(s) {unknown}; see `dataforge-download {self.config.command} --list-remote`")
        wanted: set[str] = {
            episode
            for episode in selected
            if not (self.config.skip_converted and all(target.is_file() for target in self.targets(self.identity(episode)).values()))
        }
        print(f"{self.config.command}: {len(wanted)} of {len(selected)} selected episode(s) still to fetch into {self.config.root}")
        if wanted:
            report: transports.FetchReport = fetch_episodes(session, self.config.edition, wanted, lambda episode, name: self.config.root / "episodes" / episode / name)
            print(f"{self.config.command}: {report.summary()}")

    def remote_sequences(self) -> list[RemoteSequence]:
        """Every episode of the manifest, fetched first when it is not on disk."""
        if not self.manifest_path.is_file():
            fetch_manifest(GsiSession.sign_in(self.config.api_url), self.config.edition, self.manifest_path)
        return [
            RemoteSequence(
                episode.episode_id,
                episode.hands_rrd.size_bytes + episode.slam_rrd.size_bytes,
                tuple(f"episodes/{episode.episode_id}/{name}" for name in FILES),
            )
            for episode in read_manifest(self.manifest_path).episodes
        ]

    def identity(self, episode_id: str) -> SequenceIdentity:
        return SequenceIdentity(self.config.name, (episode_id,))

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        """Every downloaded episode holding both files; a converted one has none left."""
        selected: set[str] | None = None if self.config.sequences is None else set(self.config.sequences)
        return [
            (self.identity(directory.name), directory)
            for directory in sorted((self.config.root / "episodes").glob("*"))
            if (selected is None or directory.name in selected) and all((directory / name).is_file() for name in FILES)
        ]

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write each pending layer, then delete the episode's download once every layer is published (unless ``keep_download``)."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        episode: ManifestEpisode = self.manifest_episodes[identity.parts[0]]
        self.timer.capture_s = episode.duration_s
        if pending:
            self.write_episode(identity, source, targets, pending, self.manifest, episode)
        if not self.config.keep_download and all(target.is_file() for target in targets.values()):
            for name in FILES:
                (source / name).unlink(missing_ok=True)
            source.rmdir()
        return targets[paths.BASE_LAYER]

    def write_episode(
        self, identity: SequenceIdentity, source: Path, targets: dict[str, Path], pending: list[str], manifest: Manifest, episode: ManifestEpisode
    ) -> None:
        """The base layer from GSI's cameras and SLAM rig, then the hand layers from one column-wise read."""
        hands_rrd, slam_rrd = (source / name for name in FILES)
        with self.timer.stage("read"):
            track: Episode = read_episode(hands_rrd, slam_rrd)
        if paths.BASE_LAYER in pending:
            with self.timer.stage(f"write:{paths.BASE_LAYER}"):
                self.write_base(identity, hands_rrd, slam_rrd, targets[paths.BASE_LAYER], manifest, episode, track)
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            POSE_LAYER: lambda recording: write_pose_layer(recording, track),
            paths.HAND_MESH_LAYER: lambda recording: write_hand_mesh_layer(recording, track),
            paths.PROJECTIONS_LAYER: lambda recording: write_projections_layer(recording, track),
        }
        self.write_layers(identity, targets, [layer for layer in pending if layer != paths.BASE_LAYER], writers)

    def write_base(
        self, identity: SequenceIdentity, hands_rrd: Path, slam_rrd: Path, target: Path, manifest: Manifest, episode: ManifestEpisode, track: Episode
    ) -> None:
        """The base layer (``write_base_layer``), outside ``write_layers`` so that it keeps GSI's ``RecordingInfo`` start time, the capture's own."""
        with writing.atomic_recording(target, recording_id=identity.recording_id, default_blueprint=self.default_blueprint(), send_properties=False) as recording:
            write_base_layer(recording, identity, hands_rrd, slam_rrd, track, manifest=manifest, episode=episode, edition=self.config.edition)

    def world_contents(self) -> list[str]:
        """The rig with its frustums and videos, the SLAM path, the hands; no 2D projections on the image planes."""
        return ["+ /world/**", *(f"- {schema.coco133_uv_projected_path(RIG, cam)}" for cam in range(len(VIEWS)))]

    def pane_contents(self, cam: int) -> list[str]:
        """A view's video, its projected keypoints, and both meshes drawn live through its pinhole."""
        return [schema.video_path(RIG, cam), schema.coco133_uv_projected_path(RIG, cam), *(schema.hand_mesh_path(side) for side in SIDES)]

    def default_blueprint(self) -> rrb.Blueprint:
        """The wearer's hands and path in 3D, riding the rig, beside the six views in eye, front and side pairs."""
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="GSI six-view", origin=schema.rig_path(RIG), contents=self.world_contents(), eye_controls=FOLLOW_EYE, line_grid=False, overrides=trail_overrides()
                ),
                rrb.Grid(
                    *(blueprints.camera_view(VIEWS[cam], RIG, cam, contents=self.pane_contents(cam)) for cam in PANE_ORDER),
                    grid_columns=2,
                    name="Cameras: eye, front, side",
                ),
                column_shares=[2, 3],
            ),
            rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
            collapse_panels=True,
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """The hands in 3D without any video, beside left_front with its hands: the one stream a card decodes."""
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Hands",
                origin=schema.rig_path(RIG),
                contents=[*self.world_contents(), *blueprints.video_exclusions((RIG, cam) for cam in range(len(VIEWS)))],
                eye_controls=FOLLOW_EYE,
                line_grid=False,
                overrides=trail_overrides(),
            ),
            blueprints.camera_view(VIEWS[0], RIG, 0, contents=self.pane_contents(0)),
        )

    def table_fields(self) -> writing.TableFields:
        shown: tuple[writing.TableField, ...] = (
            writing.TableField("property:capture:category", "category"),
            writing.TableField("property:episode:caption", "caption"),
            writing.TableField("property:episode:duration_s", "seconds"),
        )
        return writing.TableFields(cards=shown, table=shown)

