"""GSI Egocentric Six-View: download through GSI's API, and repack its shipped rrds as dataforge layers.

GSI ships every episode as two Rerun 0.37 rrds under one recording id (application ``gsi_off_the_shelf``, recording
id = episode id): ``hands.rrd`` holds six H.265 camera streams with their pinholes and rig extrinsics, both hands'
MANO meshes, keypoints and parameters, a calibration blob and the episode properties; ``slam.rrd`` holds the rig
trajectory that poses ``/camera`` and ``/hand`` in the world. That is already a base and a derived layer, so convert
decodes nothing: it streams each file's chunks into a dataforge recording id, compacts them for the catalog, and
keeps GSI's entity paths, timelines and samples as shipped. See docs/gsi-six-view.md.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from rerun.encodings import EntityPathLike

from dataforge import hands, paths, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig, RemoteSequence
from dataforge.datasets.gsi_six_view_remote import API_URL, GsiSession, Manifest, ManifestEpisode, fetch_episodes, fetch_manifest, read_manifest
from dataforge.identity import SequenceIdentity

SLAM_LAYER: str = "slam"
"""GSI's ``slam.rrd``: the rig trajectory, stacked onto base."""
FILES: dict[str, str] = {paths.BASE_LAYER: "hands.rrd", SLAM_LAYER: "slam.rrd"}
"""The shipped file each layer repacks."""
CAMERAS: tuple[str, ...] = ("left_eye", "right_eye", "left_front", "right_front", "left_side", "right_side")
"""The six views, in the pairs the default layout shows them: eye, front, side."""
SIDES: tuple[hands.Side, ...] = ("left", "right")
"""GSI's two hands, ``/hand/<side>``."""
TIMELINE: str = "capture_time"
"""GSI's device clock; ``frame`` and ``source_frame`` index the same samples."""
KEPT: list[str] = ["+ /**", "+ /__properties/episode", "- /__properties"]
"""Everything a file ships except its ``RecordingInfo`` name: the base takes a dataforge name, and a derived layer restates no properties."""
OPTIMIZE: rrc.OptimizationProfile = dataclasses.replace(rrc.OptimizationProfile.OBJECT_STORE, fix_keyframe=True)
"""Catalog compaction. GSI marks every non-key video sample ``is_keyframe=false``, which Rerun 0.42 rejects (the flag is a
sparse marker) and then leaves the video un-batched; ``fix_keyframe`` derives the flags from the H.265 samples instead."""


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


class GsiSixViewDataset(DataforgeDataset[GsiSixViewConfig, Path]):
    """One catalog dataset; one segment per episode."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, SLAM_LAYER)
    timeline: ClassVar[str] = TIMELINE

    @property
    def manifest_path(self) -> Path:
        """Where ``download`` keeps the edition manifest."""
        return self.config.root / "manifest.json"

    def download(self) -> None:
        """Fetch the manifest, then every selected episode that is neither on disk nor converted."""
        session: GsiSession = GsiSession.sign_in(self.config.api_url)
        fetch_manifest(session, self.config.edition, self.manifest_path)
        episodes: list[str] = [episode.episode_id for episode in read_manifest(self.manifest_path).episodes]
        selected: list[str] = episodes if self.config.sequences is None else list(dict.fromkeys(self.config.sequences))
        unknown: list[str] = sorted(set(selected) - set(episodes))
        if unknown:
            raise ValueError(f"{self.config.command}: the edition has no episode(s) {unknown}; see `dataforge-download {self.config.command} --list-remote`")
        wanted: set[str] = {episode for episode in selected if not all(target.is_file() for target in self.targets(self.identity(episode)).values())}
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
                tuple(f"episodes/{episode.episode_id}/{name}" for name in FILES.values()),
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
            if (selected is None or directory.name in selected) and all((directory / name).is_file() for name in FILES.values())
        ]

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Repack each pending layer, then delete the episode's download once both layers are published."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        manifest: Manifest = read_manifest(self.manifest_path)
        episode: ManifestEpisode = next(entry for entry in manifest.episodes if entry.episode_id == identity.parts[0])
        self.timer.capture_s = episode.duration_s
        for layer in pending:
            with self.timer.stage(f"write:{layer}"):
                self.repack(identity, source / FILES[layer], targets[layer], manifest if layer == paths.BASE_LAYER else None, episode)
        if all(target.is_file() for target in targets.values()):
            for name in FILES.values():
                (source / name).unlink(missing_ok=True)
            source.rmdir()
        return targets[paths.BASE_LAYER]

    def repack(self, identity: SequenceIdentity, shipped: Path, target: Path, manifest: Manifest | None, episode: ManifestEpisode) -> None:
        """Write ``shipped``'s chunks, compacted, to ``target`` under the dataforge recording id; ``manifest`` marks the base.

        The base keeps GSI's ``RecordingInfo`` start time, the capture's own, and adds the dataforge name, the capture
        properties and the embedded default blueprint.
        """
        compacted: rrc.ChunkStore = rrc.RrdReader(shipped).stream().filter(content=KEPT).collect(optimize=OPTIMIZE)
        base: bool = manifest is not None
        with (
            writing.atomic_write(target) as temp_path,
            writing.recording_to(temp_path, recording_id=identity.recording_id, default_blueprint=self.default_blueprint() if base else None, send_properties=False) as recording,
        ):
            recording.send_chunks(compacted)
            if manifest is not None:
                recording.send_chunks(rrc.RrdReader(shipped).stream().filter(content="/__properties", components="RecordingInfo:start_time"))
                frames: int = int(compacted.reader(index=None, contents="/__properties/episode").to_arrow_table().column("property:episode:n_frames")[0][0].as_py())
                writing.send_capture_properties(
                    recording,
                    identity,
                    num_cameras=episode.views,
                    num_frames=frames,
                    category=episode.category,
                    gsi_edition=self.config.edition,
                    gsi_source_release=manifest.source_release,
                    gsi_rrd_contract=manifest.rrd_contract,
                    gsi_privacy_status=manifest.privacy_status,
                    hands_rrd_sha256=episode.hands_rrd.sha256,
                    slam_rrd_sha256=episode.slam_rrd.sha256,
                )

    def default_blueprint(self) -> rrb.Blueprint:
        """The world with the six camera frustums beside the six views in eye, front and side pairs, each with the hands projected."""
        frustums: dict[EntityPathLike, rr.Pinhole | rr.Mesh3D] = {f"/camera/{camera}/image": rr.Pinhole.from_fields() for camera in CAMERAS}
        world: rrb.Spatial3DView = rrb.Spatial3DView(name="World", origin="/", contents=["+ /**"], overrides=frustums | see_through_hands())
        views: rrb.Grid = rrb.Grid(*(camera_pane(camera) for camera in CAMERAS), grid_columns=2)
        return rrb.Blueprint(rrb.Horizontal(world, views, column_shares=[2, 3]), rrb.TimePanel(timeline=TIMELINE, state="collapsed"), collapse_panels=True)

    def table_blueprint(self) -> rrb.Blueprint:
        """The hands in the world beside the left front view: the one stream a card decodes."""
        world: rrb.Spatial3DView = rrb.Spatial3DView(name="World", origin="/", contents=["+ /hand/**", "+ /slam/**"], overrides=see_through_hands())
        return rrb.Blueprint(rrb.Horizontal(world, camera_pane("left_front")), collapse_panels=True)

    def table_fields(self) -> writing.TableFields:
        shown: tuple[writing.TableField, ...] = (
            writing.TableField("property:capture:category", "category"),
            writing.TableField("property:episode:caption", "caption"),
            writing.TableField("property:episode:duration_s", "seconds"),
        )
        return writing.TableFields(cards=shown, table=shown)


def camera_pane(camera: str) -> rrb.Spatial2DView:
    """One view's video with both hands' meshes and keypoints projected through its pinhole.

    The pane is rooted at the pinhole, so the hands, which hang off the same rig as ``/camera``, land in image space.
    """
    hand_parts: list[str] = [f"+ /hand/{side}/{part}" for side in SIDES for part in ("vertices", "keypoints3d")]
    return rrb.Spatial2DView(name=camera, origin=f"/camera/{camera}/image", contents=[f"+ /camera/{camera}/image", *hand_parts], overrides=see_through_hands())


def see_through_hands() -> dict[EntityPathLike, rr.Pinhole | rr.Mesh3D]:
    """Display-only translucent albedo for GSI's opaque MANO meshes, so the keypoints show through; the data stays as shipped."""
    return {f"/hand/{side}/vertices": rr.Mesh3D.from_fields(albedo_factor=hands.HAND_ALBEDO[side]) for side in SIDES}
