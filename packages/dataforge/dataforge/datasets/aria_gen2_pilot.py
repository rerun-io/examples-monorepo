"""Aria Gen2 Pilot v1.0: the release fetched from Meta's CDN (or verified in place) and three native-clock layers."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb
from serde import serde
from serde.json import to_json

from dataforge import blueprints, meta_cdn, paths, schema, transports, writing
from dataforge.datasets.aria_gen2_pilot_layers import write_base, write_hands, write_projections
from dataforge.datasets.aria_gen2_pilot_source import CAMERAS, SEQUENCES, Scene, read_scene
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.identity import SequenceIdentity
from dataforge.records import read_json
from dataforge.transports import FileIntegrity, matches_file


@serde
@dataclass(frozen=True, slots=True)
class SourceFile:
    """The release manifest's canonical main VRS integrity fields."""

    file_size_bytes: int
    """Expected byte length."""
    sha1sum: str
    """Upstream digest."""


@serde
@dataclass(frozen=True, slots=True)
class SequenceFiles:
    """Only the download types this port reads (``FETCHED``); depth, scene, HOI and the rest stay upstream."""

    main_vrs: SourceFile
    """Lands as ``video.vrs`` (the official layout's name)."""
    mps_slam_trajectories: SourceFile
    """Zip holding ``closed_loop_trajectory.csv``."""
    mps_hand_tracking: SourceFile
    """Zip holding ``hand_tracking_results.csv``."""


@serde
@dataclass(frozen=True, slots=True)
class Manifest:
    """Release inventory ``download`` writes beside the sequences: sizes and SHA-1s, never a URL."""

    sequences: dict[str, SequenceFiles]
    """Integrity metadata keyed by sequence name."""


MANIFEST_FILE: str = "AriaGen2PilotDataset_manifest.json"
"""The inventory's name in the raw root."""
FETCHED: dict[str, str] = {
    "main_vrs": "video.vrs",
    "mps_slam_trajectories": "mps/slam/closed_loop_trajectory.csv",
    "mps_hand_tracking": "mps/hand_tracking/hand_tracking_results.csv",
}
"""Each fetched download type and the sequence-relative file the converter reads from it (for a zip: the one member
extracted, into the official layout's folder)."""
CDN: meta_cdn.CdnSource = meta_cdn.CdnSource(
    "Aria Gen2 Pilot", "download a fresh URL file from https://www.projectaria.com/datasets/gen2pilot/ (links expire after about a month)"
)
"""Names the dataset in every CDN message."""


def read_url_file(path: Path | None) -> meta_cdn.UrlFile:
    """The ``--url-file`` a user downloaded from projectaria.com, refusing another dataset's."""
    urls: meta_cdn.UrlFile = meta_cdn.read_url_file(meta_cdn.url_file_at(path, "--url-file", CDN), CDN)
    if urls.sequence_config.dataset_name != "AriaGen2PilotDataset":
        raise ValueError(f"the URL file is {urls.sequence_config.dataset_name}'s; aria_gen2_pilot needs AriaGen2PilotDataset_download_urls*.json")
    return urls


def manifest_of(urls: meta_cdn.UrlFile) -> Manifest:
    """The URL file's fetched types without their URLs."""
    return Manifest(
        sequences={
            name: SequenceFiles(**{kind: SourceFile(files[kind].file_size_bytes, files[kind].sha1sum) for kind in FETCHED})
            for name, files in sorted(urls.sequences.items())
        }
    )


RIG_FORWARD: tuple[float, float, float] = (0.34, -0.27, 0.9)
"""Headset forward in the rig frame: camera-rgb's optical axis (measured from its shipped pose)."""
RIG_UP: tuple[float, float, float] = (0.05, -0.95, -0.3)
"""Headset up in the rig frame: minus camera-rgb's image y axis."""
FOLLOW_EYE: rrb.EyeControls3D = blueprints.headset_eye_controls(RIG_FORWARD, RIG_UP)
"""The 3D views' eye in the rig frame, riding the headset; default and table layouts."""


@dataclass
class AriaGen2PilotConfig(FrameLimitedConfig):
    """Source selection; ``download`` writes only beneath ``root``, convert output follows DATAFORGE_OUTPUT_ROOT."""

    command: ClassVar[str] = "aria_gen2_pilot"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: AriaGen2PilotDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "aria_gen2_pilot")
    """Release root: ``MANIFEST_FILE`` and one directory per sequence (the source may live on the NAS)."""
    sequences: tuple[str, ...] | None = None
    """Subset of the 12 release names; None downloads every one and discovers all complete local sources."""
    url_file: Path | None = None
    """``AriaGen2PilotDataset_download_urls*.json`` from projectaria.com: ``download`` fetches with it; discovery and
    convert never need it."""


class AriaGen2PilotDataset(DataforgeDataset[AriaGen2PilotConfig, Path]):
    """One recording identity across sensors, measured hands and derived pixels."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, paths.PROJECTIONS_LAYER)

    def manifest(self) -> Manifest:
        """The release inventory ``download`` wrote beside the sequences."""
        path: Path = self.config.root / MANIFEST_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no {MANIFEST_FILE} in {self.config.root}; run `dataforge-download aria_gen2_pilot --url-file <URL file>` first")
        return read_json(path, Manifest)

    def remote_sequences(self) -> list[RemoteSequence]:
        """Each release sequence with the bytes ``download`` fetches for it and the files it leaves."""
        manifest: Manifest = manifest_of(read_url_file(self.config.url_file)) if self.config.url_file is not None else self.manifest()
        return [
            RemoteSequence(
                key=name,
                size_bytes=sum(getattr(files, kind).file_size_bytes for kind in FETCHED),
                files=tuple(f"{name}/{relative}" for relative in FETCHED.values()),
            )
            for name, files in manifest.sequences.items()
        ]

    def discover(self, manifest: Manifest | None = None) -> list[tuple[SequenceIdentity, Path]]:
        """Find all complete release sequences without traversing _simplecv."""
        manifest = self.manifest() if manifest is None else manifest
        selected: tuple[str, ...] = SEQUENCES if self.config.sequences is None else self.config.sequences
        unknown: set[str] = set(selected) - set(SEQUENCES)
        if unknown:
            raise ValueError(f"{self.config.command}: unknown sequence(s) {sorted(unknown)}; the release has {list(SEQUENCES)}")
        found: list[tuple[SequenceIdentity, Path]] = []
        for name in selected:
            source: Path = self.config.root / name
            required: list[Path] = [source / relative for relative in FETCHED.values()]
            missing: list[Path] = [path for path in required if not path.is_file()]
            if missing:
                if self.config.sequences is not None:
                    raise FileNotFoundError(f"{name}: missing {missing[0]}")
                print(f"skip {name}: missing {missing[0]}")
                continue
            if name not in manifest.sequences:
                raise ValueError(f"{name}: absent from the release manifest")
            if required[0].stat().st_size != manifest.sequences[name].main_vrs.file_size_bytes:
                raise ValueError(f"{required[0]}: size differs from release manifest")
            found.append((SequenceIdentity(self.config.name, (name,)), source))
        return found

    def download(self) -> None:
        """With ``url_file``: write the manifest, then fetch what convert reads; rerunning resumes and skips what is there.

        Without it: verify each local ``video.vrs`` against the manifest's size and SHA-1, changing nothing.
        """
        if self.config.url_file is not None:
            self.fetch(read_url_file(self.config.url_file))
            return
        manifest: Manifest = self.manifest()
        found: list[tuple[SequenceIdentity, Path]] = self.discover(manifest)
        for _, source in found:
            listed: SourceFile = manifest.sequences[source.name].main_vrs
            # discover() already checked the size, so only the SHA-1 can differ.
            if not matches_file(source / "video.vrs", FileIntegrity(listed.file_size_bytes, "sha1", listed.sha1sum)):
                raise ValueError(f"{source}/video.vrs: SHA-1 differs from release manifest")
        print(f"aria_gen2_pilot v1.0: verified {len(found)} local sequences")

    def fetch(self, urls: meta_cdn.UrlFile) -> None:
        """The selected sequences' main VRS and the one needed member of each MPS zip, SHA-1-checked as they land."""
        selected: list[str] = sorted(urls.sequences) if self.config.sequences is None else sorted(set(self.config.sequences))
        unknown: list[str] = [name for name in selected if name not in urls.sequences]
        if unknown:
            raise ValueError(f"aria_gen2_pilot: unknown sequence(s) {unknown} in the URL file; see `dataforge-download aria_gen2_pilot --list-remote`")
        with writing.atomic_write(self.config.root / MANIFEST_FILE) as temp_path:
            temp_path.write_text(to_json(manifest_of(urls)))
        report: transports.FetchReport = transports.FetchReport()
        for name in selected:
            folder: Path = self.config.root / name
            files: dict[str, meta_cdn.CdnFile] = urls.sequences[name]
            meta_cdn.fetch_file(files["main_vrs"], folder / FETCHED["main_vrs"], report, CDN)
            for kind in ("mps_slam_trajectories", "mps_hand_tracking"):
                member: Path = Path(FETCHED[kind])
                meta_cdn.fetch_zip(files[kind], [member.name], [member.name], folder / member.parent, report, CDN)
            print(f"  {name}: done")
        print(f"aria_gen2_pilot: {len(selected)} sequence(s); {report.summary()} → {self.config.root}")

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write atomic local layers and never follow raw symlinks for output."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root, source])
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("fetch"):
            scene: Scene = read_scene(source, self.config.frame_limit)
        self.timer.capture_s = (
            max(int(camera.times_ns[-1]) for camera in scene.cameras) - min(int(camera.times_ns[0]) for camera in scene.cameras)
        ) / 1e9
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: write_base(recording, scene, identity, self.timer),
            paths.HAND_POSE_LAYER: lambda recording: write_hands(recording, scene),
            paths.PROJECTIONS_LAYER: lambda recording: write_projections(recording, scene),
        }
        self.write_layers(identity, targets, pending, writers)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """The 3D view follows the headset, so walking sequences keep it and the hands in shot."""
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Aria Gen2 (follows the headset)",
                origin=schema.rig_path(0),
                contents=["/world/**"],
                eye_controls=FOLLOW_EYE,
            ),
            ego_panes=[
                blueprints.camera_view(label, 0, index, contents=[schema.video_path(0, index), schema.coco133_uv_projected_path(0, index)])
                for index, (_, label, _) in enumerate(CAMERAS)
            ],
            exo_panes=[],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the 3D scene following the headset beside RGB."""
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Scene",
                origin=schema.rig_path(0),
                contents=["+ /world/**", *blueprints.video_exclusions((0, index) for index in range(len(CAMERAS)))],
                eye_controls=FOLLOW_EYE,
            ),
            blueprints.camera_view("RGB", 0, 0, contents=[schema.video_path(0, 0), schema.coco133_uv_projected_path(0, 0)]),
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:episode:sequence", "sequence"),
            writing.TableField("property:capture:dataset_version", "version"),
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
