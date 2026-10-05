"""Aria Gen2 Pilot v1.0: verify-only source and three native-clock layers."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb
from serde import serde

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.aria_gen2_pilot_layers import write_base, write_hands, write_projections
from dataforge.datasets.aria_gen2_pilot_source import CAMERAS, SEQUENCES, Scene, read_scene
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
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
    """Only files needed by this port; other download types remain upstream."""

    main_vrs: SourceFile
    """Canonical video.vrs (the CDN filename is a duplicate)."""


@serde
@dataclass(frozen=True, slots=True)
class Manifest:
    """Release download inventory; expired URLs are never used."""

    sequences: dict[str, SequenceFiles]
    """Integrity metadata keyed by sequence name."""


RIG_FORWARD: tuple[float, float, float] = (0.34, -0.27, 0.9)
"""Headset forward in the rig frame: camera-rgb's optical axis (measured from its shipped pose)."""
RIG_UP: tuple[float, float, float] = (0.05, -0.95, -0.3)
"""Headset up in the rig frame: minus camera-rgb's image y axis."""
FOLLOW_EYE: rrb.EyeControls3D = blueprints.headset_eye_controls(RIG_FORWARD, RIG_UP)
"""The 3D views' eye in the rig frame, riding the headset; default and table layouts."""


@dataclass
class AriaGen2PilotConfig(FrameLimitedConfig):
    """Read-only source selection; output follows DATAFORGE_OUTPUT_ROOT."""

    command: ClassVar[str] = "aria_gen2_pilot"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: AriaGen2PilotDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "aria_gen2_pilot")
    """Release root: the download manifest and one directory per sequence (the source may live on the NAS)."""
    sequences: tuple[str, ...] | None = None
    """Subset of the 12 release names; None discovers all complete local sources."""


class AriaGen2PilotDataset(DataforgeDataset[AriaGen2PilotConfig, Path]):
    """One recording identity across sensors, measured hands and derived pixels."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, paths.PROJECTIONS_LAYER)

    def manifest(self) -> Manifest:
        """The release inventory beside the sequences."""
        return read_json(self.config.root / "AriaGen2PilotDataset_download_urls.json", Manifest)

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
            required: list[Path] = [
                source / "video.vrs",
                source / "mps/slam/closed_loop_trajectory.csv",
                source / "mps/hand_tracking/hand_tracking_results.csv",
            ]
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
        """Verify local VRS size and SHA-1; never fetch, extract or modify raw data."""
        manifest: Manifest = self.manifest()
        found: list[tuple[SequenceIdentity, Path]] = self.discover(manifest)
        for _, source in found:
            listed: SourceFile = manifest.sequences[source.name].main_vrs
            # discover() already checked the size, so only the SHA-1 can differ.
            if not matches_file(source / "video.vrs", FileIntegrity(listed.file_size_bytes, "sha1", listed.sha1sum)):
                raise ValueError(f"{source}/video.vrs: SHA-1 differs from release manifest")
        print(f"aria_gen2_pilot v1.0: verified {len(found)} local sequences")

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
