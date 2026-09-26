"""HO-Cap original archive discovery and layered conversion."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from zipfile import ZipFile

import rerun as rr
import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.hocap_layers import write_base, write_hand_meshes, write_hands, write_object_meshes, write_object_poses
from dataforge.datasets.hocap_source import EGO_RIG, EXO_RIGS, FRAME_RATE, SequenceData, read_sequence
from dataforge.identity import SequenceIdentity


@dataclass
class HocapConfig(DataforgeDatasetConfig):
    """Read the original release directly from its zip archives."""

    command: ClassVar[str] = "hocap"
    """CLI and catalog name."""
    _target: type = field(default_factory=lambda: HocapDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "hocap")
    """Read-only directory of original zip archives."""
    sequences: tuple[str, ...] | None = None
    """Subject/sequence keys, or all available subjects."""
    frame_limit: int | None = None
    """Retain the first N frames under output_root/preview-first<N>/, without resampling."""


@dataclass(frozen=True, slots=True)
class Archive:
    """Read-only handle and member index, held for the dataset run."""

    handle: ZipFile
    """Open archive."""
    members: frozenset[str]
    """Cached central-directory names."""


class HocapDataset(DataforgeDataset[HocapConfig, str]):
    """Five layers sharing one source frame clock and recording identity."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, paths.HAND_MESH_LAYER, paths.OBJECT_POSE_LAYER, paths.OBJECT_MESH_LAYER)

    def __init__(self, config: HocapConfig) -> None:
        super().__init__(config)
        self.archives: dict[str, Archive] = {}

    def archive(self, name: str) -> Archive:
        """Open each read-only archive and member set once; handles live for the run."""
        if name not in self.archives:
            handle: ZipFile = ZipFile(self.config.root / f"{name}.zip")
            self.archives[name] = Archive(handle, frozenset(handle.namelist()))
        return self.archives[name]

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        """Keep limited prefixes separate from full recordings."""
        if self.config.frame_limit is None:
            return super().targets(identity)
        return {
            layer: paths.rrd_path(paths.output_root() / f"preview-first{self.config.frame_limit}", layer=layer, identity=identity)
            for layer in self.layers
        }

    def download(self) -> None:
        """Verify archive directories without extracting or downloading anything."""
        required: list[Path] = [self.config.root / f"{name}.zip" for name in ("calibration", "models", "poses", "labels")]
        subjects: list[Path] = sorted(self.config.root.glob("subject_*.zip"))
        if not subjects:
            raise FileNotFoundError(f"{self.config.root}: missing subject_<n>.zip archives")
        for path in [*required, *subjects]:
            if not self.archive(path.stem).members:
                raise ValueError(f"empty HOCap archive: {path}")
        print(f"hocap: verified central directories; {len(self.discover())} sequences available")

    def discover(self) -> list[tuple[SequenceIdentity, str]]:
        keys: list[str] = sorted({name.rsplit("/", 1)[0] for name in self.archive("poses").members if name.endswith("/poses_m.npy")})
        if self.config.sequences is not None:
            missing: set[str] = set(self.config.sequences) - set(keys)
            if missing:
                raise ValueError(f"HOCap selections absent from poses.zip: {sorted(missing)}")
            keys = [key for key in keys if key in self.config.sequences]
        pairs: list[tuple[SequenceIdentity, str]] = []
        for key in keys:
            subject: str = key.split("/")[0]
            path: Path = self.config.root / f"{subject}.zip"
            if not path.is_file():
                if self.config.sequences is not None:
                    raise FileNotFoundError(f"{key}: missing image archive {path}")
                print(f"skip {key}: missing image archive {path}")
                continue
            pairs.append((SequenceIdentity("hocap", tuple(key.split("/"))), key))
        return pairs

    def convert(self, identity: SequenceIdentity, source: str, *, force: bool) -> Path:
        """Open each archive at most once and publish each requested layer atomically."""
        targets: dict[str, Path] = self.targets(identity)
        pending: list[str] = [layer for layer, target in targets.items() if not writing.should_skip(target, force=force)]
        if not pending:
            return targets[paths.BASE_LAYER]
        root: Path = self.config.root
        for target in targets.values():
            if target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"refusing to write beneath raw root: {target}")
        subject: str = source.split("/")[0]
        with self.timer.stage("fetch"):
            images: Archive = self.archive(subject)
            scene: SequenceData = read_sequence(
                images.handle,
                self.archive("calibration").handle,
                self.archive("poses").handle,
                source,
                self.config.frame_limit,
                members=images.members,
            )
        self.timer.capture_s = scene.count / FRAME_RATE
        layers: dict[str, tuple[str | None, Callable[[rr.RecordingStream], None]]] = {
            paths.BASE_LAYER: (
                None,
                lambda recording: write_base(recording, scene, images.handle, source, identity, self.timer, paths.output_root() / "work"),
            ),
            paths.HAND_POSE_LAYER: (
                "labels",
                lambda recording: write_hands(recording, scene, self.archive("labels").handle, source, members=self.archive("labels").members),
            ),
            paths.HAND_MESH_LAYER: (None, lambda recording: write_hand_meshes(recording, scene)),
            paths.OBJECT_POSE_LAYER: (None, lambda recording: write_object_poses(recording, scene)),
            paths.OBJECT_MESH_LAYER: ("models", lambda recording: write_object_meshes(recording, scene, self.archive("models").handle)),
        }
        with self.timer.stage("fetch"):
            for name in sorted({name for layer in pending if (name := layers[layer][0]) is not None}):
                self.archive(name)
        for layer in pending:
            writer = layers[layer][1]
            with (
                self.timer.stage(f"write:{layer}"),
                writing.atomic_recording(
                    targets[layer],
                    recording_id=identity.recording_id,
                    default_blueprint=self.default_blueprint() if layer == paths.BASE_LAYER else None,
                    send_properties=layer == paths.BASE_LAYER,
                ) as recording,
            ):
                writer(recording)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Tag 1 world",
                origin="/world",
                contents=world_contents(),
                eye_controls=table_eye(),
            ),
            ego_panes=[blueprints.camera_view("HoloLens", EGO_RIG, 0, contents=pane_contents(EGO_RIG))],
            exo_panes=[blueprints.camera_view(f"RealSense {rig:02}", rig, 0, contents=pane_contents(rig)) for rig in EXO_RIGS],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Scene",
                    origin="/world",
                    contents=[*world_contents(), *(f"- {schema.video_path(rig, 0)}/**" for rig in (*EXO_RIGS, EGO_RIG))],
                    eye_controls=table_eye(),
                ),
                blueprints.camera_view("HoloLens", EGO_RIG, 0, contents=pane_contents(EGO_RIG)),
            ),
            collapse_panels=True,
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:episode:subject_id", "subject"),
            writing.TableField("property:episode:object_ids", "objects"),
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)


def table_eye() -> rrb.EyeControls3D:
    """Tightest oblique eye with all eight cameras and the table in a 2:1 card; default and table layouts.

    Solved from the shipped calibration, which puts the cameras at the same tag-1 positions in every session.
    """
    return blueprints.eye_controls_from_pose((-0.03, -0.91, 1.07), (-0.03, 0.04, 0.52), (0.0, 0.0, 1.0))


def world_contents() -> list[str]:
    """Pixel measurements only belong in their source camera panes."""
    return ["+ /world/**", *(f"- {schema.coco133_uv_path(rig, 0)}" for rig in EXO_RIGS)]


def pane_contents(rig: int) -> list[str]:
    """Show shipped pixels on exo cameras; HoloLens uses world geometry only."""
    return [
        "+ /world/**",
        *(f"- {schema.pinhole_path(other, 0)}/**" for other in (*EXO_RIGS, EGO_RIG) if other != rig),
        *([f"- {schema.coco133_xyz_path()}"] if rig in EXO_RIGS else []),
    ]
