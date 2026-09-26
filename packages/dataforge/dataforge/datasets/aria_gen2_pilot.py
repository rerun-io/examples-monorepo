"""Aria Gen2 Pilot v1.0: verify-only source and three native-clock layers."""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray
from serde import serde

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.aria_gen2_pilot_layers import write_base, write_hands, write_projections
from dataforge.datasets.aria_gen2_pilot_source import CAMERAS, SEQUENCES, Scene, read_scene
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.show3d_source import read_json
from dataforge.identity import SequenceIdentity


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


@dataclass
class AriaGen2PilotConfig(DataforgeDatasetConfig):
    """Read-only source selection; output follows DATAFORGE_OUTPUT_ROOT."""

    command: ClassVar[str] = "aria_gen2_pilot"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: AriaGen2PilotDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: Path(os.environ.get("DATAFORGE_RAW_ROOT", "/mnt/nas/datasets/aria-gen2-pilot")))
    """Dataset root, directly overridden by DATAFORGE_RAW_ROOT."""
    sequences: tuple[str, ...] | None = None
    """Subset of the 12 release names; None discovers all complete local sources."""
    frame_limit: int | None = None
    """First N frames of each camera, isolated under preview-firstN."""


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
            raise ValueError(f"unknown Aria Gen2 Pilot sequences: {sorted(unknown)}")
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
            with (source / "video.vrs").open("rb") as stream:
                digest: str = hashlib.file_digest(stream, "sha1").hexdigest()
            if digest != manifest.sequences[source.name].main_vrs.sha1sum:
                raise ValueError(f"{source}/video.vrs: SHA-1 differs from release manifest")
        print(f"aria_gen2_pilot v1.0: verified {len(found)} local sequences")

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        root: Path = paths.output_root()
        if self.config.frame_limit is not None:
            if self.config.frame_limit < 1:
                raise ValueError("frame_limit must be positive")
            root /= f"preview-first{self.config.frame_limit}"
        return {layer: paths.rrd_path(root, layer=layer, identity=identity) for layer in self.layers}

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write atomic local layers and never follow raw symlinks for output."""
        targets: dict[str, Path] = self.targets(identity)
        forbidden: tuple[Path, ...] = (self.config.root.resolve(), source.resolve())
        for target in [*targets.values(), paths.output_root() / "work"]:
            if any(target.resolve().is_relative_to(root) for root in forbidden):
                raise ValueError(f"refusing output under the raw data: {target}")
        pending: list[str] = [layer for layer, target in targets.items() if not writing.should_skip(target, force=force)]
        if pending:
            with self.timer.stage("fetch"):
                scene: Scene = read_scene(source, self.config.frame_limit)
            self.timer.capture_s = (
                max(int(camera.times_ns[-1]) for camera in scene.cameras) - min(int(camera.times_ns[0]) for camera in scene.cameras)
            ) / 1e9
            for layer in pending:
                with (
                    self.timer.stage(f"write:{layer}"),
                    writing.atomic_recording(
                        targets[layer],
                        recording_id=identity.recording_id,
                        default_blueprint=self.scene_blueprint(scene) if layer == paths.BASE_LAYER else None,
                        send_properties=layer == paths.BASE_LAYER,
                    ) as recording,
                ):
                    if layer == paths.BASE_LAYER:
                        write_base(recording, scene, identity, self.timer)
                    elif layer == paths.HAND_POSE_LAYER:
                        write_hands(recording, scene)
                    else:
                        write_projections(recording, scene)
        return targets[paths.BASE_LAYER]

    def scene_blueprint(self, scene: Scene) -> rrb.Blueprint:
        """Fit the orbital eye to the measured scene instead of a fixed world point."""
        points: Float64[ndarray, "n 3"] = scene.trajectory.poses[:, :3, 3]
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return self.default_blueprint()
        # 5-95th percentile bounds keep one SLAM excursion from pushing the eye away; the
        # target sits 0.3 m below head height, where the hands are.
        low: Float64[ndarray, "3"] = np.percentile(points, 5.0, axis=0)
        high: Float64[ndarray, "3"] = np.percentile(points, 95.0, axis=0)
        target: Float64[ndarray, "3"] = (low + high) / 2.0 - np.array([0.0, 0.0, 0.3])
        direction: Float64[ndarray, "3"] = np.array([1.0, -1.5, 1.0]) / np.linalg.norm([1.0, -1.5, 1.0])
        eye: Float64[ndarray, "3"] = target + direction * max(1.0, 1.1 * float(np.linalg.norm(high - low)))
        return self._blueprint(
            blueprints.eye_controls_from_pose(
                (float(eye[0]), float(eye[1]), float(eye[2])),
                (float(target[0]), float(target[1]), float(target[2])),
                (0.0, 0.0, 1.0),
            )
        )

    def default_blueprint(self) -> rrb.Blueprint:
        """Catalog scenes auto-fit their own bounds, with orbital controls."""
        return self._blueprint(rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, eye_up=(0.0, 0.0, 1.0), spin_speed=0.0))

    def _blueprint(self, eye: rrb.EyeControls3D) -> rrb.Blueprint:
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(name="Aria Gen2 world", origin="/world", contents=["/world/**"], eye_controls=eye),
            ego_panes=[
                blueprints.camera_view(label, 0, index, contents=[schema.video_path(0, index), schema.coco133_uv_projected_path(0, index)])
                for index, (_, label, _) in enumerate(CAMERAS)
            ],
            exo_panes=[],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the 3D scene (auto-fit, videos excluded so a card decodes one stream) beside the RGB camera."""
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Scene",
                    origin="/world",
                    contents=["+ /world/**", *(f"- {schema.video_path(0, index)}" for index in range(len(CAMERAS)))],
                    eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, eye_up=(0.0, 0.0, 1.0), spin_speed=0.0),
                ),
                blueprints.camera_view("RGB", 0, 0, contents=[schema.video_path(0, 0), schema.coco133_uv_projected_path(0, 0)]),
            ),
            collapse_panels=True,
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:episode:sequence", "sequence"),
            writing.TableField("property:capture:dataset_version", "version"),
            writing.TableField("property:capture:num_frames", "RGB frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
