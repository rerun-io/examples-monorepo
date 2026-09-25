"""UmeTrack discovery and verification of the read-only raw mirror."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.umetrack_layers import HandKeypoints, hand_keypoints, write_base, write_hands, write_projections
from dataforge.datasets.umetrack_source import SequenceData, read_sequence
from dataforge.identity import SequenceIdentity


@dataclass
class UmetrackConfig(DataforgeDatasetConfig):
    """Read stacked source MP4s and their label files."""

    command: ClassVar[str] = "umetrack"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: UmetrackDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "umetrack-data/raw_data")
    """Read-only raw recording root."""
    sequences: tuple[str, ...] | None = None
    """Domain/interaction/split/user/recording keys; None selects all."""
    frame_limit: int | None = None
    """First N source frames, isolated under preview-first<N>."""

    def __post_init__(self) -> None:
        if self.frame_limit is not None and self.frame_limit <= 0:
            raise ValueError("frame_limit must be positive")


class UmetrackDataset(DataforgeDataset[UmetrackConfig, Path]):
    """One catalog dataset for real and synthetic captures."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, paths.PROJECTIONS_LAYER)

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        sources: dict[str, Path] = {
            p.relative_to(self.config.root).with_suffix("").as_posix(): p for p in self.config.root.glob("*/*/*/user_*/recording_*.json")
        }
        keys: list[str] = sorted(sources if self.config.sequences is None else self.config.sequences)
        result: list[tuple[SequenceIdentity, Path]] = []
        for key in keys:
            if key not in sources:
                raise FileNotFoundError(f"UmeTrack label absent: {self.config.root / key}.json")
            source: Path = sources[key]
            if not source.with_suffix(".mp4").is_file():
                raise FileNotFoundError(source.with_suffix(".mp4"))
            result.append((SequenceIdentity("umetrack", tuple(key.split("/"))), source))
        return result

    def download(self) -> None:
        """Verify manifest members; never fetch or write source data."""
        selected: set[str] | None = None if self.config.sequences is None else set(self.config.sequences)
        seen: set[str] = set()
        for domain in ("real", "synthetic"):
            manifest: Path = self.config.root.parent / f"raw_data_{domain}_manifest.txt"
            for line in manifest.read_text().splitlines():
                if not line.startswith("https://"):
                    continue
                relative: str = urlparse(line).path.split("/raw_data/", 1)[1]
                key: str = str(Path(relative).with_suffix(""))
                if selected is not None and key not in selected:
                    continue
                path: Path = self.config.root / relative
                if not path.resolve().is_relative_to(self.config.root.resolve()):
                    raise ValueError(f"unsafe manifest path: {relative}")
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"UmeTrack manifest asset missing or empty: {path}")
                seen.add(relative)
        if selected is not None:
            missing: set[str] = {f"{key}.{ext}" for key in selected for ext in ("json", "mp4")} - seen
            if missing:
                raise ValueError(f"UmeTrack selections absent from manifests: {sorted(missing)}")
        if not seen:
            raise ValueError("UmeTrack manifests contain no selected assets")

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        """Return potential layer paths; projections may be absent without hand GT. Keep previews separate."""
        root: Path = paths.output_root()
        if self.config.frame_limit is not None:
            root = root / f"preview-first{self.config.frame_limit}"
        return {layer: paths.rrd_path(root, layer=layer, identity=identity) for layer in self.layers}

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Publish available layers atomically and time fetch, transcode and writes."""
        targets: dict[str, Path] = self.targets(identity)
        work_root: Path = paths.output_root() / "work"
        for target in [*targets.values(), work_root]:
            if target.resolve().is_relative_to(self.config.root.resolve()):
                raise ValueError(f"refusing to write beneath raw root: {target}")
        pending: list[str] = [layer for layer, target in targets.items() if not writing.should_skip(target, force=force)]
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("fetch"):
            scene: SequenceData = read_sequence(source, self.config.frame_limit)
        if not np.any(scene.labels.hand_confidences > 0):
            pending = [layer for layer in pending if layer != paths.PROJECTIONS_LAYER]
        keypoints: HandKeypoints | None = (
            hand_keypoints(scene) if any(layer in pending for layer in (paths.HAND_POSE_LAYER, paths.PROJECTIONS_LAYER)) else None
        )
        self.timer.capture_s = scene.count / scene.fps
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: write_base(recording, scene, identity, self.timer, work_root),
            paths.HAND_POSE_LAYER: lambda recording: write_hands(recording, scene, keypoints),
            paths.PROJECTIONS_LAYER: lambda recording: write_projections(recording, scene, keypoints),
        }
        for layer in pending:
            with (
                self.timer.stage(f"write:{layer}"),
                writing.atomic_recording(
                    targets[layer],
                    recording_id=identity.recording_id,
                    send_properties=layer == paths.BASE_LAYER,
                    default_blueprint=self.default_blueprint() if layer == paths.BASE_LAYER else None,
                ) as recording,
            ):
                writers[layer](recording)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(name="Hands", origin="/world", contents=["+ /world/**"], eye_controls=hand_eye()),
            ego_panes=[blueprints.camera_view(f"cam_{camera:02}", 0, camera, contents=pane_contents(camera)) for camera in range(4)],
            exo_panes=[],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Hands", origin="/world", contents=["+ /world/gt/**"], eye_controls=hand_eye()),
                blueprints.camera_view("cam_00", 0, 0, contents=pane_contents(0)),
            ),
            collapse_panels=True,
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            *(writing.TableField(f"property:episode:{name}", name) for name in ("domain", "interaction", "split", "user")),
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)


def hand_eye() -> rrb.EyeControls3D:
    """Orbital view of hands below a +Y-up headset."""
    return blueprints.eye_controls_from_pose((1.2, 0.9, 1.2), (0.3, 0.0, 0.0), (0.0, 1.0, 0.0))


def pane_contents(camera: int) -> list[str]:
    """Show only this camera's video and lens-model projections; exclude 3D hands and meshes."""
    return [f"+ {schema.video_path(0, camera)}", f"+ {schema.coco133_uv_projected_path(0, camera)}", "- /world/gt/**"]
