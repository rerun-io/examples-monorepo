"""UmeTrack: download from GitHub, discovery, and conversion of the stacked recordings."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.datasets.umetrack_layers import HandKeypoints, hand_keypoints, write_base, write_hands, write_meshes, write_projections
from dataforge.datasets.umetrack_remote import REVISION, RemoteFile, exclusive, fetch_files, load_index
from dataforge.datasets.umetrack_source import SequenceData, read_sequence
from dataforge.identity import SequenceIdentity

SCENE_EYE: rrb.EyeControls3D = blueprints.eye_controls_from_pose((0.39, 0.52, 0.49), (0.0, 0.2, 0.0), (0.0, 1.0, 0.0))
"""Orbital view 0.7 m from the hands below the +Y-up headset, which barely moves within a recording."""


@dataclass
class UmetrackConfig(FrameLimitedConfig):
    """Read stacked source MP4s and their label files."""

    command: ClassVar[str] = "umetrack"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: UmetrackDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "umetrack/raw_data")
    """Raw recording root, the source repository's ``raw_data/`` inside the dataset directory; ``download()`` fills it and conversion only reads it."""
    sequences: tuple[str, ...] | None = None
    """Domain/interaction/split/user/recording keys; None selects all."""


class UmetrackDataset(DataforgeDataset[UmetrackConfig, Path]):
    """One catalog dataset for real and synthetic captures."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, paths.HAND_MESH_LAYER, paths.PROJECTIONS_LAYER)

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        """Pair every recording whose label and video are both present; a selected key must be."""
        sources: dict[str, Path] = {
            p.relative_to(self.config.root).with_suffix("").as_posix(): p
            for p in self.config.root.glob("*/*/*/user_*/recording_*.json")
            if p.with_suffix(".mp4").is_file()
        }
        keys: list[str] = sorted(sources if self.config.sequences is None else self.config.sequences)
        missing: list[str] = [key for key in keys if key not in sources]
        if missing:
            raise FileNotFoundError(f"UmeTrack recordings absent or incomplete under {self.config.root}: {missing}")
        return [(SequenceIdentity("umetrack", tuple(key.split("/"))), sources[key]) for key in keys]

    def download(self) -> None:
        """Fetch the selected recordings (all when ``sequences`` is None) at the pinned revision."""
        with exclusive(self.config.root):
            recordings: dict[str, list[RemoteFile]] = load_index(self.config.root).recordings()
            keys: list[str] = list(recordings) if self.config.sequences is None else list(dict.fromkeys(self.config.sequences))
            unknown: list[str] = sorted(set(keys) - recordings.keys())
            if unknown:
                raise ValueError(f"{self.config.command}: unknown sequence(s) {unknown} at {REVISION[:12]}; see `dataforge-download {self.config.command} --list-remote`")
            fetch_files(self.config.root, [remote for key in keys for remote in recordings[key]])

    def remote_sequences(self) -> list[RemoteSequence]:
        """One entry per recording: its label JSON and stacked MP4, sized from the index."""
        return [
            RemoteSequence(key, sum(remote.size_bytes for remote in files), tuple(remote.path for remote in files))
            for key, files in load_index(self.config.root).recordings().items()
        ]

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Publish available layers atomically and time fetch, hands, transcode and writes."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("fetch"):
            scene: SequenceData = read_sequence(source, self.config.frame_limit)
        with self.timer.stage("hands"):
            keypoints: HandKeypoints = hand_keypoints(scene)
        self.timer.capture_s = scene.count / scene.fps
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: write_base(recording, scene, identity, self.timer),
            paths.HAND_POSE_LAYER: lambda recording: write_hands(recording, scene, keypoints),
            paths.HAND_MESH_LAYER: lambda recording: write_meshes(recording, scene),
            paths.PROJECTIONS_LAYER: lambda recording: write_projections(recording, scene, keypoints),
        }
        self.write_layers(identity, targets, pending, writers)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(name="Hands", origin="/world", contents=["+ /world/**"], eye_controls=SCENE_EYE),
            ego_panes=[blueprints.camera_view(f"cam_{camera:02}", 0, camera, contents=pane_contents(camera)) for camera in range(4)],
            exo_panes=[],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Hands", origin="/world", contents=["+ /world/**", *blueprints.video_exclusions((0, camera) for camera in range(4))], eye_controls=SCENE_EYE
            ),
            blueprints.camera_view("cam_00", 0, 0, contents=pane_contents(0)),
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            *(writing.TableField(f"property:episode:{name}", name) for name in ("domain", "interaction", "split", "user")),
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)


def pane_contents(camera: int) -> list[str]:
    """Show only this camera's video and lens-model projections; exclude 3D hands and meshes."""
    return [f"+ {schema.video_path(0, camera)}", f"+ {schema.coco133_uv_projected_path(0, camera)}"]
