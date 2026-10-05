"""HO-Cap original archive discovery and layered conversion."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from zipfile import ZipFile

import rerun as rr
import rerun.blueprint as rrb
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.datasets.hocap_layers import write_base, write_hand_meshes, write_hands, write_object_meshes, write_object_poses
from dataforge.datasets.hocap_source import EGO_RIG, EXO_RIGS, FRAME_RATE, SOURCE_REPO, SOURCE_REVISION, SequenceData, read_sequence
from dataforge.identity import SequenceIdentity
from dataforge.transports import FetchReport, HfFileInfo, hf_fetch_verified, hf_lfs_files

SHARED_ARCHIVES: tuple[str, ...] = ("calibration.zip", "models.zip", "poses.zip", "labels.zip")
"""Archives every sequence reads (1.65 GB); each ``subject_<n>.zip`` holds only that subject's images."""
SCENE_EYE: rrb.EyeControls3D = blueprints.eye_controls_from_pose((-0.03, -0.91, 1.07), (-0.03, 0.04, 0.52), (0.0, 0.0, 1.0))
"""Tightest oblique eye with all eight cameras and the table in a 2:1 card; default and table layouts.

Solved from the shipped calibration, which puts the cameras at the same tag-1 positions in every session."""


@dataclass
class HocapConfig(FrameLimitedConfig):
    """Read the original release directly from its zip archives."""

    command: ClassVar[str] = "hocap"
    """CLI and catalog name."""
    _target: type = field(default_factory=lambda: HocapDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "hocap")
    """Read-only directory of original zip archives."""
    sequences: tuple[str, ...] | None = None
    """``subject_<n>/<sequence>`` keys or whole ``subject_<n>`` archives to download and convert; None = every sequence."""


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

    def download(self) -> None:
        """Fetch the shared archives, then only the subject archives the selected sequences need.

        poses.zip comes first so an unknown selection fails before anything big is fetched. A file whose
        size and sha256 already match is skipped; see ``hf_fetch_verified``. Nothing is extracted or deleted.
        """
        report: FetchReport = FetchReport()
        root: Path = self.config.root
        poses: HfFileInfo = hf_lfs_files(SOURCE_REPO, ["poses.zip"], revision=SOURCE_REVISION)[0]
        report.count(hf_fetch_verified(SOURCE_REPO, poses, local_dir=root, revision=SOURCE_REVISION), poses.size_bytes)
        keys: list[str] = self.sequence_keys()
        subjects: list[str] = sorted({f"{key.split('/')[0]}.zip" for key in keys})
        rest: list[str] = [name for name in SHARED_ARCHIVES if name != "poses.zip"] + subjects
        for remote in hf_lfs_files(SOURCE_REPO, rest, revision=SOURCE_REVISION):
            fetched: bool = hf_fetch_verified(SOURCE_REPO, remote, local_dir=root, revision=SOURCE_REVISION)
            if fetched:
                print(f"hocap: fetched {remote.path} ({remote.size_bytes / 1e9:.2f} GB)")
            report.count(fetched, remote.size_bytes)
        print(f"hocap: {len(keys)} sequence(s); {report.summary()} under {root}")

    def remote_sequences(self) -> list[RemoteSequence]:
        """One entry per ``subject_<n>.zip`` in the pinned tree; nothing is fetched.

        Each subject archive (3.4-22 GB) holds all of that subject's sequences, so the key is the whole subject:
        ``--sequences subject_<n>`` downloads and converts it, and deleting ``files`` frees its archive.
        """
        tree = HfApi().list_repo_tree(SOURCE_REPO, repo_type="dataset", revision=SOURCE_REVISION)
        subjects: list[RepoFile] = [info for info in tree if isinstance(info, RepoFile) and info.path.startswith("subject_") and info.path.endswith(".zip")]
        return [RemoteSequence(info.path.removesuffix(".zip"), info.size, (info.path,)) for info in sorted(subjects, key=lambda info: int(info.path[8:-4]))]

    def sequence_keys(self) -> list[str]:
        """Sorted ``subject/sequence`` keys of poses.zip, narrowed to ``config.sequences`` (keys or subjects) when set."""
        keys: list[str] = sorted({name.rsplit("/", 1)[0] for name in self.archive("poses").members if name.endswith("/poses_m.npy")})
        selected: tuple[str, ...] | None = self.config.sequences
        if selected is None:
            return keys
        missing: list[str] = [sel for sel in selected if not any(key == sel or key.startswith(f"{sel}/") for key in keys)]
        if missing:
            raise ValueError(f"{self.config.command}: unknown sequence(s) {sorted(missing)}; see `dataforge-download {self.config.command} --list-remote`")
        return [key for key in keys if key in selected or key.split("/")[0] in selected]

    def discover(self) -> list[tuple[SequenceIdentity, str]]:
        pairs: list[tuple[SequenceIdentity, str]] = []
        for key in self.sequence_keys():
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
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
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
            # The label and model archives open here, so the fetch stage counts them.
            archives: dict[str, str] = {paths.HAND_POSE_LAYER: "labels", paths.OBJECT_MESH_LAYER: "models"}
            for name in sorted({archives[layer] for layer in pending if layer in archives}):
                self.archive(name)
        self.timer.capture_s = scene.count / FRAME_RATE
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: write_base(recording, scene, images.handle, source, identity, self.timer),
            paths.HAND_POSE_LAYER: lambda recording: write_hands(
                recording, scene, self.archive("labels").handle, source, members=self.archive("labels").members
            ),
            paths.HAND_MESH_LAYER: lambda recording: write_hand_meshes(recording, scene),
            paths.OBJECT_POSE_LAYER: lambda recording: write_object_poses(recording, scene),
            paths.OBJECT_MESH_LAYER: lambda recording: write_object_meshes(recording, scene, self.archive("models").handle),
        }
        self.write_layers(identity, targets, pending, writers)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Tag 1 world",
                origin="/world",
                contents=world_contents(),
                eye_controls=SCENE_EYE,
            ),
            ego_panes=[blueprints.camera_view("HoloLens", EGO_RIG, 0, contents=pane_contents(EGO_RIG))],
            exo_panes=[blueprints.camera_view(f"RealSense {rig:02}", rig, 0, contents=pane_contents(rig)) for rig in EXO_RIGS],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Scene",
                origin="/world",
                contents=[*world_contents(), *blueprints.video_exclusions((rig, 0) for rig in (*EXO_RIGS, EGO_RIG))],
                eye_controls=SCENE_EYE,
            ),
            blueprints.camera_view("HoloLens", EGO_RIG, 0, contents=pane_contents(EGO_RIG)),
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:episode:subject_id", "subject"),
            writing.TableField("property:episode:object_ids", "objects"),
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)


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
