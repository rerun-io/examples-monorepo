"""Assembly101 download, discovery and three-layer conversion from the extracted mirror."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.errors import GatedRepoError

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.assembly101_actions import Actions, read_actions
from dataforge.datasets.assembly101_download import (
    RemoteFile,
    RemoteIndex,
    fetch_hub_files,
    fetch_members,
    list_annotations,
    remote_index,
)
from dataforge.datasets.assembly101_layers import Scene, camera_sources, read_scene, write_actions, write_base, write_hands
from dataforge.datasets.assembly101_source import (
    EGO_RIG,
    EXO_SERIALS,
    FRAME_RATE,
    MIRROR_REPO,
    MIRROR_REVISION,
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    POSE_MEMBERS,
    SHARED_POSE_MEMBER,
    VIDEO_DIR,
    pose_path,
    read_manifest,
)
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.identity import SequenceIdentity
from dataforge.transports import FetchReport

SCENE_EYE: rrb.EyeControls3D = blueprints.eye_controls_from_pose((1.17, 1.57, -0.62), (0.29, 0.6, 0.13), (0.0, 1.0, 0.0))
"""Tightest oblique eye with the exo cameras and the table in a 2:1 card (+Y up); default and table layouts.

Solved from the shipped extrinsics, which put the fixed cameras within ~0.1 m of the same place every day."""


@dataclass
class Assembly101Config(FrameLimitedConfig):
    """Raw inputs (written only by download); outputs use DATAFORGE_OUTPUT_ROOT."""

    command: ClassVar[str] = "assembly101"
    """CLI and catalog name."""
    _target: type = field(default_factory=lambda: Assembly101Dataset)
    """Dataset implementation."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "assembly101")
    """Mirror layout containing videos, extracted pose members and nimble calibration."""
    annotations_root: Path | None = None
    """Official annotations directory (ends in ``annotations/``); defaults to root/official/annotations."""
    sequences: tuple[str, ...] | None = None
    """Sequence directories to download or convert, or all of them."""

    @property
    def annotations(self) -> Path:
        """annotations_root, falling back to root/official/annotations (where download() puts them)."""
        return self.annotations_root or self.root / "official/annotations"


@dataclass(frozen=True, slots=True)
class Assembly101Source:
    """What discovery found for one sequence."""

    sequence: str
    """Sequence directory."""
    has_poses: bool
    """False only for a sequence the manifest marks video-only."""


class Assembly101Dataset(DataforgeDataset[Assembly101Config, Assembly101Source]):
    """No meshes or object annotations are fabricated."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, paths.ACTIONS_LAYER)

    def __init__(self, config: Assembly101Config) -> None:
        super().__init__(config)
        self.actions: dict[str, Actions] | None = None

    def discover(self) -> list[tuple[SequenceIdentity, Assembly101Source]]:
        """Sequences whose raw files are all present; others (not fetched, or pruned after conversion) are left out.

        Complete means 12 videos plus every pose member, unless the manifest (read now, so that a download on
        this instance counts) marks the sequence video-only; a sequence without a manifest row needs its poses.
        """
        folder: Path = self.config.root / VIDEO_DIR
        present: list[str] = sorted(path.name for path in folder.iterdir() if path.is_dir()) if folder.is_dir() else []
        pose_expected: dict[str, bool] = read_manifest(self.config.root)
        sources: list[Assembly101Source] = []
        for key in present:
            if len(camera_sources(self.config.root, key)) != 12:
                continue
            if pose_expected.get(key) is False:
                sources.append(Assembly101Source(key, has_poses=False))
            elif all(pose_path(self.config.root, member, key).is_file() for member in (SHARED_POSE_MEMBER, *POSE_MEMBERS)):
                sources.append(Assembly101Source(key, has_poses=True))
        if self.config.sequences is not None:
            missing: set[str] = set(self.config.sequences) - {source.sequence for source in sources}
            if missing:
                raise FileNotFoundError(f"{folder}: missing or incomplete sequences {sorted(missing)}; run dataforge-download assembly101 first")
            sources = [source for source in sources if source.sequence in self.config.sequences]
        calibration: Path = self.config.root / "assemblyhands-toolkit/calib/nimble_json_calib"
        if any(source.has_poses for source in sources) and not any(calibration.glob("*.json")):
            raise FileNotFoundError(f"Assembly101 nimble calibration missing: {calibration}; run dataforge-download assembly101 first")
        return [(SequenceIdentity("assembly101", (source.sequence,)), source) for source in sources]

    def remote_sequences(self) -> list[RemoteSequence]:
        """Each sequence's 12 videos and 5 pose members (on-disk bytes); shared files are listed by download() only."""
        index: RemoteIndex = remote_index(HfApi(), HfFileSystem())
        return [
            RemoteSequence(key, sum(remote.integrity.size_bytes for remote in files), tuple(remote.path for remote in files))
            for key, files in index.sequences.items()
        ]

    def download(self) -> None:
        """Fetch the selected sequences and the shared files at the pinned revisions; resumable, never deletes."""
        for root in (self.config.root, self.config.annotations):
            if root.resolve().is_relative_to(paths.NAS_ROOT):
                raise ValueError(f"Assembly101 downloads go to local disk, never the NAS: {root}")
        if self.config.annotations.name != "annotations":
            raise ValueError(f"annotations_root must end in annotations/ (the official repo's layout): {self.config.annotations}")
        report: FetchReport = FetchReport()
        api: HfApi = HfApi()
        filesystem: HfFileSystem = HfFileSystem()
        index: RemoteIndex = remote_index(api, filesystem)
        keys: list[str] = list(index.sequences) if self.config.sequences is None else list(self.config.sequences)
        unknown: list[str] = sorted(set(keys) - set(index.sequences))
        if unknown:
            raise ValueError(f"{self.config.command}: unknown sequence(s) {unknown} at {MIRROR_REPO}@{MIRROR_REVISION[:8]}; see `dataforge-download {self.config.command} --list-remote`")
        # The gated repo lists anonymously but serves no content, so probe access rather than the listing.
        try:
            api.auth_check(OFFICIAL_REPO, repo_type="dataset")
            annotations: list[RemoteFile] = list_annotations(api)
        except GatedRepoError:
            print(f"warning: no access to the gated {OFFICIAL_REPO}; log in with hf auth login and accept its terms. No actions layer without it.")
            annotations = []
        folders: dict[str, Path] = {"mirror": self.config.root, "zip": self.config.root, "official": self.config.annotations.parent}
        pending: dict[str, list[RemoteFile]] = {source: [] for source in folders}
        for remote in [*index.shared, *annotations, *(remote for key in keys for remote in index.sequences[key])]:
            target: Path = folders[remote.source] / remote.path
            # Every write lands by rename, so a file of the listed size is complete.
            if target.is_file() and target.stat().st_size == remote.integrity.size_bytes:
                report.count(False, remote.integrity.size_bytes)
            else:
                pending[remote.source].append(remote)
        # The zip is one range-read stream, independent of the hub pool: run it beside the pool.
        zip_report: FetchReport = FetchReport()
        with ThreadPoolExecutor(max_workers=1) as side:
            members: Future[None] = side.submit(fetch_members, filesystem, pending["zip"], folders["zip"], zip_report)
            fetch_hub_files(MIRROR_REPO, MIRROR_REVISION, pending["mirror"], folders["mirror"], report)
            fetch_hub_files(OFFICIAL_REPO, OFFICIAL_REVISION, pending["official"], folders["official"], report)
            members.result()
        report.fetched += zip_report.fetched
        report.fetched_bytes += zip_report.fetched_bytes
        print(f"assembly101: {len(keys)} sequence(s); {report.summary()}" + ("" if annotations else "; no annotations"))

    def convert(self, identity: SequenceIdentity, source: Assembly101Source, *, force: bool) -> Path:
        """Publish only available layers, atomically, sharing the recording identity."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root, self.config.annotations, paths.NAS_ROOT])
        with self.timer.stage("fetch"):
            if self.actions is None:
                self.actions = read_actions(self.config.annotations, {found.sequence for _, found in self.discover()})
            actions: Actions = self.actions[source.sequence]
        available: list[str] = [paths.BASE_LAYER]
        if source.has_poses:
            available.append(paths.HAND_POSE_LAYER)
        if actions.coarse or actions.fine:
            available.append(paths.ACTIONS_LAYER)
        pending = [layer for layer in pending if layer in available]
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("fetch"):
            scene: Scene = read_scene(self.config.root, source.sequence, self.config.frame_limit, has_poses=source.has_poses)
        self.timer.capture_s = (
            max(
                max(camera.num_frames for camera in scene.cameras),
                int(scene.poses.frames[-1]) + 1 if scene.poses is not None and len(scene.poses.frames) else 0,
            )
            / FRAME_RATE
        )
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: write_base(recording, scene, identity, actions, self.config.frame_limit, self.timer),
            paths.HAND_POSE_LAYER: lambda recording: write_hands(recording, self.config.root, source.sequence, scene, self.config.frame_limit),
            paths.ACTIONS_LAYER: lambda recording: write_actions(recording, actions, self.config.frame_limit),
        }
        self.write_layers(identity, targets, pending, writers)
        if force:
            # A forced rebuild drops what an earlier conversion published for a layer this sequence no longer has.
            for optional in (paths.HAND_POSE_LAYER, paths.ACTIONS_LAYER):
                if optional not in available:
                    targets[optional].unlink(missing_ok=True)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(name="Assembly101", origin="/world", contents=world_contents(), eye_controls=SCENE_EYE),
            ego_panes=[blueprints.camera_view(f"Ego {cam}", EGO_RIG, cam, contents=pane_contents(EGO_RIG, cam)) for cam in range(4)],
            exo_panes=[blueprints.camera_view(serial, rig, 0, contents=pane_contents(rig, 0)) for rig, serial in enumerate(EXO_SERIALS)],
            instruction=rrb.TextDocumentView(name="Fine actions", origin=schema.actions_path("fine")),
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Scene",
                origin="/world",
                contents=[*world_contents(), *blueprints.video_exclusions(camera_slots())],
                eye_controls=SCENE_EYE,
            ),
            blueprints.camera_view("C10095", 0, 0, contents=pane_contents(0, 0)),
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = tuple(
            writing.TableField(f"property:capture:{key}", title)
            for key, title in (
                ("num_frames", "frames"),
                ("calibration_source", "exo calibration"),
                ("hand_pose_present", "hand pose"),
                ("coarse_actions_present", "coarse actions"),
                ("fine_actions_present", "fine actions"),
            )
        )
        return writing.TableFields(cards=fields, table=fields)


def camera_slots() -> list[tuple[int, int]]:
    """Stable slots independent of the headset's serial set."""
    return [(rig, 0) for rig in range(8)] + [(EGO_RIG, cam) for cam in range(4)]


def world_contents() -> list[str]:
    return ["+ /world/**", *[f"- {schema.coco133_uv_path(rig, cam)}" for rig, cam in camera_slots()]]


def pane_contents(rig: int, cam: int) -> list[str]:
    """Only shipped pixels and this camera's video; no undistorted 3D projection."""
    return [f"+ {schema.pinhole_path(rig, cam)}/**"]
