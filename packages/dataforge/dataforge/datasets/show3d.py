"""SHOW3D discovery and one-scene-at-a-time layered publication."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import rerun.blueprint as rrb
from serde import from_dict

from dataforge import archives, blueprints, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.show3d_captions import Caption, write_captions_layer
from dataforge.datasets.show3d_hands import (
    HandFrame,
    HandProfileDoc,
    read_hand_frames,
    read_hand_profile,
    write_hand_mesh_layer,
    write_hand_pose_layer,
)
from dataforge.datasets.show3d_layers import Scene, write_base_layer
from dataforge.datasets.show3d_mesh_source import MeshAsset, download_meshes, stripped_mesh
from dataforge.datasets.show3d_object_source import ObjectFrame, ObjectTrack, read_object_frames
from dataforge.datasets.show3d_objects import ObjectSanity, object_sanity, write_object_mesh_layer, write_object_pose_layer
from dataforge.datasets.show3d_source import (
    CAMERAS,
    HEADSET_CAMERAS,
    FrameClock,
    IndexRow,
    Show3dCamera,
    calibration_file,
    caption_file,
    hand_pose_file,
    hand_profile_file,
    mesh_name,
    object_pose_file,
    read_frame_clock,
    read_headset_calibrations,
    read_json,
)
from dataforge.identity import SequenceIdentity
from dataforge.timing import stage
from dataforge.writing import TableField, TableFields

REPO_ID: str = "facebook/show3d-dataset"


def world_contents() -> list[str]:
    """Everything under ``/world`` except the shipped face boxes and shipped pixel keypoints.

    Rerun content filters honour exact paths and a trailing ``/**`` only: a rule such as
    ``- /world/**/boxes/face`` matches nothing and hides nothing (verified with headless
    screenshots), so the exclusions are spelled out from the camera table.
    """
    return [
        "+ /world/**",
        *(f"- {schema.boxes_path(camera.rig, camera.cam, 'face')}" for camera in CAMERAS),
        *(f"- {schema.coco133_uv_path(camera.rig, camera.cam)}" for camera in CAMERAS),
    ]


def preview_world_contents() -> list[str]:
    """The segment-table 3D view: ``world_contents`` without any camera's video.

    Every visible table row renders this at once, so nothing here may decode video. Each
    video is **excluded**, not hidden (a hidden entity is still decoded), which leaves every
    Pinhole frustum with an empty image plane.
    """
    return [*world_contents(), *(f"- {schema.video_path(camera.rig, camera.cam)}/**" for camera in CAMERAS)]


def pane_contents(camera: Show3dCamera) -> list[str]:
    """One camera's 2D pane: the world seen through this pinhole and nothing from any other image plane.

    A 2D pane cannot lift another pinhole's video or pixel landmarks into its own image
    (the viewer reports "No transform path" per entity), and a projected ego image plane
    draws its frame and uv landmarks over the exo footage, so every other camera's
    ``pinhole/**`` subtree is excluded outright.

    A rectified pinhole pane shows the viewer projection of ``coco133_xyz`` and hides
    ``coco133_uv``. A camera with distortion would show ``coco133_uv`` and exclude
    ``coco133_xyz`` because Rerun Pinhole cannot project through distortion. SHOW3D
    cameras are all PinholePlane, so only the rectified rule applies here.
    """
    others: list[str] = [f"- {schema.pinhole_path(other.rig, other.cam)}/**" for other in CAMERAS if other is not camera]
    return ["+ /world/**", f"- {schema.boxes_path(camera.rig, camera.cam, 'face')}", f"- {schema.coco133_uv_path(camera.rig, camera.cam)}", *others]


@dataclass
class Show3dConfig(DataforgeDatasetConfig):
    """SHOW3D scenes with a fixed back rig and a moving Quest 3."""

    command: ClassVar[str] = "show3d"
    """Registry key and catalog dataset."""
    _target: type = field(default_factory=lambda: Show3dDataset)
    """Dataset instantiated by setup."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "show3d")
    """Hub layout; temporary encodes live under work/."""
    sequences: tuple[str, ...] | None = None
    """Scene IDs or subject/scene keys; None selects all scenes."""
    split: Literal["all", "train", "test"] = "all"
    """Split to discover from the union of both indexes."""
    object_scenes_first: bool = True
    """Prioritize scenes with an object-pose annotation."""
    keep_raw: bool = False
    """Keep source MP4s after success; small sidecars are always kept."""
    revision: str | None = None
    """Hub branch, tag or commit; resolved once per run to a commit SHA."""


class Show3dDataset(DataforgeDataset[Show3dConfig, IndexRow]):
    """Publish each missing layer, then remove only the scene videos."""

    layers: tuple[str, ...] = (
        paths.BASE_LAYER,
        paths.HAND_POSE_LAYER,
        paths.CAPTIONS_LAYER,
        paths.OBJECT_POSE_LAYER,
        paths.OBJECT_MESH_LAYER,
        paths.HAND_MESH_LAYER,
    )
    """SHOW3D publication and loading order."""

    @cached_property
    def commit_sha(self) -> str:
        """Resolve the one Hub commit read by every fetch in this run."""
        resolved: str | None = transports.repo_revision(REPO_ID, self.config.revision)
        if resolved is None:
            raise RuntimeError(f"{REPO_ID} resolved no commit sha for {self.config.revision or 'the default branch'}")
        return resolved

    def download(self) -> None:
        transports.hf_fetch(
            REPO_ID,
            allow_patterns=["dataset_index_train.parquet", "dataset_index_test.parquet", "hand_pose/hand_profiles/*/profile_umetrack.json"],
            local_dir=self.config.root,
            revision=self.commit_sha,
        )
        download_meshes(self.config.root)
        sources: list[tuple[SequenceIdentity, IndexRow]] = self.discover()
        counts: Counter[str] = Counter(source.split for _, source in sources)
        print(
            f"show3d: {len(sources)} scenes; train={counts['train']}, test={counts['test']}; "
            f"object scenes={sum(source.has_object_pose for _, source in sources)}"
        )
        print("  convert fetches one scene bundle at a time; calibration and annotations are retained")

    def discover(self) -> list[tuple[SequenceIdentity, IndexRow]]:
        pairs: list[tuple[SequenceIdentity, IndexRow]] = []
        seen: set[SequenceIdentity] = set()
        matched: set[str] = set()
        for split in ("train", "test"):
            path: Path = self.config.root / f"dataset_index_{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"{path}: run dataforge-download show3d first")
            table: pa.Table = pq.read_table(path)
            table = table.append_column("split", pa.array([split] * table.num_rows, type=pa.string()))
            for row in table.to_pylist():
                source: IndexRow = from_dict(IndexRow, row)
                identity: SequenceIdentity = SequenceIdentity("show3d", (source.subject_id, source.scene_id))
                if identity in seen:
                    raise ValueError(f"duplicate SHOW3D identity: {identity.recording_id}")
                seen.add(identity)
                if self.config.split != "all" and self.config.split != source.split:
                    continue
                keys: set[str] = {source.scene_id, f"{source.subject_id}/{source.scene_id}"}
                if self.config.sequences is not None and not keys.intersection(self.config.sequences):
                    continue
                matched.update(keys)
                if source.num_frames == 0:
                    print(f"skip {identity.sequence_key}: degenerate index row (num_frames==0)")
                    continue
                pairs.append((identity, source))
        if self.config.sequences is not None and (missing := set(self.config.sequences) - matched):
            raise ValueError(f"SHOW3D selections absent from split {self.config.split}: {sorted(missing)}")
        pairs.sort(
            key=lambda pair: (
                not pair[1].has_object_pose if self.config.object_scenes_first else False,
                pair[1].split != "train",
                pair[1].subject_id,
                pair[1].scene_id,
            )
        )
        return pairs

    def fetch_missing(self, files: list[str]) -> None:
        """Fetch absent raw inputs once; force applies only to published layers."""
        missing: list[str] = [name for name in files if not (self.config.root / name).is_file()]
        if missing:
            transports.hf_fetch_files(REPO_ID, missing, local_dir=self.config.root, revision=self.commit_sha)

    def convert(self, identity: SequenceIdentity, source: IndexRow, *, force: bool) -> Path:
        targets: dict[str, Path] = {layer: paths.rrd_path(paths.output_root(), layer=layer, identity=identity) for layer in self.layers}
        alias: str = source.object_alias
        mesh: str | None = mesh_name(alias)
        wants: dict[str, bool] = {
            paths.BASE_LAYER: not writing.should_skip(targets[paths.BASE_LAYER], force=force),
            paths.HAND_POSE_LAYER: source.has_hand_pose and not writing.should_skip(targets[paths.HAND_POSE_LAYER], force=force),
            paths.CAPTIONS_LAYER: source.has_caption and not writing.should_skip(targets[paths.CAPTIONS_LAYER], force=force),
            paths.OBJECT_POSE_LAYER: source.has_object_pose and not writing.should_skip(targets[paths.OBJECT_POSE_LAYER], force=force),
            paths.OBJECT_MESH_LAYER: source.has_object_pose and mesh is not None and not writing.should_skip(targets[paths.OBJECT_MESH_LAYER], force=force),
            paths.HAND_MESH_LAYER: source.has_hand_pose and not writing.should_skip(targets[paths.HAND_MESH_LAYER], force=force),
        }
        if wants[paths.OBJECT_POSE_LAYER] and mesh is None:
            print(f"{identity.sequence_key}: no object_mesh: alias {alias!r} has no HOT3D mesh mapping")
        if not any(wants.values()):
            return targets[paths.BASE_LAYER]
        key: str = identity.sequence_key
        files: set[str] = set(base_files(source, key) if wants[paths.BASE_LAYER] else [])
        if wants[paths.HAND_POSE_LAYER] or wants[paths.HAND_MESH_LAYER] or (wants[paths.OBJECT_POSE_LAYER] and source.has_hand_pose):
            files.update([*metadata_files(key), hand_pose_file(key), hand_profile_file(source.subject_id)])
        if wants[paths.CAPTIONS_LAYER]:
            files.add(caption_file(key))
        if wants[paths.OBJECT_POSE_LAYER] or wants[paths.OBJECT_MESH_LAYER]:
            files.update([*metadata_files(key), object_pose_file(key)])
        if wants[paths.OBJECT_POSE_LAYER]:
            files.update(calibration_file(key, camera) for camera in HEADSET_CAMERAS)
        with stage("fetch"):
            self.fetch_missing(sorted(files))
        scene_dir: Path = self.config.root / "scenes" / key
        written: list[str] = []
        scene: Scene | None = None
        if wants[paths.BASE_LAYER]:
            work: Path = self.config.root / "work" / identity.recording_id
            work.mkdir(parents=True, exist_ok=True)
            try:
                with stage("write:base"):
                    scene = write_base_layer(
                        identity,
                        scene_dir,
                        targets[paths.BASE_LAYER],
                        index=source,
                        work_dir=work,
                        hf_revision=self.commit_sha,
                        default_blueprint=self.default_blueprint(),
                    )
            finally:
                archives.remove_tree(work)
            written.append(paths.BASE_LAYER)
        clock: FrameClock | None = None
        hand_frames: list[HandFrame] = []
        profile: HandProfileDoc | None = None
        if wants[paths.HAND_POSE_LAYER] or wants[paths.HAND_MESH_LAYER] or wants[paths.OBJECT_POSE_LAYER] or wants[paths.OBJECT_MESH_LAYER]:
            clock = scene if scene is not None else read_frame_clock(scene_dir, key)
            hand_frames = read_hand_frames(self.config.root / hand_pose_file(key), clock) if source.has_hand_pose else []
        if wants[paths.HAND_POSE_LAYER] or wants[paths.HAND_MESH_LAYER]:
            profile = read_hand_profile(self.config.root / hand_profile_file(source.subject_id))
        if wants[paths.HAND_POSE_LAYER]:
            assert clock is not None and profile is not None
            with stage("write:hand_pose"):
                write_hand_pose_layer(identity, clock, hand_frames, profile.text, targets[paths.HAND_POSE_LAYER])
            written.append(paths.HAND_POSE_LAYER)
        caption: Caption | None = read_json(self.config.root / caption_file(key), Caption) if wants[paths.CAPTIONS_LAYER] else None
        if wants[paths.CAPTIONS_LAYER]:
            assert caption is not None
            with stage("write:captions"):
                write_captions_layer(identity, caption, targets[paths.CAPTIONS_LAYER])
            written.append(paths.CAPTIONS_LAYER)
        object_track: ObjectTrack | None = None
        if wants[paths.OBJECT_POSE_LAYER] or wants[paths.OBJECT_MESH_LAYER]:
            assert clock is not None
            object_track = read_object_frames(self.config.root / object_pose_file(key), clock)
            if object_track.clock_offset_s != 0.0:
                print(f"{identity.sequence_key}: object_pose timestamps are offset by {object_track.clock_offset_s:.6g} s from frame_info; aligned by index")
        if wants[paths.OBJECT_POSE_LAYER]:
            assert clock is not None and object_track is not None
            frames: list[ObjectFrame] = object_track.frames
            metrics: ObjectSanity = object_sanity(
                frames, list((scene.headsets if scene is not None else read_headset_calibrations(scene_dir, clock)).values()), hand_frames
            )
            with stage("write:object_pose"):
                write_object_pose_layer(identity, alias, clock, frames, metrics, targets[paths.OBJECT_POSE_LAYER], clock_offset_s=object_track.clock_offset_s)
            written.append(paths.OBJECT_POSE_LAYER)
        if wants[paths.OBJECT_MESH_LAYER]:
            assert object_track is not None
            if any(frame.posed for frame in object_track.frames):
                asset: MeshAsset = stripped_mesh(self.config.root, alias)
                assert clock is not None
                with stage("write:object_mesh"):
                    write_object_mesh_layer(identity, alias, clock, object_track.frames, asset.mesh_id, asset.path, targets[paths.OBJECT_MESH_LAYER])
                written.append(paths.OBJECT_MESH_LAYER)
            else:
                # A mesh with no pose row would sit at the world origin; the track carries no posed frame.
                print(f"{identity.sequence_key}: no object_mesh: the object track has no posed frame")
        if wants[paths.HAND_MESH_LAYER]:
            assert clock is not None and profile is not None
            with stage("write:hand_mesh"):
                write_hand_mesh_layer(identity, clock, hand_frames, profile.model, targets[paths.HAND_MESH_LAYER])
            written.append(paths.HAND_MESH_LAYER)
        if wants[paths.BASE_LAYER] and not self.config.keep_raw:
            for video in scene_dir.glob("*.mp4"):
                video.unlink()
        print(f"done {identity.sequence_key}: {', '.join(written)}")
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        panes: dict[Show3dCamera, rrb.Spatial2DView] = {
            camera: blueprints.camera_view(
                schema.cam_path(camera.rig, camera.cam).removeprefix("/world/"), camera.rig, camera.cam, contents=pane_contents(camera)
            )
            for camera in CAMERAS
        }
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Back rig frame",
                origin="/world",
                contents=world_contents(),
                eye_controls=blueprints.eye_controls_from_pose((1.4, 0.7, 1.1), (0.25, -0.2, 0.1), (0.0, 1.0, 0.0)),
            ),
            ego_panes=[panes[camera] for camera in HEADSET_CAMERAS],
            exo_panes=[pane for camera, pane in panes.items() if camera not in HEADSET_CAMERAS],
            instruction=rrb.TextDocumentView(name="Instruction", origin=schema.instruction_path(), contents=[schema.instruction_path()]),
        )

    def table_fields(self) -> TableFields:
        """Cards show what the episode is; the table adds the coverage numbers worth sorting by.

        The caption wraps inside a card but runs wide in a table row, so the table puts it last,
        where it no longer pushes the short columns off screen.
        """
        action: TableField = TableField("property:episode:action", "action")
        obj: TableField = TableField("property:episode:object_alias", "object")
        caption: TableField = TableField("property:captions:overall_caption", "caption")
        subject: TableField = TableField("property:episode:subject_id", "subject")
        split: TableField = TableField("property:episode:split", "split")
        coverage: tuple[TableField, ...] = (
            TableField("property:hand_pose:coverage_left_high_conf", "left hand coverage"),
            TableField("property:hand_pose:coverage_right_high_conf", "right hand coverage"),
            TableField("property:object_pose:coverage", "object coverage"),
        )
        return TableFields(cards=(action, obj, caption, subject, split), table=(action, obj, subject, split, *coverage, caption))

    def table_blueprint(self) -> rrb.Blueprint:
        """The scene without any video beside the one decoded headset stream with its projected overlays.

        A card gives each preview view the same width, so the container carries no shares.
        """
        headset: Show3dCamera = HEADSET_CAMERAS[0]
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Scene",
                    origin="/world",
                    contents=preview_world_contents(),
                    # Closer than the full layout's eye, so the hands and object read at card size with every frustum in frame.
                    eye_controls=blueprints.eye_controls_from_pose((1.25, 0.6, 1.0), (0.2, -0.15, 0.05), (0.0, 1.0, 0.0)),
                ),
                blueprints.camera_view(headset.source_name, headset.rig, headset.cam, contents=pane_contents(headset)),
            ),
            collapse_panels=True,
        )


def metadata_files(key: str) -> list[str]:
    """The complete frame-clock input set."""
    return [f"scenes/{key}/metadata/{name}.json" for name in ("recording_info", "frame_info")]


def base_files(source: IndexRow, key: str) -> list[str]:
    """Plan camera files from index availability before fetching metadata."""
    return [
        *metadata_files(key),
        *(
            f"scenes/{key}/{name}"
            for camera in source.cameras
            for name in (f"{camera.source_name}.mp4", f"blur_info/{camera.source_name}.mp4.json")
        ),
        *(calibration_file(key, camera) for camera in source.cameras),
        f"scenes/{key}/blur_info/config.json",
        *([object_pose_file(key)] if source.has_object_pose else []),
    ]
