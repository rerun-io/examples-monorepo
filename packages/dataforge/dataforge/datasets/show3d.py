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
from dataforge.datasets.show3d_captions import Caption, write_captions_layer, write_properties_layer
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
from dataforge.datasets.show3d_object_source import ObjectFrame, read_object_frames
from dataforge.datasets.show3d_objects import ObjectSanity, object_sanity, write_object_mesh_layer, write_object_pose_layer
from dataforge.datasets.show3d_source import (
    CAMERAS,
    HEADSET_CAMERAS,
    FrameClock,
    IndexRow,
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

REPO_ID: str = "facebook/show3d-dataset"


def pane_contents() -> list[str]:
    """Everything under ``/world`` except the face-blur boxes, one explicit exclusion per camera.

    Rerun content filters honour exact paths and a trailing ``/**`` only: a rule such as
    ``- /world/**/blur_boxes`` matches nothing and hides nothing (verified with headless
    screenshots), so the exclusions are spelled out from the camera table.
    """
    return ["+ /world/**", *(f"- {schema.pinhole_path(camera.rig, camera.cam)}/blur_boxes" for camera in CAMERAS)]


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
        paths.PROPERTIES_LAYER,
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
            paths.PROPERTIES_LAYER: not writing.should_skip(targets[paths.PROPERTIES_LAYER], force=force),
            paths.OBJECT_POSE_LAYER: source.has_object_pose and not writing.should_skip(targets[paths.OBJECT_POSE_LAYER], force=force),
            paths.OBJECT_MESH_LAYER: source.has_object_pose and mesh is not None and not writing.should_skip(targets[paths.OBJECT_MESH_LAYER], force=force),
            paths.HAND_MESH_LAYER: source.has_hand_pose and not writing.should_skip(targets[paths.HAND_MESH_LAYER], force=force),
        }
        need_caption: bool = wants[paths.CAPTIONS_LAYER] or (wants[paths.PROPERTIES_LAYER] and source.has_caption)
        if wants[paths.OBJECT_POSE_LAYER] and mesh is None:
            print(f"{identity.sequence_key}: no object_mesh: alias {alias!r} has no HOT3D mesh mapping")
        if not any(wants.values()):
            return targets[paths.BASE_LAYER]
        key: str = identity.sequence_key
        files: set[str] = set(base_files(source, key) if wants[paths.BASE_LAYER] else [])
        if wants[paths.HAND_POSE_LAYER] or wants[paths.HAND_MESH_LAYER] or (wants[paths.OBJECT_POSE_LAYER] and source.has_hand_pose):
            files.update([*metadata_files(key), hand_pose_file(key), hand_profile_file(source.subject_id)])
        if need_caption:
            files.add(caption_file(key))
        if wants[paths.OBJECT_POSE_LAYER]:
            files.update([*metadata_files(key), object_pose_file(key), *(calibration_file(key, camera) for camera in HEADSET_CAMERAS)])
        self.fetch_missing(sorted(files))
        scene_dir: Path = self.config.root / "scenes" / key
        written: list[str] = []
        scene: Scene | None = None
        if wants[paths.BASE_LAYER]:
            work: Path = self.config.root / "work" / identity.recording_id
            work.mkdir(parents=True, exist_ok=True)
            try:
                scene = write_base_layer(
                    identity,
                    scene_dir,
                    targets[paths.BASE_LAYER],
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
        if wants[paths.HAND_POSE_LAYER] or wants[paths.HAND_MESH_LAYER] or wants[paths.OBJECT_POSE_LAYER]:
            clock = scene if scene is not None else read_frame_clock(scene_dir, key)
            hand_frames = read_hand_frames(self.config.root / hand_pose_file(key), clock) if source.has_hand_pose else []
        if wants[paths.HAND_POSE_LAYER] or wants[paths.HAND_MESH_LAYER]:
            profile = read_hand_profile(self.config.root / hand_profile_file(source.subject_id))
        if wants[paths.HAND_POSE_LAYER]:
            assert clock is not None and profile is not None
            write_hand_pose_layer(identity, clock, hand_frames, profile.text, targets[paths.HAND_POSE_LAYER])
            written.append(paths.HAND_POSE_LAYER)
        caption: Caption | None = read_json(self.config.root / caption_file(key), Caption) if need_caption else None
        if wants[paths.CAPTIONS_LAYER]:
            assert caption is not None
            write_captions_layer(identity, caption, targets[paths.CAPTIONS_LAYER])
            written.append(paths.CAPTIONS_LAYER)
        if wants[paths.PROPERTIES_LAYER]:
            write_properties_layer(identity, source, caption, targets[paths.PROPERTIES_LAYER])
            written.append(paths.PROPERTIES_LAYER)
        if wants[paths.OBJECT_POSE_LAYER]:
            assert clock is not None
            frames: list[ObjectFrame] = read_object_frames(self.config.root / object_pose_file(key), clock)
            metrics: ObjectSanity = object_sanity(
                frames, list((scene.headsets if scene is not None else read_headset_calibrations(scene_dir, clock)).values()), hand_frames
            )
            write_object_pose_layer(identity, alias, clock, frames, metrics, targets[paths.OBJECT_POSE_LAYER])
            written.append(paths.OBJECT_POSE_LAYER)
        if wants[paths.OBJECT_MESH_LAYER]:
            asset: MeshAsset = stripped_mesh(self.config.root, alias)
            write_object_mesh_layer(identity, alias, asset.mesh_id, asset.path, targets[paths.OBJECT_MESH_LAYER])
            written.append(paths.OBJECT_MESH_LAYER)
        if wants[paths.HAND_MESH_LAYER]:
            assert clock is not None and profile is not None
            write_hand_mesh_layer(identity, clock, hand_frames, profile.model, targets[paths.HAND_MESH_LAYER])
            written.append(paths.HAND_MESH_LAYER)
        if wants[paths.BASE_LAYER] and not self.config.keep_raw:
            for video in scene_dir.glob("*.mp4"):
                video.unlink()
        print(f"done {identity.sequence_key}: {', '.join(written)}")
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        contents: list[str] = pane_contents()
        ego: list[rrb.Spatial2DView] = []
        exo: list[rrb.Spatial2DView] = []
        for camera in CAMERAS:
            view: rrb.Spatial2DView = blueprints.camera_view(camera.source_name, camera.rig, camera.cam, contents=contents)
            (ego if camera in HEADSET_CAMERAS else exo).append(view)
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="Back rig frame",
                        origin="/world",
                        contents=contents,
                        eye_controls=blueprints.eye_controls_from_pose((1.4, 0.7, 1.1), (0.25, -0.2, 0.1), (0.0, 1.0, 0.0)),
                    ),
                    rrb.Horizontal(*ego),
                    rrb.TextDocumentView(name="Instruction", origin=schema.instruction_path(), contents=[schema.instruction_path()]),
                    row_shares=[3, 2, 1],
                ),
                rrb.Grid(*exo, grid_columns=2),
                column_shares=[3, 2],
            ),
            rrb.TimePanel(timeline=schema.TIMELINE),
            collapse_panels=True,
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return rrb.Blueprint(blueprints.camera_view("headset0", 1, 0, contents=[f"+ {schema.video_path(1, 0)}"]), collapse_panels=True)


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
