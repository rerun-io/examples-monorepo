"""SHOW3D discovery and one-scene-at-a-time BASE publication."""

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
from dataforge.datasets.show3d_source import CAMERAS, IndexRow, RecordingInfo, read_json
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
    """Fetch one bundle, write BASE atomically, then remove only its source videos."""

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

    def convert(self, identity: SequenceIdentity, source: IndexRow, *, force: bool) -> Path:
        from dataforge.datasets.show3d_layers import write_base_layer

        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        if writing.should_skip(target, force=force):
            return target
        key: str = f"{source.subject_id}/{source.scene_id}"
        files: list[str] = [f"scenes/{key}/metadata/{name}.json" for name in ("recording_info", "frame_info")]
        transports.hf_fetch_files(REPO_ID, files, local_dir=self.config.root, revision=self.commit_sha)
        scene_dir: Path = self.config.root / "scenes" / key
        info: RecordingInfo = read_json(scene_dir / "metadata/recording_info.json", RecordingInfo)
        files = [f"scenes/{key}/blur_info/config.json"]
        for camera in info.resolution:
            files.extend(f"scenes/{key}/{name}" for name in (f"{camera}.mp4", f"camera_calibration/{camera}.json", f"blur_info/{camera}.mp4.json"))
        for present, tree, name in (
            (source.has_hand_pose, "hand_pose/v2", "hand_pose"),
            (source.has_object_pose, "object_pose/v1", "object_pose"),
            (source.has_caption, "captions/v1", "caption"),
        ):
            if present:
                files.append(f"{tree}/scenes/{key}/{name}.json")
        transports.hf_fetch_files(REPO_ID, files, local_dir=self.config.root, revision=self.commit_sha)
        work: Path = self.config.root / "work" / identity.recording_id
        work.mkdir(parents=True, exist_ok=True)
        try:
            write_base_layer(identity, scene_dir, target, work_dir=work, hf_revision=self.commit_sha, default_blueprint=self.default_blueprint())
        finally:
            archives.remove_tree(work)
        if not self.config.keep_raw:
            for video in scene_dir.glob("*.mp4"):
                video.unlink()
        print(f"done {identity.sequence_key} → {target}")
        return target

    def default_blueprint(self) -> rrb.Blueprint:
        contents: list[str] = pane_contents()
        ego: list[rrb.Spatial2DView] = []
        exo: list[rrb.Spatial2DView] = []
        for camera in CAMERAS:
            view: rrb.Spatial2DView = blueprints.camera_view(camera.source_name, camera.rig, camera.cam, contents=contents)
            (ego if camera.rig == 1 else exo).append(view)
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
                    row_shares=[3, 2],
                ),
                rrb.Grid(*exo, grid_columns=2),
                column_shares=[3, 2],
            ),
            rrb.TimePanel(timeline=schema.TIMELINE),
            collapse_panels=True,
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return rrb.Blueprint(blueprints.camera_view("headset0", 1, 0, contents=[f"+ {schema.video_path(1, 0)}"]), collapse_panels=True)
