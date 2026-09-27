"""EPFL-Smart-Kitchen-30: seven disjoint layers on the shipped device clock."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from huggingface_hub.errors import HfHubHTTPError
from jaxtyping import Int64

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.datasets.epfl_actions import read_actions
from dataforge.datasets.epfl_download import (
    SMPL_FILE,
    SMPL_REPO,
    SMPL_REVISION,
    SOURCE_REPO,
    VIDEO_DIR,
    FetchError,
    complete_sessions,
    fetch,
    list_source,
    session_files,
)
from dataforge.datasets.epfl_layers import start_parameters, write_actions, write_base, write_hand_pose, write_pose, write_projections
from dataforge.datasets.epfl_mesh import MeshWriter
from dataforge.datasets.epfl_source import (
    EGO_RIG,
    EXO_CAMERAS,
    EXO_RIGS,
    FITS,
    SOURCE_REVISION,
    FitSpec,
    PoseRow,
    pose_batches,
    read_cameras,
    read_timestamps,
)
from dataforge.identity import SequenceIdentity
from dataforge.transports import FetchReport

KITCHEN_UP: tuple[float, float, float] = (-0.03, -0.81, -0.58)
"""World up in output0's camera frame. The world frame moves between sessions, but the nine
cameras keep one layout in output0's frame (measured on the sample), so the kitchen 3D view uses it."""
SCENE_EYE: rrb.EyeControls3D = blueprints.eye_controls_from_pose((0.63, -0.53, -1.19), (0.0, -0.25, 1.31), KITCHEN_UP)
"""Tightest oblique eye with all nine exo cameras and the walking area (a standing person) in a 2:1 card, in output0's frame."""
KITCHEN_GRID: rrb.LineGrid3D = rrb.LineGrid3D(visible=True, plane=rr.components.Plane3D(normal=KITCHEN_UP, distance=-1.82))
"""The floor (shipped ankle height) in output0's frame."""
BODY_MESH_STRIDE: int = 3
"""body_mesh keeps every third 30 Hz frame (10 Hz): a display layer, by decision (2026-09-25).

Full-rate SMPL vertices cost 82 KB per frame (4.3 GB for a 29-min session, 4x its videos), and
Rerun 0.38 has no mesh skinning to pose one logged mesh from joint transforms. The SMPL
parameters (body_pose) and the keypoints (hand_pose) stay at full rate."""


@dataclass
class EpflConfig(FrameLimitedConfig):
    """Raw tree as `dataforge-download epfl` lays it out under root; videos and SMPL may live elsewhere."""

    command: ClassVar[str] = "epfl"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: EpflDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "epfl")
    """Raw root: Public_release_pose (and by default Public_release_videos and body_models), as download() writes it."""
    video_root: Path | None = None
    """Root containing Public_release_videos; defaults to root. download() fetches videos here too."""
    sequences: tuple[str, ...] | None = None
    """Split/subject/session selections."""
    smpl_model_root: Path | None = None
    """Official neutral SMPL model root, containing smpl/SMPL_NEUTRAL.pkl; defaults to root/body_models."""

    @property
    def pose_root(self) -> Path:
        """Public_release_pose under root."""
        return self.root / "Public_release_pose"

    @property
    def videos_root(self) -> Path:
        """Public_release_videos under video_root, falling back to root."""
        return (self.video_root or self.root) / VIDEO_DIR

    @property
    def smpl_root(self) -> Path:
        """smpl_model_root, falling back to root/body_models (where download() puts the model)."""
        return self.smpl_model_root or self.root / "body_models"

    def local_root(self, repo_path: str) -> Path:
        """Directory a mirror path lives under: video_root for videos, root for everything else."""
        return (self.video_root or self.root) if repo_path.startswith(f"{VIDEO_DIR}/") else self.root


class EpflDataset(DataforgeDataset[EpflConfig, str]):
    """Session discovery and atomic multi-layer conversion with one pose CSV pass."""

    layers = (
        paths.BASE_LAYER,
        paths.HAND_POSE_LAYER,
        paths.BODY_POSE_LAYER,
        paths.HAND_MESH_LAYER,
        paths.BODY_MESH_LAYER,
        paths.PROJECTIONS_LAYER,
        paths.ACTIONS_LAYER,
    )

    def remote_sequences(self) -> list[RemoteSequence]:
        """One entry per session the pinned mirror ships complete; one Hub listing call, nothing downloaded."""
        sessions = complete_sessions(list_source(SOURCE_REPO, SOURCE_REVISION))
        return [RemoteSequence(key, sum(file.size_bytes for file in files), tuple(file.path for file in files)) for key, files in sessions.items()]

    def download(self) -> None:
        """Fetch the selected sessions' files, then the SMPL model unless it is in place; never deletes a raw file.

        Files already complete are skipped. Failures are collected and raised once, after every other file ran.
        """
        sessions = complete_sessions(list_source(SOURCE_REPO, SOURCE_REVISION))
        keys = list(sessions) if self.config.sequences is None else list(self.config.sequences)
        unknown = sorted(set(keys) - set(sessions))
        if unknown:
            raise ValueError(f"{self.config.command}: unknown sequence(s) {unknown} at {SOURCE_REPO}@{SOURCE_REVISION[:8]}; see `dataforge-download {self.config.command} --list-remote`")
        jobs = []
        for key in keys:
            for file in sessions[key]:
                local_root = self.config.local_root(file.path)
                jobs.append((SOURCE_REPO, SOURCE_REVISION, file, local_root / file.path, local_root / ".download"))
        smpl_dest = self.config.smpl_root / "smpl/SMPL_NEUTRAL.pkl"
        failures: list[str] = []
        if not smpl_dest.is_file():
            # The model repo is private: a stranger places the official neutral model at smpl_dest instead.
            try:
                (smpl,) = [file for file in list_source(SMPL_REPO, SMPL_REVISION, SMPL_FILE.rsplit("/", 1)[0]) if file.path == SMPL_FILE]
                jobs.append((SMPL_REPO, SMPL_REVISION, smpl, smpl_dest, self.config.smpl_root / ".download"))
            except HfHubHTTPError as error:
                failures.append(
                    f"{SMPL_REPO}: HTTP {error.response.status_code}; log in with access or place the official SMPL_NEUTRAL.pkl at {smpl_dest}"
                )
        report = FetchReport()
        for job in jobs:
            try:
                report.count(fetch(*job), job[2].size_bytes)
            except FetchError as error:
                failures.append(str(error))
        print(f"epfl download: {len(keys)} sessions; {report.summary()}")
        if failures:
            raise ValueError(f"epfl download: {len(failures)} failed, rerun to retry:\n" + "\n".join(failures))

    def discover(self) -> list[tuple[SequenceIdentity, str]]:
        """Sessions whose raw files are all present; others (absent, partial, or pruned after conversion) are skipped."""
        pose_root = self.config.pose_root
        keys = sorted(str(path.relative_to(pose_root)) for split in ("train", "test") for path in (pose_root / split).glob("*/*") if path.is_dir())
        if self.config.sequences is not None:
            missing = set(self.config.sequences) - set(keys)
            if missing:
                raise ValueError(f"EPFL selections absent from pose root: {sorted(missing)}")
            keys = [key for key in keys if key in self.config.sequences]
        result = []
        for key in keys:
            required = session_files(key)
            local = [self.config.local_root(path) / path for path in required]
            absent = [path for path in local if not path.is_file()]
            if absent:
                print(f"skip {key}: {len(absent)} of {len(required)} raw files missing, first {absent[0]}")
                continue
            result.append((SequenceIdentity("epfl", tuple(key.split("/"))), key))
        return result

    def convert(self, identity: SequenceIdentity, source: str, *, force: bool) -> Path:
        """Open pending pose layers together so both large CSVs are read once."""
        targets, pending = self.pending_layers(
            identity, force=force, roots=[paths.NAS_ROOT, self.config.root, self.config.video_root or self.config.root]
        )
        if not pending:
            return targets[paths.BASE_LAYER]
        writer: MeshWriter | None = MeshWriter(self.config.smpl_root) if any(layer in pending for layer in (paths.HAND_MESH_LAYER, paths.BODY_MESH_LAYER)) else None
        pose = self.config.pose_root / source
        video = self.config.videos_root / source
        with self.timer.stage("fetch"):
            timestamps = read_timestamps(video / "meta_data/timestamps.txt")
            cameras, ego_camera = read_cameras(video / "meta_data/camera_matrix.json")
            actions = read_actions(pose / "annotations")
        source_count = len(timestamps)
        times = timestamps[: self.config.frame_limit]
        frames = np.arange(len(times), dtype=np.int64)
        self.timer.capture_s = float((times[-1] - times[0]) / 1e9 + 1 / 30)
        layer_specs: dict[str, tuple[FitSpec, ...]] = {paths.HAND_POSE_LAYER: FITS[:2], paths.BODY_POSE_LAYER: FITS[2:], paths.HAND_MESH_LAYER: FITS[:2], paths.BODY_MESH_LAYER: FITS[2:]}

        def write_pose_layers(recordings: dict[str, rr.RecordingStream]) -> None:
            """Every pose-fed layer from one pass over the pose CSVs, batch by batch."""
            layer_writers: dict[str, Callable[[list[PoseRow], Int64[np.ndarray, "n"], Int64[np.ndarray, "n"]], None]] = {}
            for layer, recording in recordings.items():
                if layer in (paths.HAND_POSE_LAYER, paths.BODY_POSE_LAYER):
                    start_parameters(recording, layer_specs[layer])
                    layer_writers[layer] = partial(write_hand_pose if layer == paths.HAND_POSE_LAYER else write_pose, recording, specs=layer_specs[layer])
                elif layer in (paths.HAND_MESH_LAYER, paths.BODY_MESH_LAYER):
                    assert writer is not None
                    writer.start(recording, layer_specs[layer])
                    layer_writers[layer] = partial(writer.write, recording, specs=layer_specs[layer])
                else:
                    layer_writers[layer] = partial(write_projections, recording, cameras)
            batches = iter(pose_batches(pose / "pose_3d", len(times), total=source_count))
            start: int = 0
            while True:
                with self.timer.stage("fetch:poses"):
                    rows = next(batches, None)
                if rows is None:
                    break
                stop: int = start + len(rows)
                for layer, write in layer_writers.items():
                    with self.timer.stage(f"write:{layer}"):
                        if layer == paths.BODY_MESH_LAYER:
                            keep: Int64[np.ndarray, "k"] = np.flatnonzero(frames[start:stop] % BODY_MESH_STRIDE == 0)
                            write([rows[i] for i in keep], times[start:stop][keep], frames[start:stop][keep])
                        else:
                            write(rows, times[start:stop], frames[start:stop])
                start = stop

        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            paths.BASE_LAYER: lambda recording: write_base(recording, video, cameras, ego_camera, times, source_count, identity, actions, self.timer),
            paths.ACTIONS_LAYER: lambda recording: write_actions(recording, actions, times),
        }
        self.write_layers(identity, targets, pending, writers, together=write_pose_layers)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """Exo/ego layout; the 3D eye auto-fits each session, whose world frame moves (centroid std ~0.9 m)."""
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Kitchen",
                origin=schema.cam_path(0, 0),
                contents=["+ /world/**", *(f"- {schema.coco133_uv_projected_path(rig, 0)}" for rig in EXO_RIGS)],
                eye_controls=SCENE_EYE,
                line_grid=KITCHEN_GRID,
            ),
            ego_panes=[
                blueprints.camera_view(
                    "HoloLens (shipped pose drift)", EGO_RIG, 0, contents=["+ /world/gt/**", f"+ {schema.pinhole_path(EGO_RIG, 0)}/**"]
                )
            ],
            # Exo panes also draw the meshes, through Rerun's pinhole despite the lens model: off by 1-4 px
            # mid-image, up to ~30 px at the edges (decision, 2026-09-25). The skeleton stays lens-projected.
            exo_panes=[
                blueprints.camera_view(
                    name,
                    rig,
                    0,
                    contents=[
                        f"+ {schema.video_path(rig, 0)}",
                        f"+ {schema.coco133_uv_projected_path(rig, 0)}",
                        *(f"+ {spec.mesh_path}" for spec in FITS),
                    ],
                )
                for rig, name in enumerate(EXO_CAMERAS)
            ],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the kitchen in 3D beside exo camera output0."""
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Kitchen",
                origin=schema.cam_path(0, 0),
                contents=[
                    "+ /world/**",
                    *(f"- {schema.coco133_uv_projected_path(rig, 0)}" for rig in EXO_RIGS),
                    *blueprints.video_exclusions((rig, 0) for rig in (*EXO_RIGS, EGO_RIG)),
                ],
                eye_controls=SCENE_EYE,
                line_grid=KITCHEN_GRID,
            ),
            blueprints.camera_view(
                "output0",
                0,
                0,
                contents=[
                    f"+ {schema.video_path(0, 0)}",
                    f"+ {schema.coco133_uv_projected_path(0, 0)}",
                    *(f"+ {spec.mesh_path}" for spec in FITS),
                ],
            ),
        )

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("subject", "split", "activity")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
