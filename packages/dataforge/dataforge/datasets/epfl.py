"""EPFL-Smart-Kitchen-30: seven disjoint layers on the shipped device clock."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int64

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.epfl_actions import read_actions
from dataforge.datasets.epfl_layers import start_parameters, write_actions, write_base, write_hand_pose, write_pose, write_projections
from dataforge.datasets.epfl_mesh import MeshWriter
from dataforge.datasets.epfl_source import (
    CAMERA_NAMES,
    EGO_RIG,
    EXO_CAMERAS,
    EXO_RIGS,
    FITS,
    ExoCamera,
    FitSpec,
    PoseRow,
    pose_batches,
    read_cameras,
    read_timestamps,
)
from dataforge.identity import SequenceIdentity

CORPUS_SCENE_CENTRE: tuple[float, float] = (2.56, -0.02)
"""Mean over the 51 released sessions of the nine exo-camera centroids (x, y), metres.

The world frame moves between sessions (per-session centroid std 0.86 m in x and 0.87 m
in y), so converted recordings embed their own centre; this is the catalog default."""
BODY_MID_Z: float = -0.7
"""World z of the body's mid-height: the world origin sits near head height (nose z ≈ 0,
ankles z ≈ -1.39 in the shipped SMPL keypoints)."""
FLOOR_Z: float = -1.4
"""World z of the floor, at the shipped ankle keypoints; the grid draws there, not at the head."""
BODY_MESH_STRIDE: int = 3
"""body_mesh keeps every third 30 Hz frame (10 Hz): a display layer, by decision (2026-09-25).

Full-rate SMPL vertices cost 82 KB per frame (4.3 GB for a 29-min session, 4x its videos), and
Rerun 0.38 has no mesh skinning to pose one logged mesh from joint transforms. The SMPL
parameters (body_pose) and the keypoints (hand_pose) stay at full rate."""


def scene_centre(cameras: dict[str, ExoCamera]) -> tuple[float, float]:
    """World (x, y) centroid of the session's static exo camera centres."""
    centres = [-camera.word2cam[:3, :3].T @ camera.word2cam[:3, 3] for camera in cameras.values()]
    x, y, _ = np.mean(centres, axis=0)
    return float(x), float(y)


@dataclass
class EpflConfig(DataforgeDatasetConfig):
    """Read poses from a read-only root and videos from a separate local root."""

    command: ClassVar[str] = "epfl"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: EpflDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "epfl-smart-kitchen")
    """Root containing Public_release_pose."""
    video_root: Path | None = None
    """Root containing Public_release_videos; defaults to root."""
    sequences: tuple[str, ...] | None = None
    """Split/subject/session selections."""
    frame_limit: int | None = None
    """Keep the first N frames in a separate preview tree."""
    smpl_model_root: Path = field(default_factory=lambda: paths.raw_root() / "body_models")
    """Official neutral SMPL model root, containing smpl/SMPL_NEUTRAL.pkl."""

    def __post_init__(self) -> None:
        if self.frame_limit is not None and self.frame_limit <= 0:
            raise ValueError("frame_limit must be positive")

    @property
    def pose_root(self) -> Path:
        """Public_release_pose under root."""
        return self.root / "Public_release_pose"

    @property
    def videos_root(self) -> Path:
        """Public_release_videos under video_root, falling back to root."""
        return (self.video_root or self.root) / "Public_release_videos"


class EpflDataset(DataforgeDataset[EpflConfig, str]):
    """Session discovery and atomic multi-layer conversion with one pose CSV pass."""

    layers = ("base", "hand_pose", "body_pose", "hand_mesh", "body_mesh", "projections", "actions")

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        root = paths.output_root()
        if self.config.frame_limit is not None:
            root = root / f"preview-first{self.config.frame_limit}"
        return {layer: paths.rrd_path(root, layer=layer, identity=identity) for layer in self.layers}

    def download(self) -> None:
        """Verify available source sessions without fetching or changing any source."""
        if not self.discover():
            raise FileNotFoundError(f"No complete EPFL sessions under {self.config.root} and {self.config.video_root or self.config.root}")

    def discover(self) -> list[tuple[SequenceIdentity, str]]:
        """Skip incomplete poses or absent local videos/meta with a concrete reason."""
        pose_root = self.config.pose_root
        keys = sorted(str(path.relative_to(pose_root)) for split in ("train", "test") for path in (pose_root / split).glob("*/*") if path.is_dir())
        if self.config.sequences is not None:
            missing = set(self.config.sequences) - set(keys)
            if missing:
                raise ValueError(f"EPFL selections absent from pose root: {sorted(missing)}")
            keys = [key for key in keys if key in self.config.sequences]
        result = []
        for key in keys:
            pose = pose_root / key
            video = self.config.videos_root / key
            required = [pose / "pose_3d" / f"pose3d_{model}.csv" for model in ("mano", "smpl")]
            required += [pose / "annotations" / name for name in ("actions_annotations.xlsx", "activity_annotations.json")]
            required += [video / "videos" / f"{name}.mp4" for name in CAMERA_NAMES]
            required += [video / "meta_data" / name for name in ("camera_matrix.json", "timestamps.txt", "holo_data_wpose.csv")]
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                print(f"skip {key}: missing {', '.join(absent)}")
                continue
            result.append((SequenceIdentity("epfl", tuple(key.split("/"))), key))
        return result

    def convert(self, identity: SequenceIdentity, source: str, *, force: bool) -> Path:
        """Open pending pose layers together so both large CSVs are read once."""
        targets = self.targets(identity)
        pending = [layer for layer, target in targets.items() if not writing.should_skip(target, force=force)]
        if not pending:
            return targets["base"]
        work_root: Path = (paths.output_root() / "work").resolve()
        for target in (*targets.values(), work_root):
            for raw in (Path("/mnt/nas"), self.config.root, self.config.video_root or self.config.root):
                if target.resolve().is_relative_to(raw.resolve()):
                    raise ValueError(f"EPFL conversion output must be local and outside raw roots: {target}")
        writer: MeshWriter | None = MeshWriter(self.config.smpl_model_root) if any(layer in pending for layer in ("hand_mesh", "body_mesh")) else None
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
        for layer in ("base", "actions"):
            if layer not in pending:
                continue
            with (
                self.timer.stage(f"write:{layer}"),
                writing.atomic_recording(
                    targets[layer],
                    recording_id=identity.recording_id,
                    default_blueprint=self.default_blueprint(scene_centre(cameras)) if layer == "base" else None,
                    send_properties=layer == "base",
                ) as recording,
            ):
                if layer == "base":
                    write_base(recording, video, cameras, ego_camera, times, source_count, identity, actions, self.timer, work_root)
                else:
                    write_actions(recording, actions, times)
        pose_layers: list[str] = [layer for layer in pending if layer not in ("base", "actions")]
        if not pose_layers:
            return targets["base"]
        layer_specs: dict[str, tuple[FitSpec, ...]] = {"hand_pose": FITS[:2], "body_pose": FITS[2:], "hand_mesh": FITS[:2], "body_mesh": FITS[2:]}
        with ExitStack() as stack:
            recordings: dict[str, rr.RecordingStream] = {
                layer: stack.enter_context(writing.atomic_recording(targets[layer], recording_id=identity.recording_id, send_properties=False))
                for layer in pose_layers
            }
            layer_writers: dict[str, Callable[[list[PoseRow], Int64[np.ndarray, "n"], Int64[np.ndarray, "n"]], None]] = {}
            for layer, recording in recordings.items():
                if layer in ("hand_pose", "body_pose"):
                    start_parameters(recording, layer_specs[layer])
                    layer_writers[layer] = partial(write_hand_pose if layer == "hand_pose" else write_pose, recording, specs=layer_specs[layer])
                elif layer in ("hand_mesh", "body_mesh"):
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
                        if layer == "body_mesh":
                            keep: Int64[np.ndarray, "k"] = np.flatnonzero(frames[start:stop] % BODY_MESH_STRIDE == 0)
                            write([rows[i] for i in keep], times[start:stop][keep], frames[start:stop][keep])
                        else:
                            write(rows, times[start:stop], frames[start:stop])
                start = stop
        return targets["base"]

    def default_blueprint(self, centre: tuple[float, float] = CORPUS_SCENE_CENTRE) -> rrb.Blueprint:
        """Exo/ego layout with the 3D eye orbiting the kitchen at body mid-height.

        Args:
            centre: World (x, y) the eye orbits; convert passes the session's exo-camera
                centroid, the catalog default uses the corpus mean.
        """
        target = (centre[0], centre[1], BODY_MID_Z)
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Kitchen",
                origin="/world",
                contents=["+ /world/**", *(f"- {schema.coco133_uv_projected_path(rig, 0)}" for rig in EXO_RIGS)],
                eye_controls=blueprints.eye_controls_from_pose((target[0] - 3.0, target[1] - 3.0, target[2] + 3.5), target, (0.0, 0.0, 1.0)),
                line_grid=rrb.LineGrid3D(visible=True, plane=rr.components.Plane3D.XY.with_distance(FLOOR_Z)),
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
                    contents=[f"+ {schema.video_path(rig, 0)}", f"+ {schema.coco133_uv_projected_path(rig, 0)}", *(f"+ {spec.mesh_path}" for spec in FITS)],
                )
                for rig, name in enumerate(EXO_CAMERAS)
            ],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return rrb.Blueprint(
            blueprints.camera_view("output0", 0, 0, contents=[f"+ {schema.video_path(0, 0)}", f"+ {schema.coco133_uv_projected_path(0, 0)}"]),
            collapse_panels=True,
        )

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("subject", "split", "activity")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
