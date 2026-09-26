"""Assembly101 discovery and three-layer conversion from the extracted mirror."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.assembly101_actions import Actions, read_actions
from dataforge.datasets.assembly101_layers import Scene, camera_sources, read_scene, write_actions, write_base, write_hands
from dataforge.datasets.assembly101_source import EGO_RIG, EXO_SERIALS, FRAME_RATE, NAS_ROOT, POSE_MEMBERS, pose_path, read_manifest
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity


@dataclass
class Assembly101Config(DataforgeDatasetConfig):
    """Read-only inputs; outputs use DATAFORGE_OUTPUT_ROOT."""

    command: ClassVar[str] = "assembly101"
    """CLI and catalog name."""
    _target: type = field(default_factory=lambda: Assembly101Dataset)
    """Dataset implementation."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "assembly101")
    """Mirror layout containing videos, extracted pose members and nimble calibration."""
    annotations_root: Path = field(default_factory=lambda: paths.raw_root() / "assembly101/official/annotations")
    """Independent read-only official annotations root."""
    sequences: tuple[str, ...] | None = None
    """Sequence directories to include, or all present videos."""
    frame_limit: int | None = None
    """Optional prefix, written under preview-first<N> instead of the full output tree."""

    def __post_init__(self) -> None:
        if self.frame_limit is not None and self.frame_limit <= 0:
            raise ValueError("frame_limit must be positive")


class Assembly101Dataset(DataforgeDataset[Assembly101Config, str]):
    """No meshes or object annotations are fabricated."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, "actions")

    def __init__(self, config: Assembly101Config) -> None:
        super().__init__(config)
        self.actions: dict[str, Actions] | None = None
        self.pose_expected: dict[str, bool] = read_manifest(config.root)

    def discover(self) -> list[tuple[SequenceIdentity, str]]:
        folder: Path = self.config.root / "videos/av1-720-new"
        keys: list[str] = sorted(path.name for path in folder.iterdir() if path.is_dir() and any(path.glob("*.mp4"))) if folder.is_dir() else []
        if self.config.sequences is not None:
            missing: set[str] = set(self.config.sequences) - set(keys)
            if missing:
                raise FileNotFoundError(f"{folder}: missing sequences {sorted(missing)}")
            keys = [key for key in keys if key in self.config.sequences]
        return [(SequenceIdentity("assembly101", (key,)), key) for key in keys]

    def download(self) -> None:
        """Verify only; fetch_pose_members is the separate driver-controlled transport."""
        selected: list[tuple[SequenceIdentity, str]] = self.discover()
        if not selected:
            raise FileNotFoundError(f"no Assembly101 videos under {self.config.root}")
        for _, sequence in selected:
            self.verify_sequence(sequence)
        print(f"assembly101: verified {len(selected)} local sequences; no files changed")

    def verify_sequence(self, sequence: str) -> bool:
        """Refuse partial pose trees; truly video-only sequences have no pose members."""
        if len(camera_sources(self.config.root, sequence)) != 12:
            raise FileNotFoundError(f"{sequence}: expected 12 videos in {self.config.root / 'videos/av1-720-new'}")
        members: list[Path] = [pose_path(self.config.root, member, sequence) for member in POSE_MEMBERS]
        if not any(path.exists() for path in members) and self.pose_expected.get(sequence) is False:
            return False
        if sequence not in self.pose_expected and not any(path.exists() for path in members):
            raise FileNotFoundError(f"{self.config.root / 'manifests/sequences.csv'}: explicit video_only=True row required for {sequence}")
        for path in members:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Assembly101 pose asset missing: {path}")
        calibration: Path = self.config.root / "assemblyhands-toolkit/calib/nimble_json_calib"
        if not any(calibration.glob("*.json")):
            raise FileNotFoundError(f"Assembly101 nimble assets missing: {calibration}")

        return True

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        root: Path = paths.output_root()
        if self.config.frame_limit is not None:
            root /= f"preview-first{self.config.frame_limit}"
        return {layer: paths.rrd_path(root, layer=layer, identity=identity) for layer in self.layers}

    def convert(self, identity: SequenceIdentity, source: str, *, force: bool) -> Path:
        """Publish only available layers, atomically, sharing the recording identity."""
        targets: dict[str, Path] = self.targets(identity)
        for target in targets.values():
            if any(target.resolve().is_relative_to(root.resolve()) for root in (self.config.root, self.config.annotations_root, NAS_ROOT)):
                raise ValueError(f"refusing conversion output beneath raw inputs or NAS: {target}")
        with self.timer.stage("fetch"):
            has_poses: bool = self.verify_sequence(source)
            if self.actions is None:
                self.actions = read_actions(self.config.annotations_root, {sequence for _, sequence in self.discover()})
            actions: Actions = self.actions[source]
        available: list[str] = [paths.BASE_LAYER]
        if has_poses:
            available.append(paths.HAND_POSE_LAYER)
        if actions.coarse or actions.fine:
            available.append("actions")
        pending: list[str] = [layer for layer in available if not writing.should_skip(targets[layer], force=force)]
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("fetch"):
            scene: Scene = read_scene(self.config.root, source, self.config.frame_limit, has_poses=has_poses)
        self.timer.capture_s = (
            max(
                max(camera.num_frames for camera in scene.cameras),
                int(scene.poses.frames[-1]) + 1 if scene.poses is not None and len(scene.poses.frames) else 0,
            )
            / FRAME_RATE
        )
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
                if layer == paths.BASE_LAYER:
                    write_base(recording, scene, identity, actions, self.config.frame_limit, self.timer)
                elif layer == paths.HAND_POSE_LAYER:
                    write_hands(recording, self.config.root, source, scene, self.config.frame_limit)
                else:
                    write_actions(recording, actions, self.config.frame_limit)
            if force and layer == paths.BASE_LAYER:
                for optional in (paths.HAND_POSE_LAYER, "actions"):
                    if optional not in available:
                        targets[optional].unlink(missing_ok=True)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(name="Assembly101", origin="/world", contents=world_contents(), eye_controls=table_eye()),
            ego_panes=[blueprints.camera_view(f"Ego {cam}", EGO_RIG, cam, contents=pane_contents(EGO_RIG, cam)) for cam in range(4)],
            exo_panes=[blueprints.camera_view(serial, rig, 0, contents=pane_contents(rig, 0)) for rig, serial in enumerate(EXO_SERIALS)],
            instruction=rrb.TextDocumentView(name="Fine actions", origin="/task/actions/fine"),
        )

    def table_blueprint(self) -> rrb.Blueprint:
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="Scene",
                    origin="/world",
                    contents=[*world_contents(), *[f"- {schema.video_path(rig, cam)}" for rig, cam in camera_slots()]],
                    eye_controls=table_eye(),
                ),
                blueprints.camera_view("C10095", 0, 0, contents=pane_contents(0, 0)),
            ),
            collapse_panels=True,
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


def table_eye() -> rrb.EyeControls3D:
    """Tightest oblique eye with the exo cameras and the table in a 2:1 card (+Y up).

    Solved from the shipped extrinsics, which put the fixed cameras within ~0.1 m of the same place every day.
    """
    return blueprints.eye_controls_from_pose((1.17, 1.57, -0.62), (0.29, 0.6, 0.13), (0.0, 1.0, 0.0))
