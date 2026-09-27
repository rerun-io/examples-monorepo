"""Two HOT3D catalog datasets sharing one raw reader and six layer writers."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets import hot3d_download
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.datasets.hot3d_hands import HandBatch, evaluate_hands
from dataforge.datasets.hot3d_layers import write_base, write_hand_meshes, write_hands, write_object_meshes, write_object_poses, write_projections
from dataforge.datasets.hot3d_source import (
    DEVICES,
    AssetInfo,
    Device,
    DeviceSpec,
    Hot3dSource,
    Manifest,
    complete_sequences,
    read_asset_census,
    read_manifest,
)
from dataforge.datasets.hot3d_vrs import Scene, read_scene
from dataforge.identity import SequenceIdentity
from dataforge.umetrack_hands import HandProfileDoc, read_hand_profile

FOLLOW_EYE: rrb.EyeControls3D = blueprints.headset_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
"""The 3D views' eye in camera 0's logged (rotated) frame: +z looks out, -y is the shown image's up.

Not the rig frame: HOT3D's device frame differs between sequences (Quest ``P0003_cae067da``'s
camera 0 looks along -x of it, most sequences' along +x), while camera 0 is the same everywhere."""


@dataclass
class Hot3dConfig(FrameLimitedConfig):
    """Shared read-only HOT3D source selection."""

    device: ClassVar[Device]
    """Source device and its shipped conventions."""
    _target: type = field(default_factory=lambda: Hot3dDataset)
    """Shared reader constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "hot3d")
    """Raw HOT3D root: aria/, quest3/, assets/ and the manifests download writes."""
    url_file: Path | None = None
    """The device's signed URL file from projectaria.com (Hot3DAria_download_urls*.json or Hot3DQuest_...); download needs it."""
    assets_url_file: Path | None = None
    """Hot3DAssets_download_urls*.json, for the object GLBs; download needs it until assets/ is complete."""
    sequences: tuple[str, ...] | None = None
    """Sequence names to download or convert, or every one (download: the URL file's; convert: the complete local ones)."""


@dataclass
class Hot3dAriaConfig(Hot3dConfig):
    """Read-only HOT3D Aria source selection."""

    command: ClassVar[str] = "hot3d-aria"
    """CLI and catalog name."""
    device: ClassVar[Device] = "aria"
    """Source device."""


@dataclass
class Hot3dQuest3Config(Hot3dConfig):
    """Read-only HOT3D Quest 3 source selection."""

    command: ClassVar[str] = "hot3d-quest3"
    """CLI and catalog name."""
    device: ClassVar[Device] = "quest3"
    """Source device."""


class Hot3dDataset(DataforgeDataset[Hot3dConfig, Hot3dSource]):
    """Native clocks and one recording identity across all available layers."""

    layers: tuple[str, ...] = (
        paths.BASE_LAYER,
        paths.HAND_POSE_LAYER,
        paths.HAND_MESH_LAYER,
        paths.PROJECTIONS_LAYER,
        paths.OBJECT_POSE_LAYER,
        paths.OBJECT_MESH_LAYER,
    )

    def __init__(self, config: Hot3dConfig) -> None:
        super().__init__(config)
        self.sources: dict[SequenceIdentity, Hot3dSource] = {}

    def download(self) -> None:
        """Fetch the selected sequences' VRS + ground truth + hands and the shared GLBs; see ``hot3d_download``."""
        hot3d_download.download(
            self.config.root,
            self.config.device,
            url_file=self.config.url_file,
            assets_url_file=self.config.assets_url_file,
            sequences=self.config.sequences,
        )

    def remote_sequences(self) -> list[RemoteSequence]:
        """List from the URL file when given, else from the manifest an earlier download wrote."""
        manifest: Manifest = (
            hot3d_download.manifest_of(hot3d_download.read_url_file(hot3d_download.url_file_at(self.config.url_file, "--url-file")), self.config.device)
            if self.config.url_file is not None
            else read_manifest(self.config.root, self.config.device)
        )
        return hot3d_download.remote_sequences(manifest, self.config.device)

    def discover(self) -> list[tuple[SequenceIdentity, Hot3dSource]]:
        self.sources = {
            SequenceIdentity(self.config.name, (source.path.name,)): source
            for source in complete_sequences(self.config.root, self.config.device, self.config.sequences)
        }
        return list(self.sources.items())

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        """Every layer, or the base alone for a sequence whose metadata ships no hand/object ground truth.

        ``discover`` and ``convert`` record each source; a sequence neither has seen gets every layer.
        """
        source: Hot3dSource | None = self.sources.get(identity)
        layers: tuple[str, ...] = self.layers if source is None or source.metadata.have_hand_object_pose_gt else (paths.BASE_LAYER,)
        return paths.layer_targets(identity, layers, frame_limit=self.config.frame_limit)

    def convert(self, identity: SequenceIdentity, source: Hot3dSource, *, force: bool) -> Path:
        """Publish all layers atomically, without touching the raw tree."""
        self.sources[identity] = source
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("fetch"):
            scene: Scene = read_scene(source, self.config.device, self.config.frame_limit)
        self.timer.capture_s = (int(scene.cameras[0].times_ns[-1]) - int(scene.cameras[0].times_ns[0])) / 1e9
        assets: Path = self.config.root / "assets"
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {paths.BASE_LAYER: partial(write_base, scene=scene, identity=identity, timer=self.timer)}
        # The hand layers and projections share one profile read and one forward-kinematics pass.
        if any(layer in pending for layer in (paths.HAND_POSE_LAYER, paths.HAND_MESH_LAYER, paths.PROJECTIONS_LAYER)):
            with self.timer.stage("hands"):
                profile: HandProfileDoc = read_hand_profile(scene.source / "umetrack_hand_user_profile.json")
                batch: HandBatch = evaluate_hands(profile.model, scene.hands, scene.times_ns)
            writers[paths.HAND_POSE_LAYER] = partial(write_hands, scene=scene, profile=profile, batch=batch)
            writers[paths.HAND_MESH_LAYER] = partial(write_hand_meshes, scene=scene, profile=profile, batch=batch)
            writers[paths.PROJECTIONS_LAYER] = partial(write_projections, scene=scene, batch=batch)
        if any(layer in pending for layer in (paths.OBJECT_POSE_LAYER, paths.OBJECT_MESH_LAYER)):
            with self.timer.stage("fetch"):
                census: dict[str, AssetInfo] = read_asset_census(assets, scene.metadata.object_uids)
            writers[paths.OBJECT_POSE_LAYER] = partial(write_object_poses, scene=scene, census=census)
            writers[paths.OBJECT_MESH_LAYER] = partial(write_object_meshes, scene=scene, assets=assets)
        self.write_layers(identity, targets, pending, writers)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        spec: DeviceSpec = DEVICES[self.config.device]
        cameras: list[str] = [name for _, name in spec.camera_streams]
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="HOT3D (follows the headset)",
                origin=schema.cam_path(0, 0),
                contents=["/world/**"],
                eye_controls=FOLLOW_EYE,
            ),
            ego_panes=[
                blueprints.camera_view(
                    name,
                    0,
                    index,
                    contents=[schema.video_path(0, index), schema.coco133_uv_projected_path(0, index)],
                )
                for index, name in enumerate(cameras)
            ],
            exo_panes=[],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the 3D scene following the headset beside camera 0."""
        spec: DeviceSpec = DEVICES[self.config.device]
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Scene",
                origin=schema.cam_path(0, 0),
                contents=["+ /world/**", *blueprints.video_exclusions((0, index) for index in range(len(spec.camera_streams)))],
                eye_controls=FOLLOW_EYE,
            ),
            blueprints.camera_view(spec.camera_streams[0][1], 0, 0, contents=[schema.video_path(0, 0), schema.coco133_uv_projected_path(0, 0)]),
        )

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:episode:participant_id", "participant"),
            writing.TableField("property:episode:object_ids", "objects"),
            writing.TableField("property:episode:has_gt", "has_gt"),
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
