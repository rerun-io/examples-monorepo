"""Two HOT3D catalog datasets sharing one raw reader and three layer writers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.hot3d_layers import LayerWriter
from dataforge.datasets.hot3d_source import DEVICES, URL_LIST_DATE, Device, DeviceSpec, Hot3dSource, complete_sequences
from dataforge.datasets.hot3d_vrs import read_scene
from dataforge.identity import SequenceIdentity


@dataclass
class Hot3dConfig(DataforgeDatasetConfig):
    """Shared read-only HOT3D source selection."""

    device: ClassVar[Device]
    """Source device and its shipped conventions."""
    _target: type = field(default_factory=lambda: Hot3dDataset)
    """Shared reader constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "hot3d")
    """Raw HOT3D root containing aria, quest3, assets and URL lists."""
    sequences: tuple[str, ...] | None = None
    """Sequence folder names, or all complete local folders."""
    frame_limit: int | None = None
    """First N frames per camera; kept separate under preview-firstN."""


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
        paths.PROJECTIONS_LAYER,
    )

    def __init__(self, config: Hot3dConfig) -> None:
        super().__init__(config)
        self.sources: dict[SequenceIdentity, Hot3dSource] = {}

    @property
    def device(self) -> Device:
        """Source directory and clock convention selected by the config."""
        return self.config.device

    def download(self) -> None:
        """Verify only; never download, extract, or write beneath the raw root."""
        print(f"{self.config.name}: {len(self.discover())} complete sequences (URL list {URL_LIST_DATE})")

    def discover(self) -> list[tuple[SequenceIdentity, Hot3dSource]]:
        self.sources = {
            SequenceIdentity(self.config.name, (source.path.name,)): source
            for source in complete_sequences(self.config.root, self.device, self.config.sequences)
        }
        return list(self.sources.items())

    def targets(self, identity: SequenceIdentity) -> dict[str, Path]:
        source: Hot3dSource = self.sources[identity]
        root: Path = paths.output_root()
        if self.config.frame_limit is not None:
            if self.config.frame_limit < 1:
                raise ValueError("frame_limit must be positive")
            root = root / f"preview-first{self.config.frame_limit}"
        return {
            layer: paths.rrd_path(root, layer=layer, identity=identity)
            for layer in (self.layers if source.metadata.have_hand_object_pose_gt else (paths.BASE_LAYER,))
        }

    def convert(self, identity: SequenceIdentity, source: Hot3dSource, *, force: bool) -> Path:
        """Publish all layers atomically, without touching the raw tree."""
        self.sources[identity] = source
        targets: dict[str, Path] = self.targets(identity)
        for target in [*targets.values(), paths.output_root() / "work"]:
            if target.resolve().is_relative_to(self.config.root.resolve()):
                raise ValueError(f"refusing to write beneath raw root: {target}")
        pending: list[str] = [layer for layer, target in targets.items() if not writing.should_skip(target, force=force)]
        if pending:
            with self.timer.stage("fetch"):
                scene = read_scene(source, self.device, self.config.frame_limit)
            self.timer.capture_s = (int(scene.cameras[0].times_ns[-1]) - int(scene.cameras[0].times_ns[0])) / 1e9
            writer: LayerWriter = LayerWriter(scene, identity, self.timer)
            for layer in pending:
                with (
                    self.timer.stage(f"write:{layer}"),
                    writing.atomic_recording(
                        targets[layer],
                        recording_id=identity.recording_id,
                        default_blueprint=self.default_blueprint() if layer == paths.BASE_LAYER else None,
                        send_properties=layer == paths.BASE_LAYER,
                    ) as recording,
                ):
                    writer.write_layer(layer, recording)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        spec: DeviceSpec = DEVICES[self.device]
        cameras: list[str] = [name for _, name in spec.camera_streams]
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="HOT3D world",
                origin="/world",
                contents=["/world/**"],
                eye_controls=blueprints.eye_controls_from_pose((1.5, -1.5, 1.5), spec.eye_target, spec.up),
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
        return rrb.Blueprint(blueprints.camera_view("Ego", 0, 0, contents=[schema.pinhole_path(0, 0) + "/**"]), collapse_panels=True)

    def table_fields(self) -> writing.TableFields:
        fields: tuple[writing.TableField, ...] = (
            writing.TableField("property:episode:participant_id", "participant"),
            writing.TableField("property:episode:object_ids", "objects"),
            writing.TableField("property:episode:has_gt", "has_gt"),
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
