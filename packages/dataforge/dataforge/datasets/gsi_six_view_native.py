"""TODO(gsi-native): GSI Egocentric Six-View in GSI's own layout, a temporary comparison mirror of ``gsi-six-view``.

Kept for the GSI partnership while Pablo compares the two layouts; drop it, or keep it, once he decides. Same download
as ``gsi-six-view`` (``GsiSixViewDataset``); convert repacks each shipped file as one layer, ``hands.rrd`` as base and
``slam.rrd`` as slam, under a dataforge recording id: it decodes nothing and keeps GSI's entity paths, timelines and
samples as shipped. See docs/gsi-six-view.md.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from rerun.encodings import EntityPathLike

from dataforge import hands, paths, writing
from dataforge.datasets.gsi_six_view import OPTIMIZE, GsiSixViewConfig, GsiSixViewDataset
from dataforge.datasets.gsi_six_view_remote import Manifest, ManifestEpisode
from dataforge.identity import SequenceIdentity

SLAM_LAYER: str = "slam"
"""GSI's ``slam.rrd``: the rig trajectory, stacked onto base."""
FILES: dict[str, str] = {paths.BASE_LAYER: "hands.rrd", SLAM_LAYER: "slam.rrd"}
"""The shipped file each layer repacks."""
CAMERAS: tuple[str, ...] = ("left_eye", "right_eye", "left_front", "right_front", "left_side", "right_side")
"""The six views, in the pairs the default layout shows them: eye, front, side."""
NATIVE_TIMELINE: str = "capture_time"
"""GSI's device clock; ``frame`` and ``source_frame`` index the same samples."""
KEPT: list[str] = ["+ /**", "+ /__properties/episode", "- /__properties"]
"""Everything a file ships except its ``RecordingInfo`` name: the base takes a dataforge name, and a derived layer restates no properties."""
SIDES: tuple[hands.Side, ...] = ("left", "right")
"""GSI's two hands, ``/hand/<side>``."""


@dataclass
class GsiSixViewNativeConfig(GsiSixViewConfig):
    """TODO(gsi-native): GSI's partner edition in GSI's own layout; the same download as ``gsi-six-view``."""

    command: ClassVar[str] = "gsi-six-view-native"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: GsiSixViewNativeDataset)
    """Dataset constructor."""


class GsiSixViewNativeDataset(GsiSixViewDataset):
    """TODO(gsi-native): one catalog dataset in GSI's layout; one segment per episode."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, SLAM_LAYER)
    timeline: ClassVar[str] = NATIVE_TIMELINE

    def write_episode(
        self, identity: SequenceIdentity, source: Path, targets: dict[str, Path], pending: list[str], manifest: Manifest, episode: ManifestEpisode
    ) -> None:
        """Repack each pending layer from its shipped file."""
        for layer in pending:
            with self.timer.stage(f"write:{layer}"):
                self.repack(identity, source / FILES[layer], targets[layer], manifest if layer == paths.BASE_LAYER else None, episode)

    def repack(self, identity: SequenceIdentity, shipped: Path, target: Path, manifest: Manifest | None, episode: ManifestEpisode) -> None:
        """Write ``shipped``'s chunks, compacted, to ``target`` under the dataforge recording id; ``manifest`` marks the base.

        The base keeps GSI's ``RecordingInfo`` start time, the capture's own, and adds the dataforge name, the capture
        properties and the embedded default blueprint.
        """
        compacted: rrc.ChunkStore = rrc.RrdReader(shipped).stream().filter(content=KEPT).collect(optimize=OPTIMIZE)
        base: bool = manifest is not None
        with (
            writing.atomic_write(target) as temp_path,
            writing.recording_to(temp_path, recording_id=identity.recording_id, default_blueprint=self.default_blueprint() if base else None, send_properties=False) as recording,
        ):
            recording.send_chunks(compacted)
            if manifest is not None:
                recording.send_chunks(rrc.RrdReader(shipped).stream().filter(content="/__properties", components="RecordingInfo:start_time"))
                frames: int = int(compacted.reader(index=None, contents="/__properties/episode").to_arrow_table().column("property:episode:n_frames")[0][0].as_py())
                writing.send_capture_properties(
                    recording,
                    identity,
                    num_cameras=episode.views,
                    num_frames=frames,
                    category=episode.category,
                    gsi_edition=self.config.edition,
                    gsi_source_release=manifest.source_release,
                    gsi_rrd_contract=manifest.rrd_contract,
                    gsi_privacy_status=manifest.privacy_status,
                    hands_rrd_sha256=episode.hands_rrd.sha256,
                    slam_rrd_sha256=episode.slam_rrd.sha256,
                )

    def default_blueprint(self) -> rrb.Blueprint:
        """The world with the six camera frustums beside the six views in eye, front and side pairs, each with the hands projected."""
        frustums: dict[EntityPathLike, rr.Pinhole | rr.Mesh3D] = {f"/camera/{camera}/image": rr.Pinhole.from_fields() for camera in CAMERAS}
        world: rrb.Spatial3DView = rrb.Spatial3DView(name="World", origin="/", contents=["+ /**"], overrides=frustums | see_through_hands())
        views: rrb.Grid = rrb.Grid(*(camera_pane(camera) for camera in CAMERAS), grid_columns=2)
        return rrb.Blueprint(rrb.Horizontal(world, views, column_shares=[2, 3]), rrb.TimePanel(timeline=NATIVE_TIMELINE, state="collapsed"), collapse_panels=True)

    def table_blueprint(self) -> rrb.Blueprint:
        """The hands in the world beside the left front view: the one stream a card decodes."""
        world: rrb.Spatial3DView = rrb.Spatial3DView(name="World", origin="/", contents=["+ /hand/**", "+ /slam/**"], overrides=see_through_hands())
        return rrb.Blueprint(rrb.Horizontal(world, camera_pane("left_front")), collapse_panels=True)


def camera_pane(camera: str) -> rrb.Spatial2DView:
    """One view's video with both hands' meshes and keypoints projected through its pinhole.

    The pane is rooted at the pinhole, so the hands, which hang off the same rig as ``/camera``, land in image space.
    """
    hand_parts: list[str] = [f"+ /hand/{side}/{part}" for side in SIDES for part in ("vertices", "keypoints3d")]
    return rrb.Spatial2DView(name=camera, origin=f"/camera/{camera}/image", contents=[f"+ /camera/{camera}/image", *hand_parts], overrides=see_through_hands())


def see_through_hands() -> dict[EntityPathLike, rr.Pinhole | rr.Mesh3D]:
    """Display-only translucent albedo for GSI's opaque MANO meshes, so the keypoints show through; the data stays as shipped."""
    return {f"/hand/{side}/vertices": rr.Mesh3D.from_fields(albedo_factor=hands.HAND_ALBEDO[side]) for side in SIDES}
