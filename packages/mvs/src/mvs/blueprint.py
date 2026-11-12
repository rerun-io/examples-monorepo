"""Rerun entity paths and blueprint of mvs live meshing: one layout for the CLI and the app.

Ported from the original SimpleRecon live-meshing demo by Pablo Vela (repo pablovela5620/simplerecon-inference,
``packages/simplerecon/simplerecon/blueprint.py`` at d5bc1a0). The view names, contents rules, shares, panel states
and frustum style are the original's. Differences: the camera, its pinholes and its video keep the ARKitScenes
catalog's paths (``world/rig_00/cam_00/...``), because mvs output sits beside the catalog's layers; the mesh is
:class:`simplecv.rerun_live_mesh.LiveMeshLog`'s tree under ``world/mesh``; the web page's ``tablet`` layout
(IMU plots, tracked features, landmarks, trajectory, follow eye) is left out, because mvs has no data for it; the time
panel opens on :data:`TIMELINE`, where the relayed video lives.
"""

from dataclasses import dataclass
from typing import Literal

import rerun as rr
import rerun.blueprint as rrb

TIMELINE: str = "video_time"
"""Timeline the time panel opens on: the catalog's video is relayed on it only, so on ``frame`` the Current view is empty."""
SOURCE_CAMS: int = 7
"""Source views per depth run: the model's tuple size minus the current view (``pose_stream.NUM_SOURCE_VIEWS``)."""
CURRENT_CAM_COLOR: tuple[int, int, int, int] = (40, 200, 80, 255)
"""RGBA of the current camera's frustum, the examples-monorepo green (simplecv rerun_rig_logger, slam-evals)."""
CURRENT_CAM_PLANE_M: float = 0.22
"""Image-plane distance of the current camera's frustum, metres; with the source one, the 8-view frusta do not overlap."""
SOURCE_CAM_PLANE_M: float = 0.09
"""Image-plane distance of a source camera's frustum, metres."""


@dataclass(frozen=True, slots=True)
class EntityPaths:
    """Entity paths of one segment, all below ``root``."""

    root: str = ""
    """Empty for mvs (paths start at ``world``, as in the catalog)."""
    camera: str = "rig_00/cam_00"
    """The catalog's camera below ``world`` (arkitscenes rig schema); its transform, pinholes and video are the catalog's."""

    def _under_root(self, path: str) -> str:
        """Return ``path`` below the root."""
        return f"{self.root}/{path}" if self.root else path

    @property
    def world(self) -> str:
        """ViewCoordinates RIGHT_HAND_Z_UP (static); the scene root."""
        return self._under_root("world")

    @property
    def current_cam(self) -> str:
        """Transform3D world_from_cam of the current frame."""
        return f"{self.world}/{self.camera}"

    @property
    def current_pinhole(self) -> str:
        """Pinhole at the camera image size (green frustum)."""
        return f"{self.current_cam}/pinhole"

    @property
    def video(self) -> str:
        """VideoStream of the camera (the catalog's AV1 samples, relayed once)."""
        return f"{self.current_pinhole}/video"

    @property
    def depth_pinhole(self) -> str:
        """Pinhole at the depth raster size, beside the camera pinhole (the sizes differ)."""
        return f"{self.current_cam}/pinhole_lowres"

    @property
    def depth(self) -> str:
        """Encoded depth image, millimetre PNG."""
        return f"{self.depth_pinhole}/depth"

    @property
    def normal(self) -> str:
        """JPEG of the depth normals."""
        return f"{self.depth_pinhole}/normal"

    @property
    def lowest_cost(self) -> str:
        """EncodedDepthImage (millimetre PNG) of the depth of the cost volume's best plane, at the matching size."""
        return f"{self.depth_pinhole}/lowest_cost"

    @property
    def sources(self) -> str:
        """Parent of the source views ``<index>`` of the latest depth run."""
        return f"{self.world}/source_cam"

    @property
    def mesh(self) -> str:
        """Root of :class:`simplecv.rerun_live_mesh.LiveMeshLog`: live ``blocks/<x_y_z>`` and the final ``full`` mesh."""
        return f"{self.world}/mesh"

    @property
    def timings(self) -> str:
        """Parent of the per-stage timing Scalars, ms."""
        return self._under_root("timings")

    def source_cam(self, index: int) -> str:
        """Transform3D of source view ``index``."""
        return f"{self.sources}/{index}"

    def source_pinhole(self, index: int) -> str:
        """Pinhole of source view ``index``."""
        return f"{self.source_cam(index)}/pinhole"

    def source_rgb(self, index: int) -> str:
        """VideoFrameReference of source view ``index`` into :attr:`video`."""
        return f"{self.source_pinhole(index)}/rgb"

    def timing(self, stage: str) -> str:
        """Scalars, ms, of one stage."""
        return f"{self.timings}/{stage}"


Variant = Literal["native", "phone", "phone-landscape"]
"""Layout of :func:`create_blueprint`: the desktop layout, or one of the two phone screens."""


def create_blueprint(paths: EntityPaths, *, variant: Variant, source_cams: int = SOURCE_CAMS, timings: bool = True) -> rrb.Blueprint:
    """Create the live-meshing layout.

    ``native``: left, the 3D reconstruction (fused mesh + camera frusta); right, the current view with a filmstrip of
    its source views beneath (mirroring the upstream SimpleRecon demo), the depth panel beside normals and lowest cost,
    and stage timings. The phone layouts show three views only, Reconstruction, Current view and Depth, because on a
    phone screen the filmstrip, plots and timings are too small to read: ``phone`` stacks Reconstruction above
    Current view | Depth; ``phone-landscape`` puts Reconstruction left and Current view above Depth right. Phone
    layouts hide the time panel.

    Every variant's time panel plays the video timeline: a streamed viewer would otherwise follow the newest data and
    sit at the end. Send the blueprint after the whole video is logged, so playback starts at 0 (a blueprint cannot
    seek).

    The depth pinhole is excluded from the 3D view: Rerun back-projects a depth image under a pinhole into world
    points, which hide the fused mesh. The lowest-cost view is rooted at its own entity because the cost volume is
    logged at matching resolution and would otherwise letterbox inside the depth pinhole bounds.

    Args:
        paths: Entity paths of the segment.
        variant: Layout to build.
        source_cams: Number of source views in the filmstrip; 0 leaves the filmstrip out. Phone layouts ignore it.
        timings: Show the stage timings view; False leaves it out (a tool that logs no timings).

    Returns:
        Blueprint ready for ``rr.send_blueprint`` or ``Blueprint.save``.
    """
    phone: bool = variant != "native"
    world: rrb.Spatial3DView = rrb.Spatial3DView(
        origin=paths.world,
        name="Reconstruction",
        contents=[f"+ {paths.world}/**", f"- {paths.depth_pinhole}/**"],
        line_grid=rrb.LineGrid3D(visible=True, spacing=0.5) if phone else None,
    )
    current: rrb.Spatial2DView = rrb.Spatial2DView(origin=paths.current_pinhole, contents=[paths.video], name="Current view")
    depth: rrb.Spatial2DView = rrb.Spatial2DView(
        origin=paths.depth_pinhole,
        contents=paths.depth,
        name="Depth",
        defaults=[rr.DepthImage.from_fields(colormap="Turbo", depth_range=(0.25, 5.0))] if phone else [],
    )
    if variant == "phone":
        return rrb.Blueprint(
            rrb.Vertical(world, rrb.Horizontal(current, depth), row_shares=[5.0, 3.0]),
            rrb.TimePanel(state="hidden", timeline=TIMELINE, play_state="playing"),
            collapse_panels=True,
        )
    if variant == "phone-landscape":
        return rrb.Blueprint(
            rrb.Horizontal(world, rrb.Vertical(current, depth), column_shares=[3.0, 2.0]),
            rrb.TimePanel(state="hidden", timeline=TIMELINE, play_state="playing"),
            collapse_panels=True,
        )
    sources: rrb.Horizontal = rrb.Horizontal(
        *[
            rrb.Spatial2DView(origin=paths.source_pinhole(index), contents=paths.source_rgb(index), name=f"src {index}")
            for index in range(source_cams)
        ]
    )
    panels: list[rrb.View] = [
        rrb.Spatial2DView(origin=paths.depth_pinhole, contents=paths.normal, name="Normals"),
        rrb.Spatial2DView(origin=paths.lowest_cost, name="Lowest cost"),
    ]
    camera: rrb.Vertical | rrb.Spatial2DView = rrb.Vertical(current, sources, row_shares=[4.0, 1.0]) if source_cams else current
    rows: list[rrb.Container | rrb.View] = [camera, rrb.Horizontal(depth, *panels)]
    shares: list[float] = [4.0, 2.0]
    if timings:
        rows.append(rrb.TimeSeriesView(origin=paths.timings, name="Stage timings (ms)"))
        shares.append(1.8)
    layout: rrb.Container = rrb.Horizontal(world, rrb.Vertical(*rows, row_shares=shares), column_shares=[3.0, 2.0])
    return rrb.Blueprint(layout, rrb.TimePanel(state="collapsed", timeline=TIMELINE, play_state="playing"), collapse_panels=True)
