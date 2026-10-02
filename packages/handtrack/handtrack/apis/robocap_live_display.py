"""Export robocap-live's display asset: the viewer layout and the static cap mesh that the Rust logger sends first on every stream.

The layout is DataForge's RoboCap rig layout (``dataforge.blueprints.rig_blueprint``: an overview and a follow 3D view beside a
grid of the six camera panes, plots underneath), adapted to a live stream:

- the SLAM trajectory arrives as one edge per frameset on ``/world/runs/slam_rs/trajectory``, so both 3D views show it over the
  whole time range instead of only the latest edge; ``.../trail`` keeps the follow view's cursor-relative window;
- the hands are the logger's ``/world/hands/{left,right}`` (3D) and ``<pinhole>/hands/{side}`` (2D, inside the camera panes);
- the plots are the pipeline's per-stage timings (``/timings``), its frame rate (``/fps``) and the logger's drop counters (``/log``).

The cameras themselves (poses, pinholes, video codec) are not in the asset: the logger writes them from the rig it runs with.
The Rust loader (``robocap-live/src/log/display.rs``) rejects an asset that carries them, a static rig pose, or temporal data.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
from dataforge import blueprints, schema
from dataforge.writing import atomic_recording
from simplecv.data.ego.robocap_ego import CAMERA_DISPLAY_ORDER

APPLICATION_ID: str = "robocap"
"""Shared with the live recording, so the blueprint applies to it (Rust ``display::APPLICATION_ID``)."""
RECORDING_ID: str = "robocap-live-display"
RIG: int = 0
"""RoboCap is one rig (DataForge ``robocap.RIG``)."""
MESH_RELATIVE_PATH: str = "robocap-mesh/3DModel.glb"
"""The cap scan under the RoboCap root (DataForge ``robocap.MESH_RELATIVE_PATH``)."""
RUN_SOURCE: str = "slam_rs"
"""The live SLAM's run source (DataForge RoboCap's extra run source)."""
TIMINGS_ROOT: str = "/timings"
FPS_PATH: str = "/fps"
LOG_ROOT: str = "/log"
RIG_DOT_COLOR: tuple[int, int, int] = (255, 220, 0)
RIG_DOT_RADIUS_UI_POINTS: float = 9.0
TIMINGS_RANGE_MS: tuple[float, float] = (0.0, 80.0)
"""Timing plot range: the 30 fps budget is 33 ms per frameset."""
FPS_RANGE: tuple[float, float] = (0.0, 35.0)
PLOT_SHARES: tuple[int, int, int] = (3, 1, 1)
"""Width split of the plot row: timings, fps, drops."""
RIG_FORWARD: tuple[float, float, float] = (0.0, 1.0, 0.0)
"""Where the wearer faces, in the rig frame: the front cameras' optical axes are +Y (rig.json's cam_from_rig third rows)."""
RIG_UP: tuple[float, float, float] = (0.0, 0.0, -1.0)
"""The wearer's up in the rig frame: the front cameras' image-down axes are +Z (DataForge: "-Z is the wearer's up")."""
OVERVIEW_EYE: tuple[float, float, float] = (0.8, 0.8, 0.6)
"""Overview eye in the world frame (slam-rs: gravity-aligned, Z up, origin at the start pose): above and to the side, so the cap,
the hands and a few metres of trajectory are all in shot at the start."""
OVERVIEW_TARGET: tuple[float, float, float] = (0.0, 0.0, -0.2)


@dataclass
class Config:
    """Write the display asset (layout + cap mesh) for the robocap-live logger."""

    output: Path
    """Destination ``.rrd``; replaced atomically."""
    root: Path
    """RoboCap dataset root holding the cap mesh (``robocap-mesh/3DModel.glb``)."""
    mesh: Path | None = None
    """Cap mesh to use instead of the one under ``root``; the asset has no mesh when neither is readable."""


def whole_path() -> rrb.VisibleTimeRanges:
    """Everything from the start of the stream up to the cursor: an accumulating, edge-per-frameset trajectory."""
    return rrb.VisibleTimeRanges(
        rrb.VisibleTimeRange(schema.TIMELINE, start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.cursor_relative())
    )


def build_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """The live layout: two 3D views and six camera panes over timing, fps and drop plots.

    Args:
        camera_names: Camera pane labels in ``cam_00..cam_05`` order.

    Returns:
        The blueprint the logger activates on every stream.
    """
    rig_path: str = schema.rig_path(RIG)
    trajectory: str = schema.trajectory_path(RUN_SOURCE)
    trail: str = schema.trail_path(RUN_SOURCE)
    overview = rrb.Spatial3DView(
        name="Rig",
        origin="/",
        line_grid=True,
        overrides={
            # DataForge's screen-space rig dot: the cap is centimetres on a path of metres.
            rig_path: rr.Points3D(
                [[0.0, 0.0, 0.0]], radii=rr.Radius.ui_points(RIG_DOT_RADIUS_UI_POINTS), colors=[RIG_DOT_COLOR], labels=["rig"], show_labels=True
            ),
            trajectory: whole_path(),
            trail: rrb.EntityBehavior(visible=False),
        },
        eye_controls=blueprints.eye_controls_from_pose(position=OVERVIEW_EYE, look_target=OVERVIEW_TARGET, eye_up=(0.0, 0.0, 1.0)),
    )
    follow = rrb.Spatial3DView(
        name="Follow",
        origin=rig_path,
        contents="/**",
        line_grid=False,
        overrides={
            trajectory: [
                rr.LineStrips3D.from_fields(
                    colors=blueprints.DIM_TRAJECTORY_COLOR, radii=rr.Radius.ui_points(blueprints.DIM_TRAJECTORY_RADIUS_UI_POINTS)
                ),
                whole_path(),
            ],
            trail: rrb.VisibleTimeRanges(
                rrb.VisibleTimeRange(
                    schema.TIMELINE,
                    start=rrb.TimeRangeBoundary.cursor_relative(seconds=blueprints.TRAIL_WINDOW_S),
                    end=rrb.TimeRangeBoundary.cursor_relative(),
                )
            ),
        },
        # A chase eye close behind and above the head, aimed at where the hands work: the cap and both hands fill the view.
        eye_controls=blueprints.follow_eye_controls(RIG_FORWARD, RIG_UP, back_m=0.45, up_m=0.25, ahead_m=0.35, aim_down_m=0.25),
    )
    panes: list[rrb.Spatial2DView] = [blueprints.camera_view(name, RIG, index) for index, name in enumerate(camera_names)]
    plots: list[rrb.TimeSeriesView] = [
        # Fixed ranges: a cold-start spike (the first frameset's nets take a second) must not flatten the 33 ms budget.
        rrb.TimeSeriesView(
            name="Stage timings (ms)", origin=TIMINGS_ROOT, plot_legend=rrb.PlotLegend(visible=True), axis_y=rrb.ScalarAxis(range=TIMINGS_RANGE_MS)
        ),
        rrb.TimeSeriesView(name="Pipeline fps", origin="/", contents=[FPS_PATH], axis_y=rrb.ScalarAxis(range=FPS_RANGE)),
        rrb.TimeSeriesView(name="Logger drops", origin=LOG_ROOT, plot_legend=rrb.PlotLegend(visible=True)),
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(overview, follow),
                rrb.Grid(*panes, grid_columns=blueprints.CAMERA_GRID_COLUMNS, name="Cameras"),
                column_shares=list(blueprints.RIG_COLUMN_SHARES),
            ),
            rrb.Horizontal(*plots, column_shares=list(PLOT_SHARES)),
            row_shares=list(blueprints.PLOT_ROW_SHARES),
        ),
        rrb.TimePanel(timeline=schema.TIMELINE),
        collapse_panels=True,
    )


def main(config: Config) -> None:
    """Write the layout and, when readable, the cap mesh (static, under the rig, aligned as DataForge aligns it)."""
    mesh: Path = config.mesh if config.mesh is not None else config.root / MESH_RELATIVE_PATH
    with atomic_recording(
        config.output, application_id=APPLICATION_ID, recording_id=RECORDING_ID, default_blueprint=build_blueprint(list(CAMERA_DISPLAY_ORDER))
    ) as recording:
        if os.access(mesh, os.R_OK):
            # Imported here: the RoboCap dataset module pulls dataforge's download stack, which only the dataforge env carries;
            # the layout alone needs none of it.
            from dataforge.datasets import robocap

            mesh_entity: str = f"{schema.rig_path(RIG)}/mesh"
            recording.log(mesh_entity, rr.Transform3D(translation=robocap.MESH_TRANSLATION, mat3x3=robocap.MESH_MAT3X3), static=True)
            recording.log(mesh_entity, rr.Asset3D(path=mesh), static=True)
        else:
            print(f"warning: cap mesh not readable, the asset has no mesh: {mesh}")
    print(f"wrote {config.output} ({config.output.stat().st_size} bytes)")
