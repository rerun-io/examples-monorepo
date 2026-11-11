"""Default blueprint of a MetricTrail run recording (entity layout in :mod:`metrictrail.entities`).

Left: the input video with DPVO's patches (a frame-directory input: the frames DPVO tracks), the MoGe-2 metric depth and
normals, the DINOv3 feature PCA. Middle: the
trajectory against GT after the paper's Sim(3) alignment with the fused metric cloud (a second tab colours the cloud by
its fused MoGe-2 normals), and DPVO's local BA window. Right: the
per-frame error and running ATE, the SE(3)-only (scale 1) view on a 1 m grid, and the summary card. A view whose
data the recording does not hold (no live capture, no GT) is a text tile that says what it is waiting for. With a video
the time panel plays ``video_time`` from the start, as the EgoHand app does.
"""

from dataclasses import dataclass

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray
from simplecv.rerun_eye_controls import follow_eye_controls

from metrictrail import entities
from metrictrail.palette import BACKGROUND_RGB, RunLabel

FOLLOW_EYE: rrb.EyeControls3D = follow_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), back_m=0.9, up_m=0.35, ahead_m=1.0)
"""Chase eye of the Follow view, in the camera's RDF frame (metres: MetricTrail is metric): close enough that the frustum
and the video in it fill the view (MASt3R-SLAM sits 1.5 m back)."""


@dataclass(frozen=True, slots=True)
class SceneView:
    """A fixed 3D eye: where it is, what it looks at, which way is up."""

    position: tuple[float, float, float]
    """Eye position, scene units (metres)."""
    look_target: tuple[float, float, float]
    """Point the eye looks at."""
    eye_up: tuple[float, float, float]
    """The scene's up direction."""

    def controls(self) -> rrb.EyeControls3D:
        """As blueprint eye controls."""
        return rrb.EyeControls3D(position=self.position, look_target=self.look_target, eye_up=self.eye_up)


def scene_view(points_xyz: Float64[ndarray, "n 3"], up: Float64[ndarray, "3"], back: Float64[ndarray, "3"]) -> SceneView:
    """An eye above and behind the centre of ``points_xyz`` that frames them (up and back are unit scene directions).

    Raises:
        ValueError: If there is no finite point.
    """
    finite: Float64[ndarray, "m 3"] = points_xyz[np.isfinite(points_xyz).all(axis=1)]
    if finite.shape[0] == 0:
        raise ValueError("no finite point to frame")
    centre: Float64[ndarray, "3"] = finite.mean(axis=0)
    extent_m: float = max(float(np.ptp(finite, axis=0).max()), 0.5)
    eye: Float64[ndarray, "3"] = centre + extent_m * (1.1 * back + 0.9 * up)
    return SceneView(
        position=(float(eye[0]), float(eye[1]), float(eye[2])),
        look_target=(float(centre[0]), float(centre[1]), float(centre[2])),
        eye_up=(float(up[0]), float(up[1]), float(up[2])),
    )


def waiting(name: str, what: str) -> rrb.TextDocumentView:
    """A tile for data the recording does not hold; its text is logged by :func:`log_waiting_notes`."""
    return rrb.TextDocumentView(name=name, origin=f"notes/waiting/{what}")


def waiting_notes(has_capture: bool, has_gt: bool) -> dict[str, str]:
    """``notes/waiting/<key>`` texts for the tiles the blueprint shows instead of missing data."""
    notes: dict[str, str] = {}
    if not has_capture:
        text: str = (
            "waiting for a live capture: `metrictrail run` logs this while the pipeline runs; this recording was logged from finished artifacts"
        )
        notes |= {"dino": text, "patches": text, "window": text, "normals": text}
    if not has_gt:
        notes |= {"gt": "waiting for ground truth: pass a VSLAM-LAB sequence (`--gt-sequence`)"}
    return notes


def log_waiting_notes(rec: rr.RecordingStream, has_capture: bool, has_gt: bool) -> None:
    """Log the texts of :func:`waiting_notes` as static ``TextDocument`` entities."""
    for key, text in waiting_notes(has_capture, has_gt).items():
        rec.log(f"notes/waiting/{key}", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), static=True)


def frame_view(*, has_video: bool, has_capture: bool, background: rrb.Background) -> rrb.Spatial2DView:
    """The input video with DPVO's patches over it (``input``), or for a frame-directory input the frames DPVO tracks."""
    if has_video:
        return rrb.Spatial2DView(
            name="input video + DPVO patches", origin=entities.INPUT, contents=[entities.INPUT_VIDEO, entities.INPUT_PATCHES], background=background
        )
    return rrb.Spatial2DView(
        name="frame + DPVO patches" if has_capture else "input frame",
        origin=entities.CAM,
        contents=[entities.CAM_RGB, entities.CAM_PATCHES],
        background=background,
    )


def tracking_views(background: rrb.Background) -> tuple[rrb.Spatial3DView, rrb.Spatial3DView]:
    """DPVO's live scene twice, as the dpvo and MASt3R-SLAM apps show theirs: an overview rooted at ``/``, so the
    gravity alignment on ``dpvo`` applies (rerun#9244: a view's own origin transform is invisible), and a Follow view
    rooted at the newest keyframe, whose chase eye rides the camera."""
    overview: rrb.Spatial3DView = rrb.Spatial3DView(
        name="DPVO overview (keyframes + BA window)", origin="/", contents=["/dpvo/**"], background=background
    )
    follow: rrb.Spatial3DView = rrb.Spatial3DView(
        name="DPVO follow (newest keyframe)",
        origin=entities.DPVO_CURRENT,
        contents=["/dpvo/**"],
        background=background,
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=FOLLOW_EYE,
    )
    return overview, follow


def time_panel(*, has_video: bool, live: bool = False) -> rrb.TimePanel:
    """Collapsed; with a video on ``video_time``: paused while a live run tracks (the app moves the cursor to each newly
    tracked frame, :meth:`metrictrail.gradio_ui.metrictrail_ui.RunStatus.cursor_command`), else playing in a loop (a
    streamed viewer would otherwise sit at the newest data); without a video on ``frame``."""
    if has_video and live:
        return rrb.TimePanel(state="collapsed", timeline=entities.VIDEO_TIMELINE, play_state="paused")
    if has_video:
        return rrb.TimePanel(state="collapsed", timeline=entities.VIDEO_TIMELINE, play_state="playing", loop_mode="all")
    return rrb.TimePanel(state="collapsed", timeline=entities.FRAME_TIMELINE)


def calibration_views(background: rrb.Background) -> tuple[rrb.Spatial3DView, rrb.Horizontal]:
    """SharedK calibration's DROID map in 3D (keyframe frustums with thumbnails, bundle-adjusted points, path, factor
    graph), and its two plots: the focal (MoGe-2 per head frame, then after each solve) and the keyframes kept."""
    map_view: rrb.Spatial3DView = rrb.Spatial3DView(
        name="SharedK calibration: DROID keyframe map (thrown away after the focal fit)",
        origin=entities.CALIBRATION,
        contents=[f"/{entities.CALIBRATION}/**", f"- /{entities.CALIBRATION_NEWEST}/**"],
        background=background,
    )
    plots: rrb.Horizontal = rrb.Horizontal(
        rrb.TimeSeriesView(name="focal fx (tracking-raster px)", contents=[f"/{entities.CALIBRATION_HEAD_FOCAL}", f"/{entities.CALIBRATION_FOCAL}"]),
        rrb.TimeSeriesView(name="keyframes kept", contents=[f"/{entities.CALIBRATION_KEPT}"]),
    )
    return map_view, plots


def calibration_blueprint() -> rrb.Blueprint:
    """The Gradio app's layout while SharedK calibrates, on its own timeline (``calibration_step``, the frame the sweep
    read; the app moves the cursor there). Left: the newest keyframe, its MoGe-2 depth prior at the current focal and
    its bundle-adjusted depth (the two the focal fit pulls together); right: the DROID map and the two plots."""
    background: rrb.Background = rrb.Background(color=BACKGROUND_RGB)
    map_view, plots = calibration_views(background)
    newest = entities.CALIBRATION_NEWEST
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(name="newest keyframe with a MoGe-2 prior (calibration raster)", origin=newest, contents=[entities.CALIBRATION_NEWEST_RGB], background=background),
                rrb.Spatial2DView(
                    name="MoGe-2 metric depth at the current focal (the prior)", origin=newest, contents=[entities.CALIBRATION_NEWEST_PRIOR], background=background
                ),
                rrb.Spatial2DView(name="DROID depth after bundle adjustment", origin=newest, contents=[entities.CALIBRATION_NEWEST_DEPTH], background=background),
            ),
            rrb.Vertical(map_view, plots, row_shares=[3, 1]),
            column_shares=[1, 3],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline=entities.CALIBRATION_TIMELINE, play_state="paused"),
        auto_views=False,
        auto_layout=False,
    )


def app_blueprint(final_eye: SceneView | None) -> rrb.Blueprint:
    """The Gradio app's layout once DPVO tracks (after :func:`calibration_blueprint`) and of the finished run (``ours``;
    a user video has no GT), on ``video_time``.

    Left: the input video with DPVO's patches, MoGe-2 metric depth and normals of the keyframes, the DINOv3 feature PCA.
    Right, in tabs, in the order the run makes them: SharedK calibration's map and plots (to look back at), DPVO's
    keyframe path and BA window as it tracks, then the finished trajectory with the fused metric cloud, and the same
    cloud coloured by its MoGe-2 normals; the world views show the one camera the video plays in. While tracking
    (``final_eye`` None) the DPVO tab is active; the final layout opens the trajectory tab with ``final_eye`` framing it.
    """
    background: rrb.Background = rrb.Background(color=BACKGROUND_RGB)
    calibration_view: rrb.Vertical = rrb.Vertical(*calibration_views(background), row_shares=[3, 1], name="1 · SharedK calibration")
    tracking_view: rrb.Vertical = rrb.Vertical(*tracking_views(background), name="2 · DPVO tracking")
    # One camera in every view: the live DPVO camera with the input video in its pinhole, which rides the final
    # trajectory once the run ends (metrictrail.outputs.ride_final_trajectory), instead of the world's own frustum.
    shared_camera: list[str] = ["/world/**", f"- /{entities.world_cam('ours')}/**", f"/{entities.DPVO_CURRENT}/**", f"- /{entities.INPUT_PATCHES}"]
    world_view: rrb.Spatial3DView = rrb.Spatial3DView(
        name="3 · trajectory + metric surface",
        origin=entities.WORLD_ROOT,
        contents=[*shared_camera, f"- /{entities.WORLD_CLOUD_NORMALS}", f"- /{entities.WORLD_MESH_NORMALS}"],
        background=background,
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=final_eye.controls() if final_eye is not None else None,
    )
    world_normals_view: rrb.Spatial3DView = rrb.Spatial3DView(
        name="4 · surface coloured by normals",
        origin=entities.WORLD_ROOT,
        contents=[*shared_camera, f"- /{entities.WORLD_CLOUD}", f"- /{entities.WORLD_MESH}"],
        background=background,
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=final_eye.controls() if final_eye is not None else None,
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                frame_view(has_video=True, has_capture=True, background=background),
                rrb.Spatial2DView(name="MoGe-2 metric depth (keyframes)", origin=entities.CAM, contents=[entities.CAM_DEPTH], background=background),
                rrb.Spatial2DView(name="MoGe-2 normals (keyframes)", origin=entities.CAM, contents=[entities.CAM_NORMALS], background=background),
                rrb.Spatial2DView(name="DINOv3 feature PCA", origin=entities.CAM, contents=[entities.CAM_DINO_PCA], background=background),
            ),
            rrb.Tabs(calibration_view, tracking_view, world_view, world_normals_view, active_tab=1 if final_eye is None else 2),
            column_shares=[1, 3],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        time_panel(has_video=True, live=final_eye is None),
        auto_views=False,
        auto_layout=False,
    )


def run_blueprint(
    label: RunLabel,
    *,
    has_capture: bool,
    has_gt: bool,
    has_video: bool = False,
    world_eye: SceneView | None = None,
    metric_eye: SceneView | None = None,
) -> rrb.Blueprint:
    """The default layout of one run recording; the eyes frame the trajectory (the viewer's default can sit inside it)."""
    background: rrb.Background = rrb.Background(color=BACKGROUND_RGB)
    depth_view: rrb.Spatial2DView = rrb.Spatial2DView(
        name="MoGe-2 metric depth (m)", origin=entities.CAM, contents=[entities.CAM_DEPTH], background=background
    )
    normals_view: rrb.View = (
        rrb.Spatial2DView(name="MoGe-2 normals (keyframes)", origin=entities.CAM, contents=[entities.CAM_NORMALS], background=background)
        if has_capture
        else waiting("MoGe-2 normals", "normals")
    )
    dino_view: rrb.View = (
        rrb.Spatial2DView(name="DINOv3 feature PCA", origin=entities.CAM, contents=[entities.CAM_DINO_PCA], background=background)
        if has_capture
        else waiting("DINOv3 feature PCA", "dino")
    )
    world_name: str = "trajectory vs GT, Sim(3) aligned + metric surface" if has_gt else "trajectory + metric surface (ViPE world)"
    world_view: rrb.Spatial3DView = rrb.Spatial3DView(
        name=world_name,
        origin=entities.WORLD_ROOT,
        contents=["/world/**", f"- /{entities.WORLD_CLOUD_NORMALS}", f"- /{entities.WORLD_MESH_NORMALS}"],
        background=background,
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=world_eye.controls() if world_eye is not None else None,
    )
    world_normals_view: rrb.View = (
        rrb.Spatial3DView(
            name="surface coloured by normals",
            origin=entities.WORLD_ROOT,
            contents=["/world/**", f"- /{entities.WORLD_CLOUD}", f"- /{entities.WORLD_MESH}"],
            background=background,
            line_grid=rrb.LineGrid3D(visible=False),
            eye_controls=world_eye.controls() if world_eye is not None else None,
        )
        if has_capture
        else waiting("surface coloured by normals", "normals")
    )
    window_view: rrb.Container | rrb.View = rrb.Horizontal(*tracking_views(background)) if has_capture else waiting("DPVO local BA window", "window")
    if has_gt:
        error_view: rrb.View = rrb.TimeSeriesView(name="ATE per frame, Sim(3) (m)", origin="metrics/ate_sim3")
        metric_view: rrb.View = rrb.Spatial3DView(
            name="SE(3) only, scale 1: metric scale vs GT (1 m grid)",
            origin=entities.METRIC_ROOT,
            contents=["/metric/**", f"- /{entities.METRIC_CLOUD_NORMALS}", f"- /{entities.METRIC_MESH_NORMALS}"],
            background=background,
            line_grid=rrb.LineGrid3D(visible=True, spacing=1.0),
            eye_controls=metric_eye.controls() if metric_eye is not None else None,
        )
    else:
        error_view = waiting("ATE per frame", "gt")
        metric_view = waiting("metric scale vs GT", "gt")
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(frame_view(has_video=has_video, has_capture=has_capture, background=background), depth_view, normals_view, dino_view),
            rrb.Vertical(rrb.Tabs(world_view, world_normals_view, active_tab=0), window_view, row_shares=[3, 2]),
            rrb.Vertical(error_view, metric_view, rrb.TextDocumentView(name="summary", origin=entities.SUMMARY), row_shares=[2, 3, 2]),
            column_shares=[2, 4, 3],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        time_panel(has_video=has_video),
        auto_views=False,
        auto_layout=False,
    )
