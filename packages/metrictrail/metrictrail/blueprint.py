"""Default blueprint of a MetricTrail run recording (entity layout in :mod:`metrictrail.entities`).

Left: the frame DPVO tracks with its patches, the MoGe-2 metric depth, the DINOv3 feature PCA. Middle: the trajectory
against GT after the paper's Sim(3) alignment with the fused metric cloud, and DPVO's local BA window. Right: the
per-frame error and running ATE, the SE(3)-only (scale 1) view on a 1 m grid, and the summary card. A view whose
data the recording does not hold (no live capture, no GT) is a text tile that says what it is waiting for.
"""

from dataclasses import dataclass

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float64
from numpy import ndarray

from metrictrail import entities
from metrictrail.palette import BACKGROUND_RGB, RunLabel


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
        notes |= {"dino": text, "patches": text, "window": text}
    if not has_gt:
        notes |= {"gt": "waiting for ground truth: pass a VSLAM-LAB sequence (`--gt-sequence`)"}
    return notes


def log_waiting_notes(rec: rr.RecordingStream, has_capture: bool, has_gt: bool) -> None:
    """Log the texts of :func:`waiting_notes` as static ``TextDocument`` entities."""
    for key, text in waiting_notes(has_capture, has_gt).items():
        rec.log(f"notes/waiting/{key}", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN), static=True)


def run_blueprint(
    label: RunLabel, *, has_capture: bool, has_gt: bool, world_eye: SceneView | None = None, metric_eye: SceneView | None = None
) -> rrb.Blueprint:
    """The default layout of one run recording; the eyes frame the trajectory (the viewer's default can sit inside it)."""
    background: rrb.Background = rrb.Background(color=BACKGROUND_RGB)
    frame_view: rrb.Spatial2DView = rrb.Spatial2DView(
        name="frame + DPVO patches" if has_capture else "input frame",
        origin=entities.CAM,
        contents=[entities.CAM_RGB, entities.CAM_PATCHES],
        background=background,
    )
    depth_view: rrb.Spatial2DView = rrb.Spatial2DView(
        name="MoGe-2 metric depth (m)", origin=entities.CAM, contents=[entities.CAM_DEPTH], background=background
    )
    dino_view: rrb.View = (
        rrb.Spatial2DView(name="DINOv3 feature PCA", origin=entities.CAM, contents=[entities.CAM_DINO_PCA], background=background)
        if has_capture
        else waiting("DINOv3 feature PCA", "dino")
    )
    world_name: str = "trajectory vs GT, Sim(3) aligned + metric cloud" if has_gt else "trajectory + metric cloud (ViPE world)"
    world_view: rrb.Spatial3DView = rrb.Spatial3DView(
        name=world_name,
        origin=entities.WORLD_ROOT,
        contents=["/world/**"],
        background=background,
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=world_eye.controls() if world_eye is not None else None,
    )
    window_view: rrb.View = (
        rrb.Spatial3DView(name="DPVO local BA window", origin=entities.DPVO_ROOT, contents=["/dpvo/**"], background=background)
        if has_capture
        else waiting("DPVO local BA window", "window")
    )
    if has_gt:
        error_view: rrb.View = rrb.TimeSeriesView(name="ATE per frame, Sim(3) (m)", origin="metrics/ate_sim3")
        metric_view: rrb.View = rrb.Spatial3DView(
            name="SE(3) only, scale 1: metric scale vs GT (1 m grid)",
            origin=entities.METRIC_ROOT,
            contents=["/metric/**"],
            background=background,
            line_grid=rrb.LineGrid3D(visible=True, spacing=1.0),
            eye_controls=metric_eye.controls() if metric_eye is not None else None,
        )
    else:
        error_view = waiting("ATE per frame", "gt")
        metric_view = waiting("metric scale vs GT", "gt")
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(frame_view, depth_view, dino_view),
            rrb.Vertical(world_view, window_view, row_shares=[3, 2]),
            rrb.Vertical(error_view, metric_view, rrb.TextDocumentView(name="summary", origin=entities.SUMMARY), row_shares=[2, 3, 2]),
            column_shares=[2, 4, 3],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline=entities.FRAME_TIMELINE),
        auto_views=False,
        auto_layout=False,
    )
