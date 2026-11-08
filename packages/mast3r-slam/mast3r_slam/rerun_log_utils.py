"""Rerun visualisation helpers for MASt3R-SLAM.

Provides blueprint creation, video logging, and shared constants
(``FRAME_TIMELINE``, ``VIDEO_TIMELINE``) used by both the pipeline
and the async logger (``async_logger.py``).
"""

from pathlib import Path

import numpy as np
import rerun.blueprint as rrb
from jaxtyping import Int
from numpy import ndarray
from simplecv.rerun_log_utils import log_video

from mast3r_slam.dataloader import MonocularDataset, MP4Dataset

FRAME_TIMELINE: str = "frame"
VIDEO_TIMELINE: str = "video_time"
# The source video, logged once as a VideoStream. Camera panels show the model's cropped frame instead.
VIDEO_LOG_PATH: Path = Path("/video")
# Entity layout. Under the parent log path (``/world``) every Pinhole owns only what projects in 3D: the current
# camera its image and depth, each keyframe its image. 2D-only maps live outside it, where no 3D view looks:
CURRENT_MAPS_PATH: Path = Path("/current_maps")
"""The current frame's pointmap and confidence."""
LAST_KEYFRAME_PATH: Path = Path("/last_keyframe")
"""The keyframe being tracked against: image, pointmap, confidence, depth."""
KEYFRAME_MAPS_PATH: Path = Path("/keyframe_maps")
"""Each keyframe's pointmap, confidence and depth, under a static copy of its Pinhole."""


# Chase eye of the Follow view, in the camera frame (RDF: x right, y down, z forward), like dpvo's. MASt3R's
# pointmaps are roughly metric, so this sits 0.6 m above and 1.5 m behind the camera, aimed 1.5 m ahead of it.
_FOLLOW_EYE_POSITION: tuple[float, float, float] = (0.0, -0.6, -1.5)
_FOLLOW_EYE_TARGET: tuple[float, float, float] = (0.0, 0.0, 1.5)
_FOLLOW_EYE_UP: tuple[float, float, float] = (0.0, -1.0, 0.0)


def create_blueprints(parent_log_path: Path, timeline: str = FRAME_TIMELINE) -> rrb.Blueprint:
    """Create the SLAM visualiser's blueprint, sent once at the start of a run.

    Overview and Follow 3D views beside the 2D camera panels, like dpvo. The
    overview is rooted at ``/`` so the gravity-alignment transform on
    ``parent_log_path`` shows (rerun#9244: a Transform3D on a view's own origin
    is invisible), with the ground grid. The Follow view is rooted at the
    current camera, so its eye, fixed in the camera frame, rides the camera; it
    has no grid because the camera frame has no ground plane.

    Both 3D views show ``parent_log_path`` and nothing else: the entity layout
    keeps 2D-only maps outside it (``CURRENT_MAPS_PATH``, ``LAST_KEYFRAME_PATH``,
    ``KEYFRAME_MAPS_PATH``), so the rule needs no exclusions.

    Args:
        parent_log_path: Root log path for all Rerun entities.
        timeline: Timeline shown in the Rerun time panel.

    Returns:
        A configured Rerun Blueprint.
    """
    root: str = f"/{str(parent_log_path).lstrip('/')}"
    camera_path: str = f"{root}/current_camera"
    contents_3d: str = f"+ {root}/**"
    # The camera path is logged one segment per frame; show every segment from the start up to the cursor.
    path_history: rrb.VisibleTimeRanges = rrb.VisibleTimeRanges(
        [
            rrb.VisibleTimeRange(name, start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.cursor_relative())
            for name in (FRAME_TIMELINE, VIDEO_TIMELINE)
        ]
    )
    overview = rrb.Spatial3DView(
        name="Overview",
        origin="/",
        contents=contents_3d,
        overrides={f"{root}/path": path_history},
        eye_controls=rrb.EyeControls3D(spin_speed=0.25),
    )
    follow = rrb.Spatial3DView(
        name="Follow",
        origin=camera_path,
        contents=contents_3d,
        overrides={f"{root}/path": path_history},
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=_FOLLOW_EYE_POSITION,
            look_target=_FOLLOW_EYE_TARGET,
            eye_up=_FOLLOW_EYE_UP,
            spin_speed=0.0,
        ),
    )
    # The original video on top, then what MASt3R derives from the current frame and the keyframe it
    # tracks against. The model's cropped frame shows on the camera frustum in the 3D views.
    views: rrb.Vertical = rrb.Vertical(
        contents=[
            rrb.Spatial2DView(origin=str(VIDEO_LOG_PATH), name="Video"),
            rrb.Grid(
                rrb.Spatial2DView(origin=f"{camera_path}/pinhole/depth", name="Depth"),
                rrb.Spatial2DView(origin=str(CURRENT_MAPS_PATH / "confidence"), name="Confidence"),
                rrb.Spatial2DView(origin=str(CURRENT_MAPS_PATH / "pointmap"), name="Pointmap"),
                rrb.Spatial2DView(origin=str(LAST_KEYFRAME_PATH / "image"), name="Keyframe"),
                grid_columns=2,
            ),
        ],
        row_shares=[1, 1],
        name="Views",
    )
    logs: rrb.TextLogView = rrb.TextLogView(origin=f"{root}/logs", name="Logs")
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(overview, follow),
            rrb.Tabs(views, logs, active_tab=0),
            column_shares=(3, 2),
        ),
        rrb.TimePanel(timeline=timeline, state="collapsed"),
        collapse_panels=True,
    )


def log_video_for_dataset(dataset: MonocularDataset) -> Int[ndarray, "num_frames"] | None:
    """Log the source video once as a VideoStream on ``VIDEO_TIMELINE``.

    Returns the timestamps aligned to the dataset frames, or ``None`` for datasets that are not an MP4.
    """
    if not isinstance(dataset, MP4Dataset) or dataset.dataset_path is None:
        return None

    frame_timestamps_ns: Int[ndarray, "all_video_frames"] = log_video(
        video_source=dataset.dataset_path,
        video_log_path=VIDEO_LOG_PATH,
        timeline=VIDEO_TIMELINE,
    )
    return np.asarray(frame_timestamps_ns[:: dataset.stride][: len(dataset)], dtype=np.int64)
