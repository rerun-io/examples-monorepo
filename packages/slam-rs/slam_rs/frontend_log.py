"""Rerun logging for frontend keypoints, trails, occupancy and counters."""

from dataclasses import dataclass, field

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Int32, Int64, UInt8, UInt64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import RIG_ENTITY, CameraCalib

TRAIL_LENGTH: int = 10
"""Positions kept per track for the trail behind it."""
KEYPOINT_RADIUS_PX: float = 2.0
"""Radius of a tracked keypoint, in image pixels."""
CELL_COLOR: tuple[int, int, int, int] = (70, 190, 255, 110)
"""Occupied detection cells, drawn as translucent outlines."""
STATS_ENTITY: str = "/stats/frontend"
"""Where the per-frame counters go, off the dataset's own tree."""


def camera_entity(index: int) -> str:
    """The dataset's own path for one camera's image plane."""
    return f"{RIG_ENTITY}/cam_{index:02d}/pinhole"


_rising: UInt8[ndarray, " 255"] = np.arange(255, dtype=np.uint8)
_falling: UInt8[ndarray, " 255"] = 255 - _rising
_full: UInt8[ndarray, " 255"] = np.full(255, 255, dtype=np.uint8)
_zero: UInt8[ndarray, " 255"] = np.zeros(255, dtype=np.uint8)
HUE_RAMP: UInt8[ndarray, "1275 3"] = np.concatenate(
    [
        np.stack([_full, _rising, _zero], axis=1),
        np.stack([_falling, _full, _zero], axis=1),
        np.stack([_zero, _full, _rising], axis=1),
        np.stack([_zero, _falling, _full], axis=1),
        np.stack([_rising, _zero, _full], axis=1),
    ]
)
"""Five fixed hue ramps, from red through blue."""


def track_colors(ids: Int64[ndarray, " n_tracks"]) -> UInt8[ndarray, "n_tracks 3"]:
    """A stable, saturated colour per track id, from a 32-bit hash of the id.

    Hue only, so every colour is equally readable over grey imagery, and two
    neighbouring ids get unrelated hues — which is what makes an id swap visible
    rather than a smooth gradient.

    Args:
        ids: Track ids.

    Returns:
        One ``uint8`` RGB triple per id.
    """
    hashed: UInt64[ndarray, " n_tracks"] = (ids.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return HUE_RAMP[hashed % np.uint64(len(HUE_RAMP))]


def log_keypoints(frame: _core.FlowFrame) -> None:
    """Draw one frameset's tracked keypoints on the camera images.

    Two rungs draw this layer — the frontend's own and the estimator's, whose
    frontend is the same code — so it is drawn once here: a keypoint then has the
    same colour, the same radius and the same entity path in both recordings, and
    the two views can be read side by side. What stays with the frontend rung is
    its own evidence: the trails and the occupancy grid.

    Args:
        frame: What a frontend produced for one frameset.
    """
    for index in range(frame.camera_count):
        ids: Int64[ndarray, " n_tracks"] = frame.ids(index)
        rr.log(
            f"{camera_entity(index)}/keypoints",
            rr.Points2D(frame.positions(index), colors=track_colors(ids), radii=KEYPOINT_RADIUS_PX),
        )


@dataclass(slots=True)
class FrontendLogger:
    """Logs one frameset's frontend output, and carries the trail history between them."""

    camera_count: int
    """Cameras on the rig."""
    trails: list[dict[int, list[tuple[float, float]]]] = field(init=False)
    """Per camera, the last :data:`TRAIL_LENGTH` positions of every live track."""
    def __post_init__(self) -> None:
        """Initialize each camera's trail history."""
        self.trails = [{} for _ in range(self.camera_count)]

    def log(self, frame: _core.FlowFrame, elapsed_ms: float) -> None:
        """Log one frameset's keypoints, trails, occupancy and counters.

        The caller has already set the timeline, so this writes at the current
        cursor and nowhere else.

        Args:
            frame: What the frontend produced for this frameset.
            elapsed_ms: Wall time the ``process`` call took.
        """
        log_keypoints(frame)
        for index in range(self.camera_count):
            ids: Int64[ndarray, " n_tracks"] = frame.ids(index)
            positions: Float32[ndarray, "n_tracks 2"] = frame.positions(index)
            colors: UInt8[ndarray, "n_tracks 3"] = track_colors(ids)
            self._log_trails(index, ids, positions, colors)
            self._log_cells(index, frame)
            rr.log(f"{STATS_ENTITY}/cam_{index:02d}/num_tracks", rr.Scalars(float(frame.num_tracks(index))))
            rr.log(f"{STATS_ENTITY}/cam_{index:02d}/num_new", rr.Scalars(float(frame.num_new(index))))
        rr.log(f"{STATS_ENTITY}/frontend_ms", rr.Scalars(elapsed_ms))

    def _log_trails(
        self,
        index: int,
        ids: Int64[ndarray, " n_tracks"],
        positions: Float32[ndarray, "n_tracks 2"],
        colors: UInt8[ndarray, "n_tracks 3"],
    ) -> None:
        """Extend every live track's trail by one position and log the strips."""
        # Rebuilt from the live ids alone, so a track that dies takes its history
        # with it and the dictionary cannot grow over a long segment.
        history: dict[int, list[tuple[float, float]]] = self.trails[index]
        updated: dict[int, list[tuple[float, float]]] = {}
        strips: list[Float32[ndarray, "n_points 2"]] = []
        strip_colors: list[UInt8[ndarray, " 3"]] = []
        for slot, identifier in enumerate(ids.tolist()):
            trail: list[tuple[float, float]] = history.get(identifier, [])[-(TRAIL_LENGTH - 1) :]
            trail.append((float(positions[slot, 0]), float(positions[slot, 1])))
            updated[identifier] = trail
            if len(trail) > 1:
                strips.append(np.array(trail, dtype=np.float32))
                strip_colors.append(colors[slot])
        self.trails[index] = updated
        rr.log(f"{camera_entity(index)}/trails", rr.LineStrips2D(strips, colors=strip_colors, radii=1.0))

    def _log_cells(self, index: int, frame: _core.FlowFrame) -> None:
        """Draw the occupied detection cells over the image."""
        occupancy: Int32[ndarray, "rows columns"] = frame.occupancy(index)
        occupied: tuple[Int64[ndarray, " n_cells"], ...] = np.nonzero(occupancy)
        rows: Int64[ndarray, " n_cells"] = occupied[0]
        columns: Int64[ndarray, " n_cells"] = occupied[1]
        cell: int = frame.cell_size
        origin: tuple[int, int] = frame.cell_origin
        mins: Float32[ndarray, "n_cells 2"] = np.stack([origin[0] + columns * cell, origin[1] + rows * cell], axis=1).astype(np.float32)
        rr.log(
            f"{camera_entity(index)}/cells",
            rr.Boxes2D(mins=mins, sizes=np.full_like(mins, float(cell)), colors=CELL_COLOR),
        )

def camera_views(cameras: tuple[CameraCalib, ...]) -> list[rrb.View]:
    """One 2D view per camera, named and origined on the camera's own entity.

    The panel layout of a camera is a convention both rungs share, so both
    blueprints build their views here.

    Args:
        cameras: The rig's cameras, in rig order.

    Returns:
        The views, in the same order.
    """
    return [rrb.Spatial2DView(origin=camera_entity(camera.index), name=f"cam {camera.index:02d}") for camera in cameras]


def frontend_blueprint(cameras: tuple[CameraCalib, ...]) -> rrb.Blueprint:
    """One 2D view per camera over the images, keypoints and trails, plus the counters.

    Args:
        cameras: The rig's cameras, in rig order.

    Returns:
        A blueprint with the panels collapsed, so the frame is all content.
    """
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*camera_views(cameras)),
            rrb.TimeSeriesView(origin=STATS_ENTITY, name="frontend"),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
