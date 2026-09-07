"""Rerun logging for the optical-flow frontend: keypoints, trails, occupancy and the C++ overlay.

All Rerun logging is Python (D03), so the core returns arrays and this module
decides what they look like. Everything is logged under the dataset's own entity
tree — ``/world/rig_00/cam_MM/pinhole/...`` — so a frontend run and the recording
it came from sit in one viewer without a second coordinate convention.

The C++ overlay is what makes a parity claim visible rather than asserted: the
basalt fork's ``tools/dump_flow.cpp`` dumped its own keypoints for the first eight
framesets of the smoke segment, those dumps are committed next to the Rust flow
gate, and this module draws them in one contrasting colour beside the port's own.
Where the two agree the magenta sits under the coloured dot; where they disagree
it stands alone.

An overlay is only ever drawn on the segment it was recorded from. The dumps carry
no segment of their own — ``dump_flow.cpp`` writes a frame index and a timestamp —
so the fixture directory names it in :data:`SOURCE_FILE`, and every dump is keyed
by ``(segment id, frameset t_ns)``. Timestamps alone are not an association: the
feed reports ``video_time``, which starts at zero on every segment, so the smoke
segment's first-frame keypoints landed on the first frame of every other segment
too and read as a parity claim about a recording the C++ never saw.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Int64, UInt8, UInt64
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib

TRAIL_LENGTH: int = 10
"""Positions kept per track for the trail behind it."""
KEYPOINT_RADIUS_PX: float = 2.0
"""Radius of a tracked keypoint, in image pixels."""
CPP_RADIUS_PX: float = 3.5
"""Radius of a C++ keypoint: larger, so the port's dot sits inside it when they agree."""
CPP_COLOR: tuple[int, int, int] = (255, 0, 255)
"""The one colour the C++ overlay is drawn in; nothing else in the view is magenta."""
CELL_COLOR: tuple[int, int, int, int] = (70, 190, 255, 110)
"""Occupied detection cells, drawn as translucent outlines."""
DEFAULT_DUMPS_DIR: Path = Path(__file__).resolve().parents[1] / "crates/slam-rs/tests/fixtures/flow/dumps"
"""The eight committed dumps the Rust flow gate runs off."""
SOURCE_FILE: str = "source.json"
"""Names the segment a dump directory was recorded from; its ``segment_id`` key is read."""
STATS_ENTITY: str = "/stats/frontend"
"""Where the per-frame counters go, off the dataset's own tree."""


def camera_entity(index: int) -> str:
    """The dataset's own path for one camera's image plane."""
    return f"/world/rig_00/cam_{index:02d}/pinhole"


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
"""Five of the hue circle's six 255-long ramps, red round to blue.

The sixth ramp runs from magenta back to red and starts at exactly
:data:`CPP_COLOR`, so a track drawn from it would read as the C++ overlay. Built
once at import: the ramp is fixed, and it was rebuilt per camera per frameset.
"""


def track_colors(ids: Int64[ndarray, " n_tracks"]) -> UInt8[ndarray, "n_tracks 3"]:
    """A stable, saturated colour per track id, from a 32-bit hash of the id.

    Hue only, so every colour is equally readable over grey imagery, and two
    neighbouring ids get unrelated hues — which is what makes an id swap visible
    rather than a smooth gradient.

    Args:
        ids: Track ids.

    Returns:
        One ``uint8`` RGB triple per id, never :data:`CPP_COLOR`.
    """
    hashed: UInt64[ndarray, " n_tracks"] = (ids.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)
    return HUE_RAMP[hashed % np.uint64(len(HUE_RAMP))]


def read_cpp_dumps(camera_count: int, directory: Path | None = None) -> dict[tuple[str, int], list[Float32[ndarray, "n_keypoints 2"]]]:
    """Read the basalt C++ frontend's own keypoints, keyed by segment and frameset timestamp.

    The dumps are on the same ``video_time`` clock the feed reports, which starts
    at zero on every segment, so the timestamp alone does not say which recording
    a dump belongs to. :data:`SOURCE_FILE` in the directory does, and it is half
    of every key here.

    Args:
        camera_count: Cameras on the rig being replayed; every dump must carry
            exactly that many, or one camera would keep a stale overlay.
        directory: Where ``frame_XXX.json`` and :data:`SOURCE_FILE` live; the
            committed fixtures by default.

    Returns:
        ``(segment_id, t_ns)`` to one array of ``[x, y]`` per camera; empty when
        the directory holds no dumps.

    Raises:
        ValueError: The directory holds dumps but no :data:`SOURCE_FILE` naming
            the segment they came from, so nothing could be associated with
            them; or a dump carries no timestamp, or another rig's cameras.
    """
    source: Path = directory if directory is not None else DEFAULT_DUMPS_DIR
    if not source.is_dir():
        return {}
    frames: list[Path] = sorted(source.glob("frame_*.json"))
    if not frames:
        return {}
    provenance: Path = source / SOURCE_FILE
    if not provenance.is_file():
        raise ValueError(
            f"{source} holds {len(frames)} C++ dumps but no {SOURCE_FILE}; write one with a "
            '"segment_id" key naming the segment they were dumped from, or the overlay would '
            "be drawn over whatever segment happens to be replayed"
        )
    segment_id: object = json.loads(provenance.read_text()).get("segment_id")
    if not isinstance(segment_id, str):
        raise ValueError(f'{provenance} must carry a "segment_id" string naming the segment the dumps were dumped from')
    dumps: dict[tuple[str, int], list[Float32[ndarray, "n_keypoints 2"]]] = {}
    for path in frames:
        dump: dict[str, object] = json.loads(path.read_text())
        t_ns: object = dump.get("t_ns")
        if not isinstance(t_ns, int):
            raise ValueError(f'{path} must carry an integer "t_ns", the frameset time the keypoints were dumped at')
        cameras: object = dump.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != camera_count:
            raise ValueError(f"{path} holds {len(cameras) if isinstance(cameras, list) else 0} cameras, the rig being replayed has {camera_count}")
        dumps[segment_id, t_ns] = [
            np.array([[keypoint["x"], keypoint["y"]] for keypoint in camera["keypoints"]], dtype=np.float32).reshape(-1, 2)
            for camera in cameras
        ]
    return dumps


@dataclass(slots=True)
class FrontendLogger:
    """Logs one frameset's frontend output, and carries the trail history between them."""

    camera_count: int
    """Cameras on the rig."""
    segment_id: str
    """Segment being replayed; only dumps recorded from it are drawn."""
    cpp_dumps: dict[tuple[str, int], list[Float32[ndarray, "n_keypoints 2"]]]
    """The C++ frontend's keypoints by ``(segment id, frameset t_ns)``; empty without dumps."""
    trails: list[dict[int, list[tuple[float, float]]]]
    """Per camera, the last :data:`TRAIL_LENGTH` positions of every live track."""
    cpp_logged: bool = False
    """Whether an overlay has been drawn, so it can be cleared once the dumps run out."""

    @classmethod
    def create(cls, camera_count: int, segment_id: str, dumps_dir: Path | None = None) -> "FrontendLogger":
        """Build a logger for a rig of ``camera_count`` cameras replaying ``segment_id``."""
        return cls(
            camera_count=camera_count,
            segment_id=segment_id,
            cpp_dumps=read_cpp_dumps(camera_count, dumps_dir),
            trails=[{} for _ in range(camera_count)],
        )

    def log(self, frame: _core.FlowFrame, elapsed_ms: float) -> None:
        """Log one frameset's keypoints, trails, occupancy and counters.

        The caller has already set the timeline, so this writes at the current
        cursor and nowhere else.

        Args:
            frame: What the frontend produced for this frameset.
            elapsed_ms: Wall time the ``process`` call took.
        """
        for index in range(self.camera_count):
            entity: str = camera_entity(index)
            ids: Int64[ndarray, " n_tracks"] = frame.ids(index)
            positions: Float32[ndarray, "n_tracks 2"] = frame.positions(index)
            colors: UInt8[ndarray, "n_tracks 3"] = track_colors(ids)
            rr.log(f"{entity}/keypoints", rr.Points2D(positions, colors=colors, radii=KEYPOINT_RADIUS_PX))
            self._log_trails(entity, index, ids, positions, colors)
            self._log_cells(entity, index, frame)
            rr.log(f"{STATS_ENTITY}/cam_{index:02d}/num_tracks", rr.Scalars(float(frame.num_tracks(index))))
            rr.log(f"{STATS_ENTITY}/cam_{index:02d}/num_new", rr.Scalars(float(frame.num_new(index))))
        rr.log(f"{STATS_ENTITY}/frontend_ms", rr.Scalars(elapsed_ms))
        self._log_cpp(frame.t_ns)

    def _log_trails(
        self,
        entity: str,
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
        rr.log(f"{entity}/trails", rr.LineStrips2D(strips, colors=strip_colors, radii=1.0))

    def _log_cells(self, entity: str, index: int, frame: _core.FlowFrame) -> None:
        """Draw the occupied detection cells over the image."""
        occupancy: ndarray = frame.occupancy(index)
        rows, columns = np.nonzero(occupancy)
        cell: int = frame.cell_size
        x_start, y_start = frame.cell_origin
        mins: Float32[ndarray, "n_cells 2"] = np.stack([x_start + columns * cell, y_start + rows * cell], axis=1).astype(np.float32)
        rr.log(
            f"{entity}/cells",
            rr.Boxes2D(mins=mins, sizes=np.full_like(mins, float(cell)), colors=CELL_COLOR),
        )

    def _log_cpp(self, t_ns: int) -> None:
        """Draw the C++ frontend's own keypoints for this frameset, if it dumped any.

        Keyed by segment as well as timestamp: a dump from another recording is
        not this frameset's, however well the two clocks line up.
        """
        dump: list[Float32[ndarray, "n_keypoints 2"]] | None = self.cpp_dumps.get((self.segment_id, t_ns))
        if dump is None:
            # Latest-at would keep the last overlay on screen for the rest of the
            # segment, which reads as a parity claim about framesets that have no
            # dump at all; one empty batch ends it.
            if self.cpp_logged:
                for index in range(self.camera_count):
                    rr.log(f"{camera_entity(index)}/keypoints_cpp", rr.Points2D(np.zeros((0, 2), dtype=np.float32)))
                self.cpp_logged = False
            return
        for index in range(self.camera_count):
            rr.log(
                f"{camera_entity(index)}/keypoints_cpp",
                rr.Points2D(dump[index], colors=CPP_COLOR, radii=CPP_RADIUS_PX),
            )
        self.cpp_logged = True


def frontend_blueprint(cameras: tuple[CameraCalib, ...]) -> rrb.Blueprint:
    """One 2D view per camera over the images, keypoints, trails and overlay, plus the counters.

    Args:
        cameras: The rig's cameras, in rig order.

    Returns:
        A blueprint with the panels collapsed, so the frame is all content.
    """
    views: list[rrb.View] = [
        rrb.Spatial2DView(origin=camera_entity(camera.index), name=f"cam {camera.index:02d}") for camera in cameras
    ]
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*views),
            rrb.TimeSeriesView(origin=STATS_ENTITY, name="frontend"),
            row_shares=[3, 1],
        ),
        collapse_panels=True,
    )
