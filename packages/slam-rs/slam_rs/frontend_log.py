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
so the fixture directory names it in :data:`SOURCE_FILE`, and the logger compares
that name once, when it is built, with the segment id the replayed recording
carries. Timestamps alone are not an association: the feed reports ``video_time``,
which starts at zero on every segment, so the smoke segment's first-frame
keypoints landed on the first frame of every other segment too and read as a
parity claim about a recording the C++ never saw.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

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


def log_keypoints(frame: _core.FlowFrame) -> None:
    """Draw one frameset's tracked keypoints on the camera images.

    Two rungs draw this layer — the frontend's own and the estimator's, whose
    frontend is the same code — so it is drawn once here: a keypoint then has the
    same colour, the same radius and the same entity path in both recordings, and
    the two views can be read side by side. What stays with the frontend rung is
    its own evidence: the trails, the occupancy grid and the C++ overlay.

    Args:
        frame: What a frontend produced for one frameset.
    """
    for index in range(frame.camera_count):
        ids: Int64[ndarray, " n_tracks"] = frame.ids(index)
        rr.log(
            f"{camera_entity(index)}/keypoints",
            rr.Points2D(frame.positions(index), colors=track_colors(ids), radii=KEYPOINT_RADIUS_PX),
        )


def dumps_source(directory: Path | None = None) -> str:
    """Which segment a directory of C++ dumps was recorded from.

    The dumps are on the same ``video_time`` clock the feed reports, which starts
    at zero on every segment, so the timestamp alone does not say which recording
    a dump belongs to. :data:`SOURCE_FILE` in the directory does, and it is a
    fact about the directory rather than about each dump in it.

    Args:
        directory: Where ``frame_XXX.json`` and :data:`SOURCE_FILE` live; the
            committed fixtures by default.

    Returns:
        The segment the dumps came from, or the empty string when the directory
        holds none.

    Raises:
        ValueError: The directory holds dumps but no :data:`SOURCE_FILE` naming
            the segment they came from, so nothing could be associated with them.
    """
    source: Path = directory if directory is not None else DEFAULT_DUMPS_DIR
    # A missing directory globs to nothing, which is what a directory holding no
    # dumps is: the committed fixtures are the only ones that always exist.
    frames: list[Path] = list(source.glob("frame_*.json"))
    if not frames:
        return ""
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
    return segment_id


def read_cpp_dumps(camera_count: int, directory: Path | None = None) -> dict[int, list[Float32[ndarray, "n_keypoints 2"]]]:
    """Read the basalt C++ frontend's own keypoints, by frameset ``t_ns``.

    Called once :func:`dumps_source` has said the dumps are of the recording
    being replayed, and not before: the rig check below is a fact about the
    overlay that will be drawn, so dumps of another recording must not refuse a
    replay they would draw nothing on.

    Args:
        camera_count: Cameras on the rig being replayed; every dump must carry
            exactly that many, or one camera would keep a stale overlay.
        directory: Where the ``frame_XXX.json`` live; the committed fixtures by default.

    Returns:
        One ``[x, y]`` array per camera, per frameset ``t_ns``; empty when the
        directory holds no dumps.

    Raises:
        ValueError: A dump carries no timestamp, or another rig's cameras.
    """
    source: Path = directory if directory is not None else DEFAULT_DUMPS_DIR
    dumps: dict[int, list[Float32[ndarray, "n_keypoints 2"]]] = {}
    for path in sorted(source.glob("frame_*.json")):
        dump: dict[str, object] = json.loads(path.read_text())
        t_ns: object = dump.get("t_ns")
        if not isinstance(t_ns, int):
            raise ValueError(f'{path} must carry an integer "t_ns", the frameset time the keypoints were dumped at')
        cameras: object = dump.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != camera_count:
            raise ValueError(f"{path} holds {len(cameras) if isinstance(cameras, list) else 0} cameras, the rig being replayed has {camera_count}")
        dumps[t_ns] = [
            np.array([[keypoint["x"], keypoint["y"]] for keypoint in camera["keypoints"]], dtype=np.float32).reshape(-1, 2)
            for camera in cameras
        ]
    return dumps


@dataclass(slots=True)
class FrontendLogger:
    """Logs one frameset's frontend output, and carries the trail history between them."""

    camera_count: int
    """Cameras on the rig."""
    source_segment: str
    """Segment id the replayed recording carries; only dumps recorded from it are drawn."""
    dumps_dir: Path | None = None
    """Where the C++ dumps are read from; the committed fixtures by default."""
    cpp_dumps: dict[int, list[Float32[ndarray, "n_keypoints 2"]]] = field(init=False)
    """This recording's C++ keypoints by frameset ``t_ns``; empty when the dumps are another's."""
    trails: list[dict[int, list[tuple[float, float]]]] = field(init=False)
    """Per camera, the last :data:`TRAIL_LENGTH` positions of every live track."""
    cpp_logged: bool = False
    """Whether an overlay has been drawn, so it can be cleared once the dumps run out."""

    def __post_init__(self) -> None:
        """Associate the dumps with the recording once, and read their bodies only if they are of it."""
        dumped_from: str = dumps_source(self.dumps_dir)
        of_this_recording: bool = dumped_from == self.source_segment
        if dumped_from != "" and not of_this_recording:
            print(f"dumps are from {dumped_from}, replaying {self.source_segment}: no overlay")
        # Only the dumps that will be drawn are read: their camera count is a
        # fact about the overlay, so another recording's dumps would otherwise
        # refuse the replay of a rig they say nothing about.
        self.cpp_dumps = read_cpp_dumps(self.camera_count, self.dumps_dir) if of_this_recording else {}
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
        self._log_cpp(frame.t_ns)

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

    def _log_cpp(self, t_ns: int) -> None:
        """Draw the C++ frontend's own keypoints for this frameset, if it dumped any.

        Only this recording's dumps are here: another's were never read, however
        well the two clocks line up.
        """
        dump: list[Float32[ndarray, "n_keypoints 2"]] | None = self.cpp_dumps.get(t_ns)
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
    """One 2D view per camera over the images, keypoints, trails and overlay, plus the counters.

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
