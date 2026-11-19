"""Read model proposals of one segment back from the catalog as ``Proposal`` records, keyed by grid time.

Two proposers, one round trip:

- ``egohand``: the labeler's own ``hand_proposals`` layer (written by ``posekit.labeler.propose``), one
  ``Boxes2D`` per camera at ``.../pinhole/proposals/egohand`` on the grid times, class ids 100/101 for the side
  and the detector score in ``posekit.HandProposal:score``.
- ``hands_layer``: the RoboCap ``hands`` layer (robocap-live tracker, Cap B segments only). The box is the
  bounding box of KeyNet's 21 keypoints at ``.../pinhole/hands/{left,right}/keynet`` (plus a margin) where
  KeyNet's average confidence reaches ``KEYNET_MIN_CONFIDENCE``; the side is the entity, the score that
  confidence. Rows are matched to the nearest grid time. The layer's DetNet boxes (``.../detnet``) are not
  used: DetNet only runs for an untracked side, and its circles (checked against the pixels on fe6f s13/s15)
  often cover both hands, the other hand or the background. KeyNet's sides match the pixels (the watch is on
  the wearer's left wrist) and EgoHand's per-frame sides (98 %).

Boxes are clipped to the image; a box that is empty after clipping is dropped.
"""

from collections.abc import Sequence
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import pyarrow as pa
from jaxtyping import Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry

from posekit.labeler.frames import FRAMESET_TOLERANCE_NS, TIMELINE, nearest_indices
from posekit.labeler.paths import LEFT_HAND_CLASS_ID, RIGHT_HAND_CLASS_ID, pinhole_path, proposal_path
from posekit.labeler.schema import HAND_SIDES, Box, CameraInfo, HandSide, Proposal

SCORE_ARCHETYPE: str = "posekit.HandProposal"
"""Archetype name of the proposal score component (``posekit.HandProposal:score``)."""
SCORE_DECIMALS: int = 4
"""Scores are rounded to this many decimals (the layer stores float32)."""
KEYNET_MIN_CONFIDENCE: float = 0.5
"""KeyNet average keypoint confidence from which the hands layer's keypoints make a proposal."""
KEYNET_MARGIN: float = 0.15
"""Margin added on every side of the keypoint bounding box, as a fraction of its longer side (keypoints are joint centres)."""
ProposalsByTime: TypeAlias = dict[int, list[Proposal]]
"""Proposals per grid time, ns; every grid time is a key (with an empty list when nothing was proposed)."""


@runtime_checkable
class ProposalProvider(Protocol):
    """Model proposals for one time of one segment (EgoHand and, on Cap B segments, the ``hands`` layer)."""

    def proposals(self, segment_id: str, t_ns: int) -> list[Proposal]:
        """Every proposal of every camera at ``t_ns``."""
        ...

def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    width: float = min(a.x1, b.x1) - max(a.x0, b.x0)
    height: float = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    inter: float = width * height
    union: float = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union


def keynet_path(cam: int, side: HandSide) -> str:
    """``.../pinhole/hands/<side>/keynet`` — the RoboCap hands layer's KeyNet keypoints of one side."""
    return f"{pinhole_path(cam)}/hands/{side}/keynet"


def clipped_box(x0: float, y0: float, x1: float, y1: float, cam: CameraInfo) -> Box | None:
    """The box clipped to the image, or None when nothing of it is inside."""
    left: float = min(max(x0, 0.0), float(cam.width))
    top: float = min(max(y0, 0.0), float(cam.height))
    right: float = min(max(x1, 0.0), float(cam.width))
    bottom: float = min(max(y1, 0.0), float(cam.height))
    return Box(left, top, right, bottom) if right > left and bottom > top else None


def _rows(table: pa.Table, column: str) -> list:
    return table[column].to_pylist() if column in table.column_names else [None] * table.num_rows


def egohand_proposals(table: pa.Table, cams: Sequence[CameraInfo], times_ns: Int64[ndarray, "g"]) -> ProposalsByTime:
    """``egohand`` proposals of a ``hand_proposals`` reader table (rows of any entity; other columns ignored).

    Args:
        table: Catalog reader output on ``video_time`` with the proposal entities' ``Boxes2D`` and score columns.
        cams: Cameras of the segment.
        times_ns: The grid, ns.
    """
    result: ProposalsByTime = {int(t): [] for t in times_ns.tolist()}
    times: Int64[ndarray, "r"] = table[TIMELINE].cast(pa.int64()).to_numpy() if table.num_rows else np.zeros(0, np.int64)
    columns: Int64[ndarray, "r"] = nearest_indices(times_ns, times, 1_000_000)
    for cam in cams:
        path: str = proposal_path(cam.cam, "egohand")
        centers: list = _rows(table, f"{path}:Boxes2D:centers")
        half_sizes: list = _rows(table, f"{path}:Boxes2D:half_sizes")
        class_ids: list = _rows(table, f"{path}:Boxes2D:class_ids")
        scores: list = _rows(table, f"{path}:{SCORE_ARCHETYPE}:score")
        for row in range(table.num_rows):
            if centers[row] is None or columns[row] < 0:
                continue
            t_ns: int = int(times_ns[columns[row]])
            for index, (center, half) in enumerate(zip(centers[row], half_sizes[row], strict=True)):
                box: Box | None = clipped_box(center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1], cam)
                if box is None:
                    continue
                class_id: int = int(class_ids[row][index])
                if class_id not in (LEFT_HAND_CLASS_ID, RIGHT_HAND_CLASS_ID):
                    raise ValueError(f"{path} at {t_ns}: class id {class_id} is not a hand side")
                side: HandSide = "left" if class_id == LEFT_HAND_CLASS_ID else "right"
                result[t_ns].append(Proposal(cam=cam.cam, side=side, box=box, score=round(float(scores[row][index]), SCORE_DECIMALS), model="egohand"))
    return result


def hands_layer_proposals(table: pa.Table, cams: Sequence[CameraInfo], times_ns: Int64[ndarray, "g"]) -> ProposalsByTime:
    """``hands_layer`` proposals: confident KeyNet keypoints of the RoboCap ``hands`` layer nearest each grid time.

    Args:
        table: Catalog reader output on ``video_time`` with the KeyNet entities' ``Points2D`` and confidence columns.
        cams: Cameras of the segment.
        times_ns: The grid, ns.
    """
    result: ProposalsByTime = {int(t): [] for t in times_ns.tolist()}
    times: Int64[ndarray, "r"] = table[TIMELINE].cast(pa.int64()).to_numpy() if table.num_rows else np.zeros(0, np.int64)
    for cam in cams:
        for side in HAND_SIDES:
            path: str = keynet_path(cam.cam, side)
            points: list = _rows(table, f"{path}:Points2D:positions")
            confidences: list = _rows(table, f"{path}:simplecv.KeypointConfidence2D:average_confidence")
            present: list[int] = [row for row, value in enumerate(points) if value and confidences[row] and confidences[row][0] >= KEYNET_MIN_CONFIDENCE]
            if not present:
                continue
            order: list[int] = sorted(present, key=lambda row: int(times[row]))
            rows_at: Int64[ndarray, "g"] = nearest_indices(times[order].astype(np.int64), times_ns, FRAMESET_TOLERANCE_NS)
            for column, hit in enumerate(rows_at.tolist()):
                if hit < 0:
                    continue
                row: int = order[hit]
                xy: Float64[ndarray, "k 2"] = np.asarray(points[row], dtype=np.float64).reshape(-1, 2)
                low: Float64[ndarray, "2"] = xy.min(axis=0)
                high: Float64[ndarray, "2"] = xy.max(axis=0)
                margin: float = KEYNET_MARGIN * float((high - low).max())
                box: Box | None = clipped_box(low[0] - margin, low[1] - margin, high[0] + margin, high[1] + margin, cam)
                if box is not None:
                    score: float = round(min(max(float(confidences[row][0]), 0.0), 1.0), SCORE_DECIMALS)
                    result[int(times_ns[column])].append(Proposal(cam=cam.cam, side=side, box=box, score=score, model="hands_layer"))
    return result


def read_proposals(dataset: DatasetEntry, segment_id: str, cams: Sequence[CameraInfo], times_ns: Sequence[int]) -> ProposalsByTime:
    """Every proposal of a segment (``egohand`` from ``hand_proposals``, ``hands_layer`` from ``hands``), one catalog round trip.

    Load it once per segment and keep the result: ``result[t_ns]`` lists the proposals of every camera at grid
    time ``t_ns``. A segment without one of the layers simply has no proposals from that model.

    Args:
        dataset: The labeling dataset (``robocap-labeling``).
        segment_id: Catalog segment id.
        cams: Cameras of the segment (``FrameSource.cams``).
        times_ns: The 5 fps grid (``FrameSource.times_ns``).
    """
    grid: Int64[ndarray, "g"] = np.asarray(times_ns, dtype=np.int64)
    entities: list[str] = [proposal_path(cam.cam, "egohand") for cam in cams] + [keynet_path(cam.cam, side) for cam in cams for side in HAND_SIDES]
    table: pa.Table = dataset.filter_segments(segment_id).filter_contents(entities).reader(index=TIMELINE).to_arrow_table()
    egohand: ProposalsByTime = egohand_proposals(table, cams, grid)
    hands: ProposalsByTime = hands_layer_proposals(table, cams, grid)
    return {t: egohand[t] + hands[t] for t in egohand}
