"""Track bookkeeping for SAM 3 detections propagated by a video mask tracker.

Pure functions with no model or GPU: matching re-detected instances to live tracks by mask IoU, starting tracks for
new instances, dropping duplicates and tracks that stayed empty too long.
"""

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice

import numpy as np
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray
from torch import Tensor


@dataclass(frozen=True, slots=True)
class RedetectPlan:
    """What one re-detect does to the tracks of one stream."""

    reanchor: tuple[tuple[int, int], ...]
    """``(track id, detection row)``: live tracks re-prompted with the box of the detection they match."""
    start: tuple[tuple[int, int], ...]
    """``(new track id, detection row)``: tracks started for unmatched detections."""
    drop: tuple[int, ...]
    """Live tracks that lost the match to another track on the same detection (duplicates)."""
    unmatched: tuple[int, ...]
    """Live tracks that overlap no detection (SAM 3 does not confirm them)."""
    next_id: int
    """First unused track id after this plan."""


def mask_iou(a: Bool[Tensor, "n h w"], b: Bool[Tensor, "m h w"]) -> Float32[Tensor, "n m"]:
    """Pairwise IoU of two mask sets (0 where both masks are empty)."""
    fa: Float32[Tensor, "n p"] = a.flatten(1).float()
    fb: Float32[Tensor, "m p"] = b.flatten(1).float()
    inter: Float32[Tensor, "n m"] = fa @ fb.T
    union: Float32[Tensor, "n m"] = fa.sum(1, keepdim=True) + fb.sum(1)[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=1.0), torch.zeros_like(inter))


def greedy_match(iou: Float32[ndarray, "n m"], threshold: float) -> list[tuple[int, int]]:
    """One-to-one ``(row, column)`` pairs, highest IoU first, each at or above ``threshold``."""
    pairs: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        row, col = (int(v) for v in np.unravel_index(flat, iou.shape))
        if iou[row, col] < threshold:
            break
        if row in used_rows or col in used_cols:
            continue
        pairs.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
    return pairs


def plan_redetect(
    live_ids: list[int], iou: Float32[ndarray, "n m"], detection_iou: Float32[ndarray, "m m"], *, threshold: float, next_id: int, max_tracks: int
) -> RedetectPlan:
    """Match the detections of one stream (columns) to its live tracks (rows).

    Matched tracks are re-anchored on their detection. A live track that is not matched but overlaps a detection at or
    above ``threshold`` follows the same person as the winner and is dropped. Unmatched detections start new tracks while
    the stream holds fewer than ``max_tracks``, except one that overlaps an already matched or started detection at or
    above ``threshold`` (``detection_iou``): the same person seen again, e.g. in a tile.
    """
    pairs: list[tuple[int, int]] = greedy_match(iou, threshold)
    matched_rows: set[int] = {row for row, _ in pairs}
    matched_cols: set[int] = {col for _, col in pairs}
    drop: list[int] = [live_ids[row] for row in range(len(live_ids)) if row not in matched_rows and iou.shape[1] and iou[row].max() >= threshold]
    unmatched: list[int] = [t for row, t in enumerate(live_ids) if row not in matched_rows and t not in drop]
    n_tracks: int = len(live_ids) - len(drop)
    start: list[tuple[int, int]] = []
    taken: list[int] = sorted(matched_cols)
    for col in range(iou.shape[1]):
        if col in matched_cols or n_tracks >= max_tracks or any(detection_iou[col, other] >= threshold for other in taken):
            continue
        taken.append(col)
        start.append((next_id, col))
        next_id += 1
        n_tracks += 1
    return RedetectPlan(
        reanchor=tuple((live_ids[row], col) for row, col in pairs), start=tuple(start), drop=tuple(drop), unmatched=tuple(unmatched), next_id=next_id
    )


def count_misses(misses: dict[int, int], nonempty: dict[int, bool], *, max_misses: int) -> tuple[dict[int, int], list[int]]:
    """Update each track's run of empty steps; a track empty for ``max_misses`` steps in a row is dropped.

    Returns:
        The new miss counts (dropped tracks removed) and the dropped track ids.
    """
    updated: dict[int, int] = {track: 0 if seen else misses.get(track, 0) + 1 for track, seen in nonempty.items()}
    dropped: list[int] = sorted(track for track, count in updated.items() if count >= max_misses)
    return {track: count for track, count in updated.items() if count < max_misses}, dropped


def with_lookahead[StepT](steps: Iterator[StepT], ahead: int) -> Iterator[tuple[StepT, StepT | None]]:
    """Each step with the step ``ahead`` later (None for the last ``ahead`` steps, and for all with ``ahead`` 0): what
    SAM 3's lookahead starts on. A window of ``ahead`` steps is held, so 0 holds none.

    Raises:
        ValueError: If ``ahead`` is negative.
    """
    if ahead < 0:
        raise ValueError(f"lookahead of {ahead} steps")
    if ahead == 0:
        yield from ((step, None) for step in steps)
        return
    window: deque[StepT] = deque(islice(steps, ahead))
    while window:
        step: StepT = window.popleft()
        window.extend(islice(steps, 1))
        yield step, window[-1] if len(window) == ahead else None
