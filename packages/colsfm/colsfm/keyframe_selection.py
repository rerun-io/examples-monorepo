"""cuSFM's keyframe selection, as `feature_extractor_main` performs it.

Two passes run before any feature is extracted (feature_extractor_main.md §4 and §5):

1. **Sample synchronisation.** `synced_sample_id` is recomputed from scratch:
   frames are sorted by timestamp, grouped while they stay within
   `sample_sync_threshold_microseconds` of the group's first frame, and the
   groups are numbered **1..N in time order**. Galileo's input `318..346`
   becomes `1..29` with the same partition.
2. **Selection.** A greedy walk over the samples in time order keeps the first
   sample, then keeps a sample when its **rig** pose has moved at least
   `min_inter_frame_distance` metres **or** rotated at least
   `min_inter_frame_rotation_degrees` from the last **kept** sample. The two
   gates are OR-combined, the distance is straight-line displacement rather than
   path length, and every camera of a kept sample survives.

Kept samples keep the number they were given in pass 1, so a selected file's
`synced_sample_id`s are sparse (Galileo at 0.5 m / 5 deg: `1, 8, 15, 22, 29`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pycolmap

from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.geometry import relative_rotation_degrees

DEFAULT_SAMPLE_SYNC_THRESHOLD_MICROSECONDS: Final[int] = 100
"""The isaac demo's `--sample_sync_threshold_microseconds` (the gflag default is 50)."""


@dataclass(frozen=True, slots=True)
class SyncedSamples:
    """The rig grouping `feature_extractor_main` recomputes before selecting."""

    synced_sample_id_by_keyframe_id: dict[int, int]
    """Dense 1-based rig frame id per keyframe id, in time order."""
    keyframe_ids_by_synced_sample_id: dict[int, tuple[int, ...]]
    """Member keyframe ids per rig frame id, ascending."""


@dataclass(frozen=True, slots=True)
class KeyframeSelection:
    """The outcome of the selection walk."""

    kept_keyframe_ids: tuple[int, ...]
    """Surviving keyframe ids, in the input file's order."""
    synced_sample_id_by_keyframe_id: dict[int, int]
    """Renumbered rig frame id for every keyframe of the collection, kept or not."""

    @property
    def kept_synced_sample_ids(self) -> tuple[int, ...]:
        """Surviving rig frame ids, ascending.

        Sparse within the dense 1..N numbering: Galileo at 0.5 m / 5 deg keeps
        `1, 8, 15, 22, 29`.

        Returns:
            The rig frame id of every kept keyframe, deduplicated and ascending.
        """
        return tuple(sorted({self.synced_sample_id_by_keyframe_id[keyframe_id] for keyframe_id in self.kept_keyframe_ids}))


def synchronise_samples(
    frames_meta: FramesMeta,
    sample_sync_threshold_microseconds: int = DEFAULT_SAMPLE_SYNC_THRESHOLD_MICROSECONDS,
) -> SyncedSamples:
    """Regroup keyframes into rig samples by timestamp and renumber them 1..N.

    A frame joins the open group while its timestamp is within
    `sample_sync_threshold_microseconds` of the group's **first** frame; otherwise
    it opens a new group. Anchoring on the first frame rather than the previous
    one is what stops a whole run from chaining into a single sample once the
    threshold exceeds the inter-frame gap.

    Args:
        frames_meta: The collection to regroup.
        sample_sync_threshold_microseconds: Grouping window in microseconds.

    Returns:
        The dense 1-based grouping.
    """
    ordered: list[KeyframeMeta] = sorted(
        frames_meta.keyframes, key=lambda keyframe: (keyframe.timestamp_microseconds, keyframe.keyframe_id)
    )
    synced_sample_id_by_keyframe_id: dict[int, int] = {}
    members: dict[int, list[int]] = {}
    current_id: int = 0
    group_start_microseconds: int = 0
    for keyframe in ordered:
        if current_id == 0 or keyframe.timestamp_microseconds - group_start_microseconds > sample_sync_threshold_microseconds:
            current_id += 1
            group_start_microseconds = keyframe.timestamp_microseconds
            members[current_id] = []
        synced_sample_id_by_keyframe_id[keyframe.keyframe_id] = current_id
        members[current_id].append(keyframe.keyframe_id)
    return SyncedSamples(
        synced_sample_id_by_keyframe_id=synced_sample_id_by_keyframe_id,
        keyframe_ids_by_synced_sample_id={sample_id: tuple(sorted(ids)) for sample_id, ids in members.items()},
    )


def select_keyframes(
    frames_meta: FramesMeta,
    *,
    min_inter_frame_distance_m: float,
    min_inter_frame_rotation_degrees: float,
    sample_sync_threshold_microseconds: int = DEFAULT_SAMPLE_SYNC_THRESHOLD_MICROSECONDS,
) -> KeyframeSelection:
    """Run cuSFM's keyframe selection over a metadata collection.

    Args:
        frames_meta: The input collection, with poses and rig extrinsics.
        min_inter_frame_distance_m: Displacement gate in metres; 0.0 keeps every sample.
        min_inter_frame_rotation_degrees: Rotation gate in degrees; 0.0 keeps every sample.
        sample_sync_threshold_microseconds: Rig grouping window in microseconds.

    Returns:
        The kept keyframe ids and the dense rig renumbering.
    """
    samples: SyncedSamples = synchronise_samples(frames_meta, sample_sync_threshold_microseconds)
    keyframes_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    kept_sample_ids: list[int] = []
    last_kept_pose: pycolmap.Rigid3d | None = None
    for synced_sample_id in sorted(samples.keyframe_ids_by_synced_sample_id):
        reference_id: int = samples.keyframe_ids_by_synced_sample_id[synced_sample_id][0]
        world_T_vehicle: pycolmap.Rigid3d = frames_meta.world_T_vehicle(keyframes_by_id[reference_id])
        if last_kept_pose is None:
            kept_sample_ids.append(synced_sample_id)
            last_kept_pose = world_T_vehicle
            continue
        displacement_m: float = float(np.linalg.norm(world_T_vehicle.translation - last_kept_pose.translation))
        rotation_degrees: float = relative_rotation_degrees(last_kept_pose, world_T_vehicle)
        if displacement_m >= min_inter_frame_distance_m or rotation_degrees >= min_inter_frame_rotation_degrees:
            kept_sample_ids.append(synced_sample_id)
            last_kept_pose = world_T_vehicle
    kept: set[int] = set(kept_sample_ids)
    kept_keyframe_ids: tuple[int, ...] = tuple(
        keyframe.keyframe_id
        for keyframe in frames_meta.keyframes
        if samples.synced_sample_id_by_keyframe_id[keyframe.keyframe_id] in kept
    )
    return KeyframeSelection(
        kept_keyframe_ids=kept_keyframe_ids,
        synced_sample_id_by_keyframe_id=samples.synced_sample_id_by_keyframe_id,
    )


def apply_selection(frames_meta: FramesMeta, selection: KeyframeSelection) -> FramesMeta:
    """Produce the collection `feature_extractor_main` writes to its output directory.

    Args:
        frames_meta: The input collection.
        selection: The result of `select_keyframes`.

    Returns:
        A collection holding only the kept keyframes, with `synced_sample_id`
        rewritten to the dense 1-based numbering.
    """
    return frames_meta.filtered(selection.kept_keyframe_ids, selection.synced_sample_id_by_keyframe_id)
