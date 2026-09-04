# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-view people tracker with Hungarian assignment."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from lamptrack.third_party.lamp.core.types import box_iou_xyxy, Detection2D, Person, PersonState, Skeleton
from lamptrack.third_party.lamp.io.sensor_io import PerCameraCalibration
from lamptrack.third_party.lamp.models.lifter import is_outlier_pose, SnippetData
from lamptrack.third_party.lamp.tracking.smoothing import fuse_or_store_batched
from lamptrack.third_party.lamp.tracking.snippets import build_snippets_for_lifting
from lamptrack.third_party.lamp.tracking.tracking_utils import (
    hungarian_assign,
    SensorRecord,
    transform_detection_2d,
)
from scipy.optimize import (
    linear_sum_assignment,  # pyright: ignore[reportUnknownVariableType]
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LampTrackerSettings:
    """Hungarian thresholds + track-lifecycle timeouts."""

    box_proj_iou_thres: float = 0.2
    inactive_duration_s: float = 0.5
    remove_duration_s: float = 1.0
    min_track_frame_ratio: float = 0.7
    # When False (default), lifted tracks are kept once inactive (the viewer can
    # show them faded); only never-lifted tracks are pruned after the timeout.
    # When True, inactive tracks are removed as soon as they go inactive.
    remove_inactive: bool = False
    shape_lock_after_updates: int = 20


@dataclass(slots=True)
class _TrackHandle:
    """Internal: a snapshot of a track at the start of `_track_one_cam`."""

    person_id: int
    last_obs_ts: int
    last_cam_idx: int


class LampTracker:
    """Holds the persistent set of `Person` tracks across framesets."""

    def __init__(
        self,
        num_cameras: int,
        settings: LampTrackerSettings | None = None,
        non_track_creating_cam_indices: set[int] | None = None,
    ) -> None:
        self._num_cameras: int = int(num_cameras)
        self._settings: LampTrackerSettings = settings or LampTrackerSettings()
        self._non_track_creating_cam_indices: set[int] = (
            set(non_track_creating_cam_indices)
            if non_track_creating_cam_indices
            else set()
        )
        self._people: dict[int, Person] = {}
        self._sensor_data_per_cam: dict[int, dict[int, SensorRecord]] = {}

        self._sensor_data_keep_count: int = 200
        self._current_id: int = 0

    @property
    def people(self) -> dict[int, Person]:
        return self._people

    @property
    def num_cameras(self) -> int:
        return self._num_cameras

    def reset(self) -> None:
        """Drop all tracks + sensor data; reset the id counter to factory."""
        self._people = {}
        self._sensor_data_per_cam = {}
        self._current_id = 0

    # Per-frame entry point

    def track_frameset(
        self,
        detections: dict[int, list[Detection2D]],
        T_world_cams: dict[int, np.ndarray],
        cam_models: dict[int, PerCameraCalibration | None],
        timestamp_ns: int,
    ) -> None:
        """Run per-camera tracking and lifecycle cleanup for one frameset."""
        for cam_idx, dets in detections.items():
            self._track_one_cam(
                dets,
                T_world_cams[cam_idx],
                cam_models.get(cam_idx),
                cam_idx,
                timestamp_ns,
            )
        self._lifecycle_cleanup(timestamp_ns)

    def _lifecycle_cleanup(self, ts_ns: int) -> None:
        """Inactivate stale tracks and remove them per `remove_inactive`.

        With `remove_inactive=False` (default), a lifted track is kept once it
        goes inactive so the viewer can still show it faded; only never-lifted
        tracks (spurious detections with no 3D output) are pruned after the
        timeout. With `remove_inactive=True`, inactive tracks are removed.
        """
        s = self._settings
        remove_ns = int(s.remove_duration_s * 1e9)
        inactive_ns = int(s.inactive_duration_s * 1e9)
        to_remove: list[int] = []
        for person_id, person in self._people.items():
            since_obs = ts_ns - person.last_obs_ts
            lifted = person.last_lifted_ts != -1
            if since_obs > remove_ns and (s.remove_inactive or not lifted):
                to_remove.append(person_id)
                continue
            if person.active and since_obs > inactive_ns:
                person.active = False
                person.inactive_ts = ts_ns
                if person.color is not None:
                    r, g, b, _ = person.color
                    person.color = (r, g, b, 0.3)
                if s.remove_inactive:
                    to_remove.append(person_id)
        for person_id in to_remove:
            self._people.pop(person_id, None)

        for per_ts in self._sensor_data_per_cam.values():
            if len(per_ts) <= self._sensor_data_keep_count:
                continue
            sorted_ts = sorted(per_ts.keys())
            for ts in sorted_ts[: -self._sensor_data_keep_count]:
                per_ts.pop(ts, None)

    def _track_one_cam(
        self,
        detections: list[Detection2D],
        T_world_cam: np.ndarray,
        cam_model: PerCameraCalibration | None,
        cam_idx: int,
        ts_ns: int,
    ) -> None:
        """Hungarian-match `detections` against active tracks for one camera."""
        per_ts = self._sensor_data_per_cam.setdefault(cam_idx, {})
        per_ts[ts_ns] = SensorRecord(
            T_world_cam=T_world_cam, cam_model=cam_model, cam_idx=cam_idx
        )

        # Snapshot active tracks so matrix order stays stable.
        active_handles: list[_TrackHandle] = []
        for person_id, person in self._people.items():
            if not person.active:
                continue
            if person.last_obs_ts < 0 or not person.ts_to_states:
                continue
            last_state = person.ts_to_states.get(person.last_obs_ts)
            if last_state is None or not last_state.detection2ds:
                continue
            last_cam_idx = last_state.detection2ds[-1].cam_idx
            active_handles.append(
                _TrackHandle(
                    person_id=person_id,
                    last_obs_ts=person.last_obs_ts,
                    last_cam_idx=last_cam_idx,
                )
            )

        cost = self._build_cost_matrix(
            detections, active_handles, cam_model, T_world_cam
        )
        assignments = hungarian_assign(cost)

        unmatched_threshold = 1.0 - self._settings.box_proj_iou_thres
        for assign in assignments:
            det = detections[assign.detection_idx]
            if assign.track_idx == -1 or assign.cost > unmatched_threshold:
                self._create_or_skip_new_track(det, cam_idx, ts_ns)
            else:
                handle = active_handles[assign.track_idx]
                self._append_to_track(handle.person_id, det, ts_ns)

        if not active_handles:
            for det in detections:
                self._create_or_skip_new_track(det, cam_idx, ts_ns)

    # Cost matrix + assignment helpers

    def _build_cost_matrix(
        self,
        detections: list[Detection2D],
        active_handles: list[_TrackHandle],
        new_cam: PerCameraCalibration | None,
        T_world_newcam: np.ndarray,
    ) -> np.ndarray:
        """Return `(num_dets, num_active_tracks)` cost matrix; 1 - IoU per cell."""
        n_dets = len(detections)
        n_tracks = len(active_handles)
        if n_dets == 0 or n_tracks == 0:
            return np.zeros((n_dets, n_tracks), dtype=np.float32)

        cost = np.ones((n_dets, n_tracks), dtype=np.float32)
        for t_idx, handle in enumerate(active_handles):
            person = self._people[handle.person_id]
            old_record = self._sensor_data_per_cam.get(handle.last_cam_idx, {}).get(
                handle.last_obs_ts
            )
            if old_record is None:
                continue
            transformed_box = transform_detection_2d(
                person,
                old_record.cam_model,
                new_cam,
                old_record.T_world_cam,
                T_world_newcam,
            )
            fallback_box: np.ndarray | None = None
            if (
                transformed_box is None
                and new_cam is None
                and old_record.cam_model is None
            ):
                # Degenerate-test fallback: synthetic tests often pass identity
                # SE3s and no camera models. Compare raw boxes so the Hungarian
                # path remains testable without real calibration.
                last_dets = person.ts_to_states[handle.last_obs_ts].detection2ds
                fallback_det = next(
                    (d for d in last_dets if d.cam_idx == handle.last_cam_idx),
                    last_dets[-1],
                )
                fallback_box = fallback_det.box_xyxy

            for d_idx, det in enumerate(detections):
                best_cost = 1.0
                if transformed_box is not None:
                    iou = box_iou_xyxy(det.box_xyxy, transformed_box)
                    best_cost = float(1.0 - iou)
                elif fallback_box is not None:
                    iou = box_iou_xyxy(det.box_xyxy, fallback_box)
                    best_cost = float(1.0 - iou)
                cost[d_idx, t_idx] = best_cost
        return cost

    # Track create / append

    def _create_or_skip_new_track(
        self, det: Detection2D, cam_idx: int, ts_ns: int
    ) -> None:
        """Create a new `Person` for `det`, unless this cam can't spawn tracks."""
        if cam_idx in self._non_track_creating_cam_indices:
            return
        self._current_id += 1
        new_id = self._current_id
        person = Person(id=new_id)
        det.track_id = new_id
        person.ts_to_states[ts_ns] = PersonState(detection2ds=[det])
        person.last_obs_ts = ts_ns
        self._people[new_id] = person

    def _append_to_track(self, person_id: int, det: Detection2D, ts_ns: int) -> None:
        """Attach `det` to an existing person, merging same-timestamp views."""
        person = self._people[person_id]
        det.track_id = person_id
        existing = person.ts_to_states.get(ts_ns)
        if existing is None:
            person.ts_to_states[ts_ns] = PersonState(detection2ds=[det])
        else:
            existing.detection2ds.append(det)
        person.last_obs_ts = ts_ns

    # Lifter handoff

    def get_snippets_for_lifting(
        self,
        snippet_length: int,
        T_gravityWorld_world: np.ndarray,
        kp_thres: float = 0.0,
        num_views: int | None = None,
    ) -> dict[int, SnippetData]:
        """Build per-track temporal snippets for the lifter."""
        return build_snippets_for_lifting(
            self._people,
            self._sensor_data_per_cam,
            snippet_length=snippet_length,
            T_gravity_world=T_gravityWorld_world,
            kp_thres=kp_thres,
            num_views=self._num_cameras if num_views is None else num_views,
            min_track_frame_ratio=self._settings.min_track_frame_ratio,
        )

    def attach_skeletons(
        self,
        person_id: int,
        skeletons_with_ts: list[tuple[int, Skeleton]],
        T_world_cams: dict[int, np.ndarray],
        T_gravityWorld_world: np.ndarray,
        *,
        min_pose_depth: float = 0.5,
        max_pose_depth: float = 5.0,
    ) -> None:
        """Attach a snippet's lifted skeletons with temporal fusion."""
        person = self._people.get(person_id)
        if person is None:
            logger.debug("attach_skeletons: unknown person id %d", person_id)
            return
        if not skeletons_with_ts:
            return

        # Gate the whole snippet on the latest lifted step.
        latest_ts, latest_skel = skeletons_with_ts[-1]
        if is_outlier_pose(
            latest_skel.kp_world,
            T_world_cams,
            T_gravityWorld_world,
            min_depth=min_pose_depth,
            max_depth=max_pose_depth,
        ):
            return

        shape_override = self._update_person_shape(person, skeletons_with_ts)
        fuse_or_store_batched(person, skeletons_with_ts, shape_override=shape_override)
        person.last_lifted_ts = latest_ts

    # Post-lift spatial merge

    def _update_person_shape(
        self, person: Person, skeletons_with_ts: list[tuple[int, Skeleton]]
    ) -> np.ndarray | None:
        """Update the person-level SMPL betas estimate."""
        shape_obs: np.ndarray | None = None
        for _ts, skel in reversed(skeletons_with_ts):
            if skel.shape.size > 0:
                shape_obs = skel.shape.astype(np.float32, copy=False)
                break
        if shape_obs is None:
            return None

        if person.shape_locked and person.shape_estimate.shape == shape_obs.shape:
            return person.shape_estimate.copy()

        if person.shape_estimate.shape != shape_obs.shape:
            person.shape_estimate = shape_obs.copy()
            person.shape_num_updates = 1
            person.shape_locked = False
        else:
            n = max(0, person.shape_num_updates)
            person.shape_estimate = (
                (float(n) * person.shape_estimate + shape_obs) / float(n + 1)
            ).astype(np.float32, copy=False)
            person.shape_num_updates = n + 1

        lock_after = self._settings.shape_lock_after_updates
        if lock_after > 0 and person.shape_num_updates >= lock_after:
            person.shape_locked = True
            self._stamp_shape_on_states(person)
        return person.shape_estimate.copy()

    @staticmethod
    def _stamp_shape_on_states(person: Person) -> None:
        if person.shape_estimate.size == 0:
            return
        for state in person.ts_to_states.values():
            if state.skeleton is not None:
                state.skeleton.shape = person.shape_estimate.copy()

    def merge_track(self, target_id: int, src_id: int) -> None:
        """Fold `src` into `target` and remove `src` from the tracker."""
        if target_id == src_id:
            return
        target = self._people.get(target_id)
        src = self._people.get(src_id)
        if target is None or src is None:
            logger.debug(
                "merge_track: missing person (target=%s, src=%s)", target_id, src_id
            )
            return

        # Scalar fields.
        target.last_lifted_ts = max(src.last_lifted_ts, target.last_lifted_ts)
        target.last_obs_ts = max(src.last_obs_ts, target.last_obs_ts)
        target.active = True
        if target.color is not None:
            r, g, b, _ = target.color
            target.color = (r, g, b, 1.0)
        target.inactive_ts = -1
        target.uncertainty = -1.0
        target.num_obs_3d = src.num_obs_3d
        self._merge_shape_estimates(target, src)
        # Merge temporal states.
        for ts, src_state in src.ts_to_states.items():
            tgt_state = target.ts_to_states.get(ts)
            if tgt_state is None or src_state.skeleton is not None:
                target.ts_to_states[ts] = src_state

        # Rewrite detection ids so the visualizer stops using the removed source id.
        for state in target.ts_to_states.values():
            for det in state.detection2ds:
                if det.track_id == src_id:
                    det.track_id = target_id
        if target.shape_locked:
            self._stamp_shape_on_states(target)

        # The merged target must still have a skeleton at its latest lift ts.
        if target.last_lifted_ts != -1:
            final_state = target.ts_to_states.get(target.last_lifted_ts)
            if final_state is None or final_state.skeleton is None:
                raise RuntimeError(
                    f"merge_track invariant failed: target {target_id} has no "
                    f"skeleton at last_lifted_ts={target.last_lifted_ts}"
                )

        self._people.pop(src_id, None)

    def _merge_shape_estimates(self, target: Person, src: Person) -> None:
        if target.shape_locked:
            return
        if src.shape_estimate.size == 0:
            return
        if (
            target.shape_estimate.size == 0
            or target.shape_estimate.shape != src.shape_estimate.shape
            or src.shape_locked
        ):
            target.shape_estimate = src.shape_estimate.copy()
            target.shape_num_updates = src.shape_num_updates
            target.shape_locked = src.shape_locked
            return

        target_count = max(0, target.shape_num_updates)
        src_count = max(0, src.shape_num_updates)
        total = target_count + src_count
        if total <= 0:
            return
        target.shape_estimate = (
            (target_count * target.shape_estimate + src_count * src.shape_estimate)
            / float(total)
        ).astype(np.float32, copy=False)
        target.shape_num_updates = total
        if total >= self._settings.shape_lock_after_updates > 0:
            target.shape_locked = True

    def merge_lifted_tracks(
        self,
        threshold_m: float = 0.3,
        current_ts_ns: int | None = None,
        prev_last_lifted_ts: dict[int, int] | None = None,
    ) -> int:
        """Pelvis-distance Hungarian merge between fresh + existing lifted tracks."""
        if current_ts_ns is None:
            return 0

        candidate_ids: list[int] = []
        new_ids: list[int] = []
        for person_id, person in self._people.items():
            if not person.active or person.last_lifted_ts == -1:
                continue
            last_state = person.ts_to_states.get(person.last_lifted_ts)
            if last_state is None or last_state.skeleton is None:
                continue
            if prev_last_lifted_ts is not None:
                # Strict partition: "new" iff this is the person's first lift.
                was_unlifted_before = prev_last_lifted_ts.get(person_id, -1) == -1
                lifted_now = person.last_lifted_ts == current_ts_ns
                is_new = was_unlifted_before and lifted_now
            else:
                # Fallback: any re-lift this frame counts as new.
                is_new = person.last_lifted_ts == current_ts_ns
            if is_new:
                new_ids.append(person_id)
            else:
                candidate_ids.append(person_id)

        if not candidate_ids or not new_ids:
            return 0

        # Pelvis-distance cost matrix (rows = new, cols = candidates).
        new_pelvises = np.stack(
            [
                self._people[pid]
                .ts_to_states[self._people[pid].last_lifted_ts]
                .skeleton.kp_world[0]  # pyright: ignore[reportOptionalMemberAccess]
                for pid in new_ids
            ],
            axis=0,
        )
        cand_pelvises = np.stack(
            [
                self._people[pid]
                .ts_to_states[self._people[pid].last_lifted_ts]
                .skeleton.kp_world[0]  # pyright: ignore[reportOptionalMemberAccess]
                for pid in candidate_ids
            ],
            axis=0,
        )
        # Pairwise L2: (N_new, N_cand)
        diffs = new_pelvises[:, None, :] - cand_pelvises[None, :, :]
        cost = np.linalg.norm(diffs, axis=-1).astype(np.float64, copy=False)

        rows_arr, cols_arr = linear_sum_assignment(cost)  # pyright: ignore[reportUnknownVariableType]
        rows: list[int] = [int(r) for r in rows_arr]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        cols: list[int] = [int(c) for c in cols_arr]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

        n_merges = 0
        for new_idx, cand_idx in zip(rows, cols, strict=True):
            if float(cost[new_idx, cand_idx]) > threshold_m:
                continue
            target_id = candidate_ids[cand_idx]
            src_id = new_ids[new_idx]
            logger.debug(
                "merge_lifted_tracks: %d -> %d (cost=%.3f)",
                src_id,
                target_id,
                float(cost[new_idx, cand_idx]),
            )
            self.merge_track(target_id=target_id, src_id=src_id)
            n_merges += 1
        return n_merges
