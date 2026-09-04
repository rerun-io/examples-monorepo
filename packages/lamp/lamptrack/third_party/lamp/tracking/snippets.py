# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build lifter snippets from tracked people and camera records."""

from __future__ import annotations

import logging

import numpy as np
from lamptrack.third_party.lamp.core.types import Person, PersonState
from lamptrack.third_party.lamp.models.lifter import SnippetData
from lamptrack.third_party.lamp.tracking.tracking_utils import cam_params_16, SensorRecord

logger: logging.Logger = logging.getLogger(__name__)


def build_snippets_for_lifting(
    people: dict[int, Person],
    sensor_data_per_cam: dict[int, dict[int, SensorRecord]],
    *,
    snippet_length: int,
    T_gravity_world: np.ndarray,
    kp_thres: float,
    num_views: int,
    min_track_frame_ratio: float,
) -> dict[int, SnippetData]:
    view_cam_indices = sorted(sensor_data_per_cam.keys())[:num_views]
    if not view_cam_indices:
        return {}

    view_slot_by_cam = {cam_idx: slot for slot, cam_idx in enumerate(view_cam_indices)}
    anchor_cam = view_cam_indices[0]
    ts_to_sensor = sensor_data_per_cam.get(anchor_cam, {})
    if len(ts_to_sensor) < snippet_length:
        logger.debug(
            "Not enough sensor data (%d) for snippet length %d.",
            len(ts_to_sensor),
            snippet_length,
        )
        return {}

    snippet_timestamps = sorted(ts_to_sensor.keys())[-snippet_length:]
    T_gravity_world = np.asarray(T_gravity_world, dtype=np.float32)

    snippets: dict[int, SnippetData] = {}
    for person_id, person in people.items():
        if not person.active:
            continue
        snippet = _build_person_snippet(
            person=person,
            snippet_timestamps=snippet_timestamps,
            sensor_data_per_cam=sensor_data_per_cam,
            T_gravity_world=T_gravity_world,
            kp_thres=kp_thres,
            num_views=num_views,
            view_slot_by_cam=view_slot_by_cam,
            min_track_frame_ratio=min_track_frame_ratio,
        )
        if snippet is not None:
            snippets[person_id] = snippet
    return snippets


def _build_person_snippet(
    *,
    person: Person,
    snippet_timestamps: list[int],
    sensor_data_per_cam: dict[int, dict[int, SensorRecord]],
    T_gravity_world: np.ndarray,
    kp_thres: float,
    num_views: int,
    view_slot_by_cam: dict[int, int],
    min_track_frame_ratio: float,
) -> SnippetData | None:
    snippet_length = len(snippet_timestamps)
    kp2ds_per_view = [
        np.zeros((snippet_length, 17, 3), dtype=np.float32) for _ in range(num_views)
    ]
    Ts_gravity_cam_per_view = [
        np.zeros((snippet_length, 4, 4), dtype=np.float32) for _ in range(num_views)
    ]
    # Allocated lazily so each view's width follows its camera model: 16 for
    # fisheye624, 4 for pinhole. `get_cam_ray` routes on this width, so a pinhole
    # view must stay length-4 rather than being padded to 16.
    cam_params_per_view: list[np.ndarray | None] = [None] * num_views

    num_found_ts = 0
    last_found_idx = -1

    for t_idx, ts in enumerate(snippet_timestamps):
        state = person.ts_to_states.get(ts)
        if (
            state is not None
            and state.detection2ds
            and _fill_keypoints_at_timestamp(
                state=state,
                kp2ds_per_view=kp2ds_per_view,
                t_idx=t_idx,
                kp_thres=kp_thres,
                view_slot_by_cam=view_slot_by_cam,
            )
        ):
            num_found_ts += 1
            last_found_idx = t_idx

        for cam_idx, view_slot in view_slot_by_cam.items():
            rec = sensor_data_per_cam.get(cam_idx, {}).get(ts)
            if rec is None:
                continue
            T_gravity_cam = (T_gravity_world @ rec.T_world_cam).astype(
                np.float32, copy=False
            )
            Ts_gravity_cam_per_view[view_slot][t_idx] = T_gravity_cam
            if rec.cam_model is not None:
                vec = cam_params_16(rec.cam_model)
                view_params = cam_params_per_view[view_slot]
                if view_params is None:
                    view_params = np.zeros(
                        (snippet_length, vec.shape[0]), dtype=np.float32
                    )
                    cam_params_per_view[view_slot] = view_params
                view_params[t_idx] = vec

    min_count = max(1, int(min_track_frame_ratio * snippet_length))
    if num_found_ts < min_count or last_found_idx != snippet_length - 1:
        logger.debug(
            "Drop person %d: %d/%d obs, last_found_idx=%d.",
            person.id,
            num_found_ts,
            snippet_length,
            last_found_idx,
        )
        return None

    view_cam_indices: list[int | None] = [None] * num_views
    for cam_idx, view_slot in view_slot_by_cam.items():
        view_cam_indices[view_slot] = cam_idx

    # Views with no sensor records default to the zero fisheye vector, matching
    # the previous always-length-16 behavior.
    cam_params_filled = [
        view if view is not None else np.zeros((snippet_length, 16), dtype=np.float32)
        for view in cam_params_per_view
    ]

    return SnippetData(
        person_id=person.id,
        snippet_timestamps_ns=list(snippet_timestamps),
        view_cam_indices=view_cam_indices,
        kp2ds_per_view=kp2ds_per_view,
        Ts_gw_cam_per_view=Ts_gravity_cam_per_view,
        cam_params_per_view=cam_params_filled,
        T_gravityWorld_world=T_gravity_world,
    )


def _fill_keypoints_at_timestamp(
    *,
    state: PersonState,
    kp2ds_per_view: list[np.ndarray],
    t_idx: int,
    kp_thres: float,
    view_slot_by_cam: dict[int, int],
) -> bool:
    any_obs = False
    for det in state.detection2ds:
        view_slot = view_slot_by_cam.get(det.cam_idx)
        if view_slot is None or not det.has_keypoints:
            continue
        keypoints = det.keypoints[:17]
        scores = keypoints[:, 2]
        keep_mask = scores >= kp_thres
        if not bool(keep_mask.any()):
            continue
        kp_slot = kp2ds_per_view[view_slot][t_idx]
        kp_slot_view = kp_slot[: keypoints.shape[0]]
        kp_slot_view[keep_mask, 0] = keypoints[keep_mask, 0]
        kp_slot_view[keep_mask, 1] = keypoints[keep_mask, 1]
        kp_slot_view[keep_mask, 2] = 1.0
        any_obs = True
    return any_obs
