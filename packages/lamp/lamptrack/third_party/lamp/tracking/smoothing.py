# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Temporal smoothing for lifted SMPL skeletons."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float32
from numpy import ndarray

from lamptrack.third_party.lamp.core.se3 import slerp_se3_batched, slerp_so3_batched
from lamptrack.third_party.lamp.core.types import Person, PersonState, Skeleton


def fuse_or_store_batched(
    person: Person,
    skeletons_with_ts: list[tuple[int, Skeleton]],
    *,
    shape_override: Float32[ndarray, "betas"] | None = None,
) -> None:
    """Fuse lifted snippet poses into the track timeline."""
    if shape_override is not None:
        shape_override = shape_override.astype(np.float32, copy=False)
        for _ts, skel in skeletons_with_ts:
            skel.shape = shape_override.copy()

    fuse_entries: list[tuple[PersonState, Skeleton, Skeleton, float]] = []
    for ts, new_skel in skeletons_with_ts:
        state = person.ts_to_states.get(ts)
        if state is None:
            person.ts_to_states[ts] = PersonState(
                detection2ds=[],
                skeleton=new_skel,
            )
            continue
        old_skel = state.skeleton
        if old_skel is None:
            state.skeleton = new_skel
            continue
        alpha = 1.0 / (state.num_fuses + 1.0)
        fuse_entries.append((state, old_skel, new_skel, alpha))

    if not fuse_entries:
        return

    alphas = np.array([e[3] for e in fuse_entries], dtype=np.float32)
    kfs = (1.0 / alphas - 1.0).astype(np.float32)
    invds = 1.0 / (kfs + 1.0)

    old_kps = np.stack([e[1].kp_world for e in fuse_entries], axis=0)
    new_kps = np.stack([e[2].kp_world for e in fuse_entries], axis=0)
    fused_kps = (kfs[:, None, None] * old_kps + new_kps) * invds[:, None, None]

    shape_mask = np.zeros(len(fuse_entries), dtype=bool)
    fused_shapes: Float32[ndarray, "n betas"] | None = None
    if shape_override is None:
        shape_mask = np.array(
            [
                e[1].shape.size > 0 and e[2].shape.size == e[1].shape.size
                for e in fuse_entries
            ],
            dtype=bool,
        )
    if shape_override is None and shape_mask.any():
        sub_idx = [int(i) for i in np.where(shape_mask)[0]]
        old_sh = np.stack([fuse_entries[i][1].shape for i in sub_idx], axis=0).astype(
            np.float32, copy=False
        )
        new_sh = np.stack([fuse_entries[i][2].shape for i in sub_idx], axis=0).astype(
            np.float32, copy=False
        )
        fused_shapes = (kfs[sub_idx, None] * old_sh + new_sh) * invds[sub_idx, None]

    old_Ts = np.stack([e[1].T_world_pelvis for e in fuse_entries], axis=0)
    new_Ts = np.stack([e[2].T_world_pelvis for e in fuse_entries], axis=0)
    fused_Ts = slerp_se3_batched(old_Ts, new_Ts, alphas)

    jr_mask = np.array(
        [
            e[1].joints_rot_mat.size > 0
            and e[2].joints_rot_mat.shape == e[1].joints_rot_mat.shape
            for e in fuse_entries
        ],
        dtype=bool,
    )
    fused_joints_rot: Float32[ndarray, "n joints 3 3"] | None = None
    if jr_mask.any():
        sub_idx = [int(i) for i in np.where(jr_mask)[0]]
        old_jr = np.stack(
            [fuse_entries[i][1].joints_rot_mat for i in sub_idx], axis=0
        ).astype(np.float32, copy=False)
        new_jr = np.stack(
            [fuse_entries[i][2].joints_rot_mat for i in sub_idx], axis=0
        ).astype(np.float32, copy=False)
        n_jr, j_per, _, _ = old_jr.shape
        alphas_jr = np.repeat(alphas[sub_idx], j_per)
        fused_flat = slerp_so3_batched(
            old_jr.reshape(n_jr * j_per, 3, 3),
            new_jr.reshape(n_jr * j_per, 3, 3),
            alphas_jr,
        )
        fused_joints_rot = fused_flat.reshape(n_jr, j_per, 3, 3)
        if j_per == 24:
            fused_Ts[np.array(sub_idx, dtype=np.int64), :3, :3] = fused_joints_rot[:, 0]

    shape_row = 0
    jr_row = 0
    for i, (state, old_skel, _new_skel, _alpha) in enumerate(fuse_entries):
        old_skel.kp_world = fused_kps[i]
        old_skel.T_world_pelvis = fused_Ts[i]
        if shape_override is not None:
            old_skel.shape = shape_override.copy()
        elif fused_shapes is not None and shape_mask[i]:
            old_skel.shape = fused_shapes[shape_row]
            shape_row += 1
        if fused_joints_rot is not None and jr_mask[i]:
            old_skel.joints_rot_mat = fused_joints_rot[jr_row]
            jr_row += 1
        state.num_fuses += 1
