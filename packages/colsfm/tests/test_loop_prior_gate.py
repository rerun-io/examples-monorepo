"""The loop prior-consistency gate: a candidate is kept only when its measurement agrees with the prior trajectory."""

from __future__ import annotations

import numpy as np
import pycolmap

from colsfm.loop_closure import LoopCandidate, LoopClosureConfig, agrees_with_prior
from colsfm.loop_pose import RigPoseEstimate
from colsfm.loop_shortlist import RetrievalHit


def _rigid(yaw_deg: float, x: float) -> pycolmap.Rigid3d:
    yaw: float = np.radians(yaw_deg)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])), np.array([x, 0.0, 0.0]))


def _candidate(measured: pycolmap.Rigid3d) -> LoopCandidate:
    hit = RetrievalHit(query_keyframe_id=0, candidate_keyframe_id=1, source_rig_id=10, target_rig_id=20, score=1.0, delta_seconds=80.0)
    estimate = RigPoseEstimate(
        source_T_target=measured, num_inliers=500, num_observations=600, num_landmarks=900, num_cameras=4, direction_disagreement_deg=0.0
    )
    return LoopCandidate(hit=hit, estimate=estimate)


# The prior says the target rig sits 0.5 m ahead, turned around: a revisit walked the other way.
PRIOR: dict[int, pycolmap.Rigid3d] = {10: _rigid(0.0, 0.0), 20: _rigid(178.0, 0.5)}
GATE: LoopClosureConfig = LoopClosureConfig(max_prior_translation_m=1.5, max_prior_rotation_deg=10.0)


def test_a_turned_around_revisit_that_agrees_with_the_prior_is_kept() -> None:
    assert agrees_with_prior(_candidate(_rigid(175.0, 0.9)), PRIOR, GATE)


def test_an_aliased_match_half_a_turn_off_the_prior_is_dropped() -> None:
    assert not agrees_with_prior(_candidate(_rigid(-2.0, 0.5)), PRIOR, GATE)


def test_a_match_far_from_the_prior_position_is_dropped() -> None:
    assert not agrees_with_prior(_candidate(_rigid(178.0, 6.0)), PRIOR, GATE)


def test_an_unset_half_does_not_gate() -> None:
    assert agrees_with_prior(_candidate(_rigid(178.0, 6.0)), PRIOR, LoopClosureConfig(max_prior_rotation_deg=10.0))
