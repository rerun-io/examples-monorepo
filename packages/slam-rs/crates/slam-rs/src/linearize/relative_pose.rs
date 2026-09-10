//! Relative-pose preparation shared by the window and frame linearizers.

use nalgebra::Matrix6;

use crate::ba_base::compute_rel_pose;
use crate::lie::{LieScalar, Se3};
use crate::types::PoseStateWithLin;

/// Jacobians at the linearization point, then only the transform at the current
/// state when either end is frozen (`linearization_abs_qr.cpp:219-232`).
///
/// Callers handle identical image pairs before resolving states: their transform
/// is identity and their Jacobians are zero, with no `compute_rel_pose` call.
/// Pair caches and fixed-frame Jacobian masking also belong to the callers.
/// The frame update requests only the target Jacobian.
pub(crate) fn linearize_relative_pose<S: LieScalar>(
    state_h: &PoseStateWithLin<S>,
    state_t: &PoseStateWithLin<S>,
    t_i_c_h: &Se3<S>,
    t_i_c_t: &Se3<S>,
    d_rel_d_h: Option<&mut Matrix6<S>>,
    d_rel_d_t: Option<&mut Matrix6<S>>,
) -> Se3<S> {
    let mut t_t_h: Se3<S> = compute_rel_pose(
        state_h.pose_lin(),
        t_i_c_h,
        state_t.pose_lin(),
        t_i_c_t,
        d_rel_d_h,
        d_rel_d_t,
    );
    if state_h.is_linearized() || state_t.is_linearized() {
        t_t_h = compute_rel_pose(state_h.pose(), t_i_c_h, state_t.pose(), t_i_c_t, None, None);
    }
    t_t_h
}
