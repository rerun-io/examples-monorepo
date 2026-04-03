//! Gaussian projection helpers.
//!
//! Provides shared utilities for the Gaussian splatting pipeline.

use glam::Quat;

/// Normalize a quaternion, returning the identity quaternion if the input is
/// near-zero (degenerate).
pub fn normalize_quat_or_identity(quat: Quat) -> Quat {
    if quat.length_squared() > 1e-12 {
        quat.normalize()
    } else {
        Quat::IDENTITY
    }
}
