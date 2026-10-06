//! Bearing DLT with a unit spatial norm and signed inverse distance.
use super::core;
use kornia_staging_algebra::{lie::RigidTransform, Scalar};
use nalgebra::{Vector3, Vector4};

/// Triangulate unit bearings using the pose from frame 1 to frame 0.
/// Returns `None` for parallel, non-finite, zero or unconverged geometry.
#[inline]
pub fn triangulate_bearing<S: Scalar>(
    f0: &Vector3<S>,
    f1: &Vector3<S>,
    pose: &RigidTransform<S>,
) -> Option<Vector4<S>> {
    let f1_in_0 = pose.rotation * *f1;
    core::triangulate(f0, f1, &f1_in_0, pose.inverse().matrix3x4())
}
