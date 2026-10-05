//! Reference pose lookup for dump replay parity.

use nalgebra::Isometry3;

use crate::frame::isometry_from_matrix;

/// Looks up the dump's reference poses by time (`--slam reference`), undoing a replay loop's time shift.
pub struct ReferencePoses {
    poses: Vec<(i64, [f64; 16])>,
    first_t_ns: i64,
    span_ns: Option<i64>,
}

impl ReferencePoses {
    /// `poses` from `read_reference_poses`; `first_t_ns` and `span_ns` from the replay (loop shift = k * span).
    pub fn new(mut poses: Vec<(i64, [f64; 16])>, first_t_ns: i64, span_ns: Option<i64>) -> Self {
        // Framesets without a reference pose carry NaN rows in the dump (s66-full: 7 of 1362): drop them.
        poses.retain(|(_, matrix)| matrix.iter().all(|v| v.is_finite()));
        poses.sort_by_key(|(t, _)| *t);
        Self { poses, first_t_ns, span_ns }
    }

    /// The reference pose within 2 ms of `t_ns`, if any.
    pub fn at(&self, t_ns: i64) -> Option<Isometry3<f64>> {
        let t = match self.span_ns {
            Some(span) if span > 0 && t_ns >= self.first_t_ns => self.first_t_ns + (t_ns - self.first_t_ns) % span,
            _ => t_ns,
        };
        let at = self.poses.partition_point(|(pose_t, _)| *pose_t < t);
        [at.checked_sub(1), Some(at)]
            .into_iter()
            .flatten()
            .filter_map(|i| self.poses.get(i))
            .filter(|(pose_t, _)| (pose_t - t).abs() <= 2_000_000)
            .min_by_key(|(pose_t, _)| (pose_t - t).abs())
            .and_then(|(_, matrix)| isometry_from_matrix(matrix))
    }
}
