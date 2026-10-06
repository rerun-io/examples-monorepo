//! Shared CPU/device validation and limits.
use super::TrackerError;

/// Upper bound for a valid increment.
///
/// Exported so device kernels can use it rather than re-declaring the
/// number: the two lanes have no compiler coupling otherwise.
pub const MAX_INCREMENT_INFINITY_NORM: f32 = 1e6;

/// `const int filter_margin = 2`.
///
/// Exported for the same reason as [`MAX_INCREMENT_INFINITY_NORM`].
pub const FILTER_MARGIN: f32 = 2.0;

/// Maximum tracker capacity, bounding caller-controlled preallocation.
/// A million keypoints already implies roughly 7 GB across two patch sets with
/// four Pattern51 levels, far above the default grid's needs. Rejecting larger
/// requests prevents capacity arithmetic overflow at the public boundary.
pub const MAX_CAPACITY: usize = 1 << 20;

/// Maximum pyramid level count, limiting the capacity multiplier.
/// Each level halves image dimensions. Unbounded counts could request storage
/// large enough to abort allocation instead of returning an input error.
pub const MAX_LEVELS: usize = 24;

/// The four preconditions of [`super::PatchTrackerPlan::track`], checked before any
/// mutation.
///
/// Both lanes call this rather than each spelling the four out: the seam exists
/// to keep them interchangeable, and a fifth check added to one lane and not the
/// other would be invisible.
///
/// # Errors
///
/// [`TrackerError::LengthMismatch`] when the patch set and the guesses disagree,
/// [`TrackerError::CapacityExceeded`] above the tracker's capacity, and
/// [`TrackerError::LevelMismatch`] when the patch set or either pyramid is
/// shallower than the tracker. The patch set is built by the caller, so its
/// depth is an input like any other: a one-level `PatchSoA` in a two-level
/// tracker used to index past the end of `valid`.
pub fn check_track_inputs(
    count: usize,
    patches_len: usize,
    patch_levels: usize,
    prev_levels: usize,
    next_levels: usize,
    capacity: usize,
    num_levels: usize,
) -> Result<(), TrackerError> {
    if count != patches_len {
        return Err(TrackerError::LengthMismatch {
            first_name: "patches",
            first: patches_len,
            second_name: "transforms",
            second: count,
        });
    }
    if count > capacity {
        return Err(TrackerError::CapacityExceeded {
            offered: count,
            capacity,
        });
    }
    if patch_levels < num_levels {
        return Err(TrackerError::LevelMismatch {
            what: "the patch set",
            expected: num_levels,
            actual: patch_levels,
        });
    }
    for (what, levels) in [
        ("the previous pyramid", prev_levels),
        ("the next pyramid", next_levels),
    ] {
        if levels < num_levels {
            return Err(TrackerError::LevelMismatch {
                what,
                expected: num_levels,
                actual: levels,
            });
        }
    }
    Ok(())
}

/// The element counts a patch set of this shape needs: `(flags, taps)`.
///
/// `flags` is one entry per (level, patch) and `taps` is `flags * P::SIZE`; each
/// constructor forms its own last product from them, which is the part the two
/// lanes do differently (the CPU one wants three Jacobian arrays, the GPU one
/// stores point coordinates and samples templates in registers). The ceilings and the
/// `checked_mul` ladder are the part that must not drift.
///
/// # Errors
///
/// [`TrackerError::CapacityTooLarge`] above [`MAX_CAPACITY`],
/// [`TrackerError::TooManyLevels`] above [`MAX_LEVELS`], and
/// [`TrackerError::BufferShapeOverflow`] when a count does not fit a `usize`.
pub fn checked_patch_shape(
    capacity: usize,
    num_levels: usize,
    taps_per_patch: usize,
) -> Result<(usize, usize), TrackerError> {
    if num_levels == 0 {
        return Err(TrackerError::InvalidParameter(
            "num_levels must be positive",
        ));
    }
    if capacity > MAX_CAPACITY {
        return Err(TrackerError::CapacityTooLarge {
            capacity,
            ceiling: MAX_CAPACITY,
        });
    }
    if num_levels > MAX_LEVELS {
        return Err(TrackerError::TooManyLevels {
            num_levels,
            ceiling: MAX_LEVELS,
        });
    }
    let overflow = || TrackerError::BufferShapeOverflow {
        capacity,
        num_levels,
        taps: taps_per_patch,
    };
    let flags: usize = num_levels.checked_mul(capacity).ok_or_else(overflow)?;
    let taps: usize = flags.checked_mul(taps_per_patch).ok_or_else(overflow)?;
    Ok((flags, taps))
}

/// The three preconditions of [`super::PatchSoA::build`], checked before any
/// mutation, on both lanes for the same reason as [`check_track_inputs`].
///
/// # Errors
///
/// [`TrackerError::CapacityExceeded`] when the positions do not fit,
/// [`TrackerError::LengthMismatch`] when the selection mask is shorter than the
/// positions, and [`TrackerError::LevelMismatch`] when the pyramid is shallower
/// than the patch set.
pub fn check_patch_inputs(
    count: usize,
    capacity: usize,
    selected: Option<&[bool]>,
    pyramid_levels: usize,
    num_levels: usize,
) -> Result<(), TrackerError> {
    if count > capacity {
        return Err(TrackerError::CapacityExceeded {
            offered: count,
            capacity,
        });
    }
    if let Some(flags) = selected.filter(|flags| flags.len() < count) {
        return Err(TrackerError::LengthMismatch {
            first_name: "positions",
            first: count,
            second_name: "selection flags",
            second: flags.len(),
        });
    }
    if pyramid_levels < num_levels {
        return Err(TrackerError::LevelMismatch {
            what: "the pyramid",
            expected: num_levels,
            actual: pyramid_levels,
        });
    }
    Ok(())
}

/// Check iterative tracking parameters once at backend construction.
///
/// # Arguments
/// * `iterations` - Positive per-level iteration budget.
/// * `recovery_distance2` - Finite, nonnegative squared forward/backward distance.
///
/// # Errors
/// Returns [`TrackerError::InvalidParameter`] for unsupported values.
pub fn validate_tracking_parameters(
    iterations: usize,
    recovery_distance2: f32,
) -> Result<(), TrackerError> {
    if iterations == 0 {
        return Err(TrackerError::InvalidParameter(
            "max_iterations must be positive",
        ));
    }
    if !recovery_distance2.is_finite() || recovery_distance2 < 0.0 {
        return Err(TrackerError::InvalidParameter(
            "max_recovered_dist2 must be finite and nonnegative",
        ));
    }
    Ok(())
}

/// Check an optional convergence threshold before storing it.
///
/// # Arguments
/// * `threshold` - Positive finite pixel threshold, or `None` to disable early exit.
///
/// # Errors
/// Returns [`TrackerError::InvalidExitStep`] for invalid thresholds.
pub fn validate_exit_step(threshold: Option<f32>) -> Result<(), TrackerError> {
    if threshold.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(TrackerError::InvalidExitStep);
    }
    Ok(())
}
