//! The square-root linearization: landmark blocks, the IMU block, and the
//! absolute-pose QR driver that turns a window into `H` and `b`.
//!
//! Ported from `include/basalt/linearization/` and
//! `src/linearization/linearization_abs_qr.cpp`. Only basalt's `ABS_QR` variant
//! is here (decision D13): of its three linearizations and three marginalization
//! variants, `ABS_QR` plus square-root marginalization is what every shipped
//! config selects, and `ABS_SC` / `REL_SC` are out of scope.
//!
//! The shape of one optimizer iteration, from `sqrt_keypoint_vio.cpp:1289-1470`:
//!
//! ```text
//! error = lqr.linearize_problem()      // :1297   residual + Jacobians, per block
//!         lqr.perform_qr()             // :1320   3 Householder reflections per block
//!         lqr.get_dense_h_b(&mut H, &mut b)  // :1393  H += Q2Jp^T Q2Jp, b += Q2Jp^T Q2r
//!         inc = (H + lambda diag(H))^-1 b    // :1414-1430
//!         inc = -inc                         // :1450   the residual-sign compensation
//! l_diff= lqr.back_substitute(inc)     // :1454   landmark increments + model cost change
//! ```
//!
//! **What the shipped path does not call** (decision D34). `setPoseDamping`,
//! `scaleJl_cols`, `scaleJp_cols`, `getJp_diag2` and `setLandmarkDamping` are all
//! commented out in `optimize()` — at `:1307-1317` (the Jacobian scaling),
//! `:1361-1377` (both dampings) and `:1461-1463` (undoing the scaling) — which
//! matches the ICCV 2021 paper's own statement that the Givens damping stack is
//! not used in the sliding-window VIO. Damping enters through the
//! `H.diagonal() * lambda` of the dense solve instead (`:1415-1417`), which is
//! stage S8's business. All five are implemented here anyway and none of them is
//! called by this module's own driver: the only live entry into the damping code
//! is [`LandmarkBlock::set_landmark_damping`] with `lambda = 0` from inside
//! `back_substitute` (`landmark_block_abs_dynamic.hpp:310`), where — with no
//! rotations stored — it reduces to zeroing the damping diagonal.
//!
//! **Determinism.** basalt uses `tbb::parallel_deterministic_reduce` at four
//! sites where the summation order changes the answer
//! (`linearization_abs_qr.cpp:262`, `:307`, `:354`, `:550`). Decision D31 asked
//! for a fixed order at each; what it did not say is *which*, and a sequential
//! fold is the wrong one. `parallel_deterministic_reduce` over
//! `blocked_range(0, n)` with grainsize 1 splits at `begin + size / 2` until a
//! range holds one element and then joins up a balanced tree, so four elements
//! reduce as `(x0 + x1) + (x2 + x3)`. In `f32` with `[2²⁴, 1, 1, 1]` that is
//! `16777218` where a fold gives `16777216`. The [`reduce`] module reproduces
//! the tree, pinned bit for bit against the fork's own TBB by
//! `tests/fixtures/linearize/tbb_reduce_oracle.json`, and all four sites go
//! through it: [`LinearizationAbsQR::linearize_problem`],
//! [`LinearizationAbsQR::back_substitute`], [`LinearizationAbsQR::get_jp_diag2`]
//! and [`LinearizationAbsQR::get_dense_h_b`]. A rayon version has to reproduce
//! the same tree; `par_chunks` with an ordered merge does not.

mod abs_qr;
mod eigen_qr;
mod landmark_block;
mod reduce;

pub use abs_qr::{ImuInput, LinearizationAbsQR, LinearizationInputs, LinearizationOptions};
pub use landmark_block::{LandmarkBlock, LandmarkBlockOptions, LandmarkBlockState};

/// [`reduce::deterministic_reduce_scalar`] with the error type erased, so the
/// fixture test in `tests/tbb_reduce_oracle.rs` can drive the association
/// directly.
///
/// The reduction itself is internal — it exists to be called at the four sites
/// of [`LinearizationAbsQR`] — but the *association* is a claim about basalt
/// that has to be checked against basalt, and an integration test cannot reach
/// a private module.
pub fn deterministic_reduce_scalar_for_tests<S: LieScalar>(
    n: usize,
    leaf: &mut dyn FnMut(usize, S) -> S,
) -> S {
    let result: Result<S, std::convert::Infallible> =
        reduce::deterministic_reduce_scalar(n, &mut |index: usize, acc: S| Ok(leaf(index, acc)));
    match result {
        Ok(value) => value,
        // Unreachable: the leaf above cannot fail.
        Err(never) => match never {},
    }
}

use nalgebra::{DMatrix, Matrix4, Matrix6};

use crate::ba_base::BaError;
use crate::lie::LieScalar;
use crate::types::{CamId, FrameId, LandmarkId};

/// One Householder reflection: reduce `storage.col(col).segment(start, len)` and
/// apply the reflection to every column of `storage.block(start, 0, len, ncols)`.
///
/// This is one step of `performQRHouseholder`
/// (`landmark_block_abs_dynamic.hpp:445-453`), with Eigen's `makeHouseholder`
/// and `applyHouseholderOnTheLeft` arithmetic ported rather than nalgebra's
/// (see [`eigen_qr`] for why). Exposed because the marginalization QR of
/// `marg_helper.cpp:293-317` drives the same primitive over a wider matrix, and
/// because the ported `test_qr.cpp` builds a full QR out of it.
///
/// Allocates two scratch vectors per call; the landmark block preallocates
/// instead, because it runs three of these per landmark per iteration.
pub fn reflect_column<S: LieScalar>(
    storage: &mut DMatrix<S>,
    col: usize,
    start: usize,
    len: usize,
) -> Result<(), LinearizeError> {
    // This is a public boundary, so the range arithmetic is checked rather than
    // assumed: `start + len` on a caller-supplied `start` wraps in release and
    // would then pass a bounds test it should fail (decision D32).
    let end: usize = start
        .checked_add(len)
        .ok_or(LinearizeError::LayoutOverflow)?;
    if col >= storage.ncols() || end > storage.nrows() {
        return Err(LinearizeError::StackedSystemSize {
            expected: end.max(col + 1),
            found: storage.nrows().min(storage.ncols()),
        });
    }
    if len == 0 {
        return Ok(());
    }
    let mut essential: Vec<S> = vec![S::zero(); len - 1];
    let mut work: Vec<S> = vec![S::zero(); storage.ncols()];
    let (tau, _beta) = eigen_qr::make_householder(storage, col, start, len, &mut essential);
    eigen_qr::apply_householder_on_the_left(storage, start, len, &essential, tau, &mut work);
    Ok(())
}

/// `RelPoseLin<Scalar>` (`landmark_block.hpp:16-26`): one (host, target) pair's
/// relative pose and the two 6x6 Jacobians of that pose against the two absolute
/// pose increments.
///
/// Hoisted per pair rather than per observation, because every landmark hosted
/// in the same image and seen in the same image shares it
/// (`linearization_abs_qr.cpp:207-241`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelPoseLin<S: LieScalar> {
    /// `T_t_h` (`:21`), as a 4x4 so the residual can multiply the homogeneous
    /// landmark straight through.
    pub t_t_h: Matrix4<S>,
    /// `d_rel_d_h` (`:22`).
    pub d_rel_d_h: Matrix6<S>,
    /// `d_rel_d_t` (`:23`).
    pub d_rel_d_t: Matrix6<S>,
}

impl<S: LieScalar> Default for RelPoseLin<S> {
    /// `RelPoseLin()`'s in-class initialisers: all three zero (`:21-23`).
    fn default() -> Self {
        Self {
            t_t_h: Matrix4::zeros(),
            d_rel_d_h: Matrix6::zeros(),
            d_rel_d_t: Matrix6::zeros(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use nalgebra::DMatrix;

    /// `reflect_column` is public, so its range arithmetic is checked: with
    /// `start = usize::MAX` the sum wraps to a small number that would pass a
    /// naive bounds test and then index out of range (decision D32).
    #[test]
    fn reflect_column_refuses_a_range_that_overflows() {
        let mut m: DMatrix<f64> = DMatrix::identity(1, 1);
        assert_eq!(
            reflect_column(&mut m, 0, usize::MAX, 2).unwrap_err(),
            LinearizeError::LayoutOverflow
        );
        // Out of range without overflowing is refused too.
        assert!(matches!(
            reflect_column(&mut m, 0, 0, 4).unwrap_err(),
            LinearizeError::StackedSystemSize { .. }
        ));
        assert!(matches!(
            reflect_column(&mut m, 3, 0, 1).unwrap_err(),
            LinearizeError::StackedSystemSize { .. }
        ));
        // A zero-length reflection is the identity, not an error: it is what a
        // block with no rows left asks for.
        let before: DMatrix<f64> = m.clone();
        reflect_column(&mut m, 0, 0, 0).unwrap();
        assert_eq!(m, before);
    }
}

/// What the linearizer refuses to do.
///
/// Every variant replaces a C++ assertion, an `at()` that would throw, or an
/// unchecked index. The estimator runs with the GIL released, where a panic
/// aborts the process (decision D32), so the arithmetic that sizes the blocks is
/// checked once at allocation and the per-observation path cannot fail on
/// indexing afterwards.
///
/// Two basalt behaviours that look like errors are deliberately **not** here: a
/// non-finite Jacobian is zeroed with a warning
/// (`landmark_block_abs_dynamic.hpp:153-163`) and a singular `Q1Jl` skips its
/// landmark's back-substitution with a warning (`:264-269`). Turning either into
/// an `Err` would change the estimator's behaviour (trap 11).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LinearizeError {
    /// A landmark's host frame is not in the absolute ordering
    /// (`landmark_block_abs_dynamic.hpp:62` asserts).
    #[error("landmark block host frame {frame_id} is not in the absolute ordering")]
    HostNotInOrdering {
        /// The host frame.
        frame_id: FrameId,
    },
    /// No relative pose was allocated for a (host, target) pair the landmark
    /// database says exists (`:68` asserts).
    #[error("no relative pose for host ({host}, {host_cam}) -> target ({target}, {target_cam})")]
    MissingRelativePose {
        /// Host frame.
        host: FrameId,
        /// Host camera.
        host_cam: CamId,
        /// Target frame.
        target: FrameId,
        /// Target camera.
        target_cam: CamId,
    },
    /// A landmark's observation map lost an entry between allocation and
    /// linearization.
    #[error("landmark {lm_id:?} has no observation in ({target}, {target_cam})")]
    MissingObservation {
        /// The landmark.
        lm_id: LandmarkId,
        /// Target frame.
        target: FrameId,
        /// Target camera.
        target_cam: CamId,
    },
    /// A camera id is not in the calibration.
    #[error("camera {cam_id} is out of range: the rig has {camera_count}")]
    UnknownCamera {
        /// The camera asked for.
        cam_id: CamId,
        /// How many the rig has.
        camera_count: usize,
    },
    /// The block layout arithmetic of `:83-96` overflowed.
    #[error("landmark block layout arithmetic overflowed")]
    LayoutOverflow,
    /// `num_cols % 4 != 0`, which C++ asserts (`:96`).
    #[error("landmark block has {num_cols} columns, which is not a multiple of 4")]
    UnalignedBlock {
        /// The column count that failed the check.
        num_cols: usize,
    },
    /// A pose block would be written past the end of the pose columns.
    #[error("pose block at {offset} does not fit in {total_size} pose columns")]
    PoseBlockOutOfRange {
        /// The offset asked for.
        offset: usize,
        /// The ordering's total size.
        total_size: usize,
    },
    /// A method was called out of order; C++ asserts on `state`.
    #[error("landmark block is {found:?}, expected {expected:?}")]
    WrongState {
        /// What the operation needed.
        expected: LandmarkBlockState,
        /// What the block was.
        found: LandmarkBlockState,
    },
    /// `setLandmarkDamping` was given a negative lambda (`:218` asserts).
    #[error("landmark damping must not be negative")]
    NegativeDamping,
    /// The damping rotations were not a complete set of six (`:221` asserts).
    #[error("the damping rotation stack is corrupt")]
    DampingStackCorrupt,
    /// `Jl_col_scale` is not finite (`:235` asserts).
    #[error("the landmark column scale is not finite")]
    NonFiniteColumnScale,
    /// `scaleJp_cols` was called on a damped block (`:378` asserts).
    #[error("cannot scale the pose columns of a damped block")]
    ScalingDampedBlock,
    /// A stacked system was handed in at the wrong size.
    #[error("stacked system has {found} rows or columns, expected {expected}")]
    StackedSystemSize {
        /// What the block needed.
        expected: usize,
        /// What it got.
        found: usize,
    },
    /// `backSubstitute` was given a pose increment of the wrong length
    /// (`:259` asserts).
    #[error("pose increment has {found} rows, expected {expected}")]
    PoseIncrementSize {
        /// The ordering's total size.
        expected: usize,
        /// What was passed.
        found: usize,
    },
    /// A landmark the database listed is not in it.
    #[error("landmark {0:?} is not in the database")]
    UnknownLandmark(LandmarkId),
    /// An IMU measurement names a frame the window does not have.
    #[error("IMU measurement spans frames {start} to {end}, which the window does not have")]
    UnknownImuFrames {
        /// Start timestamp.
        start: FrameId,
        /// End timestamp.
        end: FrameId,
    },
    /// An IMU interval's end timestamp does not fit in an `i64`; C++ computes
    /// `start_t + dt` unchecked (`imu_block.hpp:30`).
    #[error("IMU interval from {start} ns lasting {dt_ns} ns overflows the timestamp")]
    ImuIntervalOverflow {
        /// Start timestamp.
        start: FrameId,
        /// The measurement's duration.
        dt_ns: i64,
    },
    /// An IMU factor's endpoint has a pose-sized block in the ordering. The
    /// factor is 15 columns wide at each end (`imu_block.hpp:21`), so a 6-column
    /// slot means C++ writes over the neighbouring state.
    #[error("frame {frame} carries a {size}-column block, but an IMU factor needs 15")]
    ImuStateNotFullSize {
        /// The frame that disagrees.
        frame: FrameId,
        /// The size the ordering gave it.
        size: usize,
    },
    /// A landmark block's buffer would not fit in memory. `DMatrix::zeros`
    /// multiplies the two dimensions unchecked.
    #[error("a landmark block of {rows} x {cols} cannot be allocated")]
    BlockTooLarge {
        /// Rows asked for.
        rows: usize,
        /// Columns asked for.
        cols: usize,
    },
    /// The marginalization prior's ordering disagrees with the window's, which
    /// C++ asserts block by block (`ba_base.cpp:383-388`).
    #[error("the marginalization prior's ordering does not match the window's at frame {frame_id}")]
    MargOrderMismatch {
        /// The frame that disagrees.
        frame_id: FrameId,
    },
    /// The marginalization prior is not square-root form, which the QR path
    /// asserts (`linearization_abs_qr.cpp:578`).
    #[error("the QR linearization needs a square-root marginalization prior")]
    MargPriorNotSqrt,
    /// Something the bundle-adjustment base refused.
    #[error(transparent)]
    Ba(#[from] BaError),
}
