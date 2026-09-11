//! IMU preintegration, covariance, bias Jacobians and whitened residuals.
//!
//! The estimator uses midpoint rotation `ΔR exp(½ Δt ω)`, retains `½ a Δt²`
//! in position, and orders the state `[translation, rotation, velocity]` (D12).
//! The full state appends gyro and accelerometer biases. Gyro-bias correction
//! is pre-multiplied, requiring right inverse Jacobians for state blocks and a
//! left inverse Jacobian for the gyro-bias block. Position residuals include
//! `−v_i Δt`.
//!
//! Covariance whitening uses guarded pivoted LDLT. Tiny or negative pivots receive
//! zero weight; singular inputs yield a congruence generalized inverse, not
//! necessarily a Moore-Penrose inverse. The 9x9 factor is recomputed per request
//! to avoid interior mutability. Pivot selection precedes column updates.
//!
//! Callers request all Jacobians or none. Prediction returns a fresh state with
//! an updated timestamp, and gravity initialization uses a cross-product axis.
//! Non-monotonic samples return typed errors (D32). Accelerometer bias cannot
//! change delta rotation because its input Jacobian has zero rotation rows and
//! the transition's rotation rows are `[0 I 0]`; tests enforce this identity.

mod factor;
mod preintegration;

pub use factor::{ImuBlock, ImuLinData};
pub use preintegration::{
    ImuError, ImuNoise, ImuResidualJacobians, ImuSample, IntegratedImuMeasurement, Popped,
    PropagationJacobians, gravity, gravity_from_first_accel,
};

use crate::types::{POSE_VEL_BIAS_SIZE, POSE_VEL_SIZE};
use nalgebra::SMatrix;

/// The 9x9 blocks: covariance, state transition, residual Jacobians.
pub type Matrix9<S> = SMatrix<S, POSE_VEL_SIZE, POSE_VEL_SIZE>;
/// The 9x3 blocks: noise input and bias Jacobians.
pub type Matrix9x3<S> = SMatrix<S, POSE_VEL_SIZE, 3>;
/// The 9x6 residual Jacobian with respect to both biases at once.
pub type Matrix9x6<S> = SMatrix<S, POSE_VEL_SIZE, 6>;
/// The IMU block's Jacobian: 15 rows over the two 15-column states.
pub type Matrix15x30<S> = SMatrix<S, POSE_VEL_BIAS_SIZE, { 2 * POSE_VEL_BIAS_SIZE }>;
