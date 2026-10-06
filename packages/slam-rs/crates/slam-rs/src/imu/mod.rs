//! IMU calibration, frame-boundary accumulation, and inertial factors.
use kornia_staging_algebra::Scalar;
mod factor;
use crate::{
    calib::Calibration,
    lie::{So3, c},
};
pub use factor::{ImuBlock, ImuLinData};
use kornia_staging_sensors::SensorError;
use kornia_staging_sensors::imu::ImuNoise;
use kornia_staging_sensors::imu::IntegratedImuMeasurement;
use nalgebra::Vector3;

pub(crate) fn noise_from_calibration<S: Scalar>(
    calib: &Calibration<S>,
) -> Result<ImuNoise<S>, SensorError> {
    ImuNoise::new(
        calib.discrete_time_accel_noise_std().map(|x| x * x),
        calib.discrete_time_gyro_noise_std().map(|x| x * x),
    )
}
/// Gravity in the world frame.
pub fn gravity<S: Scalar>() -> Vector3<S> {
    Vector3::new(S::zero(), S::zero(), c(-9.81))
}
/// One calibrated sample as the two producer loops pop them: `(t_ns, gyro,
/// accel)`, the shape `accumulate_to` carries its pending sample in.
pub type Popped<S> = (i64, Vector3<S>, Vector3<S>);

/// A frame interval cannot be accumulated from the supplied IMU samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AccumulateError {
    /// Invalid sensor sample or integration result.
    #[error(transparent)]
    Sensor(#[from] SensorError),
    /// Two frame timestamps that are equal or go backwards.
    ///
    /// Zero or negative duration cannot define a preintegrated measurement.
    #[error("frame interval [{t0_ns}, {t1_ns}] ns is empty or reversed")]
    NonMonotonicFrames {
        /// Timestamp of the previous frame.
        t0_ns: i64,
        /// Timestamp of the current frame.
        t1_ns: i64,
    },
    /// The requested interval starts at a different time from the measurement.
    #[error("interval starts at {t0_ns} ns but the measurement starts at {start_timestamp_ns} ns")]
    StartTimeMismatch {
        /// Start time the measurement was constructed with.
        start_timestamp_ns: i64,
        /// Start of the interval the caller asked for.
        t0_ns: i64,
    },
    /// No sample follows the frame to close the integration interval.
    #[error("no imu sample after {t1_ns} ns to close the interval with")]
    MissingSampleAfterFrame {
        /// Timestamp the interval had to reach.
        t1_ns: i64,
    },
}

/// Accumulate samples over one frame interval.
/// Skip samples at or before `skip_past_ns`, then integrate through `until_ns`.
/// If the interval remains short, retime the first later sample to the boundary
/// and integrate its own values, without interpolation. Return that pending
/// sample at its original timestamp for the next frame.
///
/// Both frontend and estimator preintegrators use this loop (D24), with their
/// own sample source and noise model. Passing pending state by value leaves
/// the caller free to mutate its queue through `pop`.
///
/// # Errors
/// Refuse reversed intervals, a mismatched start time, missing closing samples
/// or any input rejected by [`IntegratedImuMeasurement::integrate_calibrated`].
pub fn accumulate_to<S: Scalar>(
    measurement: &mut IntegratedImuMeasurement<S>,
    pending: Option<Popped<S>>,
    mut pop: impl FnMut() -> Option<Popped<S>>,
    skip_past_ns: i64,
    until_ns: i64,
    noise: &ImuNoise<S>,
) -> Result<Option<Popped<S>>, AccumulateError> {
    // the frame gap is what the measurement integrates over, so
    // an empty or reversed one has no measurement.
    if until_ns <= skip_past_ns {
        return Err(AccumulateError::NonMonotonicFrames {
            t0_ns: skip_past_ns,
            t1_ns: until_ns,
        });
    }
    //  builds the measurement with `prev_frame->t_ns` and
    // skips past that same variable. Any other origin times every sample
    // against a start the measurement does not have.
    if skip_past_ns != measurement.start_timestamp_ns() {
        return Err(AccumulateError::StartTimeMismatch {
            start_timestamp_ns: measurement.start_timestamp_ns(),
            t0_ns: skip_past_ns,
        });
    }

    let mut pending: Option<Popped<S>> = pending.or_else(&mut pop);

    // discard everything at or before the previous frameset.
    while let Some((t_ns, _, _)) = pending {
        if t_ns > skip_past_ns {
            break;
        }
        pending = pop();
    }
    // integrate everything up to and including the frameset.
    while let Some((t_ns, gyro, accel)) = pending {
        if t_ns > until_ns {
            break;
        }
        measurement.integrate_calibrated(t_ns, &accel, &gyro, noise)?;
        pending = pop();
    }
    // A measurement stopping before the frame is invalid; require a closing sample.
    if measurement.start_timestamp_ns() + measurement.dt_ns() < until_ns {
        let Some((_, gyro, accel)) = pending else {
            return Err(AccumulateError::MissingSampleAfterFrame { t1_ns: until_ns });
        };
        measurement.integrate_calibrated(until_ns, &accel, &gyro, noise)?;
    }
    Ok(pending)
}

fn eigen_dummy_precision<S: Scalar>() -> S {
    S::from_literal(if std::mem::size_of::<S>() == 4 { 1e-5 } else { 1e-12 })
}

/// Fires the near-antiparallel warning in [`gravity_from_first_accel`] once per
/// process, not once per frame.
static ANTIPARALLEL_WARNING: std::sync::Once = std::sync::Once::new();

/// The initial orientation from one accelerometer sample
///
/// Align measured specific force with world +Z. Only roll and pitch are
/// observable. Near antiparallel inputs use a cross-product axis orthogonal
/// to both directions; for exactly antiparallel inputs an arbitrary orthogonal
/// axis resolves the unobservable yaw. Zero and non-finite samples return identity.
pub fn gravity_from_first_accel<S: Scalar>(accel: &Vector3<S>) -> So3<S> {
    let norm: S = accel.norm();
    if !norm.is_finite() || norm <= S::zero() {
        return So3::identity();
    }
    let v0: Vector3<S> = accel / norm;
    let v1: Vector3<S> = Vector3::new(S::zero(), S::zero(), S::one());
    let dot: S = v1.dot(&v0);

    if dot < c::<S>(-1.0) + eigen_dummy_precision::<S>() {
        // Report this ambiguous initial orientation once per process.
        ANTIPARALLEL_WARNING.call_once(|| {
            log::warn!(
                "gravity initialisation took the near-antiparallel branch: the rig started \
                 within milliradians of upside down, and the initial yaw and up to \
                 2*sqrt(2*eps) rad of roll and pitch differ from basalt's Eigen JacobiSVD"
            );
        });
        let clamped: S = dot.max(c::<S>(-1.0));
        let axis: Vector3<S> = axis_orthogonal_to_both(&v0, &v1);
        let w2: S = (S::one() + clamped) * c::<S>(0.5);
        let vector_scale: S = (S::one() - w2).sqrt();
        let quaternion = nalgebra::Quaternion::new(
            w2.sqrt(),
            axis.x * vector_scale,
            axis.y * vector_scale,
            axis.z * vector_scale,
        );
        // Normalize to remove rounding in the squared quaternion norm.
        return So3::from_unit_quaternion(nalgebra::UnitQuaternion::new_normalize(quaternion));
    }

    let axis: Vector3<S> = v0.cross(&v1);
    let s: S = ((S::one() + dot) * c::<S>(2.0)).sqrt();
    let inv_s: S = S::one() / s;
    let quaternion = nalgebra::Quaternion::new(
        s * c::<S>(0.5),
        axis.x * inv_s,
        axis.y * inv_s,
        axis.z * inv_s,
    );
    So3::from_unit_quaternion(nalgebra::UnitQuaternion::new_normalize(quaternion))
}

/// A unit vector orthogonal to both inputs, using their cross product.
/// For parallel inputs, project the least-aligned canonical axis off v1.
fn axis_orthogonal_to_both<S: Scalar>(v0: &Vector3<S>, v1: &Vector3<S>) -> Vector3<S> {
    let cross: Vector3<S> = v0.cross(v1);
    let norm: S = cross.norm();
    if norm > S::zero() {
        return cross / norm;
    }
    let basis: Vector3<S> = if v1.x.abs() <= v1.y.abs() && v1.x.abs() <= v1.z.abs() {
        Vector3::new(S::one(), S::zero(), S::zero())
    } else if v1.y.abs() <= v1.z.abs() {
        Vector3::new(S::zero(), S::one(), S::zero())
    } else {
        Vector3::new(S::zero(), S::zero(), S::one())
    };
    let projected: Vector3<S> = basis - v1 * basis.dot(v1);
    let projected_norm: S = projected.norm();
    if projected_norm > S::zero() {
        projected / projected_norm
    } else {
        basis
    }
}

#[cfg(test)]
mod tests;
