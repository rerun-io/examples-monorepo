//! IMU preintegration from the first frame to every frame of the window.
//!
//! The quantities are the preintegrated rotation, velocity and position of
//! Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial
//! Odometry" (T-RO 2017), taken over `[t_0, t_k]` for every frame `k`. Because
//! the first body frame is the reference, `R_0 = I` and the integrals are the
//! body-0-frame terms of the window directly:
//!
//! ```text
//! R_k   = prod Exp((w_i - b_g) dt_i)
//! dv_k  = sum R_i (a_i - b_a) dt_i
//! dp_k  = sum dv_i dt_i + 1/2 R_i (a_i - b_a) dt_i^2
//! ```
//!
//! Rotation uses the mid-point gyro rate of each sub-interval and the
//! specific force is the trapezoid of its two rotated ends, so the error is
//! second order in the 1 ms sample spacing. Frame times that fall between
//! samples are reached by linear interpolation of the two neighbouring
//! samples. The bias Jacobians are the first-order recursions of Forster et
//! al. (Appendix B), evaluated at the start of each sub-interval.
use super::{ImuSample, InertialInitError};
use kornia_staging_algebra::lie::{right_jacobian_so3, Rotation3};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

/// The window preintegrated at one bias linearisation point.
#[derive(Debug, Clone)]
pub(super) struct Preintegrated {
    /// `t_k - t_0` in seconds.
    pub dt: Vec<f64>,
    /// Body-k-to-body-0 rotation `R_k`.
    pub rotation: Vec<Matrix3<f64>>,
    /// Integrated rotated specific force `dv_k` (body-0 frame).
    pub velocity: Vec<Vector3<f64>>,
    /// Double-integrated rotated specific force `dp_k` (body-0 frame).
    pub position: Vec<Vector3<f64>>,
    /// `d R_k / d b_g` as a right perturbation `R_k Exp(J dbg)`.
    pub d_rotation_d_bg: Vec<Matrix3<f64>>,
    /// `d dp_k / d b_g`.
    pub d_position_d_bg: Vec<Matrix3<f64>>,
    /// `d dp_k / d b_a`.
    pub d_position_d_ba: Vec<Matrix3<f64>>,
    /// The gyroscope bias of the linearisation point.
    pub bias_gyro: Vector3<f64>,
    /// The accelerometer bias of the linearisation point.
    pub bias_accel: Vector3<f64>,
}

impl Preintegrated {
    /// The rotation of frame `k` at gyro bias `bias_gyro`, first order around
    /// the linearisation point.
    pub fn rotation_at(&self, k: usize, bias_gyro: &Vector3<f64>) -> Matrix3<f64> {
        let delta: Vector3<f64> = self.d_rotation_d_bg[k] * (bias_gyro - self.bias_gyro);
        self.rotation[k] * Rotation3::exp(&delta).matrix()
    }

    /// `dp_k` at biases `(bias_gyro, bias_accel)`, first order.
    pub fn position_at(
        &self,
        k: usize,
        bias_gyro: &Vector3<f64>,
        bias_accel: &Vector3<f64>,
    ) -> Vector3<f64> {
        self.position[k]
            + self.d_position_d_bg[k] * (bias_gyro - self.bias_gyro)
            + self.d_position_d_ba[k] * (bias_accel - self.bias_accel)
    }
}

/// Linear interpolation of a sample pair at `t_ns`.
fn interpolate(a: &ImuSample, b: &ImuSample, t_ns: i64) -> (Vector3<f64>, Vector3<f64>) {
    let span: f64 = (b.t_ns - a.t_ns) as f64;
    let w: f64 = if span > 0.0 {
        (t_ns - a.t_ns) as f64 / span
    } else {
        0.0
    };
    (
        a.gyro * (1.0 - w) + b.gyro * w,
        a.accel * (1.0 - w) + b.accel * w,
    )
}

/// The value of the IMU signal at `t_ns`, interpolated between the
/// neighbouring samples. The caller guarantees coverage.
pub fn sample_at(imu: &[ImuSample], t_ns: i64) -> (Vector3<f64>, Vector3<f64>) {
    // First sample strictly after t_ns.
    let upper: usize = imu.partition_point(|s| s.t_ns <= t_ns);
    if upper == 0 {
        return (imu[0].gyro, imu[0].accel);
    }
    if upper == imu.len() {
        let last = &imu[imu.len() - 1];
        return (last.gyro, last.accel);
    }
    interpolate(&imu[upper - 1], &imu[upper], t_ns)
}

/// Preintegrate the window at the given biases.
///
/// # Errors
/// The IMU must cover the first and the last frame time.
pub(super) fn preintegrate(
    imu: &[ImuSample],
    frame_times_ns: &[i64],
    bias_gyro: &Vector3<f64>,
    bias_accel: &Vector3<f64>,
) -> Result<Preintegrated, InertialInitError> {
    let (Some(&t_first), Some(&t_last)) = (frame_times_ns.first(), frame_times_ns.last()) else {
        return Err(InertialInitError::TooFewFrames { frames: 0 });
    };
    let covered = match (imu.first(), imu.last()) {
        (Some(a), Some(b)) => a.t_ns <= t_first && b.t_ns >= t_last,
        _ => false,
    };
    if !covered {
        return Err(InertialInitError::ImuDoesNotCoverWindow);
    }

    let n: usize = frame_times_ns.len();
    let mut out = Preintegrated {
        dt: Vec::with_capacity(n),
        rotation: Vec::with_capacity(n),
        velocity: Vec::with_capacity(n),
        position: Vec::with_capacity(n),
        d_rotation_d_bg: Vec::with_capacity(n),
        d_position_d_bg: Vec::with_capacity(n),
        d_position_d_ba: Vec::with_capacity(n),
        bias_gyro: *bias_gyro,
        bias_accel: *bias_accel,
    };

    let mut q: UnitQuaternion<f64> = UnitQuaternion::identity();
    let mut vel: Vector3<f64> = Vector3::zeros();
    let mut pos: Vector3<f64> = Vector3::zeros();
    let mut j_rot: Matrix3<f64> = Matrix3::zeros();
    let mut j_vel_bg: Matrix3<f64> = Matrix3::zeros();
    let mut j_vel_ba: Matrix3<f64> = Matrix3::zeros();
    let mut j_pos_bg: Matrix3<f64> = Matrix3::zeros();
    let mut j_pos_ba: Matrix3<f64> = Matrix3::zeros();

    let record = |out: &mut Preintegrated,
                  t_ns: i64,
                  q: &UnitQuaternion<f64>,
                  vel: &Vector3<f64>,
                  pos: &Vector3<f64>,
                  j_rot: &Matrix3<f64>,
                  j_pos_bg: &Matrix3<f64>,
                  j_pos_ba: &Matrix3<f64>| {
        out.dt.push((t_ns - t_first) as f64 * 1e-9);
        out.rotation.push(q.to_rotation_matrix().into_inner());
        out.velocity.push(*vel);
        out.position.push(*pos);
        out.d_rotation_d_bg.push(*j_rot);
        out.d_position_d_bg.push(*j_pos_bg);
        out.d_position_d_ba.push(*j_pos_ba);
    };

    let (mut gyro_a, mut accel_a) = sample_at(imu, t_first);
    let mut t_a: i64 = t_first;
    record(
        &mut out, t_first, &q, &vel, &pos, &j_rot, &j_pos_bg, &j_pos_ba,
    );

    // Knots: every sample strictly inside (t_a, frame), then the frame.
    let mut next_sample: usize = imu.partition_point(|s| s.t_ns <= t_first);
    for &t_frame in &frame_times_ns[1..] {
        loop {
            let (t_b, gyro_b, accel_b, is_frame) = match imu.get(next_sample) {
                Some(s) if s.t_ns < t_frame => (s.t_ns, s.gyro, s.accel, false),
                _ => {
                    let (g, a) = sample_at(imu, t_frame);
                    (t_frame, g, a, true)
                }
            };
            let dt: f64 = (t_b - t_a) as f64 * 1e-9;
            if dt > 0.0 {
                let omega: Vector3<f64> = (gyro_a + gyro_b) * 0.5 - bias_gyro;
                let phi: Vector3<f64> = omega * dt;
                let dq: UnitQuaternion<f64> = UnitQuaternion::from_scaled_axis(phi);
                let r_a: Matrix3<f64> = q.to_rotation_matrix().into_inner();
                let q_b: UnitQuaternion<f64> = q * dq;
                let r_b: Matrix3<f64> = q_b.to_rotation_matrix().into_inner();
                let f_a: Vector3<f64> = accel_a - bias_accel;
                let f_b: Vector3<f64> = accel_b - bias_accel;
                let f_mid: Vector3<f64> = (f_a + f_b) * 0.5;
                let f_world: Vector3<f64> = (r_a * f_a + r_b * f_b) * 0.5;

                // Bias Jacobians first, with the start-of-interval state.
                let f_hat: Matrix3<f64> = Rotation3::hat(&f_mid);
                let half_dt2: f64 = 0.5 * dt * dt;
                j_pos_ba += j_vel_ba * dt - r_a * half_dt2;
                j_pos_bg += j_vel_bg * dt - r_a * f_hat * j_rot * half_dt2;
                j_vel_ba -= r_a * dt;
                j_vel_bg -= r_a * f_hat * j_rot * dt;
                let dr_t: Matrix3<f64> = dq.to_rotation_matrix().into_inner().transpose();
                j_rot = dr_t * j_rot - right_jacobian_so3(&phi) * dt;

                pos += vel * dt + f_world * half_dt2;
                vel += f_world * dt;
                q = q_b;
            }
            t_a = t_b;
            gyro_a = gyro_b;
            accel_a = accel_b;
            if is_frame {
                break;
            }
            next_sample += 1;
        }
        // Samples exactly at the frame time were consumed as the frame knot.
        while imu.get(next_sample).is_some_and(|s| s.t_ns <= t_frame) {
            next_sample += 1;
        }
        record(
            &mut out, t_frame, &q, &vel, &pos, &j_rot, &j_pos_bg, &j_pos_ba,
        );
    }
    Ok(out)
}
