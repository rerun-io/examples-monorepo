//! Analytic trajectory fixtures shared by numerical and consumer factor tests.
use super::{CombinedImuSample, IntegratedImuMeasurement};
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use nalgebra::{SVector, Vector3};

pub(super) const ACCEL_STD_DEV: f64 = 0.23;
pub(super) const GYRO_STD_DEV: f64 = 0.0027;
pub(super) const GRAVITY: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);

// Test support uses a smooth analytic trajectory with closed-form pose, velocity,
// acceleration and body angular velocity, plus a seeded xorshift generator.
// This keeps tests deterministic without a spline dependency.

/// xorshift64*, so every ported test runs the same numbers each time.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x: u64 = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform value in `[-1, 1]`.
    pub(super) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// Uniform on `[low, high]`.
    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (self.uniform() + 1.0) * 0.5 * (high - low)
    }

    /// Three independent uniform values in `[-1, 1]`.
    pub(super) fn vector3(&mut self) -> Vector3<f64> {
        Vector3::new(self.uniform(), self.uniform(), self.uniform())
    }

    /// `Sophus::Vector6d::Random()`.
    pub(super) fn vector6(&mut self) -> SVector<f64, 6> {
        SVector::<f64, 6>::from_fn(|_, _| self.uniform())
    }
}

/// Smooth analytic trajectory: sinusoidal position gives exact velocity and
/// acceleration. For `R = exp(phi(t))`, body angular velocity is `J_r(phi) phi_dot`.
/// Low frequencies keep midpoint integration error bounded over the 20-second test.
pub(super) struct Trajectory {
    pos_amp: Vector3<f64>,
    pos_freq: Vector3<f64>,
    pos_phase: Vector3<f64>,
    rot_amp: Vector3<f64>,
    rot_freq: Vector3<f64>,
    rot_phase: Vector3<f64>,
}

impl Trajectory {
    pub(super) fn new(rng: &mut Rng) -> Self {
        Self {
            pos_amp: Vector3::from_fn(|_, _| rng.range(0.8, 1.5)),
            pos_freq: Vector3::from_fn(|_, _| rng.range(0.15, 0.35)),
            pos_phase: Vector3::from_fn(|_, _| rng.range(0.0, std::f64::consts::TAU)),
            rot_amp: Vector3::from_fn(|_, _| rng.range(0.15, 0.30)),
            rot_freq: Vector3::from_fn(|_, _| rng.range(0.15, 0.30)),
            rot_phase: Vector3::from_fn(|_, _| rng.range(0.0, std::f64::consts::TAU)),
        }
    }

    fn seconds(timestamp_ns: i64) -> f64 {
        timestamp_ns as f64 * 1e-9
    }

    fn rotation_vector(&self, t: f64) -> Vector3<f64> {
        Vector3::from_fn(|i, _| self.rot_amp[i] * (self.rot_freq[i] * t + self.rot_phase[i]).sin())
    }

    fn rotation_vector_dot(&self, t: f64) -> Vector3<f64> {
        Vector3::from_fn(|i, _| {
            self.rot_amp[i] * self.rot_freq[i] * (self.rot_freq[i] * t + self.rot_phase[i]).cos()
        })
    }

    pub(super) fn pose(&self, timestamp_ns: i64) -> RigidTransform<f64> {
        let t: f64 = Self::seconds(timestamp_ns);
        RigidTransform::new(
            Rotation3::exp(&self.rotation_vector(t)),
            Vector3::from_fn(|i, _| {
                self.pos_amp[i] * (self.pos_freq[i] * t + self.pos_phase[i]).sin()
            }),
        )
    }

    pub(super) fn trans_vel_world(&self, timestamp_ns: i64) -> Vector3<f64> {
        let t: f64 = Self::seconds(timestamp_ns);
        Vector3::from_fn(|i, _| {
            self.pos_amp[i] * self.pos_freq[i] * (self.pos_freq[i] * t + self.pos_phase[i]).cos()
        })
    }

    fn trans_accel_world(&self, timestamp_ns: i64) -> Vector3<f64> {
        let t: f64 = Self::seconds(timestamp_ns);
        Vector3::from_fn(|i, _| {
            -self.pos_amp[i]
                * self.pos_freq[i]
                * self.pos_freq[i]
                * (self.pos_freq[i] * t + self.pos_phase[i]).sin()
        })
    }

    /// `omega_body = J_r(phi) phi_dot` for `R(t) = exp(phi(t))`.
    fn rot_vel_body(&self, timestamp_ns: i64) -> Vector3<f64> {
        let t: f64 = Self::seconds(timestamp_ns);
        kornia_staging_algebra::lie::right_jacobian_so3(&self.rotation_vector(t))
            * self.rotation_vector_dot(t)
    }

    /// Body-frame specific force and angular velocity sampled at the interval midpoint.
    pub(super) fn sample(&self, timestamp_ns: i64, dt_ns: i64) -> CombinedImuSample {
        let pose: RigidTransform<f64> = self.pose(timestamp_ns);
        CombinedImuSample {
            timestamp_ns: timestamp_ns + dt_ns / 2,
            gyro: kornia_algebra::Vec3F64::from_array(self.rot_vel_body(timestamp_ns).into()),
            accel: kornia_algebra::Vec3F64::from_array(
                (pose.rotation.inverse() * (self.trans_accel_world(timestamp_ns) - GRAVITY)).into(),
            ),
        }
    }
}

/// The samples `BiasTest` and `ResidualBiasTest` share
/// 100 samples over 1 s with both
/// biases added in.
pub(super) fn biased_samples(
    trajectory: &Trajectory,
    bg: &Vector3<f64>,
    ba: &Vector3<f64>,
) -> Vec<CombinedImuSample> {
    let dt_ns: i64 = 10_000_000;
    let mut samples: Vec<CombinedImuSample> = Vec::new();
    let mut timestamp_ns: i64 = dt_ns / 2;
    while timestamp_ns < 1_000_000_000 {
        let mut sample: CombinedImuSample = trajectory.sample(timestamp_ns, dt_ns);
        let shift = ba;
        sample.accel.x += shift.x;
        sample.accel.y += shift.y;
        sample.accel.z += shift.z;
        let shift = bg;
        sample.gyro.x += shift.x;
        sample.gyro.y += shift.y;
        sample.gyro.z += shift.z;
        samples.push(sample);
        timestamp_ns += dt_ns;
    }
    samples
}

pub(super) fn integrate_all(
    start_timestamp_ns: i64,
    bg: &Vector3<f64>,
    ba: &Vector3<f64>,
    samples: &[CombinedImuSample],
) -> IntegratedImuMeasurement<f64> {
    let mut meas: IntegratedImuMeasurement<f64> =
        IntegratedImuMeasurement::new(start_timestamp_ns, bg, ba);
    let ones: Vector3<f64> = Vector3::repeat(1.0);
    for sample in samples {
        meas.integrate(sample, &crate::imu::ImuNoise::new(ones, ones).unwrap())
            .unwrap();
    }
    meas
}
