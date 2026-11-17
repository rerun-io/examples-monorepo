//! A synthetic handheld walk seen by a multi-centre rig, with its exact IMU.
use super::super::linear::{tangent_basis, SplitMix64};
use super::super::{BearingObservation, FeatureTrack, ImuSample};
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};

/// Gravity magnitude of the synthetic world.
pub const GRAVITY: f64 = 9.81;
/// 0.5 px at 305.6 px/rad, per tangent axis.
pub const BEARING_SIGMA: f64 = 0.5 / 305.6;
/// Gyro white noise, 0.007 deg/s/sqrt(Hz) (consumer MEMS, BMI088 class).
pub const GYRO_DENSITY: f64 = 0.007 * std::f64::consts::PI / 180.0;
/// Accelerometer white noise, 150 ug/sqrt(Hz).
pub const ACCEL_DENSITY: f64 = 150e-6 * 9.81;
/// IMU rate.
pub const IMU_HZ: f64 = 1000.0;
/// Camera rate.
pub const CAMERA_HZ: f64 = 30.0;
/// Half the lens-centre separation of the 360 rig.
pub const HALF_BASELINE: f64 = 0.01655;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Motion {
    /// Forward walk with bounce, sway and yaw.
    Walking,
    /// Straight line, constant speed, constant attitude.
    ConstantVelocity,
    /// Walking's rotation about a fixed point.
    PureRotation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rig {
    /// Back-to-back lenses facing body +Z and -Z, centres +-16.55 mm on Z.
    TwoCentre,
    /// The same lenses with one shared centre.
    Central,
    /// One forward 60-degree-half-angle camera off the IMU, and a blind second one.
    Mono,
}

#[derive(Debug, Clone, Copy)]
pub struct Scenario {
    pub motion: Motion,
    pub rig: Rig,
    pub window_s: f64,
    pub seed: u64,
    pub bearing_noise: bool,
    pub imu_noise: bool,
    pub outlier_fraction: f64,
    pub inlier_tracks: usize,
    pub gyro_bias: Vector3<f64>,
    pub accel_bias: Vector3<f64>,
    /// Bearings are rendered this much later than their frame time says.
    pub time_offset_s: f64,
    /// Added to every camera centre given to the solver (not to the truth).
    pub lever_arm_error: Vector3<f64>,
    /// Rolling shutter: a bearing is rendered `rs * b_y` seconds off its frame time.
    pub rolling_shutter_s: f64,
}

impl Scenario {
    /// Noise-free walk with a constant gyro bias of 0.5 deg/s.
    pub fn clean(rig: Rig, window_s: f64) -> Self {
        Self {
            motion: Motion::Walking,
            rig,
            window_s,
            seed: 7,
            bearing_noise: false,
            imu_noise: false,
            outlier_fraction: 0.0,
            inlier_tracks: 150,
            gyro_bias: Vector3::new(1.0, -0.6, 0.5).normalize() * 0.5f64.to_radians(),
            accel_bias: Vector3::zeros(),
            time_offset_s: 0.0,
            lever_arm_error: Vector3::zeros(),
            rolling_shutter_s: 0.0,
        }
    }

    /// The realistic case: bearing and IMU noise, 0.5 deg/s gyro bias,
    /// 0.05 m/s^2 accelerometer bias, 20 % outlier tracks fixed in the image.
    pub fn noisy(rig: Rig, window_s: f64, seed: u64) -> Self {
        let mut rng = Rng::new(seed ^ 0xB1A5);
        let gyro_dir: Vector3<f64> = rng.unit_vector();
        let accel_dir: Vector3<f64> = rng.unit_vector();
        Self {
            seed,
            bearing_noise: true,
            imu_noise: true,
            outlier_fraction: 0.2,
            gyro_bias: gyro_dir * 0.5f64.to_radians(),
            accel_bias: accel_dir * 0.05,
            ..Self::clean(rig, window_s)
        }
    }
}

pub struct Truth {
    pub velocity_body0: Vector3<f64>,
    pub gravity_body0: Vector3<f64>,
    pub positions_body0: Vec<Vector3<f64>>,
    pub bias_gyro: Vector3<f64>,
    pub bias_accel: Vector3<f64>,
}

pub struct Synthetic {
    pub imu: Vec<ImuSample>,
    pub frame_times_ns: Vec<i64>,
    pub tracks: Vec<FeatureTrack>,
    pub t_i_c: Vec<RigidTransform<f64>>,
    pub truth: Truth,
}

/// Seeded uniform and normal draws.
pub struct Rng(SplitMix64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(SplitMix64::new(seed))
    }
    pub fn uniform(&mut self) -> f64 {
        (self.0.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
    pub fn index(&mut self, n: usize) -> usize {
        self.0.below(n)
    }
    /// Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u1: f64 = self.uniform().max(1e-300);
        let u2: f64 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn unit_vector(&mut self) -> Vector3<f64> {
        loop {
            let v = Vector3::new(self.normal(), self.normal(), self.normal());
            if v.norm() > 1e-6 {
                return v.normalize();
            }
        }
    }
}

/// The trajectory's free parameters.
struct Path {
    motion: Motion,
    heading: f64,
    phase: [f64; 6],
}

fn rot_x(a: f64) -> Matrix3<f64> {
    Rotation3::exp(&Vector3::new(a, 0.0, 0.0)).matrix()
}
fn rot_y(a: f64) -> Matrix3<f64> {
    Rotation3::exp(&Vector3::new(0.0, a, 0.0)).matrix()
}
fn rot_z(a: f64) -> Matrix3<f64> {
    Rotation3::exp(&Vector3::new(0.0, 0.0, a)).matrix()
}

/// Nominal attitude: body Z (the lens axis) forward along world X, body Y down.
fn nominal() -> Matrix3<f64> {
    Matrix3::from_columns(&[
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(1.0, 0.0, 0.0),
    ])
}

const TAU: f64 = 2.0 * std::f64::consts::PI;
const SPEED: f64 = 1.2;
const BOUNCE: f64 = 0.02; // m, 2 Hz vertical
const SURGE: f64 = 0.02; // m, 2 Hz forward
const SWAY: f64 = 0.03; // m, 1 Hz lateral
const DRIFT: f64 = 0.15; // m/s^2 lateral, a gentle curve

impl Path {
    /// World position, velocity and acceleration.
    fn translation(&self, t: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let ph = &self.phase;
        let (p, v, a) = match self.motion {
            Motion::PureRotation => (
                Vector3::new(0.0, 0.0, 1.5),
                Vector3::zeros(),
                Vector3::zeros(),
            ),
            Motion::ConstantVelocity => (
                Vector3::new(SPEED * t, 0.0, 1.5),
                Vector3::new(SPEED, 0.0, 0.0),
                Vector3::zeros(),
            ),
            Motion::Walking => {
                let wb: f64 = TAU * 2.0;
                let wl: f64 = TAU * 1.0;
                let p = Vector3::new(
                    SPEED * t + SURGE * (wb * t + ph[0]).sin(),
                    SWAY * (wl * t + ph[1]).sin() + 0.5 * DRIFT * t * t,
                    1.5 + BOUNCE * (wb * t + ph[2]).sin(),
                );
                let v = Vector3::new(
                    SPEED + SURGE * wb * (wb * t + ph[0]).cos(),
                    SWAY * wl * (wl * t + ph[1]).cos() + DRIFT * t,
                    BOUNCE * wb * (wb * t + ph[2]).cos(),
                );
                let a = Vector3::new(
                    -SURGE * wb * wb * (wb * t + ph[0]).sin(),
                    -SWAY * wl * wl * (wl * t + ph[1]).sin() + DRIFT,
                    -BOUNCE * wb * wb * (wb * t + ph[2]).sin(),
                );
                (p, v, a)
            }
        };
        let h: Matrix3<f64> = rot_z(self.heading);
        (h * p, h * v, h * a)
    }

    /// World-from-body rotation.
    fn rotation(&self, t: f64) -> Matrix3<f64> {
        let ph = &self.phase;
        let (yaw, pitch, roll) = match self.motion {
            Motion::ConstantVelocity => (0.0, 0.05, -0.03),
            Motion::Walking | Motion::PureRotation => (
                0.25 * (TAU * 0.6 * t + ph[3]).sin(),
                0.08 * (TAU * 2.0 * t + ph[4]).sin(),
                0.10 * (TAU * 1.0 * t + ph[5]).sin(),
            ),
        };
        rot_z(self.heading + yaw) * rot_y(pitch) * rot_x(roll) * nominal()
    }

    /// Body angular rate.
    fn omega(&self, t: f64) -> Vector3<f64> {
        let h: f64 = 1e-5;
        let delta: Matrix3<f64> = self.rotation(t - h).transpose() * self.rotation(t + h);
        let q = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
            delta,
        ));
        q.scaled_axis() / (2.0 * h)
    }
}

struct Lens {
    r_ic: Matrix3<f64>,
    t_ic: Vector3<f64>,
    half_fov: f64,
    blind: bool,
}

fn lenses(rig: Rig) -> Vec<Lens> {
    let back: Matrix3<f64> = rot_y(std::f64::consts::PI);
    let wide: f64 = 96f64.to_radians();
    match rig {
        Rig::TwoCentre => vec![
            Lens {
                r_ic: Matrix3::identity(),
                t_ic: Vector3::new(0.0, 0.0, HALF_BASELINE),
                half_fov: wide,
                blind: false,
            },
            Lens {
                r_ic: back,
                t_ic: Vector3::new(0.0, 0.0, -HALF_BASELINE),
                half_fov: wide,
                blind: false,
            },
        ],
        Rig::Central => vec![
            Lens {
                r_ic: Matrix3::identity(),
                t_ic: Vector3::zeros(),
                half_fov: wide,
                blind: false,
            },
            Lens {
                r_ic: back,
                t_ic: Vector3::zeros(),
                half_fov: wide,
                blind: false,
            },
        ],
        Rig::Mono => vec![
            Lens {
                r_ic: Matrix3::identity(),
                t_ic: Vector3::new(0.02, -0.01, 0.05),
                half_fov: 60f64.to_radians(),
                blind: false,
            },
            Lens {
                r_ic: back,
                t_ic: Vector3::new(0.0, 0.0, -HALF_BASELINE),
                half_fov: wide,
                blind: true,
            },
        ],
    }
}

/// The bearing of a world point in a lens at time t, if visible.
fn see(path: &Path, lens: &Lens, point: &Vector3<f64>, t: f64) -> Option<Vector3<f64>> {
    let (p, _, _) = path.translation(t);
    let x: Vector3<f64> =
        lens.r_ic.transpose() * (path.rotation(t).transpose() * (point - p) - lens.t_ic);
    let d: f64 = x.norm();
    if !(1.0..=12.0).contains(&d) {
        return None;
    }
    let b: Vector3<f64> = x / d;
    (b.z.clamp(-1.0, 1.0).acos() <= lens.half_fov).then_some(b)
}

fn perturb(rng: &mut Rng, b: &Vector3<f64>, on: bool) -> Vector3<f64> {
    if !on {
        return *b;
    }
    let (u1, u2) = tangent_basis(b);
    (b + u1 * (rng.normal() * BEARING_SIGMA) + u2 * (rng.normal() * BEARING_SIGMA)).normalize()
}

/// The bearing as the frontend reports it: rendered at the frame time plus
/// the time offset and the rolling-shutter row delay, then perturbed.
fn observe(
    s: &Scenario,
    path: &Path,
    lens: &Lens,
    point: &Vector3<f64>,
    t: f64,
) -> Option<Vector3<f64>> {
    let t0: f64 = t + s.time_offset_s;
    let b: Vector3<f64> = see(path, lens, point, t0)?;
    if s.rolling_shutter_s == 0.0 {
        return Some(b);
    }
    see(path, lens, point, t0 + s.rolling_shutter_s * b.y)
}

fn room_points(rng: &mut Rng, heading: f64) -> Vec<Vector3<f64>> {
    let (x0, x1, y0, y1, z0, z1) = (-5.0, 9.0, -4.0, 4.0, 0.0, 3.0);
    let faces: [(f64, usize); 6] = [
        ((x1 - x0) * (y1 - y0), 0), // floor
        ((x1 - x0) * (y1 - y0), 1), // ceiling
        ((x1 - x0) * (z1 - z0), 2), // y0 wall
        ((x1 - x0) * (z1 - z0), 3), // y1 wall
        ((y1 - y0) * (z1 - z0), 4), // x0 wall
        ((y1 - y0) * (z1 - z0), 5), // x1 wall
    ];
    let total: f64 = faces.iter().map(|f| f.0).sum();
    let h: Matrix3<f64> = rot_z(heading);
    (0..5000)
        .map(|_| {
            let mut pick: f64 = rng.uniform() * total;
            let mut face: usize = 5;
            for (area, id) in faces {
                if pick < area {
                    face = id;
                    break;
                }
                pick -= area;
            }
            let (x, y, z) = (rng.range(x0, x1), rng.range(y0, y1), rng.range(z0, z1));
            let p = match face {
                0 => Vector3::new(x, y, z0),
                1 => Vector3::new(x, y, z1),
                2 => Vector3::new(x, y0, z),
                3 => Vector3::new(x, y1, z),
                4 => Vector3::new(x0, y, z),
                _ => Vector3::new(x1, y, z),
            };
            h * p
        })
        .collect()
}

pub fn generate(s: &Scenario) -> Synthetic {
    let mut rng = Rng::new(s.seed);
    let path = Path {
        motion: s.motion,
        heading: rng.range(0.0, TAU),
        phase: std::array::from_fn(|_| rng.range(0.0, TAU)),
    };
    let lenses: Vec<Lens> = lenses(s.rig);
    let frames: usize = (s.window_s * CAMERA_HZ).round() as usize;
    let frame_times_ns: Vec<i64> = (0..frames)
        .map(|k| (k as f64 * 1e9 / CAMERA_HZ).round() as i64)
        .collect();
    let frame_t = |k: usize| frame_times_ns[k] as f64 * 1e-9;

    // IMU.
    let g_world: Vector3<f64> = Vector3::new(0.0, 0.0, -GRAVITY);
    let first: i64 = -100;
    let last: i64 = (s.window_s * IMU_HZ).round() as i64 + 100;
    let gyro_sigma: f64 = GYRO_DENSITY * IMU_HZ.sqrt();
    let accel_sigma: f64 = ACCEL_DENSITY * IMU_HZ.sqrt();
    let mut imu_rng = Rng::new(s.seed ^ 0x1A0);
    let imu: Vec<ImuSample> = (first..=last)
        .map(|i| {
            let t_ns: i64 = i * (1e9 / IMU_HZ) as i64;
            let t: f64 = t_ns as f64 * 1e-9;
            let (_, _, a) = path.translation(t);
            let r: Matrix3<f64> = path.rotation(t);
            let mut gyro: Vector3<f64> = path.omega(t) + s.gyro_bias;
            let mut accel: Vector3<f64> = r.transpose() * (a - g_world) + s.accel_bias;
            if s.imu_noise {
                gyro +=
                    Vector3::new(imu_rng.normal(), imu_rng.normal(), imu_rng.normal()) * gyro_sigma;
                accel += Vector3::new(imu_rng.normal(), imu_rng.normal(), imu_rng.normal())
                    * accel_sigma;
            }
            ImuSample { t_ns, gyro, accel }
        })
        .collect();

    // Tracks.
    let points: Vec<Vector3<f64>> = room_points(&mut rng, path.heading);
    let seeing: Vec<usize> = (0..lenses.len()).filter(|&c| !lenses[c].blind).collect();
    let span = |rng: &mut Rng| -> (usize, usize) {
        let start: usize = rng.index(frames.saturating_sub(10).max(1));
        let length: usize = 10 + rng.index(51);
        (start, (start + length).min(frames))
    };
    let mut tracks: Vec<FeatureTrack> = Vec::new();
    let mut attempts: usize = 0;
    while tracks.len() < s.inlier_tracks && attempts < 400_000 {
        attempts += 1;
        let point: &Vector3<f64> = &points[rng.index(points.len())];
        let cam: usize = seeing[rng.index(seeing.len())];
        let (start, end) = span(&mut rng);
        if !(start..end)
            .all(|k| see(&path, &lenses[cam], point, frame_t(k) + s.time_offset_s).is_some())
        {
            continue;
        }
        let mut observations: Vec<BearingObservation> = Vec::new();
        for k in start..end {
            // The other lens reports the same id where the point is in both
            // (the overlap band past 90 degrees).
            for &c in &seeing {
                if let Some(b) = observe(s, &path, &lenses[c], point, frame_t(k)) {
                    let bearing = perturb(&mut rng, &b, s.bearing_noise);
                    observations.push(BearingObservation {
                        frame: k,
                        camera: c,
                        bearing,
                    });
                }
            }
        }
        if observations.iter().filter(|o| o.camera == cam).count() < end - start {
            continue;
        }
        tracks.push(FeatureTrack {
            id: tracks.len() as u64,
            observations,
        });
    }
    let outliers: usize =
        (s.inlier_tracks as f64 * s.outlier_fraction / (1.0 - s.outlier_fraction)).round() as usize;
    for _ in 0..outliers {
        let cam: usize = seeing[rng.index(seeing.len())];
        let (start, end) = span(&mut rng);
        let bearing: Vector3<f64> = loop {
            let b = rng.unit_vector();
            if b.z.acos() <= 0.9 * lenses[cam].half_fov {
                break b;
            }
        };
        let observations: Vec<BearingObservation> = (start..end)
            .map(|k| BearingObservation {
                frame: k,
                camera: cam,
                bearing: perturb(&mut rng, &bearing, s.bearing_noise),
            })
            .collect();
        tracks.push(FeatureTrack {
            id: tracks.len() as u64,
            observations,
        });
    }

    let t_i_c: Vec<RigidTransform<f64>> = lenses
        .iter()
        .map(|l| {
            RigidTransform::new(
                Rotation3::from_unit_quaternion(UnitQuaternion::from_rotation_matrix(
                    &nalgebra::Rotation3::from_matrix_unchecked(l.r_ic),
                )),
                l.t_ic + s.lever_arm_error,
            )
        })
        .collect();

    let r0: Matrix3<f64> = path.rotation(0.0);
    let (p0, v0, _) = path.translation(0.0);
    let truth = Truth {
        velocity_body0: r0.transpose() * v0,
        gravity_body0: r0.transpose() * g_world,
        positions_body0: (0..frames)
            .map(|k| r0.transpose() * (path.translation(frame_t(k)).0 - p0))
            .collect(),
        bias_gyro: s.gyro_bias,
        bias_accel: s.accel_bias,
    };
    Synthetic {
        imu,
        frame_times_ns,
        tracks,
        t_i_c,
        truth,
    }
}
