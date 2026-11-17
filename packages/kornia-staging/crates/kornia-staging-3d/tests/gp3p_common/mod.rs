//! Synthetic multi-camera rigs for the GP3P tests and benches.
#![allow(dead_code)]
use kornia_staging_3d::pose::gp3p::{GeneralizedCorrespondence, Gp3pPose};
use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use nalgebra::{Matrix3, Vector3};

/// xorshift64*: a tiny deterministic generator, so scenes need no RNG crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn unit_vector(&mut self) -> Vector3<f64> {
        loop {
            let v = Vector3::new(
                self.uniform(-1.0, 1.0),
                self.uniform(-1.0, 1.0),
                self.uniform(-1.0, 1.0),
            );
            let n = v.norm();
            if n > 0.1 && n <= 1.0 {
                return v / n;
            }
        }
    }

    pub fn rotation(&mut self, max_angle: f64) -> Rotation3<f64> {
        let axis = self.unit_vector();
        Rotation3::exp(&(axis * self.uniform(0.0, max_angle)))
    }
}

/// A rig of `cameras` cameras around the body, looking outward (yaw spread),
/// with centres `offset` metres from the body origin (0 = central rig).
pub fn rig(rng: &mut Rng, cameras: usize, offset: f64) -> Vec<RigidTransform<f64>> {
    (0..cameras)
        .map(|i| {
            let yaw = i as f64 * std::f64::consts::TAU / cameras as f64 * 0.5;
            let rotation = Rotation3::exp(&Vector3::new(0.0, yaw, 0.0)) * rng.rotation(0.1);
            let translation = if offset == 0.0 {
                Vector3::zeros()
            } else {
                rng.unit_vector() * offset
            };
            RigidTransform::new(rotation, translation)
        })
        .collect()
}

pub fn random_pose(rng: &mut Rng) -> RigidTransform<f64> {
    RigidTransform::new(
        rng.rotation(std::f64::consts::PI),
        Vector3::new(
            rng.uniform(-5.0, 5.0),
            rng.uniform(-5.0, 5.0),
            rng.uniform(-2.0, 2.0),
        ),
    )
}

/// One correspondence seen by camera `cam`, 1-8 m deep within a 45° cone.
/// `noise` perturbs the bearing by about that many radians.
pub fn observe(
    rng: &mut Rng,
    rig: &[RigidTransform<f64>],
    cam: usize,
    t_w_b: &RigidTransform<f64>,
    noise: f64,
) -> GeneralizedCorrespondence {
    let direction = loop {
        let d = rng.unit_vector();
        if d.z > 0.7 {
            break d;
        }
    };
    let point_c = direction * rng.uniform(1.0, 8.0);
    let point_w = *t_w_b * (rig[cam] * point_c);
    let bearing = if noise > 0.0 {
        (direction + rng.unit_vector() * rng.uniform(0.0, noise)).normalize()
    } else {
        direction
    };
    GeneralizedCorrespondence::from_camera(&rig[cam], &bearing, &point_w)
}

/// An outlier: a valid bearing paired with an unrelated world point.
pub fn outlier(
    rng: &mut Rng,
    rig: &[RigidTransform<f64>],
    t_w_b: &RigidTransform<f64>,
) -> GeneralizedCorrespondence {
    let cam = rng.below(rig.len());
    let mut c = observe(rng, rig, cam, t_w_b, 0.0);
    c.bearing = rig[cam].rotation * rng.unit_vector();
    c
}

/// `n` correspondences over every camera, the first `round(n * inlier_ratio)`
/// inliers (bearing noise `noise`), the rest outliers; then shuffled.
/// Returns the correspondences and the inlier mask.
pub fn scene(
    rng: &mut Rng,
    rig: &[RigidTransform<f64>],
    t_w_b: &RigidTransform<f64>,
    n: usize,
    inlier_ratio: f64,
    noise: f64,
) -> (Vec<GeneralizedCorrespondence>, Vec<bool>) {
    let inliers = (n as f64 * inlier_ratio).round() as usize;
    let mut items: Vec<(GeneralizedCorrespondence, bool)> = (0..n)
        .map(|i| {
            if i < inliers {
                (observe(rng, rig, i % rig.len(), t_w_b, noise), true)
            } else {
                (outlier(rng, rig, t_w_b), false)
            }
        })
        .collect();
    for i in (1..items.len()).rev() {
        items.swap(i, rng.below(i + 1));
    }
    items.into_iter().unzip()
}

/// Rotation error in radians and translation error in metres.
pub fn pose_error(estimate: &Gp3pPose, truth: &RigidTransform<f64>) -> (f64, f64) {
    // |R1 - R2|_F = 2 sqrt(2) sin(θ/2): accurate for small angles, unlike acos.
    let d: Matrix3<f64> = estimate.rotation - truth.rotation.matrix();
    let angle = 2.0
        * (d.norm() / (2.0 * std::f64::consts::SQRT_2))
            .min(1.0)
            .asin();
    (angle, (estimate.translation - truth.translation).norm())
}
