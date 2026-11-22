//! CPU reference of the TRiGS motion model (`open4dgs-trigs` ed7a84a `TrigsTrainer._deform`, `math/se3.py`).
//!
//! Each row is a canonical Gaussian (centre, rotation) that lives around its centre time `mu` and moves rigidly:
//! a se(3) twist (base + quadratic Bezier residual over the row's window `[mu - 2 sigma, mu + 2 sigma]`), gauge-fixed
//! about the row's anchor, integrated over `dt = t - mu` with the SE(3) exponential. Its temporal opacity is
//! `gamma = exp(-0.5 (dt / sigma)^2)`, renormalised by the smooth cull threshold; static rows keep gamma = 1.
//! The operation order follows the torch code in f32 so results match it to float rounding.

/// Per-row motion parameters, borrowed from the `TrigsMotion3D` components (all slices have the row count).
#[derive(Clone, Copy)]
pub struct TrigsMotion<'a> {
    /// sigmoid(opacity logit).
    pub opacities: &'a [f32],
    /// Centre time `mu` in seconds.
    pub times: &'a [f32],
    /// log sigma in seconds.
    pub log_durations: &'a [f32],
    /// (nu xyz, omega xyz) base twist and the three Bezier control points.
    pub twist_base: &'a [[f32; 6]],
    pub twist_c0: &'a [[f32; 6]],
    pub twist_c1: &'a [[f32; 6]],
    pub twist_c2: &'a [[f32; 6]],
    pub anchors: &'a [[f32; 3]],
    /// > 0.5 marks a static row; may be empty (no static rows).
    pub is_static: &'a [f32],
    /// Smooth temporal cull threshold (TRiGS `temporal_active_threshold`, 0 = off).
    pub active_threshold: f32,
}

/// One evaluated row: centre, rotation xyzw (Rerun order) and opacity = base opacity x gamma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Deformed {
    pub center: [f32; 3],
    pub quaternion: [f32; 4],
    pub opacity: f32,
}

const GAUGE_EPS: f32 = 1e-8;
const GAUGE_DENOM_CLAMP: f32 = 1e-10;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// F.normalize: q / max(|q|, 1e-12).
fn normalize4(q: [f32; 4]) -> [f32; 4] {
    let n = dot([q[0], q[1], q[2]], [q[0], q[1], q[2]]) + q[3] * q[3];
    let n = n.sqrt().max(1e-12);
    q.map(|v| v / n)
}

/// Hamilton product of wxyz quaternions.
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [w1, x1, y1, z1] = a;
    let [w2, x2, y2, z2] = b;
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [w, x, y, z] = normalize4(q);
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    let r = [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ];
    r.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
}

/// `exp_se3_quat`: twist (rho, omega) -> (unit wxyz quaternion, translation).
fn exp_se3_quat(rho: [f32; 3], omega: [f32; 3]) -> ([f32; 4], [f32; 3]) {
    let theta = dot(omega, omega).sqrt();
    let theta_sq = theta * theta;
    let small = theta_sq < 1e-8;
    let theta_safe = if small { 1.0 } else { theta };
    let theta_sq_safe = theta_safe * theta_safe;
    let theta_cb_safe = theta_sq_safe * theta_safe;
    let half = theta_safe * 0.5;
    let q_w = if small {
        1.0 - theta_sq / 8.0
    } else {
        half.cos()
    };
    let factor = if small {
        0.5 - theta_sq / 48.0
    } else {
        half.sin() / theta_safe
    };
    let (b, c) = if small {
        (0.5 - theta_sq / 24.0, 1.0 / 6.0 - theta_sq / 120.0)
    } else {
        (
            (1.0 - theta_safe.cos()) / theta_sq_safe,
            (theta_safe - theta_safe.sin()) / theta_cb_safe,
        )
    };
    let wxr = cross(omega, rho);
    let wxwxr = cross(omega, wxr);
    let p = std::array::from_fn(|i| rho[i] + b * wxr[i] + c * wxwxr[i]);
    let q = normalize4([q_w, omega[0] * factor, omega[1] * factor, omega[2] * factor]);
    (q, p)
}

impl TrigsMotion<'_> {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Whether every per-row slice has `rows` entries (`is_static` may also be empty).
    pub fn is_consistent(&self, rows: usize) -> bool {
        [
            self.opacities.len(),
            self.times.len(),
            self.log_durations.len(),
            self.twist_base.len(),
            self.twist_c0.len(),
            self.twist_c1.len(),
            self.twist_c2.len(),
            self.anchors.len(),
        ]
        .iter()
        .all(|&n| n == rows)
            && (self.is_static.is_empty() || self.is_static.len() == rows)
    }

    fn row_static(&self, i: usize) -> bool {
        self.is_static.get(i).is_some_and(|&s| s > 0.5)
    }

    /// Temporal opacity factor of row `i` at `t` seconds (after the smooth cull; 1 for static rows).
    pub fn gamma(&self, i: usize, t: f32) -> f32 {
        if self.row_static(i) {
            return 1.0;
        }
        let dt = t - self.times[i];
        let sigma = self.log_durations[i].exp().max(1e-6);
        let ratio = dt / sigma;
        let mut gamma = (-0.5 * (ratio * ratio)).exp();
        let threshold = self.active_threshold.clamp(0.0, 0.999);
        if threshold > 0.0 {
            gamma = ((gamma - threshold) / (1.0 - threshold)).max(0.0);
        }
        gamma
    }

    /// Row `i` at `t` seconds, from its canonical centre and xyzw rotation.
    pub fn deform_row(
        &self,
        i: usize,
        center: [f32; 3],
        quaternion_xyzw: [f32; 4],
        t: f32,
    ) -> Deformed {
        let [qx, qy, qz, qw] = quaternion_xyzw;
        let canonical = normalize4([qw, qx, qy, qz]);
        let gamma = self.gamma(i, t);
        let opacity = self.opacities[i] * gamma;
        if self.row_static(i) {
            let [w, x, y, z] = finite_quat(canonical);
            return Deformed {
                center: center.map(finite),
                quaternion: [x, y, z, w],
                opacity,
            };
        }
        let mu = self.times[i];
        let sigma = self.log_durations[i].exp();
        let tau = ((t - (mu - 2.0 * sigma)) / (4.0 * sigma)).clamp(0.0, 1.0);
        let one_minus = 1.0 - tau;
        let (b0, b1, b2) = (one_minus * one_minus, 2.0 * one_minus * tau, tau * tau);
        let twist: [f32; 6] = std::array::from_fn(|k| {
            let residual =
                b0 * self.twist_c0[i][k] + b1 * self.twist_c1[i][k] + b2 * self.twist_c2[i][k];
            self.twist_base[i][k] + residual
        });
        let nu = [twist[0], twist[1], twist[2]];
        let omega = [twist[3], twist[4], twist[5]];
        let anchor = self.anchors[i];
        let omega_sq = dot(omega, omega);
        let (a_perp, nu_par) = if omega_sq < GAUGE_EPS {
            (anchor, nu)
        } else {
            let safe = omega_sq.max(GAUGE_DENOM_CLAMP);
            let ka = dot(anchor, omega) / safe;
            let kn = dot(nu, omega) / safe;
            (
                std::array::from_fn(|k| anchor[k] - ka * omega[k]),
                omega.map(|w| kn * w),
            )
        };
        let dt = t - mu;
        let clamp = |v: f32| v.clamp(-1e4, 1e4);
        let rho = nu_par.map(|v| clamp(v * dt));
        let omega_dt = omega.map(|v| clamp(v * dt));
        let (q_rel, p_par) = exp_se3_quat(rho, omega_dt);
        let centered = std::array::from_fn(|k| center[k] - a_perp[k]);
        let rotated = rotate(q_rel, centered);
        let moved: [f32; 3] = std::array::from_fn(|k| finite(rotated[k] + a_perp[k] + p_par[k]));
        let [w, x, y, z] = finite_quat(quat_mul(q_rel, canonical));
        Deformed {
            center: moved,
            quaternion: [x, y, z, w],
            opacity,
        }
    }
}

/// `nan_to_num(nan=0, inf=0)`.
fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// `nan_to_num`, then identity for (near) zero quaternions.
fn finite_quat(q: [f32; 4]) -> [f32; 4] {
    let q = q.map(finite);
    if q.iter().map(|v| v * v).sum::<f32>() < 1e-16 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        q
    }
}

/// The visible rows of a clip at `t` seconds: indices with opacity above `min_opacity` and their values.
///
/// Gamma is evaluated first, so the twist is integrated only for rows that are alive at `t`.
pub fn deform_visible(
    centers: &[[f32; 3]],
    quaternions_xyzw: &[[f32; 4]],
    motion: &TrigsMotion<'_>,
    t: f32,
    min_opacity: f32,
) -> (Vec<u32>, Vec<Deformed>) {
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for (i, &center) in centers.iter().enumerate() {
        if motion.opacities[i] * motion.gamma(i, t) <= min_opacity {
            continue;
        }
        let q = crate::native::at_or_last(quaternions_xyzw, i, [0.0, 0.0, 0.0, 1.0]);
        rows.push(i as u32);
        values.push(motion.deform_row(i, center, q, t));
    }
    (rows, values)
}
