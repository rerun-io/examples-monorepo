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

/// Renormalised temporal opacity `max((gamma - c) / (1 - c), 0)`, `gamma = exp(-0.5 (dt / sigma)^2)`, cut `c`
/// clamped to [0, 0.999] (0 = off). Shared by both motion forms.
fn temporal_gamma(dt: f32, sigma: f32, threshold: f32) -> f32 {
    let ratio = dt / sigma;
    let mut gamma = (-0.5 * (ratio * ratio)).exp();
    let threshold = threshold.clamp(0.0, 0.999);
    if threshold > 0.0 {
        gamma = ((gamma - threshold) / (1.0 - threshold)).max(0.0);
    }
    gamma
}

/// Bezier parameter over the row's life `[mu - 2 sigma, mu + 2 sigma]`, clamped to [0, 1]. Beyond 2 sigma the
/// curves freeze at their end points while `dt` keeps growing (a 0.05 cut sits at 2.4477 sigma).
fn life_tau(t: f32, mu: f32, sigma: f32) -> f32 {
    ((t - (mu - 2.0 * sigma)) / (4.0 * sigma)).clamp(0.0, 1.0)
}

/// Quadratic Bernstein weights at `tau`.
fn bezier2_weights(tau: f32) -> [f32; 3] {
    let one_minus = 1.0 - tau;
    [one_minus * one_minus, 2.0 * one_minus * tau, tau * tau]
}

/// Gauge fix: the anchor without its component along omega (unchanged when `|omega|^2 < 1e-8`).
fn perp_anchor(anchor: [f32; 3], omega: [f32; 3], omega_sq: f32) -> [f32; 3] {
    if omega_sq < GAUGE_EPS {
        return anchor;
    }
    let ka = dot(anchor, omega) / omega_sq.max(GAUGE_DENOM_CLAMP);
    std::array::from_fn(|k| anchor[k] - ka * omega[k])
}

/// Screw motion of `center` about `a_perp` by the twist (`nu_par`, `omega`) over `dt` seconds: the moved centre and
/// the relative rotation `q_rel` (unit wxyz) to pre-multiply onto the canonical rotation.
fn screw(
    center: [f32; 3],
    a_perp: [f32; 3],
    nu_par: [f32; 3],
    omega: [f32; 3],
    dt: f32,
) -> ([f32; 3], [f32; 4]) {
    let clamp = |v: f32| v.clamp(-1e4, 1e4);
    let rho = nu_par.map(|v| clamp(v * dt));
    let omega_dt = omega.map(|v| clamp(v * dt));
    let (q_rel, p_par) = exp_se3_quat(rho, omega_dt);
    let centered = std::array::from_fn(|k| center[k] - a_perp[k]);
    let rotated = rotate(q_rel, centered);
    let moved = std::array::from_fn(|k| finite(rotated[k] + a_perp[k] + p_par[k]));
    (moved, q_rel)
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
        temporal_gamma(
            t - self.times[i],
            self.log_durations[i].exp().max(1e-6),
            self.active_threshold,
        )
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
        let [b0, b1, b2] = bezier2_weights(life_tau(t, mu, sigma));
        let twist: [f32; 6] = std::array::from_fn(|k| {
            let residual =
                b0 * self.twist_c0[i][k] + b1 * self.twist_c1[i][k] + b2 * self.twist_c2[i][k];
            self.twist_base[i][k] + residual
        });
        let nu = [twist[0], twist[1], twist[2]];
        let omega = [twist[3], twist[4], twist[5]];
        let anchor = self.anchors[i];
        let omega_sq = dot(omega, omega);
        let a_perp = perp_anchor(anchor, omega, omega_sq);
        let nu_par = if omega_sq < GAUGE_EPS {
            nu
        } else {
            let kn = dot(nu, omega) / omega_sq.max(GAUGE_DENOM_CLAMP);
            omega.map(|w| kn * w)
        };
        let (moved, q_rel) = screw(center, a_perp, nu_par, omega, t - mu);
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

// ---------------------------------------------------------------------------------------------------------------
// The `.asog` form (animated-sog v1, format.md section 4; reference `asog.decode.Asog.evaluate` and `asog.motion`).
//
// Same model as the twist form, parameterised differently: the rotation twist is the quadratic Bezier of O0..O2
// (the base folded in), the translation is carried only along omega as the quartic Bernstein `g` of G0..G4, so
// `nu_par = g / max(|omega|^2, 1e-24) omega` (no 1e-8 switch for nu, unlike the twist form). Clips add a fade weight
// and a clip-local time. Static rows keep their canonical pose with weight = the clip fade.

/// One clip of the manifest's clip table, in seconds on the file's timeline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipWindow {
    /// Timeline time minus clip-local time: `(global.frameStart - local.frameStart) / frameRate`.
    pub offset: f32,
    pub active_start: f32,
    pub active_end: f32,
    pub core_start: f32,
    pub core_end: f32,
    /// Fade lengths (0 = none).
    pub fade_in: f32,
    pub fade_out: f32,
}

impl ClipWindow {
    /// Fade weight at timeline time `t`: 0 outside `[active_start, active_end)`, smoothstep ramps before
    /// `core_start` and from `core_end` on (reference `Asog.clip_weights`).
    pub fn fade(&self, t: f32) -> f32 {
        let smoothstep = |x: f32| {
            let x = x.clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        if !(self.active_start <= t && t < self.active_end) {
            0.0
        } else if self.fade_in > 0.0 && t < self.core_start {
            smoothstep((t - (self.core_start - self.fade_in)) / self.fade_in)
        } else if self.fade_out > 0.0 && t >= self.core_end {
            smoothstep((self.core_end + self.fade_out - t) / self.fade_out)
        } else {
            1.0
        }
    }
}

/// Per-row `.asog` motion, de-quantised (owned: `crate::asog::decode` builds it). Static rows hold zeros.
#[derive(Clone, Debug, Default)]
pub struct AsogMotion {
    /// Linear base opacity (q / 255).
    pub opacities: Vec<f32>,
    /// Clip-local centre time `mu` in seconds.
    pub times: Vec<f32>,
    /// Temporal width `sigma` in seconds.
    pub sigmas: Vec<f32>,
    pub anchors: Vec<[f32; 3]>,
    /// Rotation-twist control points O0, O1, O2 (rad/s).
    pub omega: Vec<[[f32; 3]; 3]>,
    /// Quartic Bernstein control points G0..G4 of `g = nu . omega`.
    pub g: Vec<[f32; 5]>,
    pub is_static: Vec<bool>,
    /// Index into `clips`.
    pub clip: Vec<u32>,
    pub clips: Vec<ClipWindow>,
    /// The temporal cut (`AnimatedSogSplats3D:active_threshold`; the format's constant is 0.05).
    pub active_threshold: f32,
}

/// One evaluated `.asog` row: centre, unit rotation wxyz (w >= 0) and the opacity factor (clip fade x temporal
/// weight; 0 = not drawn, the row keeps its canonical pose).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AsogRow {
    pub center: [f32; 3],
    pub quaternion: [f32; 4],
    pub weight: f32,
}

/// Default `.asog` temporal cut (animated-sog v1 `TEMPORAL_THRESHOLD`; no manifest field).
pub const ASOG_ACTIVE_THRESHOLD: f32 = 0.05;

/// Raw logit opacity of a row that is not drawn.
pub const CULLED_RAW_OPACITY: f32 = -100.0;

impl AsogMotion {
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// (fade, clip-local time) of every clip at timeline time `t`.
    pub fn clip_states(&self, t: f32) -> Vec<(f32, f32)> {
        self.clips
            .iter()
            .map(|clip| (clip.fade(t), t - clip.offset))
            .collect()
    }

    /// Row `i` from its canonical transform (gsplat-core layout: mean, quat wxyz, log scale) and the clip states.
    pub fn row(&self, i: usize, canonical: &[f32; 10], clips: &[(f32, f32)]) -> AsogRow {
        let center = [canonical[0], canonical[1], canonical[2]];
        let quaternion = [canonical[3], canonical[4], canonical[5], canonical[6]];
        let (fade, local_t) = clips
            .get(self.clip[i] as usize)
            .copied()
            .unwrap_or((0.0, 0.0));
        let still = |weight| AsogRow {
            center,
            quaternion,
            weight,
        };
        if self.is_static[i] {
            return still(fade);
        }
        let (mu, sigma) = (self.times[i], self.sigmas[i]);
        let dt = local_t - mu;
        let weight = fade * temporal_gamma(dt, sigma, self.active_threshold);
        if weight <= 0.0 {
            return still(weight);
        }
        let tau = life_tau(local_t, mu, sigma);
        let [b0, b1, b2] = bezier2_weights(tau);
        let [o0, o1, o2] = self.omega[i];
        let omega: [f32; 3] = std::array::from_fn(|k| b0 * o0[k] + b1 * o1[k] + b2 * o2[k]);
        let u = 1.0 - tau;
        let (uu, tt) = (u * u, tau * tau);
        let quartic = [
            uu * uu,
            4.0 * uu * u * tau,
            6.0 * uu * tt,
            4.0 * u * tt * tau,
            tt * tt,
        ];
        let g: f32 = quartic.iter().zip(&self.g[i]).map(|(b, c)| b * c).sum();
        let omega_sq = dot(omega, omega);
        let k = g / omega_sq.max(1e-24);
        let nu_par = omega.map(|w| k * w);
        let a_perp = perp_anchor(self.anchors[i], omega, omega_sq);
        let (moved, q_rel) = screw(center, a_perp, nu_par, omega, dt);
        let q = finite_quat(normalize4(quat_mul(q_rel, quaternion)));
        let sign = if q[0] < 0.0 { -1.0 } else { 1.0 };
        AsogRow {
            center: moved,
            quaternion: q.map(|v| v * sign),
            weight,
        }
    }

    /// The splats at timeline time `t` into `out` (gsplat-core layout, f32): transforms and raw opacities are
    /// rewritten for every row (rows with weight 0 get [`CULLED_RAW_OPACITY`]); scales and SH come from `canonical`
    /// and are copied only when `out` does not hold them yet. The CPU oracle of the motion pass.
    #[cfg(any(test, feature = "testing"))]
    pub fn evaluate_into(&self, canonical: &crate::Splats, t: f32, out: &mut crate::Splats) {
        let clips = self.clip_states(t);
        if out.sh_coefficients.len() != canonical.sh_coefficients.len()
            || out.sh_degree != canonical.sh_degree
        {
            out.sh_coefficients.clone_from(&canonical.sh_coefficients);
            out.sh_degree = canonical.sh_degree;
        }
        out.min_scale.clone_from(&canonical.min_scale);
        out.transforms.clear();
        out.raw_opacities.clear();
        for (i, transform) in canonical.transforms.iter().enumerate() {
            let row = self.row(i, transform, &clips);
            let [x, y, z] = row.center;
            let [qw, qx, qy, qz] = row.quaternion;
            out.transforms.push([
                x,
                y,
                z,
                qw,
                qx,
                qy,
                qz,
                transform[7],
                transform[8],
                transform[9],
            ]);
            out.raw_opacities.push(if row.weight > 0.0 {
                crate::sog::logit(f64::from(self.opacities[i] * row.weight))
            } else {
                CULLED_RAW_OPACITY
            });
        }
    }

    /// [`Self::evaluate_into`] into a new [`Splats`].
    #[cfg(any(test, feature = "testing"))]
    pub fn evaluate(&self, canonical: &crate::Splats, t: f32) -> crate::Splats {
        let mut out = crate::Splats::default();
        self.evaluate_into(canonical, t, &mut out);
        out
    }
}
