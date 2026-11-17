//! The closed-form stage: per-track point elimination, robust consensus over
//! tracks, and the gravity-norm-constrained least-squares solve.
//!
//! Every observation of a point `X` (body-0 frame) in frame `k` through
//! camera `c` says that the ray direction `f = R_k R_ic b` (body-0 frame)
//! passes through `X` from the camera centre `c_k = p_k + R_k t_ic`, with
//! `p_k = v0 dt_k + 1/2 g dt_k^2 + dp_k`. With `u1, u2` an orthonormal basis of
//! the plane normal to `f`, the two equations
//!
//! ```text
//! u^T X - dt_k u^T v0 - 1/2 dt_k^2 u^T g = u^T (dp_k + R_k t_ic)
//! ```
//!
//! are linear in `(X, v0, g)` (Martinelli, "Closed-form solution of
//! visual-inertial structure from motion", IJCV 2014; Kaiser, Martinelli,
//! Fontana, Scaramuzza, "Simultaneous state initialization and gyroscope bias
//! calibration in visual inertial aided navigation", RA-L 2017). They are the
//! two independent rows of `b x (R_ic^T (R_k^T (X - p_k) - t_ic)) = 0`
//! rotated into the body-0 frame, scaled so that each row is a metric
//! distance from the ray instead of the cross product's `|x| sin(angle)`.
//!
//! The point is eliminated per track by a Householder QR of the track's
//! `[A_X | A_y | rhs]` rows: the rows below the first three of the triangular
//! factor are the track's reduced system in `y = (v0, g)` (the square-root
//! form of the Schur complement, as in square-root bundle adjustment,
//! Demmel et al. CVPR 2021), and the first three rows triangulate the point
//! for any `y`.
use super::preintegration::Preintegrated;
use super::BearingObservation;
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3, Vector6};

/// Width of a track row: point (3), velocity (3), gravity (3), right-hand side (1).
const ROW_WIDTH: usize = 10;

/// One camera of the rig as the solver uses it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Camera {
    /// Camera-to-body rotation `R_ic`.
    pub r_ic: Matrix3<f64>,
    /// Camera centre in the body frame `t_ic`.
    pub t_ic: Vector3<f64>,
}

/// A track's reduced linear system.
#[derive(Debug, Clone)]
pub(super) struct TrackBlock {
    r11: Matrix3<f64>,
    r12: SMatrix<f64, 3, 6>,
    q1: Vector3<f64>,
    /// Reduced rows in `y = (v0, g)`.
    pub r22: SMatrix<f64, 6, 6>,
    /// Reduced right-hand side.
    pub d2: Vector6<f64>,
}

impl TrackBlock {
    /// The point that best explains the track for `y = (v0, g)`.
    pub fn triangulate(&self, y: &Vector6<f64>) -> Option<Vector3<f64>> {
        let rhs: Vector3<f64> = self.q1 - self.r12 * y;
        self.r11.solve_upper_triangular(&rhs)
    }
}

/// An orthonormal pair spanning the plane normal to the unit vector `f`.
pub(super) fn tangent_basis(f: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    // The canonical axis least aligned with f, made orthogonal to it.
    let axis: Vector3<f64> = if f.x.abs() <= f.y.abs() && f.x.abs() <= f.z.abs() {
        Vector3::x()
    } else if f.y.abs() <= f.z.abs() {
        Vector3::y()
    } else {
        Vector3::z()
    };
    let u1: Vector3<f64> = (axis - f * f.dot(&axis)).normalize();
    let u2: Vector3<f64> = f.cross(&u1);
    (u1, u2)
}

/// Build a track's reduced system from the preintegrated window at its
/// linearisation biases. The track must have at least five observations.
pub(super) fn track_block(
    observations: &[BearingObservation],
    preint: &Preintegrated,
    cameras: &[Camera],
) -> Option<TrackBlock> {
    let rows: usize = 2 * observations.len();
    if rows < ROW_WIDTH {
        return None;
    }
    let mut m: DMatrix<f64> = DMatrix::zeros(rows, ROW_WIDTH);
    for (i, obs) in observations.iter().enumerate() {
        let cam: &Camera = &cameras[obs.camera];
        let r_k: &Matrix3<f64> = &preint.rotation[obs.frame];
        let dt: f64 = preint.dt[obs.frame];
        let f: Vector3<f64> = (r_k * cam.r_ic * obs.bearing).normalize();
        let offset: Vector3<f64> = preint.position[obs.frame] + r_k * cam.t_ic;
        let (u1, u2) = tangent_basis(&f);
        for (j, u) in [u1, u2].iter().enumerate() {
            let row: usize = 2 * i + j;
            for a in 0..3 {
                m[(row, a)] = u[a];
                m[(row, 3 + a)] = -dt * u[a];
                m[(row, 6 + a)] = -0.5 * dt * dt * u[a];
            }
            m[(row, 9)] = u.dot(&offset);
        }
    }
    let r: DMatrix<f64> = m.qr().r();
    Some(TrackBlock {
        r11: r.fixed_view::<3, 3>(0, 0).into_owned(),
        r12: r.fixed_view::<3, 6>(0, 3).into_owned(),
        q1: r.fixed_view::<3, 1>(0, 9).into_owned(),
        r22: r.fixed_view::<6, 6>(3, 3).into_owned(),
        d2: r.fixed_view::<6, 1>(3, 9).into_owned(),
    })
}

/// Stack the reduced systems of the chosen tracks.
fn stack<'a>(blocks: impl Iterator<Item = &'a TrackBlock>) -> (DMatrix<f64>, DVector<f64>) {
    let blocks: Vec<&TrackBlock> = blocks.collect();
    let mut m: DMatrix<f64> = DMatrix::zeros(6 * blocks.len(), 6);
    let mut d: DVector<f64> = DVector::zeros(6 * blocks.len());
    for (i, b) in blocks.iter().enumerate() {
        m.view_mut((6 * i, 0), (6, 6)).copy_from(&b.r22);
        d.rows_mut(6 * i, 6).copy_from(&b.d2);
    }
    (m, d)
}

/// The unconstrained least-squares `(v0, g)` and the condition number of the
/// column-equilibrated reduced system.
pub(super) fn solve_unconstrained<'a>(
    blocks: impl Iterator<Item = &'a TrackBlock>,
) -> Option<(Vector6<f64>, f64)> {
    let (m, d) = stack(blocks);
    if m.nrows() < 6 {
        return None;
    }
    let mut scaled: DMatrix<f64> = m.clone();
    let mut scale: Vector6<f64> = Vector6::zeros();
    for c in 0..6 {
        let norm: f64 = scaled.column(c).norm();
        if norm.is_nan() || norm <= 0.0 {
            return None;
        }
        scale[c] = 1.0 / norm;
        scaled.column_mut(c).scale_mut(scale[c]);
    }
    let svd = scaled.svd(true, true);
    let s_max: f64 = svd.singular_values.max();
    let s_min: f64 = svd.singular_values.min();
    let condition: f64 = if s_min > 0.0 {
        s_max / s_min
    } else {
        f64::INFINITY
    };
    let z: DVector<f64> = svd.solve(&d, 0.0).ok()?;
    let y: Vector6<f64> = Vector6::from_fn(|i, _| z[i] * scale[i]);
    Some((y, condition))
}

/// The least-squares `(v0, g)` subject to `|g| = gravity_magnitude`.
///
/// `v0` is eliminated by QR, leaving `min |H g - e|^2` on the sphere. Its
/// global minimum is the root of the secular equation
/// `sum (s_i c_i / (s_i^2 + l))^2 = G^2` with `l > -s_min^2` (Gander,
/// "Least squares with a quadratic constraint", Numer. Math. 1981), found by
/// bisection so the result is deterministic and needs no starting point.
pub(super) fn solve_constrained<'a>(
    blocks: impl Iterator<Item = &'a TrackBlock>,
    gravity_magnitude: f64,
) -> Option<Vector6<f64>> {
    let (m, d) = stack(blocks);
    if m.nrows() < 7 {
        return None;
    }
    let mut aug: DMatrix<f64> = DMatrix::zeros(m.nrows(), 7);
    aug.view_mut((0, 0), (m.nrows(), 6)).copy_from(&m);
    aug.set_column(6, &d);
    let r: DMatrix<f64> = aug.qr().r();
    let r_vv: Matrix3<f64> = r.fixed_view::<3, 3>(0, 0).into_owned();
    let r_vg: Matrix3<f64> = r.fixed_view::<3, 3>(0, 3).into_owned();
    let q_v: Vector3<f64> = r.fixed_view::<3, 1>(0, 6).into_owned();
    let h: Matrix3<f64> = r.fixed_view::<3, 3>(3, 3).into_owned();
    let e: Vector3<f64> = r.fixed_view::<3, 1>(3, 6).into_owned();

    let g: Vector3<f64> = gravity_on_sphere(&h, &e, gravity_magnitude)?;
    let v: Vector3<f64> = r_vv.solve_upper_triangular(&(q_v - r_vg * g))?;
    let y: Vector6<f64> = Vector6::new(v.x, v.y, v.z, g.x, g.y, g.z);
    y.iter().all(|x| x.is_finite()).then_some(y)
}

/// `argmin |H g - e|` subject to `|g| = norm`.
fn gravity_on_sphere(h: &Matrix3<f64>, e: &Vector3<f64>, norm: f64) -> Option<Vector3<f64>> {
    let svd = h.svd(true, true);
    let u: Matrix3<f64> = svd.u?;
    let v_t: Matrix3<f64> = svd.v_t?;
    let s: Vector3<f64> = svd.singular_values;
    let c: Vector3<f64> = u.transpose() * e;
    let sc: Vector3<f64> = s.component_mul(&c);
    let s2: Vector3<f64> = s.component_mul(&s);
    let i_min: usize = s2.imin();
    let lo0: f64 = -s2[i_min];
    let phi = |l: f64| -> f64 { (0..3).map(|i| (sc[i] / (s2[i] + l)).powi(2)).sum() };
    let target: f64 = norm * norm;

    // Hard case: the minimum direction carries no data, so phi stays finite
    // at the pole; the remainder goes along that direction.
    let rest: f64 = (0..3)
        .filter(|&i| i != i_min)
        .map(|i| {
            let den: f64 = s2[i] + lo0;
            if den > 0.0 {
                (sc[i] / den).powi(2)
            } else {
                f64::INFINITY
            }
        })
        .sum();
    let scale: f64 = s2.max().max(f64::MIN_POSITIVE);
    if sc[i_min].abs() <= 1e-12 * scale.sqrt() * c.norm().max(1e-300) && rest < target {
        let mut z: Vector3<f64> = Vector3::zeros();
        for i in 0..3 {
            if i != i_min {
                z[i] = sc[i] / (s2[i] + lo0);
            }
        }
        z[i_min] = (target - rest).sqrt();
        return Some(v_t.transpose() * z);
    }

    let mut lo: f64 = lo0;
    let mut hi: f64 = sc.norm() / norm;
    if hi <= lo {
        hi = lo + 1.0;
    }
    while phi(hi) > target {
        hi = lo + 2.0 * (hi - lo);
        if !hi.is_finite() {
            return None;
        }
    }
    for _ in 0..200 {
        let mid: f64 = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if phi(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let l: f64 = hi;
    let z: Vector3<f64> = Vector3::from_fn(|i, _| sc[i] / (s2[i] + l));
    // The bisection stops within rounding of the root; put g exactly on the sphere.
    let g: Vector3<f64> = v_t.transpose() * z;
    let g_norm: f64 = g.norm();
    (g_norm > 0.0).then(|| g * (norm / g_norm))
}

/// The angular residual of every observation of a track, in radians, for a
/// given state. Points behind a ray give angles above 90 degrees.
pub(super) fn track_angles(
    observations: &[BearingObservation],
    point: &Vector3<f64>,
    poses: &[(Matrix3<f64>, Vector3<f64>)],
    cameras: &[Camera],
    out: &mut Vec<f64>,
) {
    out.clear();
    for obs in observations {
        let cam: &Camera = &cameras[obs.camera];
        let (r_k, p_k) = &poses[obs.frame];
        let x: Vector3<f64> = cam.r_ic.transpose() * (r_k.transpose() * (point - p_k) - cam.t_ic);
        let norm: f64 = x.norm();
        let cos: f64 = if norm > 0.0 {
            (x.dot(&obs.bearing) / norm).clamp(-1.0, 1.0)
        } else {
            -1.0
        };
        out.push(cos.acos());
    }
}

/// Root mean square of a slice.
pub(super) fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

/// The body poses `(R_k, p_k)` of the window for a state.
pub(super) fn poses(
    preint: &Preintegrated,
    velocity: &Vector3<f64>,
    gravity: &Vector3<f64>,
    bias_gyro: &Vector3<f64>,
    bias_accel: &Vector3<f64>,
) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
    (0..preint.dt.len())
        .map(|k| {
            let dt: f64 = preint.dt[k];
            (
                preint.rotation_at(k, bias_gyro),
                velocity * dt
                    + gravity * (0.5 * dt * dt)
                    + preint.position_at(k, bias_gyro, bias_accel),
            )
        })
        .collect()
}

/// SplitMix64 (Steele, Lea, Flood, OOPSLA 2014): a small seeded generator so
/// the consensus is reproducible without a dependency.
#[derive(Debug, Clone)]
pub(super) struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z: u64 = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A uniform index below `n` (`n > 0`), by Lemire's multiply-shift.
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// The result of the consensus over tracks.
#[derive(Debug, Clone)]
pub(super) struct Consensus {
    /// `(v0, g)` refitted on the inliers, with `|g|` constrained.
    pub y: Vector6<f64>,
    /// Inlier flag per block.
    pub inlier: Vec<bool>,
}

/// Parameters of the consensus.
#[derive(Debug, Clone, Copy)]
pub(super) struct ConsensusParams {
    pub gravity_magnitude: f64,
    pub threshold_rad: f64,
    pub sample_tracks: usize,
    pub max_iterations: usize,
    pub seed: u64,
}

/// MSAC over tracks (Torr and Zisserman, CVIU 2000): hypotheses from all
/// tracks and from random track subsets, each scored by the truncated RMS
/// angle of every track, with the adaptive stopping rule at 99.9 %
/// confidence. The winner is refitted on its inliers.
pub(super) fn consensus(
    tracks: &[&[BearingObservation]],
    blocks: &[TrackBlock],
    preint: &Preintegrated,
    cameras: &[Camera],
    params: &ConsensusParams,
) -> Option<Consensus> {
    let n: usize = blocks.len();
    let threshold2: f64 = params.threshold_rad * params.threshold_rad;
    let mut angles: Vec<f64> = Vec::new();
    let bg: Vector3<f64> = preint.bias_gyro;
    let ba: Vector3<f64> = preint.bias_accel;

    let mut score = |y: &Vector6<f64>, inlier: &mut Vec<bool>| -> f64 {
        let v: Vector3<f64> = y.fixed_rows::<3>(0).into_owned();
        let g: Vector3<f64> = y.fixed_rows::<3>(3).into_owned();
        let poses = poses(preint, &v, &g, &bg, &ba);
        inlier.clear();
        let mut cost: f64 = 0.0;
        for (block, obs) in blocks.iter().zip(tracks) {
            let r2: f64 = match block.triangulate(y) {
                Some(x) => {
                    track_angles(obs, &x, &poses, cameras, &mut angles);
                    rms(&angles).powi(2)
                }
                None => f64::INFINITY,
            };
            inlier.push(r2 < threshold2);
            cost += r2.min(threshold2);
        }
        cost
    };

    let mut best_cost: f64 = f64::INFINITY;
    let mut best_inlier: Vec<bool> = Vec::new();
    let mut scratch: Vec<bool> = Vec::new();
    if let Some(y) = solve_constrained(blocks.iter(), params.gravity_magnitude) {
        best_cost = score(&y, &mut best_inlier);
    }

    let sample: usize = params.sample_tracks.clamp(1, n);
    let mut rng = SplitMix64::new(params.seed);
    let mut indices: Vec<usize> = (0..n).collect();
    let mut needed: usize = params.max_iterations;
    let mut iteration: usize = 0;
    while iteration < needed.min(params.max_iterations) && n > sample {
        iteration += 1;
        // Partial Fisher-Yates over a fresh identity permutation.
        for (i, slot) in indices.iter_mut().enumerate() {
            *slot = i;
        }
        for i in 0..sample {
            let j: usize = i + rng.below(n - i);
            indices.swap(i, j);
        }
        let Some(y) = solve_constrained(
            indices[..sample].iter().map(|&i| &blocks[i]),
            params.gravity_magnitude,
        ) else {
            continue;
        };
        let cost: f64 = score(&y, &mut scratch);
        if cost < best_cost {
            best_cost = cost;
            std::mem::swap(&mut best_inlier, &mut scratch);
            let ratio: f64 = best_inlier.iter().filter(|&&b| b).count() as f64 / n as f64;
            let all_good: f64 = ratio.powi(sample as i32);
            needed = if all_good >= 1.0 {
                iteration
            } else if all_good <= 0.0 {
                params.max_iterations
            } else {
                ((1.0f64 - 0.999).ln() / (1.0 - all_good).ln()).ceil() as usize
            };
        }
    }
    if best_inlier.is_empty() {
        return None;
    }
    let y: Vector6<f64> = solve_constrained(
        blocks
            .iter()
            .zip(&best_inlier)
            .filter(|(_, &keep)| keep)
            .map(|(b, _)| b),
        params.gravity_magnitude,
    )?;
    let mut inlier: Vec<bool> = Vec::new();
    score(&y, &mut inlier);
    Some(Consensus { y, inlier })
}
