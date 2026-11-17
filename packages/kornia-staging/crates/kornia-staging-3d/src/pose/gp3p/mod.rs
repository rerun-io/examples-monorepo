//! Generalized (multi-camera, non-central) absolute pose from three rays.
//!
//! A rig observes world points through several cameras. Each observation is a
//! *generalized ray* in the body frame: the camera centre `origin` (`t_b_c`) and
//! the unit bearing `bearing` rotated into the body frame (`R_b_c f_c`). The
//! solver finds every body pose `T_w_b` such that each world point lies on its
//! ray, `X_w = T_w_b (origin + λ bearing)` with `λ > 0`.
//!
//! This is the problem OpenGV's `gp3p` solves for RootSLAM's loop verification
//! (`loop_closing.cpp:518-618`). The formulation here is different from OpenGV's
//! Gröbner-basis template: three distance constraints on the ray depths
//! `|Y_i - Y_j| = |X_i - X_j|` are reduced to one univariate octic in the first
//! depth (two eliminations and one quadratic resultant), its positive roots come
//! from a Sturm sequence, each root is polished with Newton on the three
//! constraints, and the pose follows from the congruent triangles. A central rig
//! (all origins equal) is the P3P special case of the same equations.
//!
//! The RANSAC estimator around the solver lives in `gp3p::ransac` (feature `ransac`,
//! on top of upstream `kornia_3d::ransac`). [`refine`] is the nonlinear
//! refinement on inliers that RootSLAM runs after RANSAC.

use kornia_staging_algebra::lie::{RigidTransform, Rotation3};
use nalgebra::{Matrix2x3, Matrix3, Matrix6, UnitQuaternion, Vector3, Vector6};

#[cfg(feature = "ransac")]
pub mod ransac;

/// Most real solutions the octic can produce.
pub const MAX_SOLUTIONS: usize = 8;

/// Collinearity bound on the world triangle: `|e1 x e2| <= this * max(|e|)^2`.
const COLLINEAR_SINE: f64 = 1e-9;
/// Leading octic coefficients below this fraction of the largest are dropped.
const POLY_TRIM: f64 = 1e-13;
/// Largest constraint residual (squared distance, normalized units) a polished
/// solution may keep.
const CONSTRAINT_TOL: f64 = 1e-7;
/// Two solutions whose depths agree to this (normalized units) are one.
const DUPLICATE_TOL: f64 = 1e-7;
/// Largest first depth searched, in units of the mean world-triangle edge.
const MAX_DEPTH: f64 = 1e6;
/// A bearing is unit when `| |f| - 1 |` is below this.
const UNIT_TOL: f64 = 1e-6;
const NEWTON_ITERATIONS: usize = 4;

/// One 2D-3D correspondence of a (possibly non-central) rig, in the body frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedCorrespondence {
    /// Centre of the observing camera in the body frame (`t_b_c`).
    pub origin: Vector3<f64>,
    /// Unit bearing in the body frame (`R_b_c f_c`).
    pub bearing: Vector3<f64>,
    /// The observed point in the world frame.
    pub point: Vector3<f64>,
}

impl GeneralizedCorrespondence {
    /// A correspondence from a camera-frame bearing and the camera's extrinsics
    /// `T_b_c` (camera to body). The bearing is normalized here; a zero bearing
    /// becomes non-finite and is refused by the solvers.
    pub fn from_camera(
        t_b_c: &RigidTransform<f64>,
        bearing_c: &Vector3<f64>,
        point_w: &Vector3<f64>,
    ) -> Self {
        Self {
            origin: t_b_c.translation,
            bearing: t_b_c.rotation * (bearing_c / bearing_c.norm()),
            point: *point_w,
        }
    }

    /// `1 - cos` of the angle between the bearing and the direction to the
    /// point under `pose` (OpenGV's `AbsolutePoseSacProblem` distance). 0 is a
    /// perfect fit; a point behind the camera scores above 1.
    #[inline]
    pub fn angular_residual(&self, pose: &Gp3pPose) -> f64 {
        let v: Vector3<f64> = pose.point_in_body(&self.point) - self.origin;
        1.0 - v.dot(&self.bearing) / v.norm()
    }
}

/// A body pose `T_w_b` as a rotation matrix and a translation: `X_w = R X_b + t`.
///
/// The solvers and the RANSAC hot loop work on matrices; convert with
/// [`Gp3pPose::to_rigid_transform`] at the boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gp3pPose {
    /// `R_w_b`.
    pub rotation: Matrix3<f64>,
    /// `t_w_b`.
    pub translation: Vector3<f64>,
}

impl Gp3pPose {
    /// The pose from a [`RigidTransform`] `T_w_b`.
    pub fn from_rigid_transform(t_w_b: &RigidTransform<f64>) -> Self {
        Self {
            rotation: t_w_b.rotation.matrix(),
            translation: t_w_b.translation,
        }
    }

    /// The pose as a [`RigidTransform`] `T_w_b`.
    pub fn to_rigid_transform(&self) -> RigidTransform<f64> {
        let rotation = nalgebra::Rotation3::from_matrix_unchecked(self.rotation);
        RigidTransform::new(
            Rotation3::from_unit_quaternion(UnitQuaternion::from_rotation_matrix(&rotation)),
            self.translation,
        )
    }

    /// `T_w_b^-1 X_w`.
    #[inline]
    pub fn point_in_body(&self, point_w: &Vector3<f64>) -> Vector3<f64> {
        self.rotation.tr_mul(&(point_w - self.translation))
    }
}

/// Why a generalized pose could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Gp3pError {
    /// Fewer correspondences than the method needs.
    #[error("{got} correspondences, at least {needed} needed")]
    TooFewCorrespondences {
        /// Correspondences given.
        got: usize,
        /// Correspondences needed.
        needed: usize,
    },
    /// More correspondences than a minimal solver takes.
    #[error("{got} correspondences, the minimal solver takes exactly {expected}")]
    TooManyCorrespondences {
        /// Correspondences given.
        got: usize,
        /// Correspondences the solver takes.
        expected: usize,
    },
    /// A coordinate is NaN or infinite.
    #[error("correspondence {index} is not finite")]
    NonFinite {
        /// Index of the offending correspondence.
        index: usize,
    },
    /// A bearing is not of unit length.
    #[error("correspondence {index} has a bearing that is not unit length")]
    NonUnitBearing {
        /// Index of the offending correspondence.
        index: usize,
    },
    /// The three world points are collinear or coincident: the pose is not determined.
    #[error("the world points are collinear")]
    CollinearPoints,
}

/// Check every correspondence: finite coordinates and unit bearings.
pub(crate) fn validate(correspondences: &[GeneralizedCorrespondence]) -> Result<(), Gp3pError> {
    for (index, c) in correspondences.iter().enumerate() {
        let finite = c
            .origin
            .iter()
            .chain(c.bearing.iter())
            .chain(c.point.iter())
            .all(|v| v.is_finite());
        if !finite {
            return Err(Gp3pError::NonFinite { index });
        }
        if (c.bearing.norm() - 1.0).abs() > UNIT_TOL {
            return Err(Gp3pError::NonUnitBearing { index });
        }
    }
    Ok(())
}

fn collinear(p0: &Vector3<f64>, p1: &Vector3<f64>, p2: &Vector3<f64>) -> bool {
    let e1: Vector3<f64> = p1 - p0;
    let e2: Vector3<f64> = p2 - p0;
    let e3: Vector3<f64> = p2 - p1;
    let longest = e1
        .norm_squared()
        .max(e2.norm_squared())
        .max(e3.norm_squared());
    longest == 0.0 || e1.cross(&e2).norm() <= COLLINEAR_SINE * longest
}

/// Every pose `T_w_b` that puts the three world points on their rays in front
/// of the cameras: 0 to [`MAX_SOLUTIONS`] of them.
///
/// # Errors
/// [`Gp3pError::TooFewCorrespondences`] / [`Gp3pError::TooManyCorrespondences`]
/// unless exactly three are given; [`Gp3pError::NonFinite`],
/// [`Gp3pError::NonUnitBearing`] for malformed input;
/// [`Gp3pError::CollinearPoints`] when the world triangle is degenerate.
///
/// ```
/// use kornia_staging_3d::pose::gp3p::{gp3p, GeneralizedCorrespondence};
/// use nalgebra::Vector3;
/// let origins = [Vector3::new(0.1, 0.0, 0.0), Vector3::new(-0.1, 0.0, 0.0), Vector3::zeros()];
/// let points = [Vector3::new(0.3, 0.2, 3.0), Vector3::new(-0.5, 0.1, 4.0), Vector3::new(0.0, -0.4, 2.5)];
/// // The body sits at the world origin: the bearings point straight at the points.
/// let c: Vec<GeneralizedCorrespondence> = (0..3)
///     .map(|i| GeneralizedCorrespondence {
///         origin: origins[i],
///         bearing: (points[i] - origins[i]).normalize(),
///         point: points[i],
///     })
///     .collect();
/// let poses = gp3p(&c).unwrap();
/// assert!(poses.iter().any(|p| p.translation.norm() < 1e-9));
/// ```
pub fn gp3p(correspondences: &[GeneralizedCorrespondence]) -> Result<Vec<Gp3pPose>, Gp3pError> {
    if correspondences.len() < 3 {
        return Err(Gp3pError::TooFewCorrespondences {
            got: correspondences.len(),
            needed: 3,
        });
    }
    if correspondences.len() > 3 {
        return Err(Gp3pError::TooManyCorrespondences {
            got: correspondences.len(),
            expected: 3,
        });
    }
    validate(correspondences)?;
    let triple = [correspondences[0], correspondences[1], correspondences[2]];
    if collinear(&triple[0].point, &triple[1].point, &triple[2].point) {
        return Err(Gp3pError::CollinearPoints);
    }
    let mut solutions = Solutions::default();
    solve(&triple, &mut solutions);
    Ok(solutions.as_slice().to_vec())
}

/// A fixed-capacity solution list, so the RANSAC loop never allocates.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Solutions {
    len: usize,
    poses: [Gp3pPose; MAX_SOLUTIONS],
    depths: [Vector3<f64>; MAX_SOLUTIONS],
}

impl Default for Solutions {
    fn default() -> Self {
        let zero = Gp3pPose {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        };
        Self {
            len: 0,
            poses: [zero; MAX_SOLUTIONS],
            depths: [Vector3::zeros(); MAX_SOLUTIONS],
        }
    }
}

impl Solutions {
    pub(crate) fn as_slice(&self) -> &[Gp3pPose] {
        &self.poses[..self.len]
    }

    fn push(&mut self, depths: Vector3<f64>, pose: Gp3pPose) {
        let duplicate = self.depths[..self.len]
            .iter()
            .any(|d| (d - depths).amax() <= DUPLICATE_TOL * (1.0 + depths.amax()));
        if !duplicate && self.len < MAX_SOLUTIONS {
            self.depths[self.len] = depths;
            self.poses[self.len] = pose;
            self.len += 1;
        }
    }
}

/// Ascending polynomial coefficients, degree <= 8.
type Poly = [f64; 9];

fn poly(coefficients: &[f64]) -> Poly {
    let mut p = [0.0; 9];
    p[..coefficients.len()].copy_from_slice(coefficients);
    p
}

fn mul(a: &Poly, b: &Poly) -> Poly {
    let mut out = [0.0; 9];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate().take(9 - i) {
            out[i + j] += ai * bj;
        }
    }
    out
}

fn add(a: &Poly, b: &Poly) -> Poly {
    std::array::from_fn(|i| a[i] + b[i])
}

fn sub(a: &Poly, b: &Poly) -> Poly {
    std::array::from_fn(|i| a[i] - b[i])
}

fn scale(a: &Poly, s: f64) -> Poly {
    a.map(|v| v * s)
}

fn eval(p: &Poly, x: f64) -> f64 {
    p.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

fn eval_derivative(p: &Poly, x: f64) -> f64 {
    p.iter()
        .enumerate()
        .skip(1)
        .rev()
        .fold(0.0, |acc, (i, &c)| acc * x + c * i as f64)
}

/// Degree of `p` after dropping leading coefficients below [`POLY_TRIM`] of the
/// largest; `None` for the zero polynomial.
fn degree(p: &Poly) -> Option<usize> {
    let largest = p.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    if largest == 0.0 || !largest.is_finite() {
        return None;
    }
    (0..9).rev().find(|&d| p[d].abs() > POLY_TRIM * largest)
}

/// `-(a mod b)`, scaled to unit largest coefficient. `b` has degree `db >= 1`.
fn negated_remainder(a: &Poly, b: &Poly, db: usize) -> Poly {
    let mut r = *a;
    let Some(mut dr) = degree(&r) else {
        return [0.0; 9];
    };
    while dr >= db {
        let factor = r[dr] / b[db];
        for i in 0..=db {
            r[dr - db + i] -= factor * b[i];
        }
        r[dr] = 0.0;
        match degree(&r) {
            Some(d) if d < dr => dr = d,
            Some(_) => break,
            None => return [0.0; 9],
        }
    }
    let largest = r.iter().fold(0.0f64, |m, c| m.max(c.abs()));
    if largest == 0.0 {
        return r;
    }
    r.map(|c| -c / largest)
}

/// Sturm sequence of `p`: `p`, `p'`, then negated remainders.
struct Sturm {
    seq: [Poly; 9],
    len: usize,
}

impl Sturm {
    fn new(p: &Poly, dp: usize) -> Self {
        let mut seq = [[0.0; 9]; 9];
        seq[0] = *p;
        seq[1] = std::array::from_fn(|i| {
            if i < 8 {
                p[i + 1] * (i + 1) as f64
            } else {
                0.0
            }
        });
        let mut len = 2;
        let mut db = dp - 1;
        while len < 9 && db >= 1 {
            let next = negated_remainder(&seq[len - 2], &seq[len - 1], db);
            let Some(dn) = degree(&next) else { break };
            seq[len] = next;
            len += 1;
            db = dn;
            if dn == 0 {
                break;
            }
        }
        Self { seq, len }
    }

    /// Sign changes of the sequence at `x`.
    fn changes(&self, x: f64) -> u32 {
        let mut count = 0;
        let mut last = 0.0f64;
        for p in &self.seq[..self.len] {
            let v = eval(p, x);
            if v != 0.0 {
                if last != 0.0 && (v > 0.0) != (last > 0.0) {
                    count += 1;
                }
                last = v;
            }
        }
        count
    }
}

/// The distinct real roots of `p` in `(0, ∞)`: Sturm isolation, then bisection
/// and Newton. Deterministic, and blind to the equal-modulus root pairs
/// (`±r` of a central rig) that stall a QR eigen-iteration on the companion matrix.
fn positive_roots(p: &Poly, out: &mut Vec<f64>) {
    out.clear();
    let Some(dp) = degree(p) else { return };
    if dp == 0 {
        return;
    }
    // Drop the trimmed leading terms so the sequence sees one consistent polynomial.
    let p: Poly = std::array::from_fn(|i| if i <= dp { p[i] } else { 0.0 });
    let p = &p;
    // Cauchy bound on |roots|, capped: a depth beyond `MAX_DEPTH` triangle sizes
    // is no use to a pose and only widens the search.
    let bound = 1.0 + (0..dp).fold(0.0f64, |m, i| m.max((p[i] / p[dp]).abs()));
    let bound = bound.min(MAX_DEPTH);
    let sturm = Sturm::new(p, dp);
    let mut stack: [(f64, f64, u32, u32, u32); 64] = [(0.0, 0.0, 0, 0, 0); 64];
    let mut top = 0;
    let (lo, hi) = (0.0, bound);
    stack[top] = (lo, hi, sturm.changes(lo), sturm.changes(hi), 0);
    top += 1;
    while top > 0 {
        top -= 1;
        let (a, b, va, vb, depth) = stack[top];
        let count = va.saturating_sub(vb);
        if count == 0 {
            continue;
        }
        if count == 1 || depth >= 50 || b - a <= 1e-12 * (1.0 + b.abs()) {
            out.push(refine_root(p, a, b));
            continue;
        }
        let m = 0.5 * (a + b);
        let vm = sturm.changes(m);
        if top + 2 <= stack.len() {
            stack[top] = (m, b, vm, vb, depth + 1);
            stack[top + 1] = (a, m, va, vm, depth + 1);
            top += 2;
        }
    }
}

/// One root of `p` in `[a, b]`: bisection while the bracket holds, Newton polish.
fn refine_root(p: &Poly, mut a: f64, mut b: f64) -> f64 {
    let mut fa = eval(p, a);
    let fb = eval(p, b);
    if fa * fb < 0.0 {
        for _ in 0..200 {
            let m = 0.5 * (a + b);
            let fm = eval(p, m);
            if fm == 0.0 || b - a <= 1e-15 * (1.0 + m.abs()) {
                a = m;
                b = m;
                break;
            }
            if (fm > 0.0) == (fa > 0.0) {
                a = m;
                fa = fm;
            } else {
                b = m;
            }
        }
    }
    let mut x = 0.5 * (a + b);
    for _ in 0..3 {
        let d = eval_derivative(p, x);
        if d == 0.0 {
            break;
        }
        let step = eval(p, x) / d;
        if step.is_finite() {
            x -= step;
        }
    }
    x
}

/// The three depth constraints `|c_ij + λ_i f_i - λ_j f_j|² - d_ij²`, pair order (0,1), (0,2), (1,2).
struct Constraints {
    origin: [Vector3<f64>; 3],
    bearing: [Vector3<f64>; 3],
    distance_sq: [f64; 3],
}

const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

impl Constraints {
    fn residuals_and_jacobian(&self, depth: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        let mut r = Vector3::zeros();
        let mut j = Matrix3::zeros();
        for (row, &(a, b)) in PAIRS.iter().enumerate() {
            let w: Vector3<f64> = self.origin[a] - self.origin[b] + self.bearing[a] * depth[a]
                - self.bearing[b] * depth[b];
            r[row] = w.norm_squared() - self.distance_sq[row];
            j[(row, a)] = 2.0 * w.dot(&self.bearing[a]);
            j[(row, b)] = -2.0 * w.dot(&self.bearing[b]);
        }
        (r, j)
    }

    /// Newton on the three constraints; returns the polished depths and the
    /// largest absolute residual.
    fn polish(&self, depth: Vector3<f64>) -> (Vector3<f64>, f64) {
        let mut best = depth;
        let (r, _) = self.residuals_and_jacobian(&best);
        let mut best_err = r.amax();
        let mut current = depth;
        for _ in 0..NEWTON_ITERATIONS {
            let (r, j) = self.residuals_and_jacobian(&current);
            let Some(step) = j.lu().solve(&r) else {
                break;
            };
            current -= step;
            let (r, _) = self.residuals_and_jacobian(&current);
            let err = r.amax();
            if err.is_nan() || err >= best_err {
                break;
            }
            best = current;
            best_err = err;
        }
        (best, best_err)
    }
}

/// The rotation and translation that map triangle `y` (body) onto triangle `x`
/// (world). Exact for congruent triangles.
fn align_triangles(y: &[Vector3<f64>; 3], x: &[Vector3<f64>; 3]) -> Option<Gp3pPose> {
    fn frame(p: &[Vector3<f64>; 3]) -> Option<Matrix3<f64>> {
        let e1: Vector3<f64> = (p[1] - p[0]).try_normalize(0.0)?;
        let e3: Vector3<f64> = e1.cross(&(p[2] - p[0])).try_normalize(0.0)?;
        let e2: Vector3<f64> = e3.cross(&e1);
        Some(Matrix3::from_columns(&[e1, e2, e3]))
    }
    let rotation: Matrix3<f64> = frame(x)? * frame(y)?.transpose();
    let centroid_y: Vector3<f64> = (y[0] + y[1] + y[2]) / 3.0;
    let centroid_x: Vector3<f64> = (x[0] + x[1] + x[2]) / 3.0;
    Some(Gp3pPose {
        rotation,
        translation: centroid_x - rotation * centroid_y,
    })
}

/// The real roots of `c2 x² + c1 x + c0`, or `None` when they are complex.
fn quadratic_roots(c2: f64, c1: f64, c0: f64) -> Option<[f64; 2]> {
    let disc = c1 * c1 - 4.0 * c2 * c0;
    if c2 == 0.0 || disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    // Numerically stable pair.
    let q = -0.5 * (c1 + c1.signum() * s);
    if q == 0.0 {
        return Some([0.0, 0.0]);
    }
    Some([q / c2, c0 / q])
}

/// The minimal solver on a validated, non-collinear triple. Pushes 0 to 8 poses.
pub(crate) fn solve(triple: &[GeneralizedCorrespondence; 3], out: &mut Solutions) {
    out.len = 0;
    let x = [triple[0].point, triple[1].point, triple[2].point];
    let distance = PAIRS.map(|(a, b)| (x[a] - x[b]).norm());
    let mean = (distance[0] + distance[1] + distance[2]) / 3.0;
    if mean.is_nan() || mean <= 0.0 {
        return;
    }
    // Work in units of the mean triangle edge, so every coefficient is O(1).
    let s = 1.0 / mean;
    let constraints = Constraints {
        origin: [
            triple[0].origin * s,
            triple[1].origin * s,
            triple[2].origin * s,
        ],
        bearing: [triple[0].bearing, triple[1].bearing, triple[2].bearing],
        distance_sq: distance.map(|d| (d * s) * (d * s)),
    };
    let [f1, f2, f3] = constraints.bearing;
    let [o1, o2, o3] = constraints.origin;
    let c12: Vector3<f64> = o1 - o2;
    let c13: Vector3<f64> = o1 - o3;
    let c23: Vector3<f64> = o2 - o3;
    let [d12, d13, d23] = constraints.distance_sq;

    // Q12 as a monic quadratic in λ2 and Q13 in λ3, with coefficients in x = λ1:
    // λ2² + B2 λ2 + C2 = 0, λ3² + B3 λ3 + C3 = 0.
    let b2 = poly(&[-2.0 * f2.dot(&c12), -2.0 * f1.dot(&f2)]);
    let c2 = poly(&[c12.norm_squared() - d12, 2.0 * f1.dot(&c12), 1.0]);
    let b3 = poly(&[-2.0 * f3.dot(&c13), -2.0 * f1.dot(&f3)]);
    let c3 = poly(&[c13.norm_squared() - d13, 2.0 * f1.dot(&c13), 1.0]);
    // Q23 with λ2², λ3² replaced through Q12, Q13: α λ2 λ3 + β λ2 + γ λ3 + δ = 0.
    let alpha = poly(&[-2.0 * f2.dot(&f3)]);
    let beta = sub(&poly(&[2.0 * f2.dot(&c23)]), &b2);
    let gamma = sub(&poly(&[-2.0 * f3.dot(&c23)]), &b3);
    let delta = sub(&sub(&poly(&[c23.norm_squared() - d23]), &c2), &c3);
    // λ3 = -(β λ2 + δ) / (α λ2 + γ) into Q13, times (α λ2 + γ)²: P2 λ2² + P1 λ2 + P0 = 0.
    let p2 = add(
        &sub(&mul(&beta, &beta), &mul(&mul(&b3, &beta), &alpha)),
        &mul(&c3, &mul(&alpha, &alpha)),
    );
    let p1 = add(
        &sub(
            &scale(&mul(&beta, &delta), 2.0),
            &mul(&b3, &add(&mul(&beta, &gamma), &mul(&delta, &alpha))),
        ),
        &scale(&mul(&c3, &mul(&alpha, &gamma)), 2.0),
    );
    let p0 = add(
        &sub(&mul(&delta, &delta), &mul(&mul(&b3, &delta), &gamma)),
        &mul(&c3, &mul(&gamma, &gamma)),
    );
    // Resultant in λ2 of (1, B2, C2) and (P2, P1, P0): an octic in λ1.
    let e = sub(&p0, &mul(&c2, &p2));
    let octic = sub(
        &mul(&e, &e),
        &mul(
            &sub(&p1, &mul(&b2, &p2)),
            &sub(&mul(&b2, &p0), &mul(&c2, &p1)),
        ),
    );

    let mut roots = Vec::with_capacity(8);
    positive_roots(&octic, &mut roots);
    for &l1 in roots.iter() {
        if l1.is_nan() || l1 <= 0.0 {
            continue;
        }
        let (vb2, vc2) = (eval(&b2, l1), eval(&c2, l1));
        let (vp2, vp1, vp0) = (eval(&p2, l1), eval(&p1, l1), eval(&p0, l1));
        let den = vp2 * vb2 - vp1;
        let mut l2_list: [Option<f64>; 2] = [None, None];
        if den.abs() > 1e-12 * (vp2.abs() * vb2.abs() + vp1.abs() + 1e-300) {
            l2_list[0] = Some((vp0 - vp2 * vc2) / den);
        } else if let Some(r) = quadratic_roots(1.0, vb2, vc2) {
            l2_list = [Some(r[0]), Some(r[1])];
        }
        for l2 in l2_list.into_iter().flatten() {
            let (va, vb, vg, vd) = (
                alpha[0],
                eval(&beta, l1),
                eval(&gamma, l1),
                eval(&delta, l1),
            );
            let den3 = va * l2 + vg;
            let mut l3_list: [Option<f64>; 2] = [None, None];
            if den3.abs() > 1e-12 {
                l3_list[0] = Some(-(vb * l2 + vd) / den3);
            } else if let Some(r) = quadratic_roots(1.0, eval(&b3, l1), eval(&c3, l1)) {
                l3_list = [Some(r[0]), Some(r[1])];
            }
            for l3 in l3_list.into_iter().flatten() {
                let (depth, err) = constraints.polish(Vector3::new(l1, l2, l3));
                if err.is_nan()
                    || err > CONSTRAINT_TOL
                    || depth.iter().any(|d| d.is_nan() || *d <= 0.0)
                {
                    continue;
                }
                let depth_world = depth / s;
                let y = [0, 1, 2].map(|i| triple[i].origin + triple[i].bearing * depth_world[i]);
                if let Some(pose) = align_triangles(&y, &x) {
                    out.push(depth, pose);
                }
            }
        }
    }
}

/// Levenberg–Marquardt iterations of [`refine`].
const REFINE_ITERATIONS: usize = 20;

/// Nonlinear least-squares refinement of `T_w_b` on the given correspondences
/// (RootSLAM refines on the RANSAC inliers, `loop_closing.cpp:601-602`).
///
/// The residual is the bearing error in the tangent plane of each bearing, two
/// rows per correspondence, `sin` of the angle to first order. OpenGV's
/// `optimize_nonlinear` minimises the squared `1 - cos` instead, whose gradient
/// vanishes quadratically at the optimum, so its numeric-difference LM stops on
/// tolerance short of the minimum. Returns `initial` unchanged with fewer than
/// three correspondences; never returns a pose with a higher cost than `initial`.
pub fn refine(correspondences: &[GeneralizedCorrespondence], initial: &Gp3pPose) -> Gp3pPose {
    if correspondences.len() < 3 {
        return *initial;
    }
    let tangent: Vec<Matrix2x3<f64>> = correspondences
        .iter()
        .map(|c| tangent_basis(&c.bearing))
        .collect();
    let cost = |pose: &Gp3pPose| -> f64 {
        correspondences
            .iter()
            .zip(&tangent)
            .map(|(c, b)| {
                let v: Vector3<f64> = pose.point_in_body(&c.point) - c.origin;
                (b * v.normalize()).norm_squared()
            })
            .sum()
    };
    let mut pose = *initial;
    let mut current = cost(&pose);
    let mut lambda = 1e-4;
    for _ in 0..REFINE_ITERATIONS {
        let mut h: Matrix6<f64> = Matrix6::zeros();
        let mut g: Vector6<f64> = Vector6::zeros();
        for (c, b) in correspondences.iter().zip(&tangent) {
            let pb: Vector3<f64> = pose.point_in_body(&c.point);
            let v: Vector3<f64> = pb - c.origin;
            let norm = v.norm();
            let u: Vector3<f64> = v / norm;
            let r = b * u;
            // dv/d(ν, ω) for R' = R exp(ω), t' = t + R ν: [-I | hat(pb)].
            let du: Matrix3<f64> = (Matrix3::identity() - u * u.transpose()) / norm;
            let bdu: Matrix2x3<f64> = b * du;
            let mut j = nalgebra::Matrix2x6::<f64>::zeros();
            j.fixed_view_mut::<2, 3>(0, 0).copy_from(&(-bdu));
            j.fixed_view_mut::<2, 3>(0, 3)
                .copy_from(&(bdu * Rotation3::hat(&pb)));
            h += j.transpose() * j;
            g += j.transpose() * r;
        }
        let mut improved = false;
        while lambda < 1e10 {
            let mut damped = h;
            for i in 0..6 {
                damped[(i, i)] += lambda * h[(i, i)].max(1e-12);
            }
            let Some(step) = damped.cholesky().map(|ch| ch.solve(&(-g))) else {
                lambda *= 10.0;
                continue;
            };
            let omega: Vector3<f64> = step.fixed_rows::<3>(3).into_owned();
            let candidate = Gp3pPose {
                rotation: pose.rotation * nalgebra::Rotation3::new(omega).into_inner(),
                translation: pose.translation + pose.rotation * step.fixed_rows::<3>(0),
            };
            let next = cost(&candidate);
            if next < current {
                let converged = (current - next) <= 1e-15 * current || step.amax() < 1e-14;
                pose = candidate;
                current = next;
                lambda = (lambda * 0.1).max(1e-10);
                improved = !converged;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    pose
}

/// Two unit vectors orthogonal to unit `f`, as rows.
fn tangent_basis(f: &Vector3<f64>) -> Matrix2x3<f64> {
    let helper = if f.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let a: Vector3<f64> = f.cross(&helper).normalize();
    let b: Vector3<f64> = f.cross(&a);
    Matrix2x3::from_rows(&[a.transpose(), b.transpose()])
}
