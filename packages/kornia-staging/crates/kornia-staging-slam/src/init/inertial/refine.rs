//! The non-linear stage: Levenberg-Marquardt over velocity, gravity direction,
//! gyroscope bias, optionally accelerometer bias, and the points, with a
//! Huber loss on the bearing residual.
//!
//! The residual of an observation is the chordal bearing error `n - b`, with
//! `n` the unit direction of the point in the camera and `b` the observed
//! bearing. Its norm is `2 sin(angle / 2)`: monotonic over the whole sphere,
//! so it needs no projection, works beyond 90 degrees from the axis, and does
//! not confuse a point with its antipode the way a tangent-plane residual
//! does. Gravity stays on the sphere of radius `|g|` by a rotation update in
//! its 2-D tangent plane. The points are eliminated with the Schur complement
//! at every iteration; the reduced system has 8 (or 11) unknowns.
use super::linear::{tangent_basis, Camera};
use super::preintegration::Preintegrated;
use super::BearingObservation;
use kornia_staging_algebra::lie::Rotation3;
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

/// The state being refined.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct State {
    pub velocity: Vector3<f64>,
    pub gravity: Vector3<f64>,
    pub bias_gyro: Vector3<f64>,
    pub bias_accel: Vector3<f64>,
    pub points: Vec<Vector3<f64>>,
}

/// The fixed parts of the problem.
#[derive(Debug, Clone, Copy)]
pub(super) struct Problem<'a> {
    pub preint: &'a Preintegrated,
    pub cameras: &'a [Camera],
    pub tracks: &'a [&'a [BearingObservation]],
    pub huber: f64,
    /// Bearing noise, rad: whitens the data against the bias priors.
    pub bearing_sigma: f64,
    pub estimate_accel_bias: bool,
    pub gyro_bias_prior: Vector3<f64>,
    pub gyro_bias_sigma: f64,
    pub accel_bias_sigma: f64,
}

/// What the refinement returns beside the state.
#[derive(Debug, Clone)]
pub(super) struct Refined {
    pub state: State,
    pub converged: bool,
    /// Undamped reduced information matrix at the solution, in the order
    /// velocity (3), gravity tangent (2), gyro bias (3), accel bias (3).
    pub information: DMatrix<f64>,
    /// Robust data cost at the solution, whitened by the bearing sigma.
    pub data_cost: f64,
    /// Number of observations.
    pub observations: usize,
}

const MAX_Y: usize = 11;
type RowJ = SMatrix<f64, 3, MAX_Y>;

/// Huber cost and IRLS weight for a squared residual norm.
fn huber(r2: f64, delta: f64) -> (f64, f64) {
    if r2 <= delta * delta {
        (r2, 1.0)
    } else {
        let r: f64 = r2.sqrt();
        (2.0 * delta * r - delta * delta, delta / r)
    }
}

struct Frames {
    rotation: Vec<Matrix3<f64>>,
    position: Vec<Vector3<f64>>,
}

fn frames(problem: &Problem<'_>, state: &State) -> Frames {
    let p = problem.preint;
    let mut rotation = Vec::with_capacity(p.dt.len());
    let mut position = Vec::with_capacity(p.dt.len());
    for k in 0..p.dt.len() {
        let dt: f64 = p.dt[k];
        rotation.push(p.rotation_at(k, &state.bias_gyro));
        position.push(
            state.velocity * dt
                + state.gravity * (0.5 * dt * dt)
                + p.position_at(k, &state.bias_gyro, &state.bias_accel),
        );
    }
    Frames { rotation, position }
}

fn prior_cost(problem: &Problem<'_>, state: &State) -> f64 {
    let mut cost: f64 =
        ((state.bias_gyro - problem.gyro_bias_prior) / problem.gyro_bias_sigma).norm_squared();
    if problem.estimate_accel_bias {
        cost += (state.bias_accel / problem.accel_bias_sigma).norm_squared();
    }
    cost
}

/// Robust data cost of a state.
pub(super) fn data_cost(problem: &Problem<'_>, state: &State) -> f64 {
    let fr = frames(problem, state);
    let mut cost: f64 = 0.0;
    for (track, point) in problem.tracks.iter().zip(&state.points) {
        for obs in track.iter() {
            let cam: &Camera = &problem.cameras[obs.camera];
            let z: Vector3<f64> =
                fr.rotation[obs.frame].transpose() * (point - fr.position[obs.frame]);
            let x: Vector3<f64> = cam.r_ic.transpose() * (z - cam.t_ic);
            let norm: f64 = x.norm();
            let r: Vector3<f64> = if norm > 0.0 {
                x / norm - obs.bearing
            } else {
                -obs.bearing * 2.0
            };
            cost += huber(r.norm_squared(), problem.huber).0;
        }
    }
    cost / (problem.bearing_sigma * problem.bearing_sigma)
}

/// Accumulated normal equations at one state.
pub(super) struct Normal {
    pub h_yy: DMatrix<f64>,
    pub g_y: DVector<f64>,
    h_pp: Vec<Matrix3<f64>>,
    h_py: Vec<SMatrix<f64, 3, MAX_Y>>,
    g_p: Vec<Vector3<f64>>,
}

pub(super) fn linearize(
    problem: &Problem<'_>,
    state: &State,
    gravity_basis: &SMatrix<f64, 3, 2>,
    ny: usize,
) -> Normal {
    let p = problem.preint;
    let fr = frames(problem, state);
    let mut h_yy: DMatrix<f64> = DMatrix::zeros(ny, ny);
    let mut g_y: DVector<f64> = DVector::zeros(ny);
    let mut h_pp = Vec::with_capacity(problem.tracks.len());
    let mut h_py = Vec::with_capacity(problem.tracks.len());
    let mut g_p = Vec::with_capacity(problem.tracks.len());
    let g_hat_b: SMatrix<f64, 3, 2> = Rotation3::hat(&state.gravity) * gravity_basis;
    let inv_bearing_var: f64 = 1.0 / (problem.bearing_sigma * problem.bearing_sigma);

    for (track, point) in problem.tracks.iter().zip(&state.points) {
        let mut hpp: Matrix3<f64> = Matrix3::zeros();
        let mut hpy: SMatrix<f64, 3, MAX_Y> = SMatrix::zeros();
        let mut gp: Vector3<f64> = Vector3::zeros();
        for obs in track.iter() {
            let k: usize = obs.frame;
            let cam: &Camera = &problem.cameras[obs.camera];
            let r_k: &Matrix3<f64> = &fr.rotation[k];
            let dt: f64 = p.dt[k];
            let z: Vector3<f64> = r_k.transpose() * (point - fr.position[k]);
            let x: Vector3<f64> = cam.r_ic.transpose() * (z - cam.t_ic);
            let norm: f64 = x.norm();
            if norm.is_nan() || norm <= 0.0 {
                continue;
            }
            let n: Vector3<f64> = x / norm;
            let r: Vector3<f64> = n - obs.bearing;
            let w: f64 = huber(r.norm_squared(), problem.huber).1 * inv_bearing_var;
            // d r / d z
            let a: Matrix3<f64> =
                (Matrix3::identity() - n * n.transpose()) / norm * cam.r_ic.transpose();
            let a_rt: Matrix3<f64> = a * r_k.transpose();
            let j_x: Matrix3<f64> = a_rt;
            let mut j_y: RowJ = RowJ::zeros();
            j_y.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-a_rt * dt));
            j_y.fixed_view_mut::<3, 2>(0, 3)
                .copy_from(&(a_rt * g_hat_b * (0.5 * dt * dt)));
            let j_bg: Matrix3<f64> =
                a * Rotation3::hat(&z) * p.d_rotation_d_bg[k] - a_rt * p.d_position_d_bg[k];
            j_y.fixed_view_mut::<3, 3>(0, 5).copy_from(&j_bg);
            if problem.estimate_accel_bias {
                j_y.fixed_view_mut::<3, 3>(0, 8)
                    .copy_from(&(-a_rt * p.d_position_d_ba[k]));
            }
            hpp += j_x.transpose() * j_x * w;
            hpy += j_x.transpose() * j_y * w;
            gp += j_x.transpose() * r * w;
            let jy_t_jy: SMatrix<f64, MAX_Y, MAX_Y> = j_y.transpose() * j_y * w;
            let jy_t_r: SMatrix<f64, MAX_Y, 1> = j_y.transpose() * r * w;
            for i in 0..ny {
                g_y[i] += jy_t_r[i];
                for j in 0..ny {
                    h_yy[(i, j)] += jy_t_jy[(i, j)];
                }
            }
        }
        h_pp.push(hpp);
        h_py.push(hpy);
        g_p.push(gp);
    }

    // Priors on the biases.
    let inv_sg2: f64 = 1.0 / (problem.gyro_bias_sigma * problem.gyro_bias_sigma);
    let dg: Vector3<f64> = state.bias_gyro - problem.gyro_bias_prior;
    for i in 0..3 {
        h_yy[(5 + i, 5 + i)] += inv_sg2;
        g_y[5 + i] += dg[i] * inv_sg2;
    }
    if problem.estimate_accel_bias {
        let inv_sa2: f64 = 1.0 / (problem.accel_bias_sigma * problem.accel_bias_sigma);
        for i in 0..3 {
            h_yy[(8 + i, 8 + i)] += inv_sa2;
            g_y[8 + i] += state.bias_accel[i] * inv_sa2;
        }
    }
    Normal {
        h_yy,
        g_y,
        h_pp,
        h_py,
        g_p,
    }
}

/// Schur-reduce the damped system and solve for the state increment.
fn solve_step(
    normal: &Normal,
    lambda: f64,
    ny: usize,
) -> Option<(DVector<f64>, Vec<Vector3<f64>>)> {
    let mut s: DMatrix<f64> = normal.h_yy.clone();
    let mut rhs: DVector<f64> = -normal.g_y.clone();
    for i in 0..ny {
        s[(i, i)] += lambda * (normal.h_yy[(i, i)] + 1e-9);
    }
    let mut inverses: Vec<Matrix3<f64>> = Vec::with_capacity(normal.h_pp.len());
    for ((hpp, hpy), gp) in normal.h_pp.iter().zip(&normal.h_py).zip(&normal.g_p) {
        let mut damped: Matrix3<f64> = *hpp;
        for i in 0..3 {
            damped[(i, i)] += lambda * hpp[(i, i)] + 1e-12;
        }
        let inv: Matrix3<f64> = damped.try_inverse()?;
        let hpy_n = hpy.columns(0, ny);
        let t = inv * hpy_n; // 3 x ny
        let reduce = hpy_n.transpose() * t; // ny x ny
        s -= reduce;
        let tg: Vector3<f64> = inv * gp;
        rhs += hpy_n.transpose() * tg;
        inverses.push(inv);
    }
    let dy: DVector<f64> = s.cholesky()?.solve(&rhs);
    let mut dps: Vec<Vector3<f64>> = Vec::with_capacity(inverses.len());
    for ((inv, hpy), gp) in inverses.iter().zip(&normal.h_py).zip(&normal.g_p) {
        let hpy_n = hpy.columns(0, ny);
        dps.push(-(inv * (gp + hpy_n * &dy)));
    }
    Some((dy, dps))
}

pub(super) fn apply(
    state: &State,
    dy: &DVector<f64>,
    dps: &[Vector3<f64>],
    basis: &SMatrix<f64, 3, 2>,
    estimate_ba: bool,
) -> State {
    let dtheta: Vector3<f64> = basis * nalgebra::Vector2::new(dy[3], dy[4]);
    let gravity: Vector3<f64> = Rotation3::exp(&dtheta).matrix() * state.gravity;
    State {
        velocity: state.velocity + Vector3::new(dy[0], dy[1], dy[2]),
        gravity,
        bias_gyro: state.bias_gyro + Vector3::new(dy[5], dy[6], dy[7]),
        bias_accel: if estimate_ba {
            state.bias_accel + Vector3::new(dy[8], dy[9], dy[10])
        } else {
            state.bias_accel
        },
        points: state.points.iter().zip(dps).map(|(p, d)| p + d).collect(),
    }
}

pub(super) fn gravity_basis(gravity: &Vector3<f64>) -> SMatrix<f64, 3, 2> {
    let (u1, u2) = tangent_basis(&gravity.normalize());
    SMatrix::<f64, 3, 2>::from_columns(&[u1, u2])
}

/// Levenberg-Marquardt with Marquardt diagonal scaling and a multiplicative
/// damping schedule.
pub(super) fn refine(problem: &Problem<'_>, initial: State, max_iterations: usize) -> Refined {
    let ny: usize = if problem.estimate_accel_bias { 11 } else { 8 };
    let mut state: State = initial;
    let mut cost: f64 = data_cost(problem, &state) + prior_cost(problem, &state);
    let mut lambda: f64 = 1e-4;
    let mut converged: bool = false;
    for _ in 0..max_iterations {
        let basis = gravity_basis(&state.gravity);
        let normal = linearize(problem, &state, &basis, ny);
        let mut accepted: bool = false;
        for _ in 0..12 {
            let Some((dy, dps)) = solve_step(&normal, lambda, ny) else {
                lambda *= 10.0;
                continue;
            };
            let candidate = apply(&state, &dy, &dps, &basis, problem.estimate_accel_bias);
            let new_cost: f64 = data_cost(problem, &candidate) + prior_cost(problem, &candidate);
            if new_cost.is_finite() && new_cost < cost {
                let decrease: f64 = (cost - new_cost) / cost.max(f64::MIN_POSITIVE);
                state = candidate;
                cost = new_cost;
                lambda = (lambda / 3.0).max(1e-12);
                accepted = true;
                if decrease < 1e-10 || dy.norm() < 1e-12 {
                    converged = true;
                }
                break;
            }
            lambda *= 4.0;
        }
        if !accepted {
            // No decrease at any damping: the state is a minimum to rounding.
            converged = true;
        }
        if converged {
            break;
        }
    }
    // The information always carries the accelerometer bias with its prior,
    // estimated or not: a bias held at zero is still uncertain, and leaving it
    // out would let a fixed wrong bias look like scale information.
    let consider = Problem {
        estimate_accel_bias: true,
        ..*problem
    };
    let basis = gravity_basis(&state.gravity);
    let normal = linearize(&consider, &state, &basis, MAX_Y);
    let mut information: DMatrix<f64> = normal.h_yy.clone();
    for (hpp, hpy) in normal.h_pp.iter().zip(&normal.h_py) {
        if let Some(inv) = hpp.try_inverse() {
            let hpy_n = hpy.columns(0, MAX_Y);
            information -= hpy_n.transpose() * (inv * hpy_n);
        }
    }
    let observations: usize = problem.tracks.iter().map(|t| t.len()).sum();
    Refined {
        data_cost: data_cost(problem, &state),
        state,
        converged,
        information,
        observations,
    }
}
