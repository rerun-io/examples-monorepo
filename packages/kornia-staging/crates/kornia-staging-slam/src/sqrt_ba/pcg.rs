//! Implicit preconditioned conjugate gradients for the reduced camera system.
//!
//! Termination follows rootba / Ceres: the quadratic model value `Q` must stop
//! improving, `i (Q_i - Q_{i-1}) / Q_i < eta`. rootba's CG (from Ceres) also
//! recomputes the residual from `b - A x`, every 10 iterations; we do it every
//! [`GBA_CG_CHECK_EVERY`] = 8. Ours for f32 (DESIGN B24): every PCG dot product
//! accumulates in f64 in index order and the block-Jacobi inverses are factored
//! in f64. The matvec itself (in `global`) stays in `S`.
use kornia_staging_algebra::Scalar;
use nalgebra::{Matrix6, Vector6};

/// Iterations between explicit residual recomputations. One constant for every
/// lane (CPU and GPU) so the result never depends on the host.
pub const GBA_CG_CHECK_EVERY: usize = 8;

/// Limits of one PCG solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgOptions {
    /// Hard iteration cap (rootba 500).
    pub max_iterations: usize,
    /// Forcing sequence `eta` of the `Q` criterion (rootba 0.1).
    pub eta: f64,
}

impl Default for PcgOptions {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            eta: 0.1,
        }
    }
}

/// Why a PCG solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcgTermination {
    /// The right-hand side is zero; the solution is zero.
    ZeroRhs,
    /// The `Q` criterion holds.
    Converged,
    /// The iteration cap was reached.
    MaxIterations,
    /// `p' A p <= 0`: no more progress (rootba keeps the iterate).
    Indefinite,
    /// `rho`, `beta` or `alpha` became zero (with a nonzero residual) or
    /// non-finite.
    NumericalFailure,
}

/// Iteration count and termination of one PCG solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcgSummary {
    /// Matrix-vector products in the loop (one per iteration).
    pub iterations: usize,
    /// Why the solve stopped.
    pub termination: PcgTermination,
}

/// Inverted 6x6 diagonal blocks of the reduced system.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockJacobi<S: Scalar> {
    inverses: Vec<Matrix6<S>>,
    fallbacks: usize,
}

impl<S: Scalar> BlockJacobi<S> {
    /// Invert each block by an f64 Cholesky; a block that is not positive
    /// definite falls back to its inverted diagonal (zero where the diagonal is not
    /// positive) and is counted.
    pub fn from_blocks(blocks: &[Matrix6<f64>]) -> Self {
        let mut fallbacks = 0;
        let inverses = blocks
            .iter()
            .map(|block| {
                let inverse = block
                    .cholesky()
                    .map(|c| c.inverse())
                    .filter(|m| m.iter().all(|v| v.is_finite()));
                let inverse = inverse.unwrap_or_else(|| {
                    fallbacks += 1;
                    Matrix6::from_diagonal(&block.diagonal().map(|d| {
                        if d > 0.0 {
                            1.0 / d
                        } else {
                            0.0
                        }
                    }))
                });
                inverse.map(S::from_literal)
            })
            .collect();
        Self {
            inverses,
            fallbacks,
        }
    }

    /// Blocks that were not positive definite.
    pub fn fallbacks(&self) -> usize {
        self.fallbacks
    }

    /// `z = M^-1 r`.
    pub fn apply(&self, r: &[S], z: &mut [S]) {
        for (k, inv) in self.inverses.iter().enumerate() {
            let rk = Vector6::from_column_slice(&r[6 * k..6 * k + 6]);
            z[6 * k..6 * k + 6].copy_from_slice((inv * rk).as_slice());
        }
    }
}

/// Fixed-order dot product accumulated in f64.
pub(crate) fn dot<S: Scalar>(a: &[S], b: &[S]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x.to_f64() * y.to_f64()).sum()
}

/// Solve `A x = b` from `x = 0`. `matvec(v, out)` writes `A v` into `out`.
pub fn pcg<S: Scalar>(
    mut matvec: impl FnMut(&[S], &mut [S]),
    preconditioner: &BlockJacobi<S>,
    b: &[S],
    x: &mut [S],
    options: &PcgOptions,
) -> PcgSummary {
    let n = b.len();
    x.fill(S::zero());
    if dot(b, b) == 0.0 {
        return PcgSummary {
            iterations: 0,
            termination: PcgTermination::ZeroRhs,
        };
    }
    let mut r: Vec<S> = b.to_vec();
    let mut z: Vec<S> = vec![S::zero(); n];
    let mut p: Vec<S> = vec![S::zero(); n];
    let mut q: Vec<S> = vec![S::zero(); n];
    let mut rho = 1.0f64;
    // Q(x) = -x'(b + r) / 2 up to the factor; x = 0 gives Q0 = 0.
    let mut q0 = 0.0f64;
    let mut iterations = 0;
    let termination = loop {
        iterations += 1;
        preconditioner.apply(&r, &mut z);
        let last_rho = rho;
        rho = dot(&r, &z);
        if rho == 0.0 && dot(&r, &r) == 0.0 {
            // The residual is exactly zero: `x` solves the system.
            break PcgTermination::Converged;
        }
        if rho == 0.0 || !rho.is_finite() {
            break PcgTermination::NumericalFailure;
        }
        if iterations == 1 {
            p.copy_from_slice(&z);
        } else {
            let beta = rho / last_rho;
            if beta == 0.0 || !beta.is_finite() {
                break PcgTermination::NumericalFailure;
            }
            let beta = S::from_literal(beta);
            for (pi, zi) in p.iter_mut().zip(&z) {
                *pi = *zi + beta * *pi;
            }
        }
        matvec(&p, &mut q);
        let pq = dot(&p, &q);
        if pq <= 0.0 || pq.is_infinite() {
            break PcgTermination::Indefinite;
        }
        let alpha = rho / pq;
        if !alpha.is_finite() {
            break PcgTermination::NumericalFailure;
        }
        let alpha_s = S::from_literal(alpha);
        for (xi, pi) in x.iter_mut().zip(&p) {
            *xi += alpha_s * *pi;
        }
        if iterations % GBA_CG_CHECK_EVERY == 0 {
            // Residual replacement: drop the drift of the recursive update.
            matvec(x, &mut q);
            for ((ri, bi), qi) in r.iter_mut().zip(b).zip(&q) {
                *ri = *bi - *qi;
            }
        } else {
            for (ri, qi) in r.iter_mut().zip(&q) {
                *ri -= alpha_s * *qi;
            }
        }
        let q1 = -(dot(x, b) + dot(x, &r));
        let zeta = iterations as f64 * (q1 - q0) / q1;
        if zeta < options.eta {
            break PcgTermination::Converged;
        }
        q0 = q1;
        if iterations >= options.max_iterations {
            break PcgTermination::MaxIterations;
        }
    };
    PcgSummary {
        iterations,
        termination,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn spd(n_blocks: usize, seed: u64) -> DMatrix<f64> {
        let n = 6 * n_blocks;
        let j = DMatrix::from_fn(2 * n, n, |r, c| {
            (((r * 7 + c * 13) as u64 + seed) as f64 * 0.37).sin()
        });
        j.transpose() * j + DMatrix::identity(n, n)
    }

    fn blocks(a: &DMatrix<f64>) -> Vec<Matrix6<f64>> {
        (0..a.nrows() / 6)
            .map(|k| a.fixed_view::<6, 6>(6 * k, 6 * k).into_owned())
            .collect()
    }

    #[test]
    fn matches_a_dense_solve_when_run_to_tolerance() {
        for seed in 0..4 {
            let a = spd(5, seed);
            let b = DVector::from_fn(a.nrows(), |i, _| ((i as u64 + seed) as f64).cos());
            let expected = a.clone().lu().solve(&b).unwrap();
            let mut x = vec![0.0; a.nrows()];
            let options = PcgOptions {
                max_iterations: 200,
                eta: 1e-14,
            };
            let summary = pcg(
                |v, out| out.copy_from_slice((&a * DVector::from_column_slice(v)).as_slice()),
                &BlockJacobi::from_blocks(&blocks(&a)),
                b.as_slice(),
                &mut x,
                &options,
            );
            assert!(summary.iterations <= 200, "{summary:?}");
            let err = (DVector::from_vec(x) - &expected).amax() / expected.amax();
            assert!(err < 1e-8, "seed {seed}: {err} {summary:?}");
        }
    }

    #[test]
    fn exact_block_preconditioner_converges_in_one_iteration() {
        // A block-diagonal system is its own preconditioner.
        let mut a = DMatrix::zeros(18, 18);
        for (k, block) in blocks(&spd(3, 1)).into_iter().enumerate() {
            a.fixed_view_mut::<6, 6>(6 * k, 6 * k).copy_from(&block);
        }
        let b = DVector::from_fn(18, |i, _| i as f64 - 4.0);
        let mut x = vec![0.0; 18];
        let summary = pcg(
            |v, out| out.copy_from_slice((&a * DVector::from_column_slice(v)).as_slice()),
            &BlockJacobi::from_blocks(&blocks(&a)),
            b.as_slice(),
            &mut x,
            &PcgOptions::default(),
        );
        // Ceres' Q test needs a second iteration to see the model stop moving.
        assert!(summary.iterations <= 2, "{summary:?}");
        assert!((&a * DVector::from_vec(x) - b).amax() < 1e-9);
    }

    #[test]
    fn an_exactly_zero_residual_converges() {
        // A = 2 I, b = 1: one step gives x = 1/2 and r = 0 exactly, so the
        // second iteration sees rho = 0. That is convergence, not a failure.
        let a = DMatrix::<f64>::identity(6, 6) * 2.0;
        let mut x = vec![0.0f32; 6];
        let summary = pcg(
            |v, out| out.iter_mut().zip(v).for_each(|(o, vi)| *o = 2.0 * vi),
            &BlockJacobi::from_blocks(&blocks(&a)),
            &[1.0f32; 6],
            &mut x,
            &PcgOptions::default(),
        );
        assert_eq!(
            summary.termination,
            PcgTermination::Converged,
            "{summary:?}"
        );
        assert_eq!(summary.iterations, 2);
        assert_eq!(x, vec![0.5f32; 6]);
    }

    #[test]
    fn zero_rhs_and_indefinite_are_reported() {
        let a = spd(1, 0);
        let mut x = vec![1.0; 6];
        let pre = BlockJacobi::from_blocks(&blocks(&a));
        let s = pcg(
            |_, out| out.fill(0.0),
            &pre,
            &[0.0; 6],
            &mut x,
            &PcgOptions::default(),
        );
        assert_eq!(s.termination, PcgTermination::ZeroRhs);
        assert_eq!(x, vec![0.0; 6]);
        let s = pcg(
            |v, out| out.iter_mut().zip(v).for_each(|(o, vi)| *o = -vi),
            &pre,
            &[1.0; 6],
            &mut x,
            &PcgOptions::default(),
        );
        assert_eq!(s.termination, PcgTermination::Indefinite);
        let singular = BlockJacobi::<f32>::from_blocks(&[Matrix6::zeros()]);
        assert_eq!(singular.fallbacks(), 1);
    }
}
