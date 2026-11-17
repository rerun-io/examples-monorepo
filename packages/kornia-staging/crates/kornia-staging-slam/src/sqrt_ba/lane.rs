//! Where the PCG of the global BA runs (DESIGN §7.3).
//!
//! [`super::solve_global`] runs the PCG on the host, on the landmark blocks it
//! owns. [`super::solve_global_with`] hands each LM attempt's reduced camera
//! system to a [`PcgLane`] instead: the same landmark blocks, viewed as dense
//! `J` blocks over ascending keyframes, and the pose-graph rows as two-keyframe
//! blocks. Stage 1 (Jacobians, QR), stage 2 (damping, preconditioner, `b`) and
//! the back-substitution stay on the host for every lane, so a lane changes
//! only how `(J^T J + lambda I) x = b` is solved.
use kornia_staging_algebra::Scalar;
use nalgebra::Matrix6;

use super::{BlockJacobi, PcgOptions, PcgSummary};

/// One dense block `J` of the reduced camera system: `rows x 6 k` over `k`
/// ascending keyframes. A landmark block is its `Q2^T J_p` rows; a pose-graph
/// row is a `6 x 12` block.
#[derive(Debug, Clone, Copy)]
pub struct ReducedBlock<'a, S: Scalar> {
    /// Keyframe of each six-column group, strictly ascending.
    pub keyframes: &'a [usize],
    /// Column `c < 6 k` is `values[c * stride + offset..][..rows]`.
    pub values: &'a [S],
    /// Distance between column starts.
    pub stride: usize,
    /// First row of every column.
    pub offset: usize,
    /// Rows per column.
    pub rows: usize,
}

impl<S: Scalar> ReducedBlock<'_, S> {
    /// Column `c`.
    pub fn column(&self, c: usize) -> &[S] {
        &self.values[c * self.stride + self.offset..c * self.stride + self.offset + self.rows]
    }
}

/// The reduced camera system of one LM attempt: `H = J^T J + lambda I` over
/// `6 * keyframes` scaled pose unknowns.
#[derive(Debug, Clone)]
pub struct ReducedSystem<'a, S: Scalar> {
    /// Keyframe count.
    pub keyframes: usize,
    /// Damping on the scaled pose diagonal.
    pub lambda: S,
    /// Landmark blocks in landmark order, then the pose-graph rows.
    pub blocks: Vec<ReducedBlock<'a, S>>,
}

impl<S: Scalar> ReducedSystem<'_, S> {
    /// The 6x6 diagonal blocks of `H`, in f64: the block-Jacobi input.
    pub fn diagonal_blocks(&self) -> Vec<Matrix6<f64>> {
        let mut diagonal = vec![Matrix6::<f64>::identity() * self.lambda.to_f64(); self.keyframes];
        for block in &self.blocks {
            for (s, &kf) in block.keyframes.iter().enumerate() {
                for i in 0..6 {
                    for j in 0..6 {
                        diagonal[kf][(i, j)] += block
                            .column(6 * s + i)
                            .iter()
                            .zip(block.column(6 * s + j))
                            .map(|(a, b)| a.to_f64() * b.to_f64())
                            .sum::<f64>();
                    }
                }
            }
        }
        diagonal
    }

    /// `y = H x`, accumulated in f64 block by block: a reference for lanes and
    /// tests, not the fast path.
    pub fn matvec(&self, x: &[S], y: &mut [S]) {
        let mut acc: Vec<f64> = x
            .iter()
            .map(|v| self.lambda.to_f64() * v.to_f64())
            .collect();
        let mut t: Vec<f64> = Vec::new();
        for block in &self.blocks {
            t.clear();
            t.resize(block.rows, 0.0);
            for (s, &kf) in block.keyframes.iter().enumerate() {
                for j in 0..6 {
                    let xc = x[6 * kf + j].to_f64();
                    for (tr, v) in t.iter_mut().zip(block.column(6 * s + j)) {
                        *tr += v.to_f64() * xc;
                    }
                }
            }
            for (s, &kf) in block.keyframes.iter().enumerate() {
                for j in 0..6 {
                    let col = block.column(6 * s + j);
                    acc[6 * kf + j] += col
                        .iter()
                        .zip(&t)
                        .map(|(v, tr)| v.to_f64() * tr)
                        .sum::<f64>();
                }
            }
        }
        for (yi, a) in y.iter_mut().zip(acc) {
            *yi = S::from_literal(a);
        }
    }
}

/// What a lane reports for one solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaneSolve {
    /// Iterations and termination, with [`super::pcg`]'s meaning.
    pub summary: PcgSummary,
    /// Seconds spent moving the system to the lane (0 on the host).
    pub upload_s: f64,
    /// Host waits on the lane (0 on the host).
    pub syncs: usize,
}

/// A lane that could not solve: the global BA stops with this error rather
/// than switching lanes.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("the {lane} PCG lane failed: {reason}")]
pub struct PcgLaneError {
    /// The lane's name.
    pub lane: &'static str,
    /// The lane's own reason.
    pub reason: String,
}

/// A place to run the PCG of the reduced camera system.
///
/// A lane follows [`super::pcg`]: start from `x = 0`, block-Jacobi
/// preconditioner, residual replacement every
/// [`super::GBA_CG_CHECK_EVERY`] iterations, the `Q` stopping rule and the same
/// [`super::PcgTermination`] cases (a zero `b` gives `ZeroRhs` and `x = 0`).
pub trait PcgLane<S: Scalar> {
    /// The lane's name, for reports and errors.
    fn name(&self) -> &'static str;

    /// Solve `system x = b` into `x`.
    ///
    /// # Errors
    /// The lane could not run.
    fn solve(
        &mut self,
        system: &ReducedSystem<'_, S>,
        preconditioner: &BlockJacobi<S>,
        b: &[S],
        x: &mut [S],
        options: &PcgOptions,
    ) -> Result<LaneSolve, PcgLaneError>;
}
