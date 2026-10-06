//! Square-root prior arithmetic with fixed left folds.
use super::SqrtBaError;
use kornia_staging_algebra::Scalar;
use nalgebra::{DMatrix, DVector};
/// Borrowed prior Jacobian, residual and current displacement from its frozen origin.
/// The Jacobian is packed column-major; no frame ordering or state is retained.
#[derive(Debug, Clone, Copy)]
pub struct PriorLinearization<'a, S: Scalar> {
    j: &'a DMatrix<S>,
    residual: &'a DVector<S>,
    delta: &'a DVector<S>,
}
impl<'a, S: Scalar> PriorLinearization<'a, S> {
    /// Borrow a prior and validate its vector lengths once.
    /// # Errors
    /// Rejects residual or displacement lengths that differ from the matrix shape.
    pub fn new(
        j: &'a DMatrix<S>,
        residual: &'a DVector<S>,
        delta: &'a DVector<S>,
    ) -> Result<Self, SqrtBaError> {
        if residual.len() != j.nrows() || delta.len() != j.ncols() {
            return Err(SqrtBaError::Shape);
        }
        Ok(Self { j, residual, delta })
    }
    #[inline]
    fn row_dots<const N: usize>(&self, row: usize, vectors: [&DVector<S>; N]) -> [S; N] {
        let mut sums = [S::zero(); N];
        for j in 0..self.j.ncols() {
            let value = self.j[(row, j)];
            for (sum, vector) in sums.iter_mut().zip(vectors) {
                *sum += value * vector[j];
            }
        }
        sums
    }
    fn displacement(&self) -> DVector<S> {
        DVector::from_iterator(
            self.j.nrows(),
            (0..self.j.nrows()).map(|i| self.row_dots(i, [self.delta])[0]),
        )
    }
    fn error_from_displacement(&self, h_delta: &[S]) -> S {
        (0..self.j.nrows()).fold(S::zero(), |acc, i| {
            acc + h_delta[i] * (S::from_literal(0.5) * h_delta[i] + self.residual[i])
        })
    }
    /// Constant-free prior cost `deltaᵀ Jᵀ(0.5 J delta + r)`; it can be negative.
    pub fn error(&self) -> S {
        (0..self.j.nrows()).fold(S::zero(), |acc, i| {
            let value = self.row_dots(i, [self.delta])[0];
            acc + value * (S::from_literal(0.5) * value + self.residual[i])
        })
    }
    /// Add the prior normal equations into an already-validated destination and return its cost.
    /// The caller guarantees room for the prior columns in both matrix dimensions
    /// and the vector. The destination need not be square.
    pub fn add_dense(&self, h: &mut DMatrix<S>, b: &mut DVector<S>) -> S {
        let h_delta = self.displacement();
        for i in 0..self.j.ncols() {
            for j in 0..self.j.ncols() {
                let mut acc = S::zero();
                for k in 0..self.j.nrows() {
                    acc += self.j[(k, i)] * self.j[(k, j)];
                }
                h[(i, j)] += acc;
            }
            let mut acc = S::zero();
            for (k, &value) in h_delta.iter().enumerate() {
                acc += self.j[(k, i)] * (self.residual[k] + value);
            }
            b[i] += acc;
        }
        self.error_from_displacement(h_delta.as_slice())
    }
    /// Write the prior rows and re-anchor the residual as `J delta + r`.
    /// The caller guarantees enough columns and space through
    /// `start + j.nrows()` in both outputs.
    pub fn write_stacked(&self, output: &mut DMatrix<S>, residual: &mut DVector<S>, start: usize) {
        for i in 0..self.j.nrows() {
            for j in 0..self.j.ncols() {
                output[(start + i, j)] = self.j[(i, j)];
            }
            let acc = self.row_dots(i, [self.delta])[0];
            residual[start + i] = acc + self.residual[i];
        }
    }
    /// Predicted cost change, subtracting one row at a time.
    /// The increment must have one entry per prior column.
    pub fn model_cost_change(&self, increment: &DVector<S>) -> S {
        debug_assert_eq!(increment.len(), self.j.ncols());
        let mut l_diff = S::zero();
        for k in 0..self.j.nrows() {
            let [mut b_jdelta, j_inc] = self.row_dots(k, [self.delta, increment]);
            b_jdelta += self.residual[k];
            l_diff -= j_inc * (b_jdelta + S::from_literal(0.5) * j_inc);
        }
        l_diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prior_preserves_constant_free_cost_and_rejects_shapes() {
        let j = DMatrix::from_column_slice(2, 1, &[2.0, 3.0]);
        let r = DVector::from_column_slice(&[1.0, -1.0]);
        let delta = DVector::repeat(1, 2.0);
        assert!(PriorLinearization::new(&j, &DVector::zeros(1), &delta).is_err());
        let prior = PriorLinearization::new(&j, &r, &delta).unwrap();
        assert_eq!(prior.error(), 24.0);
        assert_eq!(prior.model_cost_change(&DVector::repeat(1, -1.0)), 18.5);
    }
}
