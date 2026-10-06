//! Deterministic dense scatter from packed column-major landmark rows.
use super::{LandmarkQr, SqrtBaError};
use kornia_staging_algebra::Scalar;
use nalgebra::{DMatrix, DVector};
const LANES: usize = 8;
/// Borrowed numerical Q₂ rows; no landmark identity or state is retained.
#[derive(Clone, Copy, Debug)]
pub struct DenseBlock<'a, S: Scalar> {
    qr: &'a LandmarkQr<S>,
    storage: &'a [S],
}
impl<'a, S: Scalar> DenseBlock<'a, S> {
    /// Borrow storage belonging to this validated landmark layout.
    /// # Panics
    /// The packed storage length must match the layout.
    #[inline]
    pub fn new(qr: &'a LandmarkQr<S>, storage: &'a [S]) -> Self {
        debug_assert_eq!(storage.len(), qr.rows() * qr.columns());
        Self { qr, storage }
    }
    #[inline]
    fn q2(&self, row: usize, column: usize) -> S {
        self.storage[3 + row + column * self.qr.rows()]
    }
    fn fill_rows(&self, columns: impl ExactSizeIterator<Item = usize>, rows: &mut Vec<S>) -> usize {
        let live = columns.len();
        let stride = (live + 1).div_ceil(LANES) * LANES;
        rows.clear();
        rows.resize(self.rows() * stride, S::zero());
        for (slot, column) in columns.enumerate() {
            for r in 0..self.rows() {
                rows[r * stride + slot] = self.q2(r, column);
            }
        }
        for r in 0..self.rows() {
            rows[r * stride + live] = self.q2(r, self.qr.residual_column());
        }
        stride
    }
    /// Number of pose columns.
    pub fn pose_columns(&self) -> usize {
        self.qr.pose_columns()
    }
    /// Number of reduced rows, including damping rows.
    pub fn rows(&self) -> usize {
        self.qr.rows() - 3
    }
    /// Copy Q₂ rows into a dynamic matrix and residual vector.
    /// Caller guarantees room for `start + self.rows()` and all pose columns.
    pub fn write_stacked_unchecked(
        &self,
        output: &mut DMatrix<S>,
        residual: &mut DVector<S>,
        start: usize,
    ) {
        for r in 0..self.rows() {
            residual[start + r] = self.storage[3 + r + self.qr.residual_column() * self.qr.rows()];
            for k in 0..self.qr.pose_columns() {
                output[(start + r, k)] = self.storage[3 + r + k * self.qr.rows()];
            }
        }
    }
    /// Whether sparse symmetric scatter preserves every Q₂ coefficient, including NaNs and signed zero.
    pub fn active_writeback_is_exact(&self) -> bool {
        let rows: usize = self.qr.rows() - 3;
        let mut active = self.qr.active_columns().iter().copied().peekable();
        for column in 0..self.qr.pose_columns() {
            if active.next_if_eq(&column).is_some() {
                for r in 0..rows {
                    if !self.storage[3 + r + column * self.qr.rows()]
                        .to_f64()
                        .is_finite()
                    {
                        return false;
                    }
                }
            } else {
                // `-0.0 == 0.0`, which is what this wants: either zero makes
                // every product `±0.0`.
                for r in 0..rows {
                    if self.storage[3 + r + column * self.qr.rows()] != S::zero() {
                        return false;
                    }
                }
            }
        }
        // The residual column, the other factor of `b`.
        (0..rows).all(|r| {
            self.storage[3 + r + self.qr.residual_column() * self.qr.rows()]
                .to_f64()
                .is_finite()
        })
    }

    /// Add every pose column, preserving non-finite propagation.
    /// Caller guarantees `h` has at least the pose width in both dimensions,
    /// and `b` has at least the pose width.
    pub fn add_full_unchecked(
        &self,
        h: &mut DMatrix<S>,
        b: &mut DVector<S>,
        scratch: &mut DenseHbWorkspace<S>,
    ) {
        self.coefficients(
            self.pose_columns(),
            |i| i,
            scratch,
            |i, j, value| {
                if let Some(j) = j {
                    h[(i, j)] += value;
                } else {
                    b[i] += value;
                }
            },
        );
    }
    /// Contract selected columns in ascending row order, without an intermediate zero addition.
    /// Caller guarantees that `columns` are valid pose-column indices.
    pub fn coefficients_unchecked(
        &self,
        columns: &[usize],
        scratch: &mut DenseHbWorkspace<S>,
        write: impl FnMut(usize, Option<usize>, S),
    ) {
        self.coefficients(columns.len(), |i| columns[i], scratch, write);
    }
    fn coefficients(
        &self,
        live: usize,
        column: impl Fn(usize) -> usize,
        scratch: &mut DenseHbWorkspace<S>,
        mut write: impl FnMut(usize, Option<usize>, S),
    ) {
        let stride = self.fill_rows((0..live).map(&column), &mut scratch.rows);
        for slot in 0..live {
            let i = column(slot);
            for lo in (0..stride).step_by(LANES) {
                let partial = contract_rows(&scratch.rows, stride, slot, lo);
                for (offset, &value) in partial.iter().enumerate().take((live + 1 - lo).min(LANES))
                {
                    let j = lo + offset;
                    write(i, if j < live { Some(column(j)) } else { None }, value);
                }
            }
        }
    }
}
/// Reusable row-transpose scratch. Output buffers remain owned by the caller.
#[derive(Debug, Clone)]
pub struct DenseHbWorkspace<S: Scalar> {
    rows: Vec<S>,
}
impl<S: Scalar> Default for DenseHbWorkspace<S> {
    fn default() -> Self {
        Self { rows: Vec::new() }
    }
}
impl<S: Scalar> DenseHbWorkspace<S> {
    /// Reset the square output to positive zero, then fold blocks in iterator order.
    /// # Errors
    /// Rejects wrong output lengths or a block wider than the output. Earlier blocks
    /// may already have been accumulated when a later block has the wrong width.
    /// ```
    /// use kornia_staging_slam::sqrt_ba::{DenseHbWorkspace, DenseBlock, LandmarkQr};
    /// let qr = LandmarkQr::<f64>::new(1, 2, vec![0,1])?;
    /// let storage = vec![0.0; qr.rows()*qr.columns()];
    /// let block = DenseBlock::new(&qr, &storage);
    /// use nalgebra::{DMatrix, DVector};
    /// DenseHbWorkspace::default().reduce_into([block], &mut DMatrix::zeros(2,2), &mut DVector::zeros(2))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reduce_into<'a>(
        &mut self,
        blocks: impl IntoIterator<Item = DenseBlock<'a, S>>,
        h: &mut DMatrix<S>,
        b: &mut DVector<S>,
    ) -> Result<(), SqrtBaError> {
        let n = b.len();
        if h.shape() != (n, n) {
            return Err(SqrtBaError::Shape);
        }
        h.fill(S::zero());
        b.fill(S::zero());
        for block in blocks {
            if block.qr.pose_columns() > n {
                return Err(SqrtBaError::SystemSize {
                    expected: block.qr.pose_columns(),
                    found: n,
                });
            }
            if !block.active_writeback_is_exact() {
                block.add_full_unchecked(h, b, self);
                continue;
            }
            let columns = block.qr.active_columns();
            let live = columns.len();
            let stride = block.fill_rows(columns.iter().copied(), &mut self.rows);
            for (slot, &i) in columns.iter().enumerate() {
                for lo in ((slot / LANES * LANES)..stride).step_by(LANES) {
                    let partial = contract_rows(&self.rows, stride, slot, lo);
                    for (offset, value) in partial.into_iter().enumerate() {
                        let j = lo + offset;
                        if j < slot {
                            continue;
                        }
                        if j < live {
                            let column = columns[j];
                            h[(i, column)] += value;
                            if j != slot {
                                h[(column, i)] += value;
                            }
                        } else if j == live {
                            b[i] += value;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Each lane folds rows in input order, starting from positive zero.
#[inline]
fn contract_rows<S: Scalar>(rows: &[S], stride: usize, slot: usize, lo: usize) -> [S; LANES] {
    let mut partial = [S::zero(); LANES];
    for row in rows.chunks_exact(stride) {
        let factor = row[slot];
        for (sum, &value) in partial.iter_mut().zip(&row[lo..lo + LANES]) {
            *sum += factor * value;
        }
    }
    partial
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_rejects_wrong_output_shape_or_width() {
        let qr = LandmarkQr::<f64>::new(2, 2, vec![0, 1]).unwrap();
        let data = vec![0.0; qr.rows() * qr.columns()];
        let block = DenseBlock::new(&qr, &data);
        assert!(DenseHbWorkspace::default()
            .reduce_into([block], &mut DMatrix::zeros(1, 1), &mut DVector::zeros(2))
            .is_err());
        assert!(DenseHbWorkspace::default()
            .reduce_into([block], &mut DMatrix::zeros(1, 1), &mut DVector::zeros(1))
            .is_err());
    }
    fn compare<S: Scalar>(nan: bool, width: usize) {
        let qr = LandmarkQr::<S>::new(4, width, vec![0, 2, width - 1]).unwrap();
        let mut data = vec![S::zero(); qr.rows() * qr.columns()];
        for col in [0, 2, width - 1, qr.residual_column()] {
            for r in 3..qr.rows() {
                data[r + col * qr.rows()] = S::from_literal(((r * 13 + col * 7) as f64).sin());
            }
        }
        // Inactive signed zero is safe, but nonfinite residuals require every column.
        data[3 + qr.rows()] = -S::zero();
        let first_data = data.clone();
        let first = DenseBlock::new(&qr, &first_data);
        if nan {
            data[3 + qr.residual_column() * qr.rows()] = S::from_literal(f64::NAN);
        }
        let block = DenseBlock::new(&qr, &data);
        assert_eq!(block.active_writeback_is_exact(), !nan);
        let mut actual_h = DMatrix::repeat(width, width, S::one());
        let mut actual_b = DVector::repeat(width, S::one());
        DenseHbWorkspace::default()
            .reduce_into([first, block], &mut actual_h, &mut actual_b)
            .unwrap();
        let mut h = vec![S::zero(); width * width];
        let mut b = vec![S::zero(); width];
        for data in [&first_data, &data] {
            for i in 0..width {
                for j in 0..width {
                    let mut sum = S::zero();
                    for r in 3..qr.rows() {
                        sum += data[r + i * qr.rows()] * data[r + j * qr.rows()];
                    }
                    h[i + j * width] += sum;
                }
                let mut sum = S::zero();
                for r in 3..qr.rows() {
                    sum += data[r + i * qr.rows()] * data[r + qr.residual_column() * qr.rows()];
                }
                b[i] += sum;
            }
        }
        for (actual, expected) in actual_h.iter().chain(&actual_b).zip(h.iter().chain(&b)) {
            assert_eq!(actual.to_f64().to_bits(), expected.to_f64().to_bits());
        }
    }
    #[test]
    fn row_fold_and_full_width_nan_bits_are_preserved() {
        for width in [6, 17] {
            compare::<f32>(false, width);
            compare::<f64>(false, width);
            compare::<f32>(true, width);
            compare::<f64>(true, width);
        }
    }
}
