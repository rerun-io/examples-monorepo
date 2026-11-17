//! Block-sparse symmetric matrices over 6x6 blocks and their Cholesky factor.
//!
//! [`SymbolicCholesky`] orders the block graph by minimum degree and records the fill pattern of the
//! factor once; [`SymbolicCholesky::factor`] then runs a left-looking block `L L^T` in place, as many
//! times as the LM loop needs. The ordering is the exact minimum degree of the elimination graph,
//! which for a pose graph (a chain, short covisibility edges and a few loops) keeps the fill linear in
//! the number of keyframes. No dependency beyond nalgebra, so it builds wherever the VIO builds.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use nalgebra::{Matrix6, Vector6};

/// A diagonal block lost positive definiteness during the factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("block {block} of the factor is not positive definite")]
pub struct NotPositiveDefinite {
    /// The block, in the caller's (unpermuted) numbering.
    pub block: usize,
}

/// Values of a symmetric block matrix on a [`SymbolicCholesky`] pattern; after
/// [`SymbolicCholesky::factor`], the values of its lower Cholesky factor.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockSymmetric {
    /// Diagonal blocks by elimination position: full symmetric blocks, or lower-triangular factor blocks.
    diagonal: Vec<Matrix6<f64>>,
    /// Strictly lower blocks, by slot of the pattern.
    lower: Vec<Matrix6<f64>>,
}

/// The ordering and fill pattern of a block Cholesky factor.
#[derive(Debug, Clone)]
pub struct SymbolicCholesky {
    /// Elimination position -> caller's block index.
    order: Vec<usize>,
    /// Caller's block index -> elimination position.
    position: Vec<usize>,
    /// Column `j`'s lower blocks are slots `column_start[j]..column_start[j + 1]`.
    column_start: Vec<usize>,
    /// Row position of each slot; sorted ascending within a column, always below the diagonal.
    row: Vec<usize>,
    /// Blocks of factor row `i` left of the diagonal are `row_slot[row_start[i]..row_start[i + 1]]`.
    row_start: Vec<usize>,
    /// `(column, slot)` of each block in a factor row, by ascending column.
    row_slot: Vec<(usize, usize)>,
}

impl SymbolicCholesky {
    /// Order a block graph and compute the pattern of its factor.
    ///
    /// # Arguments
    /// * `blocks` - Number of 6x6 diagonal blocks.
    /// * `edges` - Pairs of blocks with a nonzero off-diagonal block; repeats and either order are fine.
    ///   Self pairs and indices `>= blocks` are ignored.
    pub fn new(blocks: usize, edges: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); blocks];
        for (a, b) in edges {
            if a != b && a < blocks && b < blocks {
                adjacency[a].insert(b);
                adjacency[b].insert(a);
            }
        }

        // Exact minimum degree on the elimination graph, ties to the smaller index; stale heap entries
        // are skipped when their degree no longer matches.
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = (0..blocks)
            .map(|v| Reverse((adjacency[v].len(), v)))
            .collect();
        let mut eliminated = vec![false; blocks];
        let mut order = Vec::with_capacity(blocks);
        let mut structure: Vec<Vec<usize>> = vec![Vec::new(); blocks];
        while let Some(Reverse((degree, v))) = heap.pop() {
            if eliminated[v] || adjacency[v].len() != degree {
                continue;
            }
            eliminated[v] = true;
            order.push(v);
            let neighbours: Vec<usize> = std::mem::take(&mut adjacency[v]).into_iter().collect();
            for &u in &neighbours {
                adjacency[u].remove(&v);
                adjacency[u].extend(neighbours.iter().copied().filter(|&w| w != u));
                heap.push(Reverse((adjacency[u].len(), u)));
            }
            structure[v] = neighbours;
        }

        let mut position = vec![0; blocks];
        for (p, &v) in order.iter().enumerate() {
            position[v] = p;
        }
        let mut column_start = Vec::with_capacity(blocks + 1);
        let mut row = Vec::new();
        column_start.push(0);
        for &v in &order {
            let mut rows: Vec<usize> = structure[v].iter().map(|&u| position[u]).collect();
            rows.sort_unstable();
            row.extend(rows);
            column_start.push(row.len());
        }

        let mut counts = vec![0; blocks + 1];
        for &i in &row {
            counts[i + 1] += 1;
        }
        for i in 0..blocks {
            counts[i + 1] += counts[i];
        }
        let row_start = counts.clone();
        let mut row_slot = vec![(0, 0); row.len()];
        for j in 0..blocks {
            let column = column_start[j]..column_start[j + 1];
            for (slot, &i) in column.clone().zip(&row[column]) {
                row_slot[counts[i]] = (j, slot);
                counts[i] += 1;
            }
        }

        Self {
            order,
            position,
            column_start,
            row,
            row_start,
            row_slot,
        }
    }

    /// Number of diagonal blocks.
    pub fn blocks(&self) -> usize {
        self.order.len()
    }

    /// The elimination order: caller's block indices, first eliminated first.
    pub fn ordering(&self) -> &[usize] {
        &self.order
    }

    /// Number of strictly lower blocks of the factor, fill included.
    pub fn off_diagonal_blocks(&self) -> usize {
        self.row.len()
    }

    /// Whether the factor pattern has a block for this pair.
    pub fn has_block(&self, a: usize, b: usize) -> bool {
        self.slot(a, b).is_some()
    }

    /// The slot of the lower block for a pair of distinct blocks, and whether it stores `H_ab^T`;
    /// found once, so assembly adds at the slot without a search.
    pub(crate) fn slot(&self, a: usize, b: usize) -> Option<(usize, bool)> {
        let (pa, pb) = (*self.position.get(a)?, *self.position.get(b)?);
        let (i, j, transposed) = match pa.cmp(&pb) {
            std::cmp::Ordering::Greater => (pa, pb, false),
            std::cmp::Ordering::Less => (pb, pa, true),
            std::cmp::Ordering::Equal => return None,
        };
        let column = &self.row[self.column_start[j]..self.column_start[j + 1]];
        let k = column.binary_search(&i).ok()?;
        Some((self.column_start[j] + k, transposed))
    }

    /// An all-zero matrix on this pattern.
    pub fn zeros(&self) -> BlockSymmetric {
        BlockSymmetric {
            diagonal: vec![Matrix6::zeros(); self.blocks()],
            lower: vec![Matrix6::zeros(); self.row.len()],
        }
    }

    /// `H_aa += block`.
    pub fn add_diagonal(&self, h: &mut BlockSymmetric, a: usize, block: &Matrix6<f64>) {
        h.diagonal[self.position[a]] += block;
    }

    /// The diagonal block `H_aa`.
    pub fn diagonal(&self, h: &BlockSymmetric, a: usize) -> Matrix6<f64> {
        h.diagonal[self.position[a]]
    }

    /// `H_ab += block` (and `H_ba += block^T`).
    ///
    /// # Panics
    /// If `a == b` or the pair was not an edge given to [`SymbolicCholesky::new`].
    pub fn add_off_diagonal(
        &self,
        h: &mut BlockSymmetric,
        a: usize,
        b: usize,
        block: &Matrix6<f64>,
    ) {
        let (slot, transposed) = self
            .slot(a, b)
            .unwrap_or_else(|| panic!("blocks ({a}, {b}) are not an edge of the pattern"));
        self.add_to_slot(h, slot, transposed, block);
    }

    /// Add `H_ab` at a slot from [`Self::slot`].
    pub(crate) fn add_to_slot(
        &self,
        h: &mut BlockSymmetric,
        slot: usize,
        transposed: bool,
        block: &Matrix6<f64>,
    ) {
        if transposed {
            h.lower[slot] += block.transpose();
        } else {
            h.lower[slot] += block;
        }
    }

    /// `y = H x` with `x` and `y` in the caller's numbering, six entries per block.
    pub fn multiply(&self, h: &BlockSymmetric, x: &[f64], y: &mut [f64]) {
        let xs: Vec<Vector6<f64>> = self.gather(x);
        let mut ys = vec![Vector6::zeros(); self.blocks()];
        for j in 0..self.blocks() {
            ys[j] += h.diagonal[j] * xs[j];
            for slot in self.column_start[j]..self.column_start[j + 1] {
                let i = self.row[slot];
                ys[i] += h.lower[slot] * xs[j];
                ys[j] += h.lower[slot].tr_mul(&xs[i]);
            }
        }
        self.scatter(&ys, y);
    }

    /// Factor `h` in place into its lower Cholesky factor, `H = L L^T` in elimination order.
    ///
    /// # Errors
    /// [`NotPositiveDefinite`] names the first block whose Schur complement is not positive definite.
    pub fn factor(&self, h: &mut BlockSymmetric) -> Result<(), NotPositiveDefinite> {
        let mut slot_of_row = vec![usize::MAX; self.blocks()];
        for j in 0..self.blocks() {
            let column = self.column_start[j]..self.column_start[j + 1];
            for slot in column.clone() {
                slot_of_row[self.row[slot]] = slot;
            }
            let mut d = h.diagonal[j];
            for &(k, slot_jk) in &self.row_slot[self.row_start[j]..self.row_start[j + 1]] {
                let l_jk = h.lower[slot_jk];
                d -= l_jk * l_jk.transpose();
                // Rows of column k below row j update column j.
                for slot_ik in slot_jk + 1..self.column_start[k + 1] {
                    let update = h.lower[slot_ik] * l_jk.transpose();
                    h.lower[slot_of_row[self.row[slot_ik]]] -= update;
                }
            }
            let l_jj = d
                .cholesky()
                .ok_or(NotPositiveDefinite {
                    block: self.order[j],
                })?
                .unpack();
            h.diagonal[j] = l_jj;
            for slot in column {
                // L_ij = A_ij L_jj^-T, row by row.
                let mut transposed = h.lower[slot].transpose();
                for c in 0..6 {
                    let mut v = transposed.column(c).into_owned();
                    forward_substitute(&l_jj, &mut v);
                    transposed.set_column(c, &v);
                }
                h.lower[slot] = transposed.transpose();
            }
            for slot in self.column_start[j]..self.column_start[j + 1] {
                slot_of_row[self.row[slot]] = usize::MAX;
            }
        }
        Ok(())
    }

    /// Solve `L L^T x = rhs` in place, with a factor from [`Self::factor`] and `rhs` in the caller's numbering.
    pub fn solve(&self, factor: &BlockSymmetric, rhs: &mut [f64]) {
        let mut y = self.gather(rhs);
        for j in 0..self.blocks() {
            forward_substitute(&factor.diagonal[j], &mut y[j]);
            let yj = y[j];
            for slot in self.column_start[j]..self.column_start[j + 1] {
                y[self.row[slot]] -= factor.lower[slot] * yj;
            }
        }
        for j in (0..self.blocks()).rev() {
            let mut v = y[j];
            for slot in self.column_start[j]..self.column_start[j + 1] {
                v -= factor.lower[slot].tr_mul(&y[self.row[slot]]);
            }
            backward_substitute_transposed(&factor.diagonal[j], &mut v);
            y[j] = v;
        }
        self.scatter(&y, rhs);
    }

    /// Caller's numbering -> blocks in elimination order.
    fn gather(&self, x: &[f64]) -> Vec<Vector6<f64>> {
        self.order
            .iter()
            .map(|&v| Vector6::from_column_slice(&x[6 * v..6 * v + 6]))
            .collect()
    }

    /// Blocks in elimination order -> caller's numbering.
    fn scatter(&self, blocks: &[Vector6<f64>], out: &mut [f64]) {
        for (block, &v) in blocks.iter().zip(&self.order) {
            out[6 * v..6 * v + 6].copy_from_slice(block.as_slice());
        }
    }
}

/// `v = L^-1 v` for a lower-triangular `L` with a positive diagonal.
fn forward_substitute(l: &Matrix6<f64>, v: &mut Vector6<f64>) {
    for i in 0..6 {
        let mut sum = v[i];
        for k in 0..i {
            sum -= l[(i, k)] * v[k];
        }
        v[i] = sum / l[(i, i)];
    }
}

/// `v = L^-T v` for a lower-triangular `L` with a positive diagonal.
fn backward_substitute_transposed(l: &Matrix6<f64>, v: &mut Vector6<f64>) {
    for i in (0..6).rev() {
        let mut sum = v[i];
        for k in i + 1..6 {
            sum -= l[(k, i)] * v[k];
        }
        v[i] = sum / l[(i, i)];
    }
}

impl BlockSymmetric {
    /// Marquardt damping: `H_ii += lambda * scale_i` on every diagonal entry.
    ///
    /// `scale` is in the caller's numbering, six entries per block.
    pub fn add_scaled_identity(&mut self, symbolic: &SymbolicCholesky, lambda: f64, scale: &[f64]) {
        for (p, &v) in symbolic.order.iter().enumerate() {
            for k in 0..6 {
                self.diagonal[p][(k, k)] += lambda * scale[6 * v + k];
            }
        }
    }

    /// Zero every block, keeping the pattern.
    pub fn clear(&mut self) {
        self.diagonal.fill(Matrix6::zeros());
        self.lower.fill(Matrix6::zeros());
    }
}
