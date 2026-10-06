//! The keypoint lists the tracker reads and writes: [`PointsSoA`] (positions)
//! and [`FlowTransforms`] (warps), each coordinate in its own flat array with
//! the keypoint index varying fastest.

use crate::optical_flow::patch_se2::AffineCompact2f;

/// A list of 2-D points with the coordinates in two flat arrays.
///
/// `Vec<Vector2<f32>>` would give one coordinate a stride of two floats; here a
/// warp reading every patch's `x` reads consecutive addresses.
#[derive(Debug, Default, PartialEq)]
pub struct PointsSoA {
    x: Vec<f32>,
    y: Vec<f32>,
}

/// `Clone` by hand for the sake of `clone_from`.
///
/// `#[derive(Clone)]` only writes `clone`; `clone_from` then falls back to
/// `*self = source.clone()`, which drops both buffers and allocates two more.
/// Copying field by field lets `Vec::clone_from` overwrite in place, which is
/// what makes the frontend's per-frame snapshot allocation-free.
impl Clone for PointsSoA {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.x.clone_from(&source.x);
        self.y.clone_from(&source.y);
    }
}

// Nothing asks a point list, a warp list or a patch set whether it is empty:
// they are sized to a capacity at construction and read by index.
#[allow(clippy::len_without_is_empty)]
impl PointsSoA {
    /// An empty list with room for `capacity` points.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            x: Vec::with_capacity(capacity),
            y: Vec::with_capacity(capacity),
        }
    }

    /// Points held.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Drop every point, keeping the allocation.
    pub fn clear(&mut self) {
        self.x.clear();
        self.y.clear();
    }

    /// Append one point.
    pub fn push(&mut self, point: impl Into<[f32; 2]>) {
        let point = point.into();
        self.x.push(point[0]);
        self.y.push(point[1]);
    }

    /// Overwrite point `index`.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn set(&mut self, index: usize, point: impl Into<[f32; 2]>) {
        let point = point.into();
        self.x[index] = point[0];
        self.y[index] = point[1];
    }

    /// Grow to `len` points, filling with the origin.
    pub fn resize(&mut self, len: usize) {
        self.x.resize(len, 0.0);
        self.y.resize(len, 0.0);
    }

    /// Point `index`.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    #[inline]
    pub fn get(&self, index: usize) -> [f32; 2] {
        [self.x[index], self.y[index]]
    }

    /// Every `x` coordinate, patch index fast-varying.
    pub fn xs(&self) -> &[f32] {
        &self.x
    }

    /// Every `y` coordinate, patch index fast-varying.
    pub fn ys(&self) -> &[f32] {
        &self.y
    }
}

/// Affine warps in six flat coefficient arrays, with keypoint index varying fastest.
/// [`FlowTransforms::get`] reconstructs one warp with six loads.
#[derive(Debug, Default, PartialEq)]
pub struct FlowTransforms {
    m00: Vec<f32>,
    m01: Vec<f32>,
    m10: Vec<f32>,
    m11: Vec<f32>,
    tx: Vec<f32>,
    ty: Vec<f32>,
}

/// `Clone` by hand, for the `clone_from` reason on [`PointsSoA`].
impl Clone for FlowTransforms {
    fn clone(&self) -> Self {
        Self {
            m00: self.m00.clone(),
            m01: self.m01.clone(),
            m10: self.m10.clone(),
            m11: self.m11.clone(),
            tx: self.tx.clone(),
            ty: self.ty.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.m00.clone_from(&source.m00);
        self.m01.clone_from(&source.m01);
        self.m10.clone_from(&source.m10);
        self.m11.clone_from(&source.m11);
        self.tx.clone_from(&source.tx);
        self.ty.clone_from(&source.ty);
    }
}

#[allow(clippy::len_without_is_empty)]
impl FlowTransforms {
    /// An empty list with room for `capacity` warps.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            m00: Vec::with_capacity(capacity),
            m01: Vec::with_capacity(capacity),
            m10: Vec::with_capacity(capacity),
            m11: Vec::with_capacity(capacity),
            tx: Vec::with_capacity(capacity),
            ty: Vec::with_capacity(capacity),
        }
    }

    /// Warps held.
    pub fn len(&self) -> usize {
        self.m00.len()
    }

    /// Drop every warp, keeping the allocation.
    pub fn clear(&mut self) {
        self.m00.clear();
        self.m01.clear();
        self.m10.clear();
        self.m11.clear();
        self.tx.clear();
        self.ty.clear();
    }

    /// Grow to `len` warps, filling with the identity.
    pub fn resize(&mut self, len: usize) {
        self.m00.resize(len, 1.0);
        self.m01.resize(len, 0.0);
        self.m10.resize(len, 0.0);
        self.m11.resize(len, 1.0);
        self.tx.resize(len, 0.0);
        self.ty.resize(len, 0.0);
    }

    /// Append one warp.
    pub fn push(&mut self, warp: &AffineCompact2f) {
        for (column, value) in [
            &mut self.m00,
            &mut self.m01,
            &mut self.m10,
            &mut self.m11,
            &mut self.tx,
            &mut self.ty,
        ]
        .into_iter()
        .zip(warp.coefficients())
        {
            column.push(value);
        }
    }

    /// Insert one warp at `index`, shifting the rest up.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn insert(&mut self, index: usize, warp: &AffineCompact2f) {
        for (column, value) in [
            &mut self.m00,
            &mut self.m01,
            &mut self.m10,
            &mut self.m11,
            &mut self.tx,
            &mut self.ty,
        ]
        .into_iter()
        .zip(warp.coefficients())
        {
            column.insert(index, value);
        }
    }

    /// Remove the warp at `index`, shifting the rest down.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn remove(&mut self, index: usize) {
        self.m00.remove(index);
        self.m01.remove(index);
        self.m10.remove(index);
        self.m11.remove(index);
        self.tx.remove(index);
        self.ty.remove(index);
    }

    /// Overwrite the warp at `index`.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn set(&mut self, index: usize, warp: &AffineCompact2f) {
        for (column, value) in [
            &mut self.m00,
            &mut self.m01,
            &mut self.m10,
            &mut self.m11,
            &mut self.tx,
            &mut self.ty,
        ]
        .into_iter()
        .zip(warp.coefficients())
        {
            column[index] = value;
        }
    }

    /// The warp at `index`, reassembled.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    #[inline]
    pub fn get(&self, index: usize) -> AffineCompact2f {
        AffineCompact2f::from_coefficients(self.coefficients(index))
    }

    /// The translation at `index`, without reassembling the linear part.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    #[inline]
    pub fn translation(&self, index: usize) -> [f32; 2] {
        [self.tx[index], self.ty[index]]
    }

    /// The six coefficients at `index`, in the order
    /// [`AffineCompact2f::coefficients`] uses.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    // Keep these six loads in cross-crate point loops; an outlined call adds a return buffer.
    #[inline(always)]
    pub fn coefficients(&self, index: usize) -> [f32; 6] {
        [
            self.m00[index],
            self.m01[index],
            self.m10[index],
            self.m11[index],
            self.tx[index],
            self.ty[index],
        ]
    }

    /// The six coefficient arrays, mutably, in the order
    /// [`AffineCompact2f::coefficients`] uses.
    ///
    /// This is how a backend writes a whole camera's warps without going through
    /// one warp at a time: each array is contiguous with the patch index
    /// fast-varying, which is what a device copy and a rayon `par_chunks_mut`
    /// both want.
    pub fn coefficients_mut(&mut self) -> [&mut [f32]; 6] {
        [
            &mut self.m00,
            &mut self.m01,
            &mut self.m10,
            &mut self.m11,
            &mut self.tx,
            &mut self.ty,
        ]
    }

    /// The first `len` entries of the six coefficient arrays, mutably.
    ///
    /// The prefix [`Self::fill_with`] wants
    /// when a capacity-sized buffer is carrying `len` live warps, which is the
    /// tracker's shape on both of its passes.
    ///
    /// # Panics
    ///
    /// If `len` is past the end of the arrays.
    pub fn coefficients_prefix_mut(&mut self, len: usize) -> [&mut [f32]; 6] {
        [
            &mut self.m00[..len],
            &mut self.m01[..len],
            &mut self.m10[..len],
            &mut self.m11[..len],
            &mut self.tx[..len],
            &mut self.ty[..len],
        ]
    }

    /// Every translation `x`, patch index fast-varying.
    pub fn translations_x(&self) -> &[f32] {
        &self.tx
    }

    /// Every translation `y`, patch index fast-varying.
    pub fn translations_y(&self) -> &[f32] {
        &self.ty
    }
}

impl FlowTransforms {
    /// Fill a prefix from independent warp calculations using a caller-owned pool.
    ///
    /// Each slot is evaluated once; None executes sequentially. `init` creates
    /// local state once per worker chunk, allowing borrowed image views to be
    /// reused across points without sharing mutable state between workers.
    /// # Panics
    /// If count exceeds the stored warp or validity length.
    pub fn fill_with<S>(
        &mut self,
        pool: Option<&rayon::ThreadPool>,
        count: usize,
        valid: &mut [bool],
        init: impl Fn() -> S + Sync + Send,
        body: impl Fn(&mut S, usize) -> ([f32; 6], bool) + Sync + Send,
    ) {
        assert!(
            valid.len() >= count,
            "validity storage is shorter than the requested prefix"
        );
        use rayon::prelude::*;
        let [m00, m01, m10, m11, tx, ty] = self.coefficients_prefix_mut(count);
        let len = count;

        let Some(pool) = pool else {
            let mut state = init();
            for index in 0..len {
                let (warp, flag) = body(&mut state, index);
                m00[index] = warp[0];
                m01[index] = warp[1];
                m10[index] = warp[2];
                m11[index] = warp[3];
                tx[index] = warp[4];
                ty[index] = warp[5];
                valid[index] = flag;
            }
            return;
        };

        // A fixed split for a given (len, threads): eight chunks per worker of
        // at least eight warps, so a worker that drew the expensive points does
        // not hold the others up. Which worker runs a chunk is rayon's choice
        // and cannot matter: every index is a pure function of itself.
        let chunk: usize = len.div_ceil(pool.current_num_threads() * 8).max(8);
        pool.install(|| {
            m00[..len]
                .par_chunks_mut(chunk)
                .zip(m01[..len].par_chunks_mut(chunk))
                .zip(m10[..len].par_chunks_mut(chunk))
                .zip(m11[..len].par_chunks_mut(chunk))
                .zip(tx[..len].par_chunks_mut(chunk))
                .zip(ty[..len].par_chunks_mut(chunk))
                .zip(valid[..len].par_chunks_mut(chunk))
                .enumerate()
                .for_each(|(block, ((((((m00, m01), m10), m11), tx), ty), valid))| {
                    let base: usize = block * chunk;
                    let mut state = init();
                    for offset in 0..valid.len() {
                        let (warp, flag) = body(&mut state, base + offset);
                        m00[offset] = warp[0];
                        m01[offset] = warp[1];
                        m10[offset] = warp[2];
                        m11[offset] = warp[3];
                        tx[offset] = warp[4];
                        ty[offset] = warp[5];
                        valid[offset] = flag;
                    }
                });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "validity storage is shorter than the requested prefix")]
    fn a_short_validity_buffer_is_rejected_before_writes() {
        let mut warps = FlowTransforms::with_capacity(2);
        warps.resize(2);
        warps.fill_with(
            None,
            2,
            &mut [false],
            || (),
            |_, _| panic!("must not write"),
        );
    }
}
