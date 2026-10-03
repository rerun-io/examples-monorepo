//! The keypoint lists the tracker reads and writes: [`PointsSoA`] (positions)
//! and [`FlowTransforms`] (warps), each coordinate in its own flat array with
//! the keypoint index varying fastest.

use nalgebra::{Matrix2, Vector2};

use crate::frontend::se2::AffineCompact2f;
use crate::types::KeypointId;

/// One camera's inputs and output slot in a submitted tracking batch.
/// IDs and positions are in the same order as the initial guesses.
#[derive(Debug, Default)]
pub struct TrackInput {
    /// Source camera index in the supplied previous pyramid set.
    pub source: usize,
    /// Destination camera index in the current pyramid set.
    pub destination: usize,
    /// Keypoint identities, ascending, after source masking.
    pub ids: Vec<KeypointId>,
    /// Source template positions at level zero.
    pub positions: PointsSoA,
    /// Source linear transforms with predicted destination positions.
    pub guesses: FlowTransforms,
    /// Result slot set by batch submission.
    pub result: usize,
}

/// A list of 2-D points with the coordinates in two flat arrays.
///
/// `Vec<Vector2<f32>>` would give one coordinate a stride of two floats; here a
/// warp reading every patch's `x` reads consecutive addresses (§12.2 item 1).
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
/// what makes the frontend's per-frame snapshot allocation-free (see
/// [`crate::frontend::flow::FrameToFrameOpticalFlow::process_frame`]).
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
    pub fn push(&mut self, point: Vector2<f32>) {
        self.x.push(point.x);
        self.y.push(point.y);
    }

    /// Overwrite point `index`.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn set(&mut self, index: usize, point: Vector2<f32>) {
        self.x[index] = point.x;
        self.y[index] = point.y;
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
    pub fn get(&self, index: usize) -> Vector2<f32> {
        Vector2::new(self.x[index], self.y[index])
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
        self.m00.push(warp.linear[(0, 0)]);
        self.m01.push(warp.linear[(0, 1)]);
        self.m10.push(warp.linear[(1, 0)]);
        self.m11.push(warp.linear[(1, 1)]);
        self.tx.push(warp.translation.x);
        self.ty.push(warp.translation.y);
    }

    /// Insert one warp at `index`, shifting the rest up.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn insert(&mut self, index: usize, warp: &AffineCompact2f) {
        self.m00.insert(index, warp.linear[(0, 0)]);
        self.m01.insert(index, warp.linear[(0, 1)]);
        self.m10.insert(index, warp.linear[(1, 0)]);
        self.m11.insert(index, warp.linear[(1, 1)]);
        self.tx.insert(index, warp.translation.x);
        self.ty.insert(index, warp.translation.y);
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
        self.m00[index] = warp.linear[(0, 0)];
        self.m01[index] = warp.linear[(0, 1)];
        self.m10[index] = warp.linear[(1, 0)];
        self.m11[index] = warp.linear[(1, 1)];
        self.tx[index] = warp.translation.x;
        self.ty[index] = warp.translation.y;
    }

    /// The warp at `index`, reassembled.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn get(&self, index: usize) -> AffineCompact2f {
        AffineCompact2f {
            linear: Matrix2::new(
                self.m00[index],
                self.m01[index],
                self.m10[index],
                self.m11[index],
            ),
            translation: Vector2::new(self.tx[index], self.ty[index]),
        }
    }

    /// The translation at `index`, without reassembling the linear part.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
    pub fn translation(&self, index: usize) -> Vector2<f32> {
        Vector2::new(self.tx[index], self.ty[index])
    }

    /// The six coefficients at `index`, in the order
    /// [`AffineCompact2f::coefficients`] uses.
    ///
    /// # Panics
    ///
    /// If `index` is past the end.
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
    /// both want. [`crate::frontend::parallel::WorkPool::for_each_warp`] takes
    /// exactly this shape.
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
    /// The prefix [`crate::frontend::parallel::WorkPool::for_each_warp`] wants
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
