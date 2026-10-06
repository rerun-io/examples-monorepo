//! Integer Gaussian image pyramids for the frontend.
//! A separable `[1, 4, 6, 4, 1]` filter uses reflect-101 borders, i32 accumulation
//! and one final rounding `(value + 128) >> 8`. Each level halves both dimensions.
//!
//! CPU levels own flat images; level zero is copied so pyramid lifetime and reuse
//! are independent of input frames. Three requested reductions produce four
//! levels, zero through three. The generic stage interface exposes geometry and
//! copies into caller buffers, never borrowed device memory.
//! Sparse KLT gradient sampling avoids constructing unused dense gradient images.

use crate::frontend::parallel::WorkPool;
use crate::image::ImageError;
use kornia_image::Image;
use std::sync::atomic::{AtomicU64, Ordering};

use kornia_staging_imgproc::pyramid::{PyrDownU16Scratch, pyrdown_u16_unchecked};

/// Minimum filter side length: the first output reaches source index two.
/// Refuse smaller geometry at construction (D32).
pub(crate) const MIN_SIDE: usize = 3;

/// The stage seam: build every level of one camera's pyramid in one call.
///
/// An associated `Pyramid` type from day one, so a GPU backend can carry
/// device handles instead of `Vec<u16>` without touching the frontend's
/// signature. The output is a `&mut` parameter, never a return value, so the
/// caller owns the allocation and the per-frame path never allocates.
pub trait PyramidBuilder {
    /// Build a frameset after allocating all its pyramids. Device backends keep
    /// their upload-before-dispatch ordering; CPU builders can split by camera.
    fn build_frames(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [Self::Pyramid],
        pool: &WorkPool,
    ) -> Result<(), PyramidError>;

    /// The pyramid representation this builder fills.
    ///
    /// Bounded by [`Pyramid`], so generic code can read geometry and copy
    /// pixels out but can never borrow the storage.
    type Pyramid: Pyramid;

    /// Fill `out` from `img`, camera `camera` of the rig.
    ///
    /// `camera` is the frame's place in the frameset, not a hint: a backend
    /// whose pyramid lives on a device publishes level 0 under that index so
    /// the detector's [`kornia_staging_imgproc::features::CornerScan`] — the only other
    /// stage that reads the same pixels — can read the copy already there
    /// rather than upload a second one. A backend that keeps its levels in host
    /// memory ignores it, because the caller still holds `img`.
    ///
    /// # Errors
    ///
    /// When `img`'s geometry does not match the one `out` was allocated for.
    /// (The dossier's sketch has this infallible; a `Result` is what keeps the
    /// mismatch from being a panic on a rayon worker inside the released-GIL
    /// region, which aborts the process — decision D32.)
    fn build(
        &mut self,
        camera: usize,
        img: &Image<u16, 1>,
        out: &mut Self::Pyramid,
    ) -> Result<(), PyramidError>;

    /// Allocate a pyramid this builder can fill for a `width` x `height` frame
    /// with `num_levels` halvings on top of level 0.
    ///
    /// Without this the seam is only half a seam: a caller generic over the
    /// builder could fill a pyramid but never make one, so it would have to name
    /// the concrete type to allocate. A GPU builder allocates device memory here.
    ///
    /// # Errors
    ///
    /// When the geometry cannot carry that many levels, or does not fit memory.
    fn allocate(
        &self,
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<Self::Pyramid, PyramidError>;
}

/// Reuse a frameset's pyramids while their geometry and level count match.
pub(crate) fn ensure_pyramids<B: PyramidBuilder>(
    builder: &B,
    pyramids: &mut Vec<B::Pyramid>,
    images: &[Image<u16, 1>],
    levels: usize,
) -> Result<(), PyramidError> {
    pyramids.truncate(images.len());
    for (camera, image) in images.iter().enumerate() {
        let fits = pyramids.get(camera).is_some_and(|pyramid| {
            pyramid.num_levels() == levels + 1
                && pyramid.level_size(0).is_some_and(|(width, height, _)| {
                    width == image.width() && height == image.height()
                })
        });
        if !fits {
            let pyramid = builder.allocate(image.width(), image.height(), levels)?;
            if let Some(slot) = pyramids.get_mut(camera) {
                *slot = pyramid;
            } else {
                pyramids.push(pyramid);
            }
        }
    }
    Ok(())
}

/// What generic frontend code may ask of a pyramid, whatever holds its pixels.
///
/// Geometry, and a copy into a buffer the caller owns. Deliberately no
/// accessor that hands out storage: a CubeCL pyramid's levels live in device
/// memory and there is nothing to lend, so a `&[u16]` here would be an API
/// that only the CPU backend could ever satisfy (deviation X04).
pub trait Pyramid {
    /// Stored level count, including level zero.
    fn num_levels(&self) -> usize;

    /// `(width, height, stride)` of one level, or `None` past the top.
    fn level_size(&self, level: usize) -> Option<(usize, usize, usize)>;

    /// Copy one level into `out`, resizing it to the level's geometry.
    ///
    /// `out` keeps its allocation when it is already big enough, so a caller
    /// that reads the same level every frame never allocates. Only the tests
    /// call it today — the detector reads the frame it was handed rather than a
    /// copy of level 0 — and it stays because it is the seam's forward half:
    /// the one way generic code can read a pyramid a GPU backend owns.
    ///
    /// # Errors
    ///
    /// [`PyramidError::NoSuchLevel`] past the top level, or the image error
    /// when `out` cannot be sized.
    fn copy_level_into(&self, level: usize, out: &mut Image<u16, 1>) -> Result<(), PyramidError>;
}

/// What can go wrong building a pyramid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PyramidError {
    /// A level would be too small for the 5-tap kernel's reach.
    #[error("a {width}x{height} image cannot carry {num_levels} subsampled levels")]
    TooSmall {
        /// Level-0 row length.
        width: usize,
        /// Level-0 row count.
        height: usize,
        /// Levels asked for, on top of level 0.
        num_levels: usize,
    },
    /// The frame does not have the geometry the pyramid was allocated for.
    #[error(
        "frame is {width}x{height}, the pyramid was built for {expected_width}x{expected_height}"
    )]
    GeometryMismatch {
        /// Level-0 row length the pyramid holds.
        expected_width: usize,
        /// Level-0 row count the pyramid holds.
        expected_height: usize,
        /// Row length of the frame offered.
        width: usize,
        /// Row count of the frame offered.
        height: usize,
    },
    /// A level was asked for that the pyramid does not hold.
    #[error("level {level} does not exist; the pyramid holds {num_levels}")]
    NoSuchLevel {
        /// Level asked for.
        level: usize,
        /// Levels the pyramid holds.
        num_levels: usize,
    },
    /// The `i32` accumulator for this frame size cannot be allocated.
    #[error("a {width}x{height} frame needs more scratch than one allocation may hold")]
    ScratchTooLarge {
        /// Level-0 row length.
        width: usize,
        /// Level-0 row count.
        height: usize,
    },
    /// The level geometry does not fit in memory.
    #[error("level geometry is not representable: {0}")]
    Image(#[from] ImageError),
    /// A GPU backend's device read failed.
    ///
    /// Carried here for the same reason [`crate::frontend::tracker::TrackerError`]
    /// carries it: the download in `copy_level_into` can fail on the device, and
    /// the trait's caller must get a typed error rather than a panic (D32).
    #[cfg(feature = "gpu-core")]
    #[error(transparent)]
    Gpu(#[from] crate::gpu::GpuError),
    /// A device download returned the wrong number of bytes.
    ///
    /// Only a GPU backend produces this. A CubeCL runtime whose shader
    /// compilation fails panics on its own worker thread and hands
    /// back a short buffer rather than an error, and reading that as pixels
    /// would quietly give a black pyramid; every download is length-checked and
    /// a short one is refused here instead (decision D32).
    #[error("reading level {level} returned {actual} bytes, expected {expected}")]
    ShortDeviceRead {
        /// Level asked for.
        level: usize,
        /// Bytes the device returned.
        actual: usize,
        /// Bytes the level's geometry needs.
        expected: usize,
    },
}

/// One camera's pyramid: level 0 plus `num_levels` halvings, each a flat buffer.
#[derive(Clone, Default)]
pub struct PyramidU16 {
    levels: Vec<Image<u16, 1>>,
    /// Unique across builds, including different allocations and builders.
    /// Clones keep the generation until either copy is rebuilt.
    generation: PyramidGeneration,
}

/// Opaque cache identity, not part of the pyramid's numerical state.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PyramidGeneration(u64);

impl std::fmt::Debug for PyramidGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Like an allocation address, the process-wide counter differs across
        // otherwise identical runs. Keep numerical state dumps deterministic.
        f.write_str("PyramidGeneration")
    }
}

// Equality describes pixels, not the build history used by template caches.
impl PartialEq for PyramidU16 {
    fn eq(&self, other: &Self) -> bool {
        self.levels.len() == other.levels.len()
            && self
                .levels
                .iter()
                .zip(&other.levels)
                .all(|(a, b)| a.size() == b.size() && a.as_slice() == b.as_slice())
    }
}

impl Eq for PyramidU16 {}

impl PyramidU16 {
    /// Allocate zero-filled levels; three reductions give levels zero through three.
    ///
    /// # Errors
    /// Refuse sides smaller than the kernel's reach or overflowing image geometry.
    pub fn with_capacity(
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<Self, PyramidError> {
        for level in 0..num_levels {
            // The source of every subsample step must be at least 3x3.
            if (width >> level) < MIN_SIDE || (height >> level) < MIN_SIDE {
                return Err(PyramidError::TooSmall {
                    width,
                    height,
                    num_levels,
                });
            }
        }
        let mut levels: Vec<Image<u16, 1>> = Vec::with_capacity(num_levels + 1);
        for level in 0..=num_levels {
            // Level dimensions are original dimensions shifted right by the level index.
            levels.push(crate::image::zeros(width >> level, height >> level)?);
        }
        Ok(Self {
            levels,
            generation: PyramidGeneration::default(),
        })
    }

    pub(crate) fn generation(&self) -> PyramidGeneration {
        self.generation
    }

    /// Level `level`, or `None` past the top.
    ///
    /// `pub(crate)` on purpose: this lends pyramid storage, which the trait
    /// seam may not do. The in-crate KLT tracker is the only caller.
    pub(crate) fn level(&self, level: usize) -> Option<&Image<u16, 1>> {
        self.levels.get(level)
    }
}

impl Pyramid for PyramidU16 {
    fn num_levels(&self) -> usize {
        self.levels.len()
    }

    fn level_size(&self, level: usize) -> Option<(usize, usize, usize)> {
        self.level(level)
            .map(|image| (image.width(), image.height(), image.width()))
    }

    fn copy_level_into(&self, level: usize, out: &mut Image<u16, 1>) -> Result<(), PyramidError> {
        let source: &Image<u16, 1> = self.level(level).ok_or(PyramidError::NoSuchLevel {
            level,
            num_levels: self.levels.len(),
        })?;
        crate::image::copy_image(source, out)?;
        Ok(())
    }
}

/// CPU pyramid builder with reusable i32 filtering scratch storage.
#[derive(Debug, Clone, Default)]
pub struct CpuPyramidBuilder {
    scratch: PyrDownU16Scratch,
    camera_scratch: Vec<PyrDownU16Scratch>,
}

impl CpuPyramidBuilder {
    /// A builder with no scratch buffer yet; the first `build` sizes it.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PyramidBuilder for CpuPyramidBuilder {
    type Pyramid = PyramidU16;

    fn build_frames(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [PyramidU16],
        pool: &WorkPool,
    ) -> Result<(), PyramidError> {
        self.camera_scratch
            .resize_with(images.len(), PyrDownU16Scratch::default);
        let build = |(image, (scratch, pyramid)): (
            &Image<u16, 1>,
            (&mut PyrDownU16Scratch, &mut PyramidU16),
        )| { build_cpu(image, pyramid, scratch) };
        if let Some(result) = pool.install(|| {
            use rayon::prelude::*;
            images
                .par_iter()
                .zip(self.camera_scratch.par_iter_mut().zip(out.par_iter_mut()))
                .try_for_each(build)
        }) {
            result
        } else {
            images
                .iter()
                .zip(self.camera_scratch.iter_mut().zip(out.iter_mut()))
                .try_for_each(build)
        }
    }

    fn allocate(
        &self,
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<PyramidU16, PyramidError> {
        PyramidU16::with_capacity(width, height, num_levels)
    }

    /// `ManagedImagePyr::setFromImage`.
    ///
    /// `_camera` is unused here: the levels are host memory and the caller
    /// still holds `img`, so the detector reads the frame itself.
    fn build(
        &mut self,
        _camera: usize,
        img: &Image<u16, 1>,
        out: &mut PyramidU16,
    ) -> Result<(), PyramidError> {
        build_cpu(img, out, &mut self.scratch)
    }
}

/// Shared by single-camera builds and the frameset's independent camera tasks.
fn build_cpu(
    img: &Image<u16, 1>,
    out: &mut PyramidU16,
    scratch: &mut PyrDownU16Scratch,
) -> Result<(), PyramidError> {
    let Some(level0) = out.levels.first() else {
        return Err(PyramidError::GeometryMismatch {
            expected_width: 0,
            expected_height: 0,
            width: img.width(),
            height: img.height(),
        });
    };
    if level0.width() != img.width() || level0.height() != img.height() {
        return Err(PyramidError::GeometryMismatch {
            expected_width: level0.width(),
            expected_height: level0.height(),
            width: img.width(),
            height: img.height(),
        });
    }

    // Invalidate templates before any pixels change, even if a later step fails.
    // A per-pyramid counter would alias unrelated images built equally often.
    static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);
    out.generation = PyramidGeneration(
        NEXT_GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .unwrap_or_else(|_| panic!("pyramid build generation exhausted")),
    );

    // Reuse the dense level-zero allocation.
    crate::image::copy_image(img, &mut out.levels[0])?;

    scratch
        .prepare(img.width())
        .map_err(|_| PyramidError::ScratchTooLarge {
            width: img.width(),
            height: img.height(),
        })?;
    for level in 1..out.levels.len() {
        let (lower, upper) = out.levels.split_at_mut(level);
        let (source, destination) = (&lower[level - 1], &mut upper[0]);
        pyrdown_u16_unchecked(source, destination, scratch);
    }
    Ok(())
}

impl std::fmt::Debug for PyramidU16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyramidU16")
            .field(
                "levels",
                &self
                    .levels
                    .iter()
                    .map(|image| (image.size(), image.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .field("generation", &self.generation)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn random_image(width: usize, height: usize, seed: u64) -> Image<u16, 1> {
        let mut image: Image<u16, 1> = crate::image::zeros(width, height).unwrap();
        let mut state: u64 = seed | 1;
        for y in 0..height {
            for pixel in {
                let width = image.width();
                &mut image.as_slice_mut()[(y) * width..((y) + 1) * width]
            } {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *pixel = (state >> 32) as u16;
            }
        }
        image
    }

    /// Every level, coarsest last. Generic code walks `0..num_levels()` like
    /// this; the tests do the same rather than borrowing the level vector.
    fn each_level(pyramid: &PyramidU16) -> Vec<&Image<u16, 1>> {
        (0..pyramid.num_levels())
            .map(|level| pyramid.level(level).unwrap())
            .collect()
    }

    fn build(image: &Image<u16, 1>, num_levels: usize) -> PyramidU16 {
        let mut pyramid: PyramidU16 =
            PyramidU16::with_capacity(image.width(), image.height(), num_levels).unwrap();
        CpuPyramidBuilder::new()
            .build(0, image, &mut pyramid)
            .unwrap();
        pyramid
    }

    #[test]
    fn three_levels_means_four_usable_levels() {
        // `optical_flow_levels = 3` -> levels 0..3.
        let pyramid: PyramidU16 = build(&random_image(64, 48, 1), 3);
        assert_eq!(pyramid.num_levels(), 4);
        let sizes: Vec<(usize, usize)> = each_level(&pyramid)
            .iter()
            .map(|level| (level.width(), level.height()))
            .collect();
        assert_eq!(sizes, vec![(64, 48), (32, 24), (16, 12), (8, 6)]);
    }

    #[test]
    fn level_geometry_shifts_the_original_size() {
        // `lvl(l)` is `(orig_w >> l, image.h >> l)`,
        // which for an odd side is not the same as halving the level below by
        // rounding up: 41 -> 20 -> 10 -> 5.
        let pyramid: PyramidU16 = build(&random_image(41, 27, 2), 3);
        let sizes: Vec<(usize, usize)> = each_level(&pyramid)
            .iter()
            .map(|level| (level.width(), level.height()))
            .collect();
        assert_eq!(sizes, vec![(41, 27), (20, 13), (10, 6), (5, 3)]);
    }

    #[test]
    fn level_zero_is_a_copy_of_the_frame() {
        let image: Image<u16, 1> = random_image(20, 14, 3);
        let pyramid: PyramidU16 = build(&image, 2);
        assert_eq!(pyramid.level(0).unwrap().as_slice(), image.as_slice());
    }

    #[test]
    fn a_flat_image_stays_flat() {
        // The kernel sums to 256 and the rounding is `(v * 256 + 128) >> 8`,
        // so a constant image is a fixed point at every level.
        let mut image: Image<u16, 1> = crate::image::zeros(32, 32).unwrap();
        for y in 0..image.height() {
            {
                let width = image.width();
                &mut image.as_slice_mut()[(y) * width..((y) + 1) * width]
            }
            .fill(4_242);
        }
        let pyramid: PyramidU16 = build(&image, 3);
        for level in each_level(&pyramid) {
            assert!(level.as_slice().iter().all(|pixel| *pixel == 4_242));
        }
    }

    #[test]
    fn a_geometry_that_cannot_carry_the_levels_is_refused() {
        assert_eq!(
            PyramidU16::with_capacity(8, 4, 2),
            Err(PyramidError::TooSmall {
                width: 8,
                height: 4,
                num_levels: 2
            })
        );
        // 8x6 -> 4x3 is fine as a source; 4x3 -> 2x1 is not.
        assert!(PyramidU16::with_capacity(8, 6, 2).is_ok());
        assert!(PyramidU16::with_capacity(8, 6, 3).is_err());
    }

    /// The seam hands out geometry and copies, never storage.
    #[test]
    fn the_pyramid_trait_lends_nothing() {
        let image: Image<u16, 1> = random_image(32, 24, 21);
        let pyramid: PyramidU16 = build(&image, 2);
        assert_eq!(pyramid.num_levels(), 3);
        assert_eq!(pyramid.level_size(0), Some((32, 24, 32)));
        assert_eq!(pyramid.level_size(2), Some((8, 6, 8)));
        assert_eq!(pyramid.level_size(3), None);

        let mut out: Image<u16, 1> = crate::image::empty();
        pyramid.copy_level_into(1, &mut out).unwrap();
        assert_eq!((out.width(), out.height()), (16, 12));
        assert_eq!(out.as_slice(), pyramid.level(1).unwrap().as_slice());

        // Copying the same level again reuses the caller's allocation.
        let pointer: *const u16 = out.as_slice().as_ptr();
        pyramid.copy_level_into(1, &mut out).unwrap();
        assert_eq!(out.as_slice().as_ptr(), pointer);

        assert_eq!(
            pyramid.copy_level_into(9, &mut out),
            Err(PyramidError::NoSuchLevel {
                level: 9,
                num_levels: 3
            })
        );
    }

    #[test]
    fn a_frame_of_the_wrong_size_is_refused() {
        let mut pyramid: PyramidU16 = PyramidU16::with_capacity(32, 24, 2).unwrap();
        let image: Image<u16, 1> = random_image(32, 25, 4);
        assert_eq!(
            CpuPyramidBuilder::new().build(0, &image, &mut pyramid),
            Err(PyramidError::GeometryMismatch {
                expected_width: 32,
                expected_height: 24,
                width: 32,
                height: 25
            })
        );
    }

    #[test]
    fn rebuilding_allocates_nothing() {
        let image: Image<u16, 1> = random_image(96, 64, 5);
        let mut pyramid: PyramidU16 = PyramidU16::with_capacity(96, 64, 3).unwrap();
        // The first build allocates each level; later builds must retain its storage.
        // Scratch reuse is also covered by the staged kernel tests.
        let mut builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
        builder.build(0, &image, &mut pyramid).unwrap();
        let pointers: Vec<*const u16> = each_level(&pyramid)
            .iter()
            .map(|level| level.as_slice().as_ptr())
            .collect();
        let capacities: Vec<usize> = each_level(&pyramid)
            .iter()
            .map(|level| level.as_slice().len())
            .collect();
        for seed in 6..12 {
            builder
                .build(0, &random_image(96, 64, seed), &mut pyramid)
                .unwrap();
        }
        let after: Vec<*const u16> = each_level(&pyramid)
            .iter()
            .map(|level| level.as_slice().as_ptr())
            .collect();
        assert_eq!(after, pointers, "a level buffer moved");
        assert_eq!(
            capacities,
            each_level(&pyramid)
                .iter()
                .map(|level| level.as_slice().len())
                .collect::<Vec<usize>>()
        );
    }
}
