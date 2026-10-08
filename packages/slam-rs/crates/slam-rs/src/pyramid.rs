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
use crate::image::IngestError;
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::pyramid::{PyramidPlanError, PyramidPlanU16};

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
pub(crate) fn ensure_pyramid_sizes<B: PyramidBuilder>(
    builder: &B,
    pyramids: &mut Vec<B::Pyramid>,
    sizes: impl ExactSizeIterator<Item = ImageSize>,
    levels: usize,
) -> Result<(), PyramidError> {
    pyramids.truncate(sizes.len());
    for (camera, size) in sizes.enumerate() {
        let fits = pyramids.get(camera).is_some_and(|pyramid| {
            pyramid.num_levels() == levels + 1 && pyramid.level_size(0) == Some(size)
        });
        if !fits {
            let pyramid = builder.allocate(size.width, size.height, levels)?;
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

    /// Dimensions of one level, or `None` past the top.
    fn level_size(&self, level: usize) -> Option<ImageSize>;

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
    /// Staged pyramid validation or allocation failure.
    #[error(transparent)]
    Plan(#[from] PyramidPlanError),
    /// A level was asked for that the pyramid does not hold.
    #[error("level {level} does not exist; the pyramid holds {num_levels}")]
    NoSuchLevel {
        /// Level asked for.
        level: usize,
        /// Levels the pyramid holds.
        num_levels: usize,
    },
    /// The level geometry does not fit in memory.
    #[error("level geometry is not representable: {0}")]
    Image(#[from] IngestError),
    /// Staged GPU pyramid failure.
    #[cfg(feature = "gpu-core")]
    #[error(transparent)]
    Gpu(#[from] kornia_staging_gpu::pyramid::PyramidError),
}

impl Pyramid for PyramidPlanU16 {
    fn num_levels(&self) -> usize {
        self.levels().len()
    }

    fn level_size(&self, level: usize) -> Option<ImageSize> {
        self.levels().get(level).map(Image::size)
    }

    fn copy_level_into(&self, level: usize, out: &mut Image<u16, 1>) -> Result<(), PyramidError> {
        let source = self.levels().get(level).ok_or(PyramidError::NoSuchLevel {
            level,
            num_levels: self.levels().len(),
        })?;
        crate::image::copy_image(source, out)?;
        Ok(())
    }
}

/// CPU frameset scheduling adapter. Each staged plan owns its reusable storage.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuPyramidBuilder;

impl CpuPyramidBuilder {
    /// Create a stateless CPU builder.
    pub fn new() -> Self {
        Self
    }
}

impl PyramidBuilder for CpuPyramidBuilder {
    type Pyramid = PyramidPlanU16;

    fn build_frames(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [PyramidPlanU16],
        pool: &WorkPool,
    ) -> Result<(), PyramidError> {
        let build = |(image, pyramid): (&Image<u16, 1>, &mut PyramidPlanU16)| {
            pyramid.run(image).map_err(PyramidError::from)
        };
        if let Some(result) = pool.install(|| {
            use rayon::prelude::*;
            images
                .par_iter()
                .zip(out.par_iter_mut())
                .try_for_each(build)
        }) {
            result
        } else {
            images.iter().zip(out.iter_mut()).try_for_each(build)
        }
    }

    fn allocate(
        &self,
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<PyramidPlanU16, PyramidError> {
        PyramidPlanU16::new(ImageSize { width, height }, num_levels).map_err(Into::into)
    }

    fn build(
        &mut self,
        _camera: usize,
        img: &Image<u16, 1>,
        out: &mut PyramidPlanU16,
    ) -> Result<(), PyramidError> {
        out.run(img).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn the_pyramid_trait_copies_into_reusable_storage() {
        let image = Image::from_size_val(
            ImageSize {
                width: 32,
                height: 24,
            },
            4242u16,
        )
        .unwrap();
        let mut builder = CpuPyramidBuilder::new();
        let mut pyramid = builder.allocate(32, 24, 2).unwrap();
        builder.build(0, &image, &mut pyramid).unwrap();
        assert_eq!(pyramid.num_levels(), 3);
        assert_eq!(pyramid.level_size(0), Some(image.size()));
        assert_eq!(
            pyramid.level_size(2),
            Some(ImageSize {
                width: 8,
                height: 6
            })
        );
        assert_eq!(pyramid.level_size(3), None);
        let mut out = crate::image::empty();
        pyramid.copy_level_into(1, &mut out).unwrap();
        assert_eq!(
            out.size(),
            ImageSize {
                width: 16,
                height: 12
            }
        );
        assert_eq!(out.as_slice(), pyramid.levels()[1].as_slice());
        let pointer = out.as_slice().as_ptr();
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
        let mut builder = CpuPyramidBuilder::new();
        let mut pyramid = builder.allocate(32, 24, 2).unwrap();
        let image = Image::from_size_val(
            ImageSize {
                width: 32,
                height: 25,
            },
            1u16,
        )
        .unwrap();
        assert_eq!(
            builder.build(0, &image, &mut pyramid),
            Err(PyramidError::Plan(PyramidPlanError::GeometryMismatch {
                expected_width: 32,
                expected_height: 24,
                width: 32,
                height: 25,
            }))
        );
        assert_eq!(
            builder.allocate(8, 4, 2),
            Err(PyramidError::Plan(PyramidPlanError::TooSmall {
                width: 8,
                height: 4,
                max_level: 2
            }))
        );
    }
}
