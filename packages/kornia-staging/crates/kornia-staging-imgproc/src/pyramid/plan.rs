//! Reusable floor-halved Gaussian image pyramids.
use std::sync::atomic::{AtomicU64, Ordering};

use kornia_image::{Image, ImageSize};

use super::u16::pyrdown_floor_u16_with_scratch;

/// Opaque identity of one complete pyramid build.
/// Cloning a pyramid preserves its identity; rebuilding creates a new identity.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct BuildGeneration(u64);

impl BuildGeneration {
    /// No image has been built yet.
    pub const NONE: Self = Self(0);

    fn next() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self(
            NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .unwrap_or_else(|_| panic!("pyramid build generation exhausted")),
        )
    }
}

impl std::fmt::Debug for BuildGeneration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Cache identity is not numerical state and must not change state fingerprints.
        f.write_str("BuildGeneration")
    }
}

/// Invalid pyramid geometry or an unrepresentable allocation layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PyramidPlanError {
    /// A source level is smaller than the Gaussian filter's three-pixel minimum.
    #[error("a {width}x{height} image cannot carry {max_level} floor-halved levels")]
    TooSmall {
        /// Input width.
        width: usize,
        /// Input height.
        height: usize,
        /// Requested number of reductions.
        max_level: usize,
    },
    /// Input dimensions do not match the plan.
    #[error("frame is {width}x{height}, expected {expected_width}x{expected_height}")]
    GeometryMismatch {
        /// Plan width.
        expected_width: usize,
        /// Plan height.
        expected_height: usize,
        /// Offered width.
        width: usize,
        /// Offered height.
        height: usize,
    },
    /// A level cannot fit in a single u16 allocation.
    #[error("a {width}x{height} u16 image exceeds the allocation layout limit")]
    LayoutOverflow {
        /// Requested width.
        width: usize,
        /// Requested height.
        height: usize,
    },
}

/// A Gaussian u16 pyramid with reusable levels and filter scratch.
/// Each reduction floor-halves both dimensions. This differs from the ceil-half
/// rule of upstream `pyrdown_u8` and `pyrdown_f32`. Filtering uses reflect-101,
/// exact i32 intermediate sums and one final rounding `(sum + 128) >> 8`.
#[derive(Clone)]
pub struct PyramidPlanU16 {
    levels: Vec<Image<u16, 1>>,
    scratch: Vec<i32>,
    generation: BuildGeneration,
}

impl PyramidPlanU16 {
    /// Allocate level zero and `max_level` floor-halved levels.
    ///
    /// # Arguments
    /// * `size` - Input dimensions in pixels.
    /// * `max_level` - Number of reductions after level zero; zero only copies input.
    ///
    /// # Errors
    /// Rejects source levels smaller than three pixels per side and allocation overflow.
    ///
    /// # Examples
    /// ```
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_staging_imgproc::pyramid::PyramidPlanU16;
    /// let size = ImageSize { width: 9, height: 7 };
    /// let source = Image::from_size_val(size, 1234u16)?;
    /// let mut pyramid = PyramidPlanU16::new(size, 2)?;
    /// pyramid.run(&source)?;
    /// assert_eq!(pyramid.levels()[2].size(), ImageSize { width: 2, height: 1 });
    /// assert_eq!(pyramid.levels()[2].as_slice(), &[1234, 1234]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(size: ImageSize, max_level: usize) -> Result<Self, PyramidPlanError> {
        let (mut width, mut height) = (size.width, size.height);
        for _ in 0..max_level {
            if width < 3 || height < 3 {
                return Err(PyramidPlanError::TooSmall {
                    width: size.width,
                    height: size.height,
                    max_level,
                });
            }
            width /= 2;
            height /= 2;
        }
        let layout_error = || PyramidPlanError::LayoutOverflow {
            width: size.width,
            height: size.height,
        };
        let len = size
            .width
            .checked_mul(size.height)
            .ok_or_else(layout_error)?;
        if len > isize::MAX as usize / size_of::<u16>() {
            return Err(layout_error());
        }
        let mut levels = Vec::with_capacity(max_level + 1);
        let (mut width, mut height) = (size.width, size.height);
        for _ in 0..=max_level {
            let level = Image::new(ImageSize { width, height }, vec![0u16; width * height])
                .map_err(|_| PyramidPlanError::LayoutOverflow { width, height })?;
            levels.push(level);
            width /= 2;
            height /= 2;
        }
        Ok(Self {
            levels,
            scratch: vec![0; if max_level == 0 { 0 } else { size.width }],
            generation: BuildGeneration::NONE,
        })
    }

    /// Copy `source` and rebuild each reduced level without allocating.
    ///
    /// # Arguments
    /// * `source` - Dense host image matching the dimensions passed to [`Self::new`].
    ///
    /// # Errors
    /// Returns [`PyramidPlanError::GeometryMismatch`] without changing the plan on a size mismatch.
    pub fn run(&mut self, source: &Image<u16, 1>) -> Result<(), PyramidPlanError> {
        let expected = self.levels[0].size();
        if source.size() != expected {
            return Err(PyramidPlanError::GeometryMismatch {
                expected_width: expected.width,
                expected_height: expected.height,
                width: source.width(),
                height: source.height(),
            });
        }
        self.generation = BuildGeneration::next();
        self.levels[0]
            .as_slice_mut()
            .copy_from_slice(source.as_slice());
        for level in 1..self.levels.len() {
            let (lower, upper) = self.levels.split_at_mut(level);
            pyrdown_floor_u16_with_scratch(&lower[level - 1], &mut upper[0], &mut self.scratch);
        }
        Ok(())
    }

    /// All levels in resolution order, starting with a copy of the input.
    #[inline]
    pub fn levels(&self) -> &[Image<u16, 1>] {
        &self.levels
    }

    /// Cache identity of the last build; clones retain it until rebuilt.
    #[inline]
    pub fn generation(&self) -> BuildGeneration {
        self.generation
    }
}

impl PartialEq for PyramidPlanU16 {
    fn eq(&self, other: &Self) -> bool {
        self.levels.len() == other.levels.len()
            && self.levels.iter().zip(&other.levels).all(|(left, right)| {
                left.size() == right.size() && left.as_slice() == right.as_slice()
            })
    }
}
impl Eq for PyramidPlanU16 {}

impl std::fmt::Debug for PyramidPlanU16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyramidPlanU16")
            .field(
                "levels",
                &self
                    .levels
                    .iter()
                    .map(|level| (level.size(), level.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .field("generation", &self.generation)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn plan_reuses_levels_and_changes_generation_only_after_a_valid_run() {
        let size = ImageSize {
            width: 41,
            height: 27,
        };
        let source = Image::from_size_val(size, 4242u16).unwrap();
        let mut plan = PyramidPlanU16::new(size, 3).unwrap();
        plan.run(&source).unwrap();
        assert_eq!(
            plan.levels()
                .iter()
                .map(|level| level.size())
                .collect::<Vec<_>>(),
            vec![
                size,
                ImageSize {
                    width: 20,
                    height: 13
                },
                ImageSize {
                    width: 10,
                    height: 6
                },
                ImageSize {
                    width: 5,
                    height: 3
                }
            ]
        );
        let scratch_pointer = plan.scratch.as_ptr();
        let pointers = plan
            .levels()
            .iter()
            .map(|level| level.as_slice().as_ptr())
            .collect::<Vec<_>>();
        let copied = plan.clone();
        let first = plan.generation();
        assert_eq!(copied.generation(), first);
        plan.run(&source).unwrap();
        assert_ne!(plan.generation(), first);
        assert_eq!(plan.scratch.as_ptr(), scratch_pointer);
        assert_eq!(
            pointers,
            plan.levels()
                .iter()
                .map(|level| level.as_slice().as_ptr())
                .collect::<Vec<_>>()
        );
        assert!(plan
            .levels()
            .iter()
            .all(|level| level.as_slice().iter().all(|&pixel| pixel == 4242)));
        let generation = plan.generation();
        let wrong = Image::from_size_val(
            ImageSize {
                width: 42,
                height: 27,
            },
            0u16,
        )
        .unwrap();
        assert!(plan.run(&wrong).is_err());
        assert_eq!(plan.generation(), generation);
        assert_eq!(plan, copied);
    }

    #[test]
    fn zero_reductions_copies_pixels_and_has_no_scratch() {
        let size = ImageSize {
            width: 2,
            height: 1,
        };
        let source = Image::new(size, vec![17, 65535]).unwrap();
        let mut plan = PyramidPlanU16::new(size, 0).unwrap();
        assert_eq!(plan.generation(), BuildGeneration::NONE);
        assert!(plan.scratch.is_empty());
        plan.run(&source).unwrap();
        assert_eq!(plan.levels()[0].as_slice(), source.as_slice());
        assert_ne!(
            plan.levels()[0].as_slice().as_ptr(),
            source.as_slice().as_ptr()
        );
        let mut other = PyramidPlanU16::new(size, 0).unwrap();
        other.run(&source).unwrap();
        assert_ne!(plan.generation(), other.generation());
        assert_eq!(plan, other);
        assert_eq!(format!("{plan:?}"), format!("{other:?}"));
    }

    #[test]
    fn constructor_rejects_invalid_geometry_and_layout_before_allocation() {
        let small = ImageSize {
            width: 8,
            height: 4,
        };
        assert_eq!(
            PyramidPlanU16::new(small, 2),
            Err(PyramidPlanError::TooSmall {
                width: 8,
                height: 4,
                max_level: 2
            })
        );
        let minimum = ImageSize {
            width: 8,
            height: 6,
        };
        assert!(PyramidPlanU16::new(minimum, 2).is_ok());
        assert!(PyramidPlanU16::new(minimum, 3).is_err());
        assert!(matches!(
            PyramidPlanU16::new(minimum, usize::MAX),
            Err(PyramidPlanError::TooSmall { .. })
        ));
        for size in [
            ImageSize {
                width: usize::MAX,
                height: 2,
            },
            ImageSize {
                width: isize::MAX as usize / 2 + 1,
                height: 1,
            },
        ] {
            assert_eq!(
                PyramidPlanU16::new(size, 0),
                Err(PyramidPlanError::LayoutOverflow {
                    width: size.width,
                    height: size.height
                })
            );
        }
    }
}
