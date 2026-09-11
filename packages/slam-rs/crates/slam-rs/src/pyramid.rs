//! The frontend's integer Gaussian image pyramid.
//!
//! Ported from `thirdparty/basalt-headers/include/basalt/image/image_pyr.h`.
//! `subsample` (`image_pyr.h:99-140`) is reproduced operation for operation:
//! a separable 5-tap `[1, 4, 6, 4, 1]` Gaussian with BORDER_REFLECT_101
//! extrapolation, integer accumulation in an `i32` scratch buffer laid out
//! row-major over `dst_height` x `src_width` — which reproduces C++'s
//! transposed *arithmetic*, not its layout, see `subsample` — and one
//! rounding at the very end, `(val + (1 << 7)) >> 8` (`image_pyr.h:135`).
//! Each level halves both dimensions after filtering and rounds once to u16.
//!
//! ## What is not reproduced: the mipmap allocation
//!
//! basalt packs every level into one `w + w/2` wide buffer (`image_pyr.h:71`)
//! and hands out sub-images (`lvl`, `image_pyr.h:146-152`). That is an
//! allocation trick, not a numerical one, and it is the one thing the GPU seam
//! forbids: a level must be a flat buffer with a stateable stride so it uploads
//! as one `create_from_slice`. So [`PyramidU16`] owns one [`ImageU16`] per
//! level and keeps basalt's level *geometry* — level `l` is
//! `(width >> l, height >> l)` (`image_pyr.h:149-150`) — exactly (deviation X04).
//!
//! ## Level 0 is a copy, not a borrow
//!
//! basalt copies too (`lvl_internal(0).CopyFrom(other)`, `image_pyr.h:73`), and
//! the seam rule is that no public signature returns a borrowed view into
//! pyramid memory: the whole pyramid is one residency unit that a GPU backend
//! uploads and owns. A borrowed level 0 would also tie the pyramid's lifetime
//! to the input frame, which the per-frame reuse in
//! [`CpuPyramidBuilder::build`] is built to avoid.
//!
//! ## Level counting
//!
//! `setFromImage(other, num_levels)` builds levels `1..=num_levels`
//! (`image_pyr.h:75-79`), so basalt's `optical_flow_levels = 3` means **four**
//! usable levels, 0 through 3. [`PyramidU16::with_capacity`] takes the same
//! `num_levels` and allocates `num_levels + 1` levels.
//!
//! ## The seam lends nothing
//!
//! [`PyramidBuilder`] and its associated [`Pyramid`] type expose **no borrows**:
//! only geometry ([`Pyramid::num_levels`], [`Pyramid::level_size`]) and a copy
//! into a buffer the caller already owns ([`Pyramid::copy_level_into`]). That is
//! the §12.3.2 rule — a GPU pyramid holds device handles and cannot lend a
//! `&[u16]`, so generic frontend code written against `P: PyramidBuilder` must
//! never be able to ask for one. The concrete CPU [`PyramidU16`] does keep one
//! borrowing accessor, `level`, because the KLT tracker lands in this crate and
//! reads pixels straight out of a CPU pyramid, but it is `pub(crate)`: the
//! borrow stops at the crate boundary and cannot reach a public signature.
//!
//! ## No fused gradient here yet
//!
//! The CubeCL review wants "downsample + gradient" fused into one
//! [`PyramidBuilder::build`] call. The KLT tracker samples a sparse 52-tap
//! pattern and calls [`ImageU16::interp_grad`] per tap, so a dense gradient
//! image would be built and discarded; the fused pass lands with the tracker,
//! when there is a consumer that reads it.

use crate::image::{ImageError, ImageU16};

/// The 5-tap Gaussian, `image_pyr.h:102`.
const KERNEL: [i32; 5] = [1, 4, 6, 4, 1];

/// Smallest side `subsample` can read: it indexes `abs(2 * 0 - 2) == 2`.
///
/// `image_pyr.h:110` reaches two rows above the first output row without a
/// bounds check; in C++ that is out-of-range for a two-row image. The port
/// refuses the geometry at construction instead (decision D32).
pub(crate) const MIN_SIDE: usize = 3;

/// The stage seam: build every level of one camera's pyramid in one call.
///
/// An associated `Pyramid` type from day one, so a GPU backend can carry
/// device handles instead of `Vec<u16>` without touching the frontend's
/// signature. The output is a `&mut` parameter, never a return value, so the
/// caller owns the allocation and the per-frame path never allocates.
pub trait PyramidBuilder {
    /// Whether images are uploaded together before any pyramid dispatch.
    const PREPARE_IMAGES: bool = false;

    /// Prepare this frameset's uploads after all pyramid allocations.
    fn prepare_images(&mut self, _images: &[ImageU16]) -> Result<(), PyramidError> {
        Ok(())
    }

    /// The pyramid representation this builder fills.
    ///
    /// Bounded by [`Pyramid`], so generic code can read geometry and copy
    /// pixels out but can never borrow the storage.
    type Pyramid: Pyramid;

    /// Fill `out` from `img`, camera `camera` of the rig.
    ///
    /// `camera` is the frame's place in the frameset, not a hint: a backend
    /// whose pyramid lives on a device publishes level 0 under that index so
    /// the detector's [`crate::frontend::detect::CornerScan`] — the only other
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
        img: &ImageU16,
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

/// What generic frontend code may ask of a pyramid, whatever holds its pixels.
///
/// Geometry, and a copy into a buffer the caller owns. Deliberately no
/// accessor that hands out storage: a CubeCL pyramid's levels live in device
/// memory and there is nothing to lend, so a `&[u16]` here would be an API
/// that only the CPU backend could ever satisfy (deviation X04).
pub trait Pyramid {
    /// Levels held, which is basalt's `num_levels + 1`.
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
    fn copy_level_into(&self, level: usize, out: &mut ImageU16) -> Result<(), PyramidError>;
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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PyramidU16 {
    levels: Vec<ImageU16>,
}

impl PyramidU16 {
    /// Allocate every level for a `width` x `height` frame, zero-filled.
    ///
    /// `num_levels` is basalt's: `with_capacity(w, h, 3)` gives four levels,
    /// 0 through 3, matching `optical_flow_levels = 3`.
    ///
    /// # Errors
    ///
    /// [`PyramidError::TooSmall`] when a level would be narrower or shorter
    /// than the kernel's reach, [`PyramidError::Image`] when the geometry does
    /// not fit in a `usize`.
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
        let mut levels: Vec<ImageU16> = Vec::with_capacity(num_levels + 1);
        for level in 0..=num_levels {
            // basalt's `lvl(l)`: `(orig_w >> lvl, image.h >> lvl)` (`image_pyr.h:149-150`).
            levels.push(ImageU16::zeros(width >> level, height >> level)?);
        }
        Ok(Self { levels })
    }

    /// Level `level`, or `None` past the top.
    ///
    /// `pub(crate)` on purpose: this lends pyramid storage, which the trait
    /// seam may not do. The in-crate KLT tracker is the only caller.
    pub(crate) fn level(&self, level: usize) -> Option<&ImageU16> {
        self.levels.get(level)
    }
}

impl Pyramid for PyramidU16 {
    fn num_levels(&self) -> usize {
        self.levels.len()
    }

    fn level_size(&self, level: usize) -> Option<(usize, usize, usize)> {
        self.level(level)
            .map(|image| (image.width(), image.height(), image.stride()))
    }

    fn copy_level_into(&self, level: usize, out: &mut ImageU16) -> Result<(), PyramidError> {
        let source: &ImageU16 = self.level(level).ok_or(PyramidError::NoSuchLevel {
            level,
            num_levels: self.levels.len(),
        })?;
        out.copy_from(source)?;
        Ok(())
    }
}

/// The CPU [`PyramidBuilder`]: basalt's `subsample`, level by level.
///
/// Owns the `i32` accumulator `subsample` needs (`image_pyr.h:105`) so the
/// per-frame path allocates nothing once the builder has seen one frame of a
/// given size.
#[derive(Debug, Clone, Default)]
pub struct CpuPyramidBuilder {
    scratch: Vec<i32>,
}

impl CpuPyramidBuilder {
    /// A builder with no scratch buffer yet; the first `build` sizes it.
    pub fn new() -> Self {
        Self::default()
    }
}

impl PyramidBuilder for CpuPyramidBuilder {
    type Pyramid = PyramidU16;

    fn allocate(
        &self,
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<PyramidU16, PyramidError> {
        PyramidU16::with_capacity(width, height, num_levels)
    }

    /// `ManagedImagePyr::setFromImage`, `image_pyr.h:70-80`.
    ///
    /// `_camera` is unused here: the levels are host memory and the caller
    /// still holds `img`, so the detector reads the frame itself.
    fn build(
        &mut self,
        _camera: usize,
        img: &ImageU16,
        out: &mut PyramidU16,
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

        // `lvl_internal(0).CopyFrom(other)` (`image_pyr.h:73`). `copy_from` is
        // the same row-by-row copy, which is what a possibly strided source
        // needs; the geometry check above makes its resize a no-op.
        out.levels[0].copy_from(img)?;

        // `for (i = 0; i < num_levels; i++) subsample(lvl(i), lvl_internal(i + 1))`.
        self.scratch
            .resize(scratch_len(img.width(), img.height())?, 0);
        for level in 1..out.levels.len() {
            let (lower, upper) = out.levels.split_at_mut(level);
            // `level >= 1`, so `lower` holds level - 1 and `upper` starts at level.
            let (source, destination) = (&lower[level - 1], &mut upper[0]);
            subsample(source, destination, &mut self.scratch);
        }
        Ok(())
    }
}

/// Scratch elements `subsample` needs at level 0, which is its largest use.
///
/// `ManagedImage<int> tmp(img_sub.h, img.w)` (`image_pyr.h:105`) is
/// `img.w * (img.h / 2)` integers whichever way they are laid out; [`subsample`]
/// holds them row-major over `dst_height` x `src_width`. A saturating product
/// would turn an impossible geometry into a `vec!` that aborts the process, so
/// the overflow and the `isize::MAX`-byte allocation cap are both typed errors.
fn scratch_len(width: usize, height: usize) -> Result<usize, PyramidError> {
    let len: usize = width
        .checked_mul(height >> 1)
        .ok_or(PyramidError::ScratchTooLarge { width, height })?;
    if len > crate::image::max_elements::<i32>() {
        return Err(PyramidError::ScratchTooLarge { width, height });
    }
    Ok(len)
}

/// BORDER_REFLECT_101 for an index past the *high* end, `image_pyr.h:83`.
///
/// `h - 1 - |h - 1 - x|`. It is only a reflection for `x >= 0`; for a negative
/// index basalt uses `std::abs` instead (`image_pyr.h:110-111`), which is the
/// same reflection about 0 but a different expression. Trap 3 of the
/// architecture dossier is precisely that these two are not one function.
#[inline]
fn border101(x: i64, h: i64) -> i64 {
    h - 1 - (h - 1 - x).abs()
}

/// basalt's `subsample`, `image_pyr.h:99-140`, operation for operation.
///
/// C++ writes a **transposed** accumulator: `ManagedImage<int> tmp(img_sub.h,
/// img.w)`, whose width is the destination height, holds `tmp(r, c)` with `r`
/// the destination row and `c` the source column (`image_pyr.h:117`), and the
/// horizontal pass walks its rows (`:126-136`). What that transposition decides
/// is the *arithmetic*: `tmp.h` is the source width, so the second pass's
/// `border101(2 * c + 2, tmp.h)` reflects about the source width, and that is
/// reproduced here.
///
/// The **layout** is not reproduced, because it costs a cache line per
/// coefficient in both directions: the vertical pass would stride its writes by
/// `dst_height` and the horizontal pass would stride its `dst` writes by the
/// row. The accumulator here is row-major over `dst_height` x `src_width`, so
/// both passes run along their rows. Every value and every index is the one C++
/// computes — the taps are integers, where an order is not a rounding.
///
/// There is exactly one rounding, at the end of the horizontal pass, so the
/// separable form is bit-identical to a direct 5x5 convolution.
///
/// # Panics
///
/// If `src` is narrower or shorter than [`MIN_SIDE`], or `dst` is not
/// `(src.width() >> 1, src.height() >> 1)`, or `scratch` is short. All three
/// are established by [`PyramidU16::with_capacity`] and checked by
/// [`CpuPyramidBuilder::build`].
fn subsample(src: &ImageU16, dst: &mut ImageU16, scratch: &mut [i32]) {
    let src_width: usize = src.width();
    let src_height: usize = src.height();
    let dst_width: usize = dst.width();
    let dst_height: usize = dst.height();
    debug_assert_eq!(dst_width, src_width >> 1);
    debug_assert_eq!(dst_height, src_height >> 1);

    // Vertical convolution, `image_pyr.h:108-121`, one accumulator row per
    // destination row.
    for r in 0..dst_height {
        let row2: i64 = 2 * r as i64;
        // `std::abs(2 * r - 2)` and `std::abs(2 * r - 1)`, not `border101`.
        let rows: [usize; 5] = [
            (row2 - 2).unsigned_abs() as usize,
            (row2 - 1).unsigned_abs() as usize,
            row2 as usize,
            border101(row2 + 1, src_height as i64) as usize,
            border101(row2 + 2, src_height as i64) as usize,
        ];
        let [row_m2, row_m1, row_0, row_p1, row_p2]: [&[u16]; 5] = [
            src.row(rows[0]),
            src.row(rows[1]),
            src.row(rows[2]),
            src.row(rows[3]),
            src.row(rows[4]),
        ];
        // `tmp(r, c)`, one contiguous run of `c` rather than one column of it.
        let band: &mut [i32] = &mut scratch[r * src_width..(r + 1) * src_width];
        for c in 0..src_width {
            band[c] = KERNEL[0] * i32::from(row_m2[c])
                + KERNEL[1] * i32::from(row_m1[c])
                + KERNEL[2] * i32::from(row_0[c])
                + KERNEL[3] * i32::from(row_p1[c])
                + KERNEL[4] * i32::from(row_p2[c]);
        }
    }

    // Horizontal convolution, `image_pyr.h:123-139`. `tmp.h` is `src_width`, so
    // the reflection is about the **source** width whichever way `tmp` is laid
    // out.
    for r in 0..dst_height {
        let band: &[i32] = &scratch[r * src_width..(r + 1) * src_width];
        for (c, pixel) in dst.row_mut(r).iter_mut().enumerate() {
            // An interior column's five taps are the contiguous window
            // `band[2c - 2 ..= 2c + 2]`. The reflection only bites at `c == 0`,
            // where C++ takes `abs` of a negative index, and at the last column
            // or two, where `2c + 2` runs past `src_width - 1`; peeling those
            // out keeps the two `border101` calls and their four casts off the
            // 230,400 interior columns of a level-0 pass.
            let value: i32 = match (2 * c)
                .checked_sub(2)
                .and_then(|first| band.get(first..)?.first_chunk::<5>())
            {
                Some(window) => {
                    KERNEL[0] * window[0]
                        + KERNEL[1] * window[1]
                        + KERNEL[2] * window[2]
                        + KERNEL[3] * window[3]
                        + KERNEL[4] * window[4]
                }
                None => {
                    let col2: i64 = 2 * c as i64;
                    let columns: [usize; 5] = [
                        (col2 - 2).unsigned_abs() as usize,
                        (col2 - 1).unsigned_abs() as usize,
                        col2 as usize,
                        border101(col2 + 1, src_width as i64) as usize,
                        border101(col2 + 2, src_width as i64) as usize,
                    ];
                    KERNEL[0] * band[columns[0]]
                        + KERNEL[1] * band[columns[1]]
                        + KERNEL[2] * band[columns[2]]
                        + KERNEL[3] * band[columns[3]]
                        + KERNEL[4] * band[columns[4]]
                }
            };
            // `T val = ((val_int + (1 << 7)) >> 8)` (`image_pyr.h:135`). The
            // accumulator peaks at 65535 * 16 * 16, so the shift lands back in
            // `u16` exactly and the cast never truncates.
            *pixel = ((value + (1 << 7)) >> 8) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    use proptest::prelude::*;

    /// True reflect-101, written the obvious way: mirror until the index lands
    /// inside `[0, n)`. Independent of [`border101`] and of basalt's `abs`.
    fn reflect101_naive(mut index: i64, n: i64) -> i64 {
        assert!(n >= 2, "reflect-101 needs at least two samples");
        loop {
            if index < 0 {
                index = -index;
            } else if index >= n {
                index = 2 * (n - 1) - index;
            } else {
                return index;
            }
        }
    }

    /// A direct 5x5 convolution with true reflect-101 borders and basalt's
    /// single rounding. Independent of [`subsample`]: no separability, no
    /// accumulator, no `abs`/`border101` split.
    fn subsample_naive(src: &ImageU16) -> ImageU16 {
        let width: usize = src.width() >> 1;
        let height: usize = src.height() >> 1;
        let mut dst: ImageU16 = ImageU16::zeros(width, height).unwrap();
        for r in 0..height {
            for c in 0..width {
                let mut sum: i64 = 0;
                for (dy, ky) in KERNEL.iter().enumerate() {
                    let y: i64 =
                        reflect101_naive(2 * r as i64 + dy as i64 - 2, src.height() as i64);
                    for (dx, kx) in KERNEL.iter().enumerate() {
                        let x: i64 =
                            reflect101_naive(2 * c as i64 + dx as i64 - 2, src.width() as i64);
                        let pixel: i64 = i64::from(src.get(x as usize, y as usize).unwrap());
                        sum += i64::from(*ky) * i64::from(*kx) * pixel;
                    }
                }
                dst.set(c, r, ((sum + 128) >> 8) as u16);
            }
        }
        dst
    }

    fn random_image(width: usize, height: usize, seed: u64) -> ImageU16 {
        let mut image: ImageU16 = ImageU16::zeros(width, height).unwrap();
        let mut state: u64 = seed | 1;
        for y in 0..height {
            for pixel in image.row_mut(y) {
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
    fn each_level(pyramid: &PyramidU16) -> Vec<&ImageU16> {
        (0..pyramid.num_levels())
            .map(|level| pyramid.level(level).unwrap())
            .collect()
    }

    fn build(image: &ImageU16, num_levels: usize) -> PyramidU16 {
        let mut pyramid: PyramidU16 =
            PyramidU16::with_capacity(image.width(), image.height(), num_levels).unwrap();
        CpuPyramidBuilder::new()
            .build(0, image, &mut pyramid)
            .unwrap();
        pyramid
    }

    #[test]
    fn three_levels_means_four_usable_levels() {
        // `optical_flow_levels = 3` -> levels 0..3 (`image_pyr.h:75-79`).
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
        // `lvl(l)` is `(orig_w >> l, image.h >> l)` (`image_pyr.h:149-150`),
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
        let image: ImageU16 = random_image(20, 14, 3);
        let pyramid: PyramidU16 = build(&image, 2);
        assert_eq!(pyramid.level(0).unwrap(), &image);
    }

    #[test]
    fn a_flat_image_stays_flat() {
        // The kernel sums to 256 and the rounding is `(v * 256 + 128) >> 8`,
        // so a constant image is a fixed point at every level.
        let mut image: ImageU16 = ImageU16::zeros(32, 32).unwrap();
        for y in 0..image.height() {
            image.row_mut(y).fill(4_242);
        }
        let pyramid: PyramidU16 = build(&image, 3);
        for level in each_level(&pyramid) {
            assert!(level.data().iter().all(|pixel| *pixel == 4_242));
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
        let image: ImageU16 = random_image(32, 24, 21);
        let pyramid: PyramidU16 = build(&image, 2);
        assert_eq!(pyramid.num_levels(), 3);
        assert_eq!(pyramid.level_size(0), Some((32, 24, 32)));
        assert_eq!(pyramid.level_size(2), Some((8, 6, 8)));
        assert_eq!(pyramid.level_size(3), None);

        let mut out: ImageU16 = ImageU16::default();
        pyramid.copy_level_into(1, &mut out).unwrap();
        assert_eq!((out.width(), out.height()), (16, 12));
        assert_eq!(out.data(), pyramid.level(1).unwrap().data());

        // Copying the same level again reuses the caller's allocation.
        let pointer: *const u16 = out.data().as_ptr();
        pyramid.copy_level_into(1, &mut out).unwrap();
        assert_eq!(out.data().as_ptr(), pointer);

        assert_eq!(
            pyramid.copy_level_into(9, &mut out),
            Err(PyramidError::NoSuchLevel {
                level: 9,
                num_levels: 3
            })
        );
    }

    /// A frame size whose scratch buffer cannot be allocated is an error, not
    /// an abort. `usize::MAX * 2` wraps; `max_elements::<i32>() + 1` fits a
    /// `usize` but not one allocation. Neither reaches the `vec!` `build` sizes.
    #[test]
    fn an_unallocatable_scratch_is_an_error_not_an_abort() {
        assert_eq!(
            scratch_len(usize::MAX, 4).err(),
            Some(PyramidError::ScratchTooLarge {
                width: usize::MAX,
                height: 4
            })
        );
        let too_wide: usize = crate::image::max_elements::<i32>() + 1;
        assert_eq!(
            scratch_len(too_wide, 2).err(),
            Some(PyramidError::ScratchTooLarge {
                width: too_wide,
                height: 2
            })
        );
    }

    #[test]
    fn a_frame_of_the_wrong_size_is_refused() {
        let mut pyramid: PyramidU16 = PyramidU16::with_capacity(32, 24, 2).unwrap();
        let image: ImageU16 = random_image(32, 25, 4);
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
        let image: ImageU16 = random_image(96, 64, 5);
        let mut pyramid: PyramidU16 = PyramidU16::with_capacity(96, 64, 3).unwrap();
        // `new`'s scratch is empty; the first `build` sizes it, and the pointers
        // are captured after that build, so this still measures re-use.
        let mut builder: CpuPyramidBuilder = CpuPyramidBuilder::new();
        builder.build(0, &image, &mut pyramid).unwrap();
        let pointers: Vec<*const u16> = each_level(&pyramid)
            .iter()
            .map(|level| level.data().as_ptr())
            .collect();
        let capacities: Vec<usize> = each_level(&pyramid)
            .iter()
            .map(|level| level.data().len())
            .collect();
        let scratch: *const i32 = builder.scratch.as_ptr();
        let scratch_capacity: usize = builder.scratch.capacity();
        for seed in 6..12 {
            builder
                .build(0, &random_image(96, 64, seed), &mut pyramid)
                .unwrap();
        }
        let after: Vec<*const u16> = each_level(&pyramid)
            .iter()
            .map(|level| level.data().as_ptr())
            .collect();
        assert_eq!(after, pointers, "a level buffer moved");
        assert_eq!(
            capacities,
            each_level(&pyramid)
                .iter()
                .map(|level| level.data().len())
                .collect::<Vec<usize>>()
        );
        assert_eq!(builder.scratch.as_ptr(), scratch, "the scratch moved");
        assert_eq!(builder.scratch.capacity(), scratch_capacity);
    }

    #[test]
    fn border101_matches_the_naive_reflection_over_its_whole_domain() {
        for n in 2i64..24 {
            // basalt calls `border101` with indices in `[0, 2 * (n - 1)]`
            // (`image_pyr.h:112-113`) and `std::abs` below zero
            // (`image_pyr.h:110-111`). Both agree with the naive reflection
            // there, and `border101` does *not* below zero, which is trap 3.
            for x in 0..=2 * (n - 1) {
                assert_eq!(
                    border101(x, n),
                    reflect101_naive(x, n),
                    "border101({x}, {n})"
                );
            }
            for x in -(n - 1)..0 {
                assert_eq!(x.abs(), reflect101_naive(x, n), "abs({x}) for n = {n}");
                assert_ne!(
                    border101(x, n),
                    reflect101_naive(x, n),
                    "border101({x}, {n})"
                );
            }
        }
    }

    /// kornia-rs is the second, independent oracle for the same arithmetic.
    ///
    /// `pyrdown_u8` (`crates/kornia-imgproc/src/pyramid.rs:469`) runs the same
    /// integer `[1,4,6,4,1]` kernel with reflect-101 borders and the same
    /// single rounding, `(sum + 128) >> 8` (`:650`), against basalt's
    /// `(val_int + (1 << 7)) >> 8` (`image_pyr.h:135`). It transposes the pass
    /// order (horizontal first, `:498`) where basalt goes vertical first, which
    /// changes nothing: both passes accumulate in integers and round once, so
    /// each is exactly the 5x5 convolution.
    ///
    /// The comparison runs on `u16` pixels holding **0..255 unshifted**, not
    /// the `<< 8` frontend values. The two are not comparable: rounding happens
    /// on different magnitudes, so `subsample(v << 8) >> 8` is not
    /// `subsample(v)` — the `<< 8` form keeps eight more bits of the weighted
    /// sum before the shift discards them, and the two disagree by one LSB
    /// wherever the discarded remainder crosses a half. That extra headroom is
    /// exactly why basalt widens (kornia-rs inventory §2.1).
    ///
    /// Sizes are even on purpose: kornia sizes its output `div_ceil(2)`
    /// (`pyramid.rs:473-474`) where basalt shifts right (`image_pyr.h:149`), so
    /// on an odd side the two produce different geometries and only basalt's is
    /// the reference.
    #[test]
    fn subsample_matches_kornia_pyrdown_u8_on_byte_valued_pixels() {
        use kornia_image::{Image, ImageSize};

        for (width, height, seed) in [(16usize, 12usize, 11u64), (64, 64, 12), (34, 18, 13)] {
            let bytes: Vec<u8> = {
                let mut state: u64 = seed | 1;
                (0..width * height)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1);
                        (state >> 33) as u8
                    })
                    .collect()
            };

            let source: Image<u8, 1> =
                Image::new(ImageSize { width, height }, bytes.clone()).unwrap();
            let mut kornia_out: Image<u8, 1> = Image::from_size_val(
                ImageSize {
                    width: width / 2,
                    height: height / 2,
                },
                0u8,
            )
            .unwrap();
            kornia_imgproc::pyramid::pyrdown_u8(&source, &mut kornia_out).unwrap();

            // Our own subsample over the same values, held in `u16` with no shift.
            let mut ours: ImageU16 = ImageU16::zeros(width, height).unwrap();
            for (y, row) in bytes.chunks_exact(width).enumerate() {
                for (pixel, byte) in ours.row_mut(y).iter_mut().zip(row) {
                    *pixel = u16::from(*byte);
                }
            }
            let mut got: ImageU16 = ImageU16::zeros(width / 2, height / 2).unwrap();
            let mut scratch: Vec<i32> = vec![0; scratch_len(width, height).unwrap()];
            subsample(&ours, &mut got, &mut scratch);

            let expected: Vec<u16> = kornia_out
                .as_slice()
                .iter()
                .map(|byte| u16::from(*byte))
                .collect();
            assert_eq!(got.data(), expected.as_slice(), "{width}x{height}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]

        /// The separable, `abs`/`border101` implementation with its
        /// row-major accumulator and a direct 5x5 convolution with true
        /// reflect-101 borders agree bit for bit, on even and odd sizes alike.
        #[test]
        fn subsample_matches_a_naive_5x5_convolution(
            width in 3usize..40,
            height in 3usize..40,
            seed in any::<u64>(),
        ) {
            let image: ImageU16 = random_image(width, height, seed);
            let expected: ImageU16 = subsample_naive(&image);
            let mut got: ImageU16 = ImageU16::zeros(width >> 1, height >> 1).unwrap();
            let mut scratch: Vec<i32> = vec![0; scratch_len(width, height).unwrap()];
            subsample(&image, &mut got, &mut scratch);
            prop_assert_eq!(got.data(), expected.data());
        }

        /// Every level of a whole pyramid, not just the first subsample.
        #[test]
        fn every_pyramid_level_matches_the_naive_reference(
            width in 24usize..70,
            height in 24usize..70,
            seed in any::<u64>(),
        ) {
            let image: ImageU16 = random_image(width, height, seed);
            let pyramid: PyramidU16 = build(&image, 3);
            let mut expected: ImageU16 = image.clone();
            for level in 1..pyramid.num_levels() {
                expected = subsample_naive(&expected);
                prop_assert_eq!(pyramid.level(level).unwrap().data(), expected.data());
            }
        }

        /// A subsampled level never exceeds the source's range: the kernel is a
        /// normalized average, so it cannot overshoot and cannot wrap the cast.
        #[test]
        fn subsample_stays_inside_the_source_range(
            width in 3usize..40,
            height in 3usize..40,
            seed in any::<u64>(),
        ) {
            let image: ImageU16 = random_image(width, height, seed);
            let pyramid: PyramidU16 = build(&image, 1);
            let low: u16 = *image.data().iter().min().unwrap();
            let high: u16 = *image.data().iter().max().unwrap();
            for pixel in pyramid.level(1).unwrap().data() {
                prop_assert!(*pixel >= low && *pixel <= high);
            }
        }
    }
}
