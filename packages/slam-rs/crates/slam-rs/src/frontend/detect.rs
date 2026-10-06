//! Application frame preparation around the shared cell detector policy.
use super::flow::FrontendError;
use super::input::{FrameImage, FrameImages};
use kornia_staging_imgproc::features::{
    CellSelect, CornerScan, FastCorner, Occupancy, SelectionStatus,
};

/// Frame scheduling and image ownership remain in the application.
pub trait FrameCornerScan: CornerScan<Error: Into<FrontendError>> {
    fn fork_frame(&self) -> Option<Box<Self>> where Self: Sized { None }
    fn scan_frame(&mut self, camera: usize, image: FrameImage<'_>) -> Result<(), FrontendError> {
        image
            .with_dense(|image| self.scan(camera, image))
            .map_err(FrontendError::Image)?.map_err(Into::into)
    }
    fn select_frame(
        &mut self,
        camera: usize,
        image: FrameImage<'_>,
        select: &CellSelect,
        eligibility: Option<(&Occupancy<'_>, &[bool])>,
        out: &mut Vec<Option<FastCorner>>,
    ) -> Result<SelectionStatus, FrontendError> {
        image
            .with_dense(|image| self.select_cells(camera, image, select, eligibility, out))
            .map_err(FrontendError::Image)?.map_err(Into::into)
    }
    fn submit_cells(
        &mut self,
        _images: FrameImages<'_>,
        _selects: &[Option<CellSelect>],
    ) -> Result<(), Self::Error> {
        Ok(())
    }
    fn take_cells(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl FrameCornerScan for kornia_staging_imgproc::features::CpuCornerScan {
    fn fork_frame(&self) -> Option<Box<Self>> { Some(Box::new(self.fresh())) }
}

// Device winner encoding is private to the application GPU lane.
pub const NO_CELL_WINNER: u32 = u32::MAX;
pub const KEY_SCORE_SHIFT: u32 = 24;
pub const KEY_ROW_SHIFT: u32 = 12;
pub const KEY_FIELD_MASK: u32 = 0xFFF;
pub const CELL_KEY_LIMIT: usize = 1 << KEY_ROW_SHIFT;
pub fn decode_key(key: u32) -> Option<FastCorner> {
    (key != NO_CELL_WINNER).then(|| FastCorner {
        xy: [
            (key & KEY_FIELD_MASK) as f32,
            ((key >> KEY_ROW_SHIFT) & KEY_FIELD_MASK) as f32,
        ],
        response: kornia_staging_imgproc::features::opencv_corner_score(
            (255 - (key >> KEY_SCORE_SHIFT)) as f32 / 255.0,
        ),
    })
}

use kornia_staging_imgproc::features::FAST_BORDER;
pub const FAST_RING_ROW: [i32; 16] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1];
/// The column half of [`FAST_RING_ROW`]'s ring.
pub const FAST_RING_COLUMN: [i32; 16] = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3];

/// Lanes in one block of kornia's in-block local-maximum filter (`fast.rs`).
pub const FAST_FILTER_LANES: usize = 16;

/// The width at which kornia turns that filter on (`fast.rs`).
const FAST_FILTER_WIDTH: usize = 800;

/// Where kornia's local-maximum filter stops, and whether it runs at all.
///
/// The filter keeps a candidate only when its score beats both neighbours
/// *inside its own sixteen-lane block*, the blocks are aligned to the image's
/// own left margin, and the scalar tail past the last whole block is
/// unfiltered — kornia's SIMD loop runs while `x + 16 <= width - margin`
/// (`fast.rs`). So the alignment and the tail are part of the corner set,
/// not an implementation detail, and this is the one place that arithmetic
/// lives: the CPU sweep gets it from kornia itself, and the GPU kernel and
/// `tests/fast_model.rs` read it here rather than each spelling it out.
/// The filter exists only in kornia's SIMD loop. Keep its runtime gate in
/// sync with kornia-imgproc `features/fast.rs`: NEON unless
/// `KORNIA_FAST_NEON == "0"` on aarch64, AVX2 on x86_64, otherwise scalar.
/// The scalar loop has no block filter, even on images at least 800 pixels wide.
///
/// # Arguments
/// * `width` - Full image row length; the block filter is width-dependent.
pub fn block_filter_end(width: usize) -> (usize, bool) {
    let blocks: usize = width.saturating_sub(2 * FAST_BORDER) / FAST_FILTER_LANES;
    (
        FAST_BORDER + blocks * FAST_FILTER_LANES,
        width >= FAST_FILTER_WIDTH && kornia_uses_simd(),
    )
}

fn kornia_uses_simd() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::env::var("KORNIA_FAST_NEON").map_or(true, |value| value != "0")
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

