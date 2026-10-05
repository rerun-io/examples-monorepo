//! The 1920x1080 -> 640x360 "small" image: an exact integer area (box) downscale, and the per-frameset step that makes it once
//! per present camera.
//!
//! kornia-imgproc has no area mode: `resize_fast_mono_aa` with Bilinear at exactly /3 samples pixel `3x+1` (point sampling,
//! aliases), and Bicubic/Lanczos with `antialias` are not the box mean the SPEC fixes (each small pixel = the rounded mean of a
//! 3x3 block, which is also what PR #270's `slam_luma` and the SLAM calibration's `downscale3` assume). [`resize_area_u8`] is
//! written in kornia's style (a free function over `Image`, writing into `dst`, `Result<(), ImageError>`, rayon over row chunks,
//! a NEON kernel with a scalar path) for upstreaming to kornia-imgproc's `resize` (see UPSTREAM.md).
#![deny(missing_docs)]

use std::collections::VecDeque;
use std::sync::Arc;

use kornia_image::{Image, ImageError};
use rayon::prelude::*;

use crate::frame::{Frameset, Luma, NUM_CAMERAS, SMALL_SIZE};

pub use slam_rs::area::resize_area_u8;

/// The small images of one frameset, by camera index.
pub type SmallImages = [Option<Luma>; NUM_CAMERAS];

/// Earlier framesets' small images, kept so that one no stage holds any more is written again instead of allocating (and
/// zero-filling) a new 640x360 image: the `Arc::strong_count == 1` re-use of kornia-io's V4L2 `MmapStream`.
#[derive(Default)]
pub struct SmallImagePool {
    held: VecDeque<Luma>,
}

/// Images the pool keeps at most: sixteen framesets of six cameras. A lossless replay keeps up to six framesets in flight in
/// the stage queues (four were too few: 23 of 1800 images were re-used), a realtime run about one.
const POOL_IMAGES: usize = 16 * NUM_CAMERAS;

impl SmallImagePool {
    /// An image to write a small image into: a held one that nobody else holds any more, else a new one.
    fn take(&mut self) -> Result<Image<u8, 1>, ImageError> {
        if let Some(index) = self.held.iter().position(|luma| Arc::strong_count(luma) == 1)
            && let Some(Ok(image)) = self.held.remove(index).map(Arc::try_unwrap)
        {
            return Ok(image);
        }
        Image::from_size_val(SMALL_SIZE, 0)
    }

    /// Keeps a frameset's images for re-use once every stage has dropped them.
    fn keep(&mut self, small: &SmallImages) {
        self.held.extend(small.iter().flatten().cloned());
        while self.held.len() > POOL_IMAGES {
            self.held.pop_front();
        }
    }
}

/// Make the 640x360 small image of every present camera (in parallel on the current rayon pool), or of `only` cameras when
/// given (SLAM-only runs need four).
///
/// # Arguments
///
/// * `frameset` - The frameset.
/// * `only` - The cameras to make, or `None` for all.
/// * `pool` - Earlier small images to write again when no stage holds them any more; it keeps these ones too.
///
/// # Returns
///
/// The small images by camera index (`None` for a missing or unwanted camera).
///
/// # Errors
///
/// [`ImageError`] when a camera's full image is not 1920x1080 (3x the small size).
///
/// A frame turned 180 degrees ([`crate::frame::FrameMeta::turned_180`]) comes out upright: its small image is reversed in place
/// after the downsample, which is exact, as the 3x3 area mean commutes with the turn.
pub fn small_images(frameset: &Frameset, only: Option<&[usize]>, pool: &mut SmallImagePool) -> Result<SmallImages, ImageError> {
    let wanted = |camera: usize| only.is_none_or(|cameras| cameras.contains(&camera));
    let mut jobs: Vec<(usize, &Luma, bool, Image<u8, 1>)> = Vec::with_capacity(NUM_CAMERAS);
    for (camera, frame) in frameset.cameras.iter().enumerate() {
        if let Some(frame) = frame.as_ref().filter(|_| wanted(camera)) {
            jobs.push((camera, &frame.full, frame.meta.turned_180, pool.take()?));
        }
    }
    let results: Vec<(usize, Result<Luma, ImageError>)> = jobs
        .into_par_iter()
        .map(|(camera, full, turned_180, mut small)| {
            let result = resize_area_u8(full, &mut small).map(|()| {
                if turned_180 {
                    small.as_slice_mut().reverse();
                }
                Arc::new(small)
            });
            (camera, result)
        })
        .collect();
    let mut out: SmallImages = Default::default();
    for (camera, small) in results {
        out[camera] = Some(small?);
    }
    pool.keep(&out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use kornia_image::ImageSize;

    use super::*;

    fn pseudo_random(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect()
    }

    fn frameset(seed: u64) -> Result<Frameset, ImageError> {
        let full = Arc::new(Image::<u8, 1>::new(ImageSize { width: 1920, height: 1080 }, pseudo_random(1920 * 1080, seed))?);
        let cameras = std::array::from_fn(|camera| {
            Some(crate::frame::CameraFrame { meta: crate::frame::FrameMeta { seq: 0, pts_ns: 0, source_id: camera as u32, turned_180: false }, full: full.clone() })
        });
        Ok(Frameset { index: 0, t_ns: 0, cameras })
    }

    #[test]
    fn an_upside_down_cameras_small_image_is_exactly_its_upright_frames() -> Result<(), ImageError> {
        let upright = frameset(4)?;
        let mut turned = upright.clone();
        let Some(frame) = turned.cameras[2].as_mut() else { unreachable!() };
        let reversed: Vec<u8> = frame.full.as_slice().iter().rev().copied().collect();
        frame.full = Arc::new(Image::<u8, 1>::new(ImageSize { width: 1920, height: 1080 }, reversed)?);
        frame.meta.turned_180 = true;
        let expected = small_images(&upright, None, &mut SmallImagePool::default())?;
        let got = small_images(&turned, None, &mut SmallImagePool::default())?;
        let pixels = |images: &SmallImages, camera: usize| images[camera].as_ref().map(|luma| luma.as_slice().to_vec());
        assert!(pixels(&got, 2) == pixels(&expected, 2), "the turned camera comes out upright, bit for bit");
        assert!(pixels(&got, 0) == pixels(&expected, 0), "the others are untouched");
        Ok(())
    }

    #[test]
    fn the_pool_rewrites_only_images_no_stage_holds() -> Result<(), ImageError> {
        let mut pool = SmallImagePool::default();
        let first = small_images(&frameset(1)?, None, &mut pool)?;
        let addresses: Vec<*const u8> = first.iter().flatten().map(|luma| luma.as_slice().as_ptr()).collect();
        // A stage still holds the first frameset: the second gets new images.
        let second = small_images(&frameset(2)?, None, &mut pool)?;
        assert!(second.iter().flatten().all(|luma| !addresses.contains(&luma.as_slice().as_ptr())));
        // Once the stages drop it, the next frameset is written into the first frameset's memory, with the right pixels.
        drop(first);
        let third = small_images(&frameset(3)?, None, &mut pool)?;
        assert!(third.iter().flatten().all(|luma| addresses.contains(&luma.as_slice().as_ptr())));
        let mut expected = Image::<u8, 1>::from_size_val(SMALL_SIZE, 0)?;
        resize_area_u8(&frameset(3)?.cameras[0].as_ref().map(|frame| frame.full.clone()).unwrap_or_else(|| unreachable!()), &mut expected)?;
        assert!(third.iter().flatten().all(|luma| luma.as_slice() == expected.as_slice()));
        drop(second);
        assert!(pool.held.len() <= POOL_IMAGES);
        Ok(())
    }

}
