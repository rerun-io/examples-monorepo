//! Consumer scheduling and input adapters for the staged GPU pyramid.
use kornia_staging_gpu::runtime::GpuError;
use super::{ guarded};
impl From<GpuError> for PyramidError {
    fn from(error: GpuError) -> Self { Self::Gpu(error.into()) }
}

use crate::frontend::input::{FrameImages, PackedImages};
use crate::pyramid::{Pyramid, PyramidBuilder, PyramidError};
use cubecl::prelude::*;
use kornia_image::{Image, ImageSize};
pub(super) use kornia_staging_gpu::pyramid::FrameArena;
pub use kornia_staging_gpu::pyramid::GpuPyramid;

/// Schedule persistent device pyramids with the frontend frame queue.
pub struct GpuPyramidBuilder<R: Runtime> {
    pub(super) inner: kornia_staging_gpu::pyramid::GpuPyramidBuilder<R>,
    client: ComputeClient<R>,
    launches: super::submission::LaunchList,
}
impl<R: Runtime> GpuPyramidBuilder<R> {
    pub fn new(client: ComputeClient<R>, launches: super::submission::LaunchList) -> Self {
        Self {
            inner: kornia_staging_gpu::pyramid::GpuPyramidBuilder::new(client.clone()),
            client,
            launches,
        }
    }
    pub fn prepare_images(&mut self, images: &[Image<u16, 1>]) -> Result<(), PyramidError> {
        self.inner.prepare_images(images).map_err(Into::into)
    }
    pub(super) fn build_images(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [GpuPyramid<R>],
    ) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "frameset pyramid",
            },
            || {
                self.launches.flush(&self.client);
                self.inner.build_images(images, out).map_err(Into::into)
            },
        )
    }
}
impl<R: Runtime> GpuPyramidBuilder<R> {
    pub(super) fn build_packed(
        &mut self,
        packed: &PackedImages,
        out: &mut [GpuPyramid<R>],
    ) -> Result<(), PyramidError> {
        let images = FrameImages::Packed(packed);
        match self.inner.prepare_packed(
            packed.bytes(),
            images.iter().map(|image| image.size()),
            out,
        )? {
            Some(launch) => self
                .launches
                .dispatch(&self.client, super::submission::Launch::Pyramid(launch)),
            None => {
                self.launches.flush(&self.client);
                for (camera, (image, pyramid)) in images.iter().zip(out).enumerate() {
                    image.with_dense(|image| self.build(camera, image, pyramid))??;
                }
            }
        }
        Ok(())
    }
}
impl<R: Runtime> PyramidBuilder for GpuPyramidBuilder<R> {
    type Pyramid = GpuPyramid<R>;
    fn allocate(
        &self,
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<Self::Pyramid, PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "pyramid allocation",
            },
            || {
                self.inner
                    .allocate(width, height, num_levels)
                    .map_err(Into::into)
            },
        )
    }
    fn build(
        &mut self,
        camera: usize,
        image: &Image<u16, 1>,
        out: &mut Self::Pyramid,
    ) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "pyramid build",
            },
            || self.inner.build(camera, image, out).map_err(Into::into),
        )
    }
    fn build_frames(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [Self::Pyramid],
        _pool: &crate::frontend::parallel::WorkPool,
    ) -> Result<(), PyramidError> {
        self.build_images(images, out)
    }
}
impl<R: Runtime> Pyramid for GpuPyramid<R> {
    fn num_levels(&self) -> usize {
        self.num_levels()
    }
    fn level_size(&self, level: usize) -> Option<ImageSize> {
        self.level_size(level)
    }
    fn copy_level_into(&self, level: usize, out: &mut Image<u16, 1>) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "a pyramid level read",
            },
            || self.read_level_into(level, out).map_err(Into::into),
        )
    }
}
#[cfg(all(test, feature = "gpu-wgpu"))]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::pyramid::PyramidBuilder;

    #[test]
    fn odd_packed_cameras_use_one_arena_and_preserve_visible_pixels() {
        let client = kornia_staging_gpu::runtime::gpu_client().unwrap();
        let mut builder = GpuPyramidBuilder::new(client.clone(), Default::default());
        let cameras: Vec<Vec<u8>> = (0..2)
            .map(|camera| {
                (0..65 * 49)
                    .map(|i| ((i * 37 + camera * 29) % 256) as u8)
                    .collect()
            })
            .collect();
        let views: Vec<_> = cameras
            .iter()
            .map(|pixels| crate::ImageView {
                data: pixels,
                width: 65,
                height: 49,
                stride: 65,
            })
            .collect();
        let mut packed = PackedImages::default();
        packed.fill(&views);
        let mut pyramids: Vec<_> = (0..2)
            .map(|_| builder.allocate(65, 49, 2).unwrap())
            .collect();
        builder.build_packed(&packed, &mut pyramids).unwrap();
        builder.launches.flush(&client);
        for (camera, pyramid) in pyramids.iter().enumerate() {
            assert!(pyramid.arena().is_some());
            let mut level0 = crate::image::empty();
            pyramid.copy_level_into(0, &mut level0).unwrap();
            let expected: Vec<u16> = cameras[camera]
                .iter()
                .map(|&pixel| u16::from(pixel) << 8)
                .collect();
            assert_eq!(level0.as_slice(), expected);
        }
    }
}

impl<R: Runtime> std::fmt::Debug for GpuPyramidBuilder<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
