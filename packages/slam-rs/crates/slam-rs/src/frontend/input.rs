//! Explicit dense or retained byte inputs for one frontend frame.
use crate::{ImageView, image};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::panorama::{TileLut, assemble_panorama, lens_level1_into, remap_tile};
use std::sync::{Arc, Mutex};

/// The dual-fisheye ingest: two lens images become the two hemisphere tiles through tables built once on the CPU
/// (`kornia_staging_imgproc::panorama::build_tile_lut`). Tile `k` reads lens `k`. The GPU lane uploads the tables
/// once and stitches on the device; the CPU lane and the GPU fallback run `lens_level1` + `remap_tile`. Both give the
/// same tile pixels bit for bit.
pub struct LensStitch {
    luts: [TileLut; 2],
    lens: ImageSize,
    /// Host remap scratch, reused every frameset: one lens widened to `u16`, then its level 1.
    scratch: Mutex<[Image<u16, 1>; 2]>,
}
impl std::fmt::Debug for LensStitch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LensStitch")
            .field("luts", &self.luts)
            .field("lens", &self.lens)
            .finish_non_exhaustive()
    }
}
impl LensStitch {
    /// # Errors
    /// [`image::IngestError::StitchTables`] unless `luts` are tile 0 then tile 1 of one geometry whose source is
    /// `lens` floor-halved, with one entry per tile pixel; [`image::IngestError::LensSize`] for a lens below 4x4.
    pub fn new(luts: [TileLut; 2], lens: ImageSize) -> Result<Self, image::IngestError> {
        if lens.width < 4 || lens.height < 4 {
            return Err(image::IngestError::LensSize {
                width: lens.width,
                height: lens.height,
                expected_width: 4,
                expected_height: 4,
            });
        }
        let source = ImageSize {
            width: lens.width / 2,
            height: lens.height / 2,
        };
        let consistent = luts.iter().enumerate().all(|(tile, lut)| {
            lut.tile == tile
                && lut.geometry == luts[0].geometry
                && lut.source_size == source
                && lut.sample.len() == lut.geometry.width() * lut.geometry.height
        });
        if !consistent {
            return Err(image::IngestError::StitchTables);
        }
        let scratch = Mutex::new([
            image::zeros(lens.width, lens.height)?,
            image::zeros(source.width, source.height)?,
        ]);
        Ok(Self {
            luts,
            lens,
            scratch,
        })
    }
    /// The two tables, tile 0 then tile 1.
    pub fn luts(&self) -> &[TileLut; 2] {
        &self.luts
    }
    /// Each lens image's size.
    pub fn lens_size(&self) -> ImageSize {
        self.lens
    }
    /// Each tile's size: level 0 of a tile camera's pyramid.
    pub fn tile_size(&self) -> ImageSize {
        ImageSize {
            width: self.luts[0].geometry.width(),
            height: self.luts[0].geometry.height,
        }
    }
    /// Tile `tile` from lens `tile`'s bytes (`stride` bytes per row), on the host: `lens_level1` + `remap_tile`, the
    /// device stitch's integer rule, through reused scratch and without copying the lens first.
    fn remap(
        &self,
        tile: usize,
        lens: &[u8],
        stride: usize,
        out: &mut Image<u16, 1>,
    ) -> Result<(), image::IngestError> {
        let layout = |_| image::IngestError::StitchTables;
        let needed = stride * (self.lens.height - 1) + self.lens.width;
        if stride < self.lens.width || lens.len() < needed {
            return Err(image::IngestError::StitchTables);
        }
        if out.size() != self.tile_size() {
            *out = image::zeros(self.tile_size().width, self.tile_size().height)?;
        }
        let mut scratch = self
            .scratch
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let [level0, level1] = &mut *scratch;
        lens_level1_into(lens, stride, level0, level1).map_err(layout)?;
        remap_tile(level1, &self.luts[tile], out).map_err(layout)
    }
    /// The viewing panorama of two lens images without a pipeline: [`Self::remap`] of each tile, then
    /// [`assemble_panorama`], the lane's own host stitch (what `PanoramaRig::panorama_image` logs).
    ///
    /// # Errors
    /// A lens view of the wrong size or count.
    pub fn panorama(&self, lenses: &[ImageView<'_>]) -> Result<Image<u8, 1>, image::IngestError> {
        self.check(lenses)?;
        let mut tiles = [image::empty(), image::empty()];
        for (tile, (out, view)) in tiles.iter_mut().zip(lenses).enumerate() {
            self.remap(tile, view.data, view.stride, out)?;
        }
        assemble_panorama([&tiles[0], &tiles[1]], self.luts[0].geometry)
            .map_err(|_| image::IngestError::StitchTables)
    }
    fn check(&self, lenses: &[ImageView<'_>]) -> Result<(), image::IngestError> {
        match lenses
            .iter()
            .find(|view| view.width != self.lens.width || view.height != self.lens.height)
        {
            Some(view) => Err(image::IngestError::LensSize {
                width: view.width,
                height: view.height,
                expected_width: self.lens.width,
                expected_height: self.lens.height,
            }),
            None if lenses.len() == 2 => Ok(()),
            None => Err(image::IngestError::StitchTables),
        }
    }
}

/// Retained byte frames share one upload allocation. CPU storage is touched only
/// when a detector needs host pixels.
#[derive(Default)]
pub struct PackedImages {
    bytes: bytes::Bytes,
    frames: Vec<(ImageSize, usize)>,
    host: Vec<Mutex<Image<u16, 1>>>,
    /// Set when the bytes are two lens images and the frames are their tiles.
    stitch: Option<Arc<LensStitch>>,
}
impl std::fmt::Debug for PackedImages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedImages")
            .field("frames", &self.frames)
            .field("bytes", &self.bytes)
            .field("stitch", &self.stitch.is_some())
            .finish()
    }
}
impl PackedImages {
    /// Copy validated visible rows directly into the reusable upload buffer.
    pub(crate) fn fill(&mut self, views: &[ImageView<'_>]) {
        let mut bytes = std::mem::take(&mut self.bytes)
            .try_into_mut()
            .unwrap_or_else(|bytes| bytes::BytesMut::with_capacity(bytes.len()));
        bytes.clear();
        self.frames.clear();
        self.host
            .resize_with(views.len(), || Mutex::new(image::empty()));
        for view in views {
            self.frames.push((
                ImageSize {
                    width: view.width,
                    height: view.height,
                },
                bytes.len(),
            ));
            for row in 0..view.height {
                let start = row * view.stride;
                bytes.extend_from_slice(&view.data[start..start + view.width]);
            }
        }
        bytes.resize(bytes.len().next_multiple_of(4), 0);
        self.bytes = bytes.freeze();
        self.stitch = None;
    }
    /// Retain two lens images whose frames are the two tiles `stitch` makes of them.
    ///
    /// # Errors
    /// [`image::IngestError::LensSize`] when a view is not the tables' lens size, or
    /// [`image::IngestError::StitchTables`] when there are not two views.
    pub fn fill_lenses(
        &mut self,
        lenses: &[ImageView<'_>],
        stitch: &Arc<LensStitch>,
    ) -> Result<(), image::IngestError> {
        stitch.check(lenses)?;
        self.fill(lenses);
        self.stitch = Some(stitch.clone());
        Ok(())
    }
    #[cfg(feature = "gpu-core")]
    pub(crate) fn bytes(&self) -> &bytes::Bytes {
        &self.bytes
    }
    /// The stitch whose lens images these bytes are, if any.
    #[cfg(feature = "gpu-core")]
    pub(crate) fn stitch(&self) -> Option<&Arc<LensStitch>> {
        self.stitch.as_ref()
    }
    #[cfg(feature = "gpu-core")]
    pub(crate) fn same_pixels(&self, other: &Self) -> bool {
        let same_stitch = match (&self.stitch, &other.stitch) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (a, b) => a.is_none() && b.is_none(),
        };
        same_stitch && self.frames == other.frames && self.bytes == other.bytes
    }
}

/// A borrowed frameset; dense callers never take the packed-byte path.
#[derive(Clone, Copy)]
pub enum FrameImages<'a> {
    Dense(&'a [Image<u16, 1>]),
    Packed(&'a PackedImages),
}
impl<'a> FrameImages<'a> {
    pub fn len(self) -> usize {
        match self {
            Self::Dense(v) => v.len(),
            Self::Packed(v) => v.frames.len(),
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn get(self, camera: usize) -> FrameImage<'a> {
        match self {
            Self::Dense(v) => FrameImage::dense(&v[camera]),
            Self::Packed(v) => {
                let (size, offset) = v.frames[camera];
                let bytes = &v.bytes[offset..offset + size.width * size.height];
                let host = &v.host[camera];
                FrameImage {
                    source: match &v.stitch {
                        Some(stitch) => FrameSource::Lens {
                            stitch,
                            tile: camera,
                            bytes,
                            host,
                        },
                        None => FrameSource::Packed { size, bytes, host },
                    },
                }
            }
        }
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = FrameImage<'a>> + Clone {
        (0..self.len()).map(move |camera| self.get(camera))
    }
}
/// One image's geometry and explicit pixel source.
#[derive(Clone, Copy)]
pub struct FrameImage<'a> {
    source: FrameSource<'a>,
}
#[derive(Clone, Copy)]
enum FrameSource<'a> {
    Dense(&'a Image<u16, 1>),
    Packed {
        size: ImageSize,
        bytes: &'a [u8],
        host: &'a Mutex<Image<u16, 1>>,
    },
    /// Tile `tile`, made of lens `tile`'s bytes.
    Lens {
        stitch: &'a LensStitch,
        tile: usize,
        bytes: &'a [u8],
        host: &'a Mutex<Image<u16, 1>>,
    },
}
impl<'a> FrameImage<'a> {
    pub fn dense(image: &'a Image<u16, 1>) -> Self {
        Self {
            source: FrameSource::Dense(image),
        }
    }
    pub fn size(self) -> ImageSize {
        match self.source {
            FrameSource::Dense(v) => v.size(),
            FrameSource::Packed { size, .. } => size,
            FrameSource::Lens { stitch, .. } => stitch.tile_size(),
        }
    }
    pub fn width(self) -> usize {
        self.size().width
    }
    pub fn height(self) -> usize {
        self.size().height
    }
    /// Materialize only for a host operation, retaining its allocation.
    pub fn with_dense<T>(
        self,
        body: impl FnOnce(&Image<u16, 1>) -> T,
    ) -> Result<T, image::IngestError> {
        match self.source {
            FrameSource::Dense(v) => Ok(body(v)),
            FrameSource::Packed { size, bytes, host } => {
                let mut image = host
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                image::fill_from_u8_strided_unchecked(
                    &mut image,
                    bytes,
                    size.width,
                    size.height,
                    size.width,
                )?;
                Ok(body(&image))
            }
            FrameSource::Lens {
                stitch,
                tile,
                bytes,
                host,
            } => {
                let mut image = host
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                stitch.remap(tile, bytes, stitch.lens.width, &mut image)?;
                Ok(body(&image))
            }
        }
    }
}

/// Pipeline-owned input retained across prepare/finish.
pub enum FrameStorage {
    Dense(Vec<Image<u16, 1>>),
    Packed(PackedImages),
}
impl std::fmt::Debug for FrameStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(images) => f
                .debug_list()
                .entries(images.iter().map(|image| (image.size(), image.as_slice())))
                .finish(),
            Self::Packed(images) => images.fmt(f),
        }
    }
}
impl FrameStorage {
    pub(crate) fn new(gpu: bool) -> Self {
        if gpu {
            Self::Packed(PackedImages::default())
        } else {
            Self::Dense(Vec::new())
        }
    }
    pub(crate) fn fill(&mut self, views: &[ImageView<'_>]) -> Result<(), image::IngestError> {
        match self {
            Self::Dense(images) => {
                images.resize_with(views.len(), image::empty);
                for (image, view) in images.iter_mut().zip(views) {
                    image::fill_from_u8_strided_unchecked(
                        image,
                        view.data,
                        view.width,
                        view.height,
                        view.stride,
                    )?;
                }
            }
            Self::Packed(images) => images.fill(views),
        }
        Ok(())
    }
    /// Retain two lens images as the two tile images `stitch` makes of them: the dual-fisheye replacement for
    /// [`Self::fill`] with the tile views. The GPU lane stitches on the device; the CPU lane remaps here.
    ///
    /// # Errors
    /// A lens view of the wrong size or count.
    pub fn fill_lenses(
        &mut self,
        lenses: &[ImageView<'_>],
        stitch: &Arc<LensStitch>,
    ) -> Result<(), image::IngestError> {
        match self {
            Self::Dense(images) => {
                stitch.check(lenses)?;
                images.resize_with(2, image::empty);
                for (tile, (image, view)) in images.iter_mut().zip(lenses).enumerate() {
                    stitch.remap(tile, view.data, view.stride, image)?;
                }
                Ok(())
            }
            Self::Packed(images) => images.fill_lenses(lenses, stitch),
        }
    }
    pub(crate) fn view(&self) -> FrameImages<'_> {
        match self {
            Self::Dense(images) => FrameImages::Dense(images),
            Self::Packed(images) => FrameImages::Packed(images),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    #[test]
    fn retained_upload_bytes_do_not_change_when_the_next_frame_arrives() {
        let mut packed = PackedImages::default();
        packed.fill(&[ImageView {
            data: &[1, 2, 3, 4],
            width: 2,
            height: 2,
            stride: 2,
        }]);
        let in_flight = packed.bytes.clone();
        packed.fill(&[ImageView {
            data: &[5, 6, 7, 8],
            width: 2,
            height: 2,
            stride: 2,
        }]);
        assert_eq!(&in_flight[..], &[1, 2, 3, 4]);
        assert_eq!(&packed.bytes[..], &[5, 6, 7, 8]);
    }

    #[cfg(feature = "gpu-core")]
    #[test]
    fn upload_shares_the_packed_storage_without_a_host_copy() {
        let mut packed = PackedImages::default();
        packed.fill(&[ImageView {
            data: &[1, 2, 3, 4],
            width: 2,
            height: 2,
            stride: 2,
        }]);
        let pointer = packed.bytes.as_ptr();
        let upload = cubecl::bytes::Bytes::from_shared(
            packed.bytes.clone(),
            cubecl::bytes::AllocationProperty::Native,
        );
        assert_eq!(upload.as_ptr(), pointer);
        drop(upload);
        packed.fill(&[ImageView {
            data: &[9, 8, 7, 6],
            width: 2,
            height: 2,
            stride: 2,
        }]);
        assert_eq!(packed.bytes.as_ptr(), pointer);
    }

    #[test]
    fn odd_camera_frames_and_cpu_fallback_refill_keep_their_own_pixels() {
        let mut packed = PackedImages::default();
        let a = ImageView {
            data: &[1, 2, 3, 99],
            width: 3,
            height: 1,
            stride: 4,
        };
        let b = ImageView {
            data: &[4, 5, 6, 98],
            width: 3,
            height: 1,
            stride: 4,
        };
        packed.fill(&[a, b]);
        assert_eq!(&packed.bytes[..], &[1, 2, 3, 4, 5, 6, 0, 0]);
        let pointer = FrameImages::Packed(&packed)
            .get(1)
            .with_dense(|image| {
                assert_eq!(image.as_slice(), &[1024, 1280, 1536]);
                image.as_slice().as_ptr()
            })
            .unwrap();
        packed.fill(&[b, a]);
        FrameImages::Packed(&packed)
            .get(1)
            .with_dense(|image| {
                assert_eq!(image.as_slice(), &[256, 512, 768]);
                assert_eq!(image.as_slice().as_ptr(), pointer);
            })
            .unwrap();
    }

    #[test]
    fn packed_input_ignores_padding_and_keeps_host_storage_empty() {
        let views = [ImageView {
            data: &[1, 2, 99, 3, 4, 98],
            width: 2,
            height: 2,
            stride: 3,
        }];
        let mut packed = PackedImages::default();
        packed.fill(&views);
        assert_eq!(packed.bytes.as_ref(), &[1, 2, 3, 4]);
        assert!(packed.host[0].lock().unwrap().as_slice().is_empty());
        let pointer = packed.bytes.as_ptr();
        packed.fill(&views);
        assert_eq!(pointer, packed.bytes.as_ptr());
        FrameImages::Packed(&packed)
            .get(0)
            .with_dense(|image| {
                assert_eq!(image.as_slice(), &[256, 512, 768, 1024]);
            })
            .unwrap();
    }
}
