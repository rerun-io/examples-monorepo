//! Explicit dense or retained byte inputs for one frontend frame.
use crate::{ImageView, image};
use kornia_image::{Image, ImageSize};
use std::sync::Mutex;

/// Retained byte frames share one upload allocation. CPU storage is touched only
/// when a detector needs host pixels.
#[derive(Default)]
pub struct PackedImages {
    bytes: bytes::Bytes,
    frames: Vec<(ImageSize, usize)>,
    host: Vec<Mutex<Image<u16, 1>>>,
}
impl std::fmt::Debug for PackedImages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedImages")
            .field("frames", &self.frames)
            .field("bytes", &self.bytes)
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
    }
    #[cfg(feature = "gpu-core")]
    pub(crate) fn bytes(&self) -> &bytes::Bytes {
        &self.bytes
    }
    #[cfg(feature = "gpu-core")]
    pub(crate) fn same_pixels(&self, other: &Self) -> bool {
        self.frames == other.frames && self.bytes == other.bytes
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
                FrameImage {
                    source: FrameSource::Packed {
                        size,
                        bytes: &v.bytes[offset..offset + size.width * size.height],
                        host: &v.host[camera],
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
