//! The GPU [`PyramidBuilder`]: level 0 uploaded, every halving built on device.

use kornia_staging_imgproc::pyramid::PyramidPlanError;
use kornia_staging_gpu::runtime::GpuError;
use std::sync::Arc;

use cubecl::prelude::*;

use super::kernels;
use super::{ guarded};
use crate::frontend::input::{FrameImages, PackedImages};
use crate::pyramid::{MIN_SIDE, Pyramid, PyramidError};
use kornia_image::{Image, ImageSize};

/// Where one camera's level 0 sits on the device, for the stage that reads the
/// same pixels.
///
/// The corner scanner and the pyramid builder are handed the *same* frame: the
/// detector's input is level 0 of the pyramid this builder has just filled
/// On the host that costs nothing —
/// the caller still owns the image — but on a device it is a second upload of
/// the whole frame. So the builder publishes level 0 here under the camera
/// index [`crate::pyramid::PyramidBuilder::build`] gives it, and
/// [`super::GpuCornerScan`] reads that instead. Cloning a `Handle` keeps the
/// allocation alive, so a published entry is readable whatever happens to the
/// pyramid afterwards.
#[derive(Debug, Clone)]
pub(super) struct Level0 {
    /// A view of exactly `width * height` packed `u16` pixels. Frameset builds
    /// publish the front of each camera's even-arena slot; individual builds
    /// publish the upload buffer that their level-zero copy reads.
    pub(super) handle: cubecl::server::Handle,
    /// Level 0's width, checked against the frame the scanner was handed.
    pub(super) width: usize,
    /// Level 0's height, checked the same way.
    pub(super) height: usize,
    pub(super) arena: Option<Arc<FrameArena>>,
    pub(super) camera: usize,
}

/// One level's place inside a [`GpuPyramid`]'s two buffers.
///
/// Visible to [`super::kernels`] because the subsample launcher takes a source
/// and a target level, and this names exactly the three fields it needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Level {
    /// Offset of pixel `(0, 0)` inside the buffer this level lives in.
    pub(super) base: usize,
    /// Row length, which is also the stride: levels are packed without padding.
    pub(super) width: usize,
    /// Row count.
    pub(super) height: usize,
    /// `false` for buffer `a`, `true` for buffer `b`; the level index's parity.
    pub(super) odd: bool,
}

/// One camera's pyramid, resident on the device.
///
/// Two `u16` allocations rather than one, because a subsample that read and
/// wrote the same allocation would need CubeCL to bind it as both a
/// `const __restrict__` input and an output, which WGSL refuses; level `l` goes
/// to `a` when `l` is even and `b` when it is odd, and no kernel ever reads the
/// buffer it writes.
///
/// One configuration the CPU lane accepts and this one does not, left open by
/// the S25 review and recorded here rather than in a report only:
/// `optical_flow_levels = 0` builds level 0 alone on the CPU
/// ([`kornia_staging_imgproc::pyramid::PyramidPlanU16`]) and is [`PyramidError::TooSmall`] here,
/// because the odd buffer would then hold no level and `client.empty(0)` is a
/// zero-sized allocation wgpu rejects at validation. No shipped manifest sets
/// it, so no lane runs differently today; closing it means giving the empty
/// buffer a harmless length rather than refusing the geometry.
pub struct GpuPyramid<R: Runtime> {
    client: ComputeClient<R>,
    levels: Vec<Level>,
    even: cubecl::server::Handle,
    even_len: usize,
    odd: cubecl::server::Handle,
    odd_len: usize,
    /// Equal-geometry cameras share storage when built as a frameset. Per-camera
    /// handles remain views with local metadata for the unchanged tracker.
    arena: Option<Arc<FrameArena>>,
    arena_camera: usize,
}

#[derive(Debug)]
pub(super) struct FrameArena {
    pub(super) even: cubecl::server::Handle,
    pub(super) odd: cubecl::server::Handle,
    pub(super) strides: [usize; 2],
    pub(super) cameras: usize,
}

impl FrameArena {
    pub(super) fn bindings(&self) -> [kernels::Buffer<'_>; 2] {
        [
            (&self.even, self.strides[0] * self.cameras),
            (&self.odd, self.strides[1] * self.cameras),
        ]
    }
}

impl<R: Runtime> GpuPyramid<R> {
    /// Allocate levels zero through `num_levels`.
    ///
    /// # Errors
    /// Refuse geometry below the five-tap kernel's reach, zero requested levels,
    /// and buffers exceeding the u32 device index range.
    fn new(
        client: ComputeClient<R>,
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<Self, PyramidError> {
        // A pyramid of level 0 alone leaves the odd buffer with no level in it,
        // and `client.empty(0)` is a zero-sized allocation wgpu rejects at
        // validation — on cubecl's worker thread, so the launch would report
        // success and every read come back as zeros. Refused here rather than
        // discovered there; the CPU pyramid accepts it because nothing it
        // allocates can be empty.
        if num_levels == 0 {
            return Err(PyramidError::Plan(PyramidPlanError::TooSmall {
                width,
                height,
                max_level: num_levels,
            }));
        }
        for level in 0..num_levels {
            // The same refusal `PyramidPlanU16::new` makes, off the same
            // constant, so the two lanes cannot drift on what geometry is legal.
            if (width >> level) < MIN_SIDE || (height >> level) < MIN_SIDE {
                return Err(PyramidError::Plan(PyramidPlanError::TooSmall {
                    width,
                    height,
                    max_level: num_levels,
                }));
            }
        }
        let mut levels: Vec<Level> = Vec::with_capacity(num_levels + 1);
        let mut lengths: [usize; 2] = [0, 0];
        for level in 0..=num_levels {
            let odd: bool = level % 2 == 1;
            let (level_width, level_height): (usize, usize) = (width >> level, height >> level);
            let slot: &mut usize = &mut lengths[usize::from(odd)];
            levels.push(Level {
                base: *slot,
                width: level_width,
                height: level_height,
                odd,
            });
            *slot += level_width * level_height;
        }

        // Every integer in `meta` is an index into one of these two buffers —
        // a base, a width, a height — so refusing a buffer longer than a `u32`
        // refuses every field of every level at once, and the casts below are
        // exact by that check rather than by hope.
        for pixels in lengths {
            if u32::try_from(pixels).is_err() {
                return Err(GpuError::BufferTooLong { pixels }.into());
            }
        }

        Ok(Self {
            levels,
            even: client.empty(lengths[0] * size_of::<u16>()),
            even_len: lengths[0],
            odd: client.empty(lengths[1] * size_of::<u16>()),
            odd_len: lengths[1],
            arena: None,
            arena_camera: 0,
            client,
        })
    }

    /// The two pixel buffers and their element counts, as the launchers want them.
    pub(super) fn buffers(&self) -> [kernels::Buffer<'_>; 2] {
        [(&self.even, self.even_len), (&self.odd, self.odd_len)]
    }

    /// Shared bindings and absolute level offsets for an all-camera dispatch.
    pub(super) fn arena(&self) -> Option<&Arc<FrameArena>> {
        self.arena.as_ref()
    }

    /// Append exact integer geometry, using arena offsets when supplied.
    pub(super) fn append_geometry(&self, out: &mut Vec<u32>, arena: Option<&FrameArena>) {
        for level in &self.levels {
            out.extend([
                (level.base
                    + arena.map_or(0, |arena| {
                        self.arena_camera * arena.strides[usize::from(level.odd)]
                    })) as u32,
                level.width as u32,
                level.height as u32,
                u32::from(level.odd),
            ]);
        }
    }

    /// The handle and element count of the buffer level `level` lives in.
    fn buffer_of(&self, level: &Level) -> (&cubecl::server::Handle, usize) {
        if level.odd {
            (&self.odd, self.odd_len)
        } else {
            (&self.even, self.even_len)
        }
    }
}

/// The GPU [`crate::pyramid::PyramidBuilder`].
///
/// Holds the client and reusable staging for packed camera
/// uploads. Frameset builds share even/odd arenas when camera geometry matches.
pub struct GpuPyramidBuilder<R: Runtime> {
    launches: super::submission::LaunchList,
    client: ComputeClient<R>,
    pub(super) level0: Vec<Option<Level0>>,
    prepared: Vec<Option<Level0>>,
    packed_upload: Option<(usize, cubecl::server::Handle)>,
}

impl<R: Runtime> GpuPyramidBuilder<R> {
    /// A pyramid builder on `client`.
    pub fn new(client: ComputeClient<R>, launches: super::submission::LaunchList) -> Self {
        Self {
            client,
            level0: Default::default(),
            prepared: Vec::new(),
            packed_upload: None,
            launches,
        }
    }

    /// Prepare uploads for individually built camera pyramids.
    pub fn prepare_images(&mut self, images: &[Image<u16, 1>]) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "frameset uploads",
            },
            || {
                self.prepared.resize_with(images.len(), || None);
                for (slot, image) in self.prepared.iter_mut().zip(images) {
                    // Upload every camera before building its pyramid.
                    let (handle, _) = super::upload_frame(&self.client, image);
                    *slot = Some(Level0 {
                        handle,
                        width: image.width(),
                        height: image.height(),
                        arena: None,
                        camera: 0,
                    });
                }
                Ok(())
            },
        )
    }
}

impl<R: Runtime> crate::pyramid::PyramidBuilder for GpuPyramidBuilder<R> {
    type Pyramid = GpuPyramid<R>;
    fn build_frames(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [GpuPyramid<R>],
        _pool: &crate::frontend::parallel::WorkPool,
    ) -> Result<(), PyramidError> {
        self.build_images(images, out)
    }

    fn allocate(
        &self,
        width: usize,
        height: usize,
        num_levels: usize,
    ) -> Result<GpuPyramid<R>, PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "pyramid allocation",
            },
            || GpuPyramid::new(self.client.clone(), width, height, num_levels),
        )
    }

    /// `ManagedImagePyr::setFromImage` on the device.
    ///
    /// One upload for level 0 and one launch per halving. No synchronisation:
    /// the frame is left in flight and the tracker's own launches queue behind
    /// it, so the whole frameset costs one wait per
    /// [`kornia_staging_slam::tracking::optical_flow::PatchTracker::track`] call.
    fn build(
        &mut self,
        camera: usize,
        img: &Image<u16, 1>,
        out: &mut GpuPyramid<R>,
    ) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "pyramid build",
            },
            || {
                let Some(&level0) = out.levels.first() else {
                    return Err(PyramidError::Plan(PyramidPlanError::GeometryMismatch {
                        expected_width: 0,
                        expected_height: 0,
                        width: img.width(),
                        height: img.height(),
                    }));
                };
                if level0.width != img.width() || level0.height != img.height() {
                    return Err(PyramidError::Plan(PyramidPlanError::GeometryMismatch {
                        expected_width: level0.width,
                        expected_height: level0.height,
                        width: img.width(),
                        height: img.height(),
                    }));
                }

                // Level 0 reaches the front of the even allocation through one
                // device copy off the upload buffer, which costs microseconds and
                // lets that allocation be made once at `allocate` instead of
                // replaced every frame. `upload_frame` is what puts the frame
                // there, and its doc is where the copy count lives.
                let (upload, pixels): (cubecl::server::Handle, usize) =
                    match self.prepared.get_mut(camera).and_then(Option::take) {
                        Some(frame)
                            if frame.width == img.width() && frame.height == img.height() =>
                        {
                            (frame.handle, frame.width * frame.height)
                        }
                        _ => super::upload_frame(&out.client, img),
                    };
                // SAFETY: The upload holds this frame's pixels; the validated
                // even allocation contains level 0 and is a distinct buffer.
                unsafe {
                    kernels::launch_copy_level0::<R>(
                        &out.client,
                        (&upload, pixels),
                        (&out.even, out.even_len),
                        pixels,
                    );
                }

                if self.level0.len() <= camera {
                    self.level0.resize(camera + 1, None);
                }
                self.level0[camera] = Some(Level0 {
                    handle: upload,
                    width: level0.width,
                    height: level0.height,
                    arena: None,
                    camera,
                });

                // Build levels in source-before-target order.
                for level in 1..out.levels.len() {
                    let source: Level = out.levels[level - 1];
                    let target: Level = out.levels[level];
                    kernels::launch_subsample::<R>(
                        &out.client,
                        out.buffer_of(&source),
                        out.buffer_of(&target),
                        source,
                        target,
                    );
                }
                Ok(())
            },
        )
    }
}

impl<R: Runtime> Pyramid for GpuPyramid<R> {
    fn num_levels(&self) -> usize {
        self.levels.len()
    }

    fn level_size(&self, level: usize) -> Option<ImageSize> {
        self.levels.get(level).map(|level| ImageSize {
            width: level.width,
            height: level.height,
        })
    }

    /// Download one level.
    ///
    /// The one synchronising call on a [`GpuPyramid`], and nothing on the
    /// per-frame path uses it: it exists because the trait's forward half is how
    /// generic code — and the tolerance tests — read a pyramid a GPU backend
    /// owns. Being off the per-frame path is also why it carries its own guard
    /// rather than sitting inside a stage's: a download panics rather than
    /// returning when the device is gone (decision D32).
    fn copy_level_into(&self, level: usize, out: &mut Image<u16, 1>) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "a pyramid level read",
            },
            || {
                let Some(&geometry) = self.levels.get(level) else {
                    return Err(PyramidError::NoSuchLevel {
                        level,
                        num_levels: self.levels.len(),
                    });
                };
                let (handle, length) = self.buffer_of(&geometry);
                let bytes = self
                    .client
                    .read_one(handle.clone())
                    .map_err(|error| super::read_failed("a pyramid level", &error))?;

                let expected: usize = length * size_of::<u16>();
                if bytes.len() != expected {
                    return Err(PyramidError::ShortDeviceRead {
                        level,
                        actual: bytes.len(),
                        expected,
                    });
                }
                let pixels: &[u16] = u16::from_bytes(&bytes);
                let size = ImageSize {
                    width: geometry.width,
                    height: geometry.height,
                };
                if out.size() != size {
                    *out = crate::image::zeros(size.width, size.height)?;
                }
                let end = geometry.base + geometry.width * geometry.height;
                out.as_slice_mut()
                    .copy_from_slice(&pixels[geometry.base..end]);
                Ok(())
            },
        )
    }
}

/// `ComputeClient` is not `Debug`, so both types print their geometry instead of
/// deriving it.
impl<R: Runtime> std::fmt::Debug for GpuPyramid<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPyramid")
            .field("levels", &self.levels)
            .field("even_len", &self.even_len)
            .field("odd_len", &self.odd_len)
            .finish()
    }
}

impl<R: Runtime> std::fmt::Debug for GpuPyramidBuilder<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPyramidBuilder")
            .field("level0_cameras", &self.level0.len())
            .finish()
    }
}

pub(super) struct PyramidLaunch {
    arena: Arc<FrameArena>,
    levels: Vec<Level>,
    packed_input: (cubecl::server::Handle, usize),
}

impl PyramidLaunch {
    pub(super) fn run<R: Runtime>(self, client: &ComputeClient<R>) {
        let Self {
            arena,
            levels,
            packed_input,
        } = self;
        let strides = arena.strides;
        let cameras = arena.cameras;
        let buffers = arena.bindings();
        for pair in levels.windows(2) {
            let [source, target] = [pair[0], pair[1]];
            if source == levels[0] {
                let (handle, len) = &packed_input;
                kernels::launch_ingest(
                    client,
                    (handle, *len),
                    buffers[0],
                    buffers[1],
                    source.width,
                    source.height,
                    strides[1],
                    cameras,
                );
            } else {
                kernels::launch_subsample_batch(
                    client,
                    buffers[usize::from(source.odd)],
                    buffers[usize::from(target.odd)],
                    source,
                    target,
                    strides[usize::from(source.odd)],
                    strides[usize::from(target.odd)],
                    cameras,
                );
            }
        }
    }
}

impl<R: Runtime> GpuPyramidBuilder<R> {
    pub(super) fn build_images(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [GpuPyramid<R>],
    ) -> Result<(), PyramidError> {
        use crate::pyramid::PyramidBuilder;
        guarded(
            GpuError::DeviceLost {
                what: "frameset pyramid",
            },
            || {
                self.launches.flush(&self.client);
                self.prepare_images(images)?;
                for (camera, (image, pyramid)) in images.iter().zip(out).enumerate() {
                    self.build(camera, image, pyramid)?;
                }
                Ok(())
            },
        )
    }

    pub(super) fn build_packed(
        &mut self,
        packed: &PackedImages,
        out: &mut [GpuPyramid<R>],
    ) -> Result<(), PyramidError> {
        use crate::pyramid::PyramidBuilder;
        let images = FrameImages::Packed(packed);
        let Some(first) = out.first() else {
            return Ok(());
        };
        let cameras = out.len();
        let (alignment, limit) = super::submission::binding_limits(&self.client);
        let alignment = alignment / size_of::<u16>();
        let strides = [
            first.even_len.next_multiple_of(alignment),
            first.odd_len.next_multiple_of(alignment),
        ];
        let lengths = strides.map(|stride| stride.saturating_mul(cameras));
        // Mixed geometry and arenas beyond a binding's limit keep the general
        // per-camera path. In the common rig every camera has the same shape.
        let batchable = images.len() == cameras
            && cameras <= kernels::MAX_CUBES_PER_DIM as usize
            && out.iter().all(|pyramid| pyramid.levels == first.levels)
            && lengths
                .iter()
                .all(|&len| len <= u32::MAX as usize && len <= limit / size_of::<u16>());
        if !batchable {
            return guarded(
                GpuError::DeviceLost {
                    what: "frameset pyramid",
                },
                || {
                    self.launches.flush(&self.client);
                    self.prepared.clear();
                    for (camera, (image, pyramid)) in images.iter().zip(out).enumerate() {
                        image.with_dense(|image| self.build(camera, image, pyramid))??;
                    }
                    Ok(())
                },
            );
        }
        guarded(
            GpuError::DeviceLost {
                what: "frameset pyramid",
            },
            || {
                for (image, pyramid) in images.iter().zip(out.iter()) {
                    let level = pyramid.levels[0];
                    if image.width() != level.width || image.height() != level.height {
                        return Err(PyramidError::Plan(PyramidPlanError::GeometryMismatch {
                            expected_width: level.width,
                            expected_height: level.height,
                            width: image.width(),
                            height: image.height(),
                        }));
                    }
                }
                let shared = out[0]
                    .arena
                    .as_ref()
                    .filter(|arena| {
                        arena.cameras == cameras
                            && arena.strides == strides
                            && out.iter().enumerate().all(|(camera, pyramid)| {
                                pyramid.arena_camera == camera
                                    && pyramid
                                        .arena
                                        .as_ref()
                                        .is_some_and(|other| Arc::ptr_eq(arena, other))
                            })
                    })
                    .cloned();
                let arena = match shared {
                    Some(arena) => arena,
                    None => {
                        let arena = Arc::new(FrameArena {
                            even: self.client.empty(lengths[0] * size_of::<u16>()),
                            odd: self.client.empty(lengths[1] * size_of::<u16>()),
                            strides,
                            cameras,
                        });
                        for (camera, pyramid) in out.iter_mut().enumerate() {
                            let start = camera * strides[0];
                            pyramid.even = arena
                                .even
                                .clone()
                                .offset_start((start * size_of::<u16>()) as u64)
                                .offset_end(
                                    ((lengths[0] - start - pyramid.even_len) * size_of::<u16>())
                                        as u64,
                                );
                            let start = camera * strides[1];
                            pyramid.odd = arena
                                .odd
                                .clone()
                                .offset_start((start * size_of::<u16>()) as u64)
                                .offset_end(
                                    ((lengths[1] - start - pyramid.odd_len) * size_of::<u16>())
                                        as u64,
                                );
                            pyramid.arena = Some(arena.clone());
                            pyramid.arena_camera = camera;
                        }
                        arena
                    }
                };
                self.prepared.clear();
                // CubeCL owns transport bytes; the frameset staging buffer is
                // reused by ingestion and also validates retained lookahead.
                let bytes = packed.bytes().len();
                let (_, handle) = match &mut self.packed_upload {
                    Some(existing) if existing.0 == bytes => existing,
                    slot => slot.insert((bytes, self.client.empty(bytes))),
                };
                self.client.write(
                    handle,
                    cubecl::bytes::Bytes::from_shared(
                        packed.bytes().clone(),
                        cubecl::bytes::AllocationProperty::Native,
                    ),
                );
                let packed_input = (handle.clone(), bytes);
                self.level0.resize(cameras, None);
                for (camera, pyramid) in out.iter().enumerate() {
                    let level = pyramid.levels[0];
                    self.level0[camera] = Some(Level0 {
                        handle: pyramid.even.clone().offset_end(
                            ((pyramid.even_len - level.width * level.height) * size_of::<u16>())
                                as u64,
                        ),
                        width: level.width,
                        height: level.height,
                        arena: Some(arena.clone()),
                        camera,
                    });
                }
                self.launches.dispatch(
                    &self.client,
                    super::submission::Launch::Pyramid(PyramidLaunch {
                        arena,
                        levels: out[0].levels.clone(),
                        packed_input,
                    }),
                );
                Ok(())
            },
        )
    }
}

#[cfg(all(test, feature = "gpu-wgpu"))]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::pyramid::PyramidBuilder;

    #[test]
    fn dense_u16_inputs_keep_the_general_path_and_all_low_bits() {
        let client = kornia_staging_gpu::runtime::gpu_client().unwrap();
        let mut builder = GpuPyramidBuilder::new(client, Default::default());
        for shift_only in [true, false] {
            let pixels: Vec<u16> = (0..64 * 48)
                .map(|i| {
                    if shift_only {
                        ((i % 256) as u16) << 8
                    } else {
                        (i * 17) as u16
                    }
                })
                .collect();
            let image = Image::new(
                ImageSize {
                    width: 64,
                    height: 48,
                },
                pixels,
            )
            .unwrap();
            let mut pyramids = vec![builder.allocate(64, 48, 2).unwrap()];
            builder
                .build_images(std::slice::from_ref(&image), &mut pyramids)
                .unwrap();
            assert!(pyramids[0].arena.is_none());
            let mut level0 = crate::image::empty();
            pyramids[0].copy_level_into(0, &mut level0).unwrap();
            assert_eq!(level0.as_slice(), image.as_slice());
        }
    }

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
            assert!(pyramid.arena.is_some());
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
