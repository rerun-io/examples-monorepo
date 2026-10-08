//! Persistent floor-halved u16 pyramids on a CubeCL device.

use std::sync::Arc;

use cubecl::prelude::*;

use super::kernels;
use crate::runtime::{guarded, GpuError};
use crate::transfer::{binding_limits, read_failed, upload_inner};
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::pyramid::PyramidPlanError;
use std::mem::size_of;

/// Invalid pyramid geometry, transfer, or device operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PyramidError {
    /// Geometry violates the CPU u16 pyramid contract.
    #[error(transparent)]
    Geometry(#[from] PyramidPlanError),
    /// Runtime failure.
    #[error(transparent)]
    Gpu(#[from] GpuError),
    /// Requested level is absent.
    #[error("level {level} does not exist; the pyramid holds {num_levels}")]
    NoSuchLevel {
        /// Requested level.
        level: usize,
        /// Stored level count.
        num_levels: usize,
    },
    /// A batch has different input and output counts.
    #[error("batch has {inputs} inputs but {outputs} outputs")]
    BatchSizeMismatch {
        /// Number of inputs.
        inputs: usize,
        /// Number of outputs.
        outputs: usize,
    },
    /// Packed host bytes do not match the camera geometry.
    #[error("packed input has {actual} bytes, expected {expected}")]
    PackedLength {
        /// Supplied bytes.
        actual: usize,
        /// Required padded bytes.
        expected: usize,
    },
    /// A level buffer exceeds the device index range.
    #[error("pyramid buffer of {pixels} pixels exceeds the u32 index range")]
    BufferTooLong {
        /// Required pixels.
        pixels: usize,
    },
}

/// Where one camera's level 0 sits on the device, for the stage that reads the
/// same pixels.
///
/// The corner scanner and the pyramid builder are handed the *same* frame: the
/// detector's input is level 0 of the pyramid this builder has just filled
/// On the host that costs nothing —
/// the caller still owns the image — but on a device it is a second upload of
/// the whole frame. So the builder publishes level 0 here under the camera
/// index [`GpuPyramidBuilder::build`] gives it, and
/// a corner scanner reads that instead. Cloning a `Handle` keeps the
/// allocation alive, so a published entry is readable whatever happens to the
/// pyramid afterwards.
#[derive(Debug, Clone)]
pub(crate) struct Level0 {
    /// A view of exactly `width * height` packed `u16` pixels. Frameset builds
    /// publish the front of each camera's even-arena slot; individual builds
    /// publish the upload buffer that their level-zero copy reads.
    pub handle: cubecl::server::Handle,
    /// Level 0's width, checked against the frame the scanner was handed.
    pub width: usize,
    /// Level 0's height, checked the same way.
    pub height: usize,
    /// Shared all-camera storage, when this view belongs to a packed build.
    pub arena: Option<Arc<FrameArena>>,
    /// Camera slot in the shared allocation.
    pub camera: usize,
}

/// One level's place inside a [`GpuPyramid`]'s two buffers.
///
/// Visible to [`super::kernels`] because the subsample launcher takes a source
/// and a target level, and this names exactly the three fields it needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Level {
    /// Offset of pixel `(0, 0)` inside the buffer this level lives in.
    pub base: usize,
    /// Row length, which is also the stride: levels are packed without padding.
    pub width: usize,
    /// Row count.
    pub height: usize,
    /// `false` for buffer `a`, `true` for buffer `b`; the level index's parity.
    pub(crate) odd: bool,
}

/// One camera's pyramid, resident on the device.
///
/// Two `u16` allocations rather than one, because a subsample that read and
/// wrote the same allocation would need CubeCL to bind it as both a
/// `const __restrict__` input and an output, which WGSL refuses; level `l` goes
/// to `a` when `l` is even and `b` when it is odd, and no kernel ever reads the
/// buffer it writes.
///
/// Zero reductions are refused because the odd-level buffer would be empty,
/// which wgpu rejects. The CPU plan accepts zero reductions.
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

/// Aligned allocations shared by equal-geometry camera pyramids.
#[derive(Debug)]
pub struct FrameArena {
    /// Even-level storage for every camera.
    pub(crate) even: cubecl::server::Handle,
    /// Odd-level storage for every camera.
    pub(crate) odd: cubecl::server::Handle,
    /// Pixel stride between cameras in each allocation.
    pub(crate) strides: [usize; 2],
    /// Number of camera slots.
    pub(crate) cameras: usize,
}

impl FrameArena {
    /// Both allocations and their u16 element counts.
    pub(crate) fn bindings(&self) -> [kernels::Buffer<'_>; 2] {
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
            return Err(PyramidPlanError::TooSmall {
                width,
                height,
                max_level: num_levels,
            }
            .into());
        }
        kornia_staging_imgproc::pyramid::check_u16_geometry(
            ImageSize { width, height },
            num_levels,
        )?;
        let layout_error = || PyramidPlanError::LayoutOverflow { width, height };
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
            *slot = slot
                .checked_add(level_width * level_height)
                .ok_or_else(layout_error)?;
        }

        // Every integer in `meta` is an index into one of these two buffers —
        // a base, a width, a height — so refusing a buffer longer than a `u32`
        // refuses every field of every level at once, and the casts below are
        // exact by that check rather than by hope.
        for pixels in lengths {
            if u32::try_from(pixels).is_err() {
                return Err(PyramidError::BufferTooLong { pixels });
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
    pub(crate) fn buffers(&self) -> [kernels::Buffer<'_>; 2] {
        [(&self.even, self.even_len), (&self.odd, self.odd_len)]
    }

    /// Shared bindings and absolute level offsets for an all-camera dispatch.
    pub fn arena(&self) -> Option<&Arc<FrameArena>> {
        self.arena.as_ref()
    }

    pub(crate) fn append_geometry_levels(
        &self,
        out: &mut Vec<u32>,
        arena: Option<&FrameArena>,
        levels: usize,
    ) {
        for level in self.levels.iter().take(levels) {
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

/// The GPU pyramid builder.
///
/// Holds the client and reusable staging for packed camera
/// uploads. Frameset builds share even/odd arenas when camera geometry matches.
pub struct GpuPyramidBuilder<R: Runtime> {
    client: ComputeClient<R>,
    level0: Vec<Option<Level0>>,
    prepared: Vec<Option<Level0>>,
    packed_upload: Option<(usize, cubecl::server::Handle, Arc<()>)>,
}

impl<R: Runtime> GpuPyramidBuilder<R> {
    /// Create a builder with reusable transfer storage on `client`.
    ///
    /// # Arguments
    /// * `client` - Probed runtime client shared by the output pyramids.
    ///
    /// ```no_run
    /// # #[cfg(feature = "wgpu")] {
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_staging_gpu::{runtime::gpu_client, pyramid::GpuPyramidBuilder};
    /// let mut builder = GpuPyramidBuilder::new(gpu_client()?);
    /// let mut pyramid = builder.allocate(64, 48, 2)?;
    /// let source = Image::from_size_val(ImageSize { width: 64, height: 48 }, 1234u16)?;
    /// builder.build(0, &source, &mut pyramid)?;
    /// let mut level = source.clone();
    /// pyramid.read_level_into(2, &mut level)?;
    /// assert!(level.as_slice().iter().all(|&v| v == 1234));
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(client: ComputeClient<R>) -> Self {
        Self {
            client,
            level0: Default::default(),
            prepared: Vec::new(),
            packed_upload: None,
        }
    }

    /// Prepare uploads for individually built camera pyramids.
    ///
    /// # Arguments
    /// * `images` - Dense source images in camera order. Their pixels are retained.
    ///
    /// # Errors
    /// Returns a typed device error when allocation or upload panics.
    pub fn prepare_images(&mut self, images: &[Image<u16, 1>]) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "frameset uploads",
            },
            || {
                self.prepared.resize_with(images.len(), || None);
                for (slot, image) in self.prepared.iter_mut().zip(images) {
                    // Upload every camera before building its pyramid.
                    let handle = upload_inner(&self.client, u16::as_bytes(image.as_slice()));
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

impl<R: Runtime> GpuPyramidBuilder<R> {
    /// Allocate level zero and `num_levels` floor-halved reductions.
    ///
    /// # Arguments
    /// * `width`, `height` - Input dimensions in pixels.
    /// * `num_levels` - Positive number of reductions, following the CPU u16 contract.
    ///
    /// # Errors
    /// Rejects invalid geometry, device-index overflow, and device failures.
    pub fn allocate(
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

    /// Upload level zero and enqueue every reduction without waiting.
    ///
    /// # Arguments
    /// * `camera` - Index of the optional prepared upload and published level-zero view.
    /// * `img` - Source matching the output geometry.
    /// * `out` - Persistent pyramid allocated on this client's device.
    ///
    /// # Errors
    /// Returns a geometry mismatch or typed device failure.
    pub fn build(
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
                let level0 = out.levels[0];
                if level0.width != img.width() || level0.height != img.height() {
                    return Err(PyramidPlanError::GeometryMismatch {
                        expected_width: level0.width,
                        expected_height: level0.height,
                        width: img.width(),
                        height: img.height(),
                    }
                    .into());
                }

                // Copy level zero into the persistent even-level allocation.
                let (upload, pixels): (cubecl::server::Handle, usize) =
                    match self.prepared.get_mut(camera).and_then(Option::take) {
                        Some(frame)
                            if frame.width == img.width() && frame.height == img.height() =>
                        {
                            (frame.handle, frame.width * frame.height)
                        }
                        _ => (
                            upload_inner(&out.client, u16::as_bytes(img.as_slice())),
                            img.as_slice().len(),
                        ),
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

impl<R: Runtime> GpuPyramid<R> {
    /// Number of stored levels, including level zero.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Dimensions at `level`, or `None` beyond the last level.
    pub fn level_size(&self, level: usize) -> Option<ImageSize> {
        self.levels.get(level).map(|level| ImageSize {
            width: level.width,
            height: level.height,
        })
    }

    /// Download one level, resizing the destination only when needed.
    ///
    /// # Arguments
    /// * `level` - Zero-based pyramid level.
    /// * `out` - Caller-owned dense image to fill.
    ///
    /// # Errors
    /// Returns a missing-level, allocation, short-read, or typed device error.
    pub fn read_level_into(
        &self,
        level: usize,
        out: &mut Image<u16, 1>,
    ) -> Result<(), PyramidError> {
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
                    .map_err(|error| read_failed("a pyramid level", &error))?;

                let expected: usize = length * size_of::<u16>();
                if bytes.len() != expected {
                    return Err(GpuError::ShortRead {
                        what: "a pyramid level",
                        actual: bytes.len(),
                        expected,
                    }
                    .into());
                }
                let pixels: &[u16] = u16::from_bytes(&bytes);
                let size = ImageSize {
                    width: geometry.width,
                    height: geometry.height,
                };
                if out.size() != size {
                    *out = Image::from_size_val(size, 0u16).map_err(|_| {
                        PyramidPlanError::LayoutOverflow {
                            width: size.width,
                            height: size.height,
                        }
                    })?;
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

/// Prepared packed upload reductions for deferred device execution.
/// Keeps source and destination allocations alive until submission.
pub struct PyramidLaunch {
    arena: Arc<FrameArena>,
    levels: Vec<Level>,
    packed_input: (cubecl::server::Handle, usize, Arc<()>),
}

impl PyramidLaunch {
    /// Submit the prepared pyramid operations without waiting.
    ///
    /// # Safety
    /// Preparation, execution, and later buffer reuse must use the same client
    /// and CubeCL stream. The buffers must not be accessed concurrently.
    pub unsafe fn run<R: Runtime>(self, client: &ComputeClient<R>) {
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
                let (handle, len, _lease) = &packed_input;
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
    /// Upload every camera before enqueueing its pyramid reductions.
    ///
    /// # Arguments
    /// * `images` - Dense sources in camera order.
    /// * `out` - Matching persistent pyramids on this device.
    ///
    /// # Errors
    /// Returns a geometry mismatch or typed device failure.
    pub fn build_images(
        &mut self,
        images: &[Image<u16, 1>],
        out: &mut [GpuPyramid<R>],
    ) -> Result<(), PyramidError> {
        if images.len() != out.len() {
            return Err(PyramidError::BatchSizeMismatch {
                inputs: images.len(),
                outputs: out.len(),
            });
        }
        self.prepare_images(images)?;
        for (camera, (image, pyramid)) in images.iter().zip(out).enumerate() {
            self.build(camera, image, pyramid)?;
        }
        Ok(())
    }

    /// Upload retained u8 pixels and prepare an equal-geometry camera batch.
    /// Returns `None` when the geometry or binding limits require individual builds.
    /// Each u8 becomes a u16 with the byte in its high half, matching CPU ingestion.
    ///
    /// # Arguments
    /// * `packed` - Contiguous visible camera pixels, padded to a four-byte length.
    /// * `sizes` - Input dimensions in camera order.
    /// * `out` - Matching persistent pyramids on this device.
    ///
    /// # Errors
    /// Returns geometry or packed-length mismatches and typed device failures.
    pub fn prepare_packed(
        &mut self,
        packed: &bytes::Bytes,
        sizes: impl ExactSizeIterator<Item = ImageSize>,
        out: &mut [GpuPyramid<R>],
    ) -> Result<Option<PyramidLaunch>, PyramidError> {
        if sizes.len() != out.len() {
            return Err(PyramidError::BatchSizeMismatch {
                inputs: sizes.len(),
                outputs: out.len(),
            });
        }
        let Some(first) = out.first() else {
            return Ok(None);
        };
        let cameras = out.len();
        let (alignment, limit) = binding_limits(&self.client);
        let alignment = alignment / size_of::<u16>();
        let strides = [
            first.even_len.next_multiple_of(alignment),
            first.odd_len.next_multiple_of(alignment),
        ];
        let lengths = strides.map(|stride| stride.saturating_mul(cameras));
        // Mixed geometry and arenas beyond a binding's limit keep the general
        // per-camera path. In the common rig every camera has the same shape.
        let batchable = cameras <= kernels::MAX_CUBES_PER_DIM as usize
            && out.iter().all(|pyramid| pyramid.levels == first.levels)
            && lengths
                .iter()
                .all(|&len| len <= u32::MAX as usize && len <= limit / size_of::<u16>());
        if !batchable {
            self.prepared.clear();
            return Ok(None);
        }
        let expected =
            (first.levels[0].width * first.levels[0].height * cameras).next_multiple_of(4);
        guarded(
            GpuError::DeviceLost {
                what: "frameset pyramid",
            },
            || {
                for (image, pyramid) in sizes.zip(out.iter()) {
                    let level = pyramid.levels[0];
                    if image.width != level.width || image.height != level.height {
                        return Err(PyramidPlanError::GeometryMismatch {
                            expected_width: level.width,
                            expected_height: level.height,
                            width: image.width,
                            height: image.height,
                        }
                        .into());
                    }
                }
                if packed.len() != expected {
                    return Err(PyramidError::PackedLength {
                        actual: packed.len(),
                        expected,
                    });
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
                let bytes = packed.len();
                // A pending launch retains the lease. Reuse needs sole ownership;
                // submitted or dropped launches release their lease without waiting.
                // Ordinary sequential frames reuse both the buffer and this Arc.
                let (_, handle, lease) = match &mut self.packed_upload {
                    Some(existing)
                        if existing.0 == bytes && Arc::strong_count(&existing.2) == 1 =>
                    {
                        existing
                    }
                    slot => slot.insert((bytes, self.client.empty(bytes), Arc::new(()))),
                };
                self.client.write(
                    handle,
                    cubecl::bytes::Bytes::from_shared(
                        packed.clone(),
                        cubecl::bytes::AllocationProperty::Native,
                    ),
                );
                let packed_input = (handle.clone(), bytes, lease.clone());
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
                Ok(Some(PyramidLaunch {
                    arena,
                    levels: out[0].levels.clone(),
                    packed_input,
                }))
            },
        )
    }

    /// Replace the caller's level-zero views with this build's views.
    pub(crate) fn take_level0(&mut self, out: &mut Vec<Option<Level0>>) {
        out.clear();
        std::mem::swap(out, &mut self.level0);
    }
}
