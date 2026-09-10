//! The GPU [`PyramidBuilder`]: level 0 uploaded, every halving built on device.

use std::sync::{Arc, Mutex};

use cubecl::prelude::*;

use super::kernels;
use super::{GpuError, guarded};
use crate::image::ImageU16;
use crate::pyramid::{MIN_SIDE, Pyramid, PyramidError};

/// Where one camera's level 0 sits on the device, for the stage that reads the
/// same pixels.
///
/// The corner scanner and the pyramid builder are handed the *same* frame: the
/// detector's input is level 0 of the pyramid this builder has just filled
/// (`keypoints.cpp:152`, `image_pyr.h:73`). On the host that costs nothing —
/// the caller still owns the image — but on a device it is a second upload of
/// the whole frame. So the builder publishes level 0 here under the camera
/// index [`crate::pyramid::PyramidBuilder::build`] gives it, and
/// [`super::GpuCornerScan`] reads that instead. Cloning a `Handle` keeps the
/// allocation alive, so a published entry is readable whatever happens to the
/// pyramid afterwards.
#[derive(Debug, Clone)]
pub struct Level0 {
    /// The upload buffer, which is the frame and nothing else: `width * height`
    /// `u16` with a stride equal to the width, the upload being exactly as long
    /// as the frame. A device copy is what puts the same pixels at the front of
    /// the pyramid's even allocation.
    pub(super) handle: cubecl::server::Handle,
    /// Level 0's width, checked against the frame the scanner was handed.
    pub(super) width: usize,
    /// Level 0's height, checked the same way.
    pub(super) height: usize,
}

/// The per-camera level-0 table a builder and a scanner on one client share.
///
/// A `Mutex` rather than a `RefCell` because [`crate::frontend::detect::CornerScan`]
/// is `Send + Sync`; it is taken twice per camera per frameset and never
/// contended, both stages running on the frontend's own thread.
pub type Level0Table = Arc<Mutex<Vec<Option<Level0>>>>;

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
/// The `meta` buffer beside them carries the level geometry and the sampling
/// pattern the per-patch kernels read (see `kernels`); it is written once,
/// when the pyramid is allocated, and never touched per frame.
///
/// One configuration the CPU lane accepts and this one does not, left open by
/// the S25 review and recorded here rather than in a report only:
/// `optical_flow_levels = 0` builds level 0 alone on the CPU
/// ([`crate::pyramid::PyramidU16`]) and is [`PyramidError::TooSmall`] here,
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
    meta: cubecl::server::Handle,
    meta_len: usize,
}

impl<R: Runtime> GpuPyramid<R> {
    /// Allocate every level for a `width` x `height` frame and write `meta`.
    ///
    /// `num_levels` is basalt's, so the pyramid holds `num_levels + 1` levels
    /// (0 through `num_levels`), matching [`crate::pyramid::PyramidU16`].
    ///
    /// # Errors
    ///
    /// [`PyramidError::TooSmall`] when a level would be under the 5-tap
    /// kernel's reach, as the CPU pyramid refuses it, and for `num_levels == 0`,
    /// which the CPU pyramid accepts and this one cannot allocate;
    /// [`GpuError::BufferTooLong`] when a pyramid buffer would be longer than
    /// the `u32` its device metadata indexes it with.
    fn new(
        client: ComputeClient<R>,
        width: usize,
        height: usize,
        num_levels: usize,
        pattern: &[[f32; 2]],
    ) -> Result<Self, PyramidError> {
        // A pyramid of level 0 alone leaves the odd buffer with no level in it,
        // and `client.empty(0)` is a zero-sized allocation wgpu rejects at
        // validation — on cubecl's worker thread, so the launch would report
        // success and every read come back as zeros. Refused here rather than
        // discovered there; the CPU pyramid accepts it because nothing it
        // allocates can be empty.
        if num_levels == 0 {
            return Err(PyramidError::TooSmall {
                width,
                height,
                num_levels,
            });
        }
        for level in 0..num_levels {
            // The same refusal `PyramidU16::with_capacity` makes, off the same
            // constant, so the two lanes cannot drift on what geometry is legal.
            if (width >> level) < MIN_SIDE || (height >> level) < MIN_SIDE {
                return Err(PyramidError::TooSmall {
                    width,
                    height,
                    num_levels,
                });
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

        // The `meta` layout `kernels` documents: four integers per level, then
        // the pattern taps as their bit patterns. `u32` and not `f32`, because
        // a base is an index the kernels add to a pixel offset and `f32` holds
        // only every second integer above 2^24: a 4097x4097 frame puts level 2
        // at base 16,785,409, which `f32` stores as 16,785,408, and every
        // level-2 sample then read one pixel early on both lanes with nothing
        // reporting it (the S25 review). The taps ride in the same buffer as
        // bits rather than in a second one because both per-patch kernels are at
        // the six-buffer ceiling `kernels` documents and a seventh binding is a
        // question on every device the portable lane runs on, while a bitcast is
        // one instruction on all three of its shader compilers.
        let mut meta: Vec<u32> = Vec::with_capacity((num_levels + 1) * 4 + pattern.len() * 2);
        for level in &levels {
            meta.push(level.base as u32);
            meta.push(level.width as u32);
            meta.push(level.height as u32);
            meta.push(u32::from(level.odd));
        }
        for tap in pattern {
            meta.push(tap[0].to_bits());
            meta.push(tap[1].to_bits());
        }

        Ok(Self {
            levels,
            even: super::empty(&client, lengths[0] * size_of::<u16>()),
            even_len: lengths[0],
            odd: super::empty(&client, lengths[1] * size_of::<u16>()),
            odd_len: lengths[1],
            meta: super::submission::upload(&client, u32::as_bytes(&meta)),
            meta_len: meta.len(),
            client,
        })
    }

    /// The two pixel buffers and their element counts, as the launchers want them.
    pub(super) fn buffers(
        &self,
    ) -> (
        &cubecl::server::Handle,
        usize,
        &cubecl::server::Handle,
        usize,
    ) {
        (&self.even, self.even_len, &self.odd, self.odd_len)
    }

    /// The `meta` buffer and its element count.
    pub(super) fn meta(&self) -> (&cubecl::server::Handle, usize) {
        (&self.meta, self.meta_len)
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
/// Holds the client, the pattern the per-patch kernels need in every pyramid's
/// `meta`, and a repack buffer used only by a frame whose stride exceeds its
/// width — a level never is strided, and one upload of a packed buffer beats
/// one upload per row.
pub struct GpuPyramidBuilder<R: Runtime> {
    client: ComputeClient<R>,
    pattern: Vec<[f32; 2]>,
    staging: Vec<u16>,
    level0: Level0Table,
    prepared: Vec<Option<Level0>>,
}

impl<R: Runtime> GpuPyramidBuilder<R> {
    /// A builder on `client` for a frontend using `pattern`.
    pub fn new(client: ComputeClient<R>, pattern: &[[f32; 2]]) -> Self {
        Self {
            client,
            pattern: pattern.to_vec(),
            staging: Vec::new(),
            level0: Level0Table::default(),
            prepared: Vec::new(),
        }
    }

    /// The client, for the tracker that shares it.
    pub fn client(&self) -> ComputeClient<R> {
        self.client.clone()
    }

    /// The level-0 table this builder publishes into, for the corner scanner
    /// that reads the same frames — see [`Level0`]. Handed over by
    /// [`super::gpu_backends`] when it builds the two on one client.
    pub fn level0_table(&self) -> Level0Table {
        Arc::clone(&self.level0)
    }
}

impl<R: Runtime> crate::pyramid::PyramidBuilder for GpuPyramidBuilder<R> {
    type Pyramid = GpuPyramid<R>;
    const PREPARE_IMAGES: bool = true;

    fn prepare_images(&mut self, images: &[ImageU16]) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "frameset uploads",
            },
            || {
                self.prepared.resize_with(images.len(), || None);
                for (slot, image) in self.prepared.iter_mut().zip(images) {
                    // upload_frame reserves its task here, before any camera builds.
                    let (handle, _) = super::upload_frame(&self.client, image, &mut self.staging);
                    *slot = Some(Level0 {
                        handle,
                        width: image.width(),
                        height: image.height(),
                    });
                }
                Ok(())
            },
        )
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
            || {
                GpuPyramid::new(
                    self.client.clone(),
                    width,
                    height,
                    num_levels,
                    &self.pattern,
                )
            },
        )
    }

    /// `ManagedImagePyr::setFromImage` (`image_pyr.h:70-80`) on the device.
    ///
    /// One upload for level 0 and one launch per halving. No synchronisation:
    /// the frame is left in flight and the tracker's own launches queue behind
    /// it, so the whole frameset costs one wait per
    /// [`crate::frontend::tracker::PatchTracker::track`] call.
    fn build(
        &mut self,
        camera: usize,
        img: &ImageU16,
        out: &mut GpuPyramid<R>,
    ) -> Result<(), PyramidError> {
        guarded(
            GpuError::DeviceLost {
                what: "pyramid build",
            },
            || {
                let Some(&level0) = out.levels.first() else {
                    return Err(PyramidError::GeometryMismatch {
                        expected_width: 0,
                        expected_height: 0,
                        width: img.width(),
                        height: img.height(),
                    });
                };
                if level0.width != img.width() || level0.height != img.height() {
                    return Err(PyramidError::GeometryMismatch {
                        expected_width: level0.width,
                        expected_height: level0.height,
                        width: img.width(),
                        height: img.height(),
                    });
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
                        _ => super::upload_frame(&out.client, img, &mut self.staging),
                    };
                kernels::launch_copy_level0::<R>(
                    &out.client,
                    (&upload, pixels),
                    (&out.even, out.even_len),
                    pixels,
                );

                // Level 0 is now on the device and the detector wants exactly it — the
                // upload buffer, which is the frame and nothing else. A poisoned lock is
                // left to fall through: the scanner then uploads its own copy, which is
                // slower and correct.
                if let Ok(mut table) = self.level0.lock() {
                    if table.len() <= camera {
                        table.resize(camera + 1, None);
                    }
                    table[camera] = Some(Level0 {
                        handle: upload,
                        width: level0.width,
                        height: level0.height,
                    });
                } else {
                    log::warn!(
                        "the shared level-0 table is poisoned: camera {camera} uploads its frame twice from here on"
                    );
                }

                // Each launch reserves one task; deeper pyramids split at the queue ceiling.
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

    fn level_size(&self, level: usize) -> Option<(usize, usize, usize)> {
        self.levels
            .get(level)
            .map(|level| (level.width, level.height, level.width))
    }

    /// Download one level.
    ///
    /// The one synchronising call on a [`GpuPyramid`], and nothing on the
    /// per-frame path uses it: it exists because the trait's forward half is how
    /// generic code — and the tolerance tests — read a pyramid a GPU backend
    /// owns. Being off the per-frame path is also why it carries its own guard
    /// rather than sitting inside a stage's: a download panics rather than
    /// returning when the device is gone (decision D32).
    fn copy_level_into(&self, level: usize, out: &mut ImageU16) -> Result<(), PyramidError> {
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
                super::drained(&self.client);
                let expected: usize = length * size_of::<u16>();
                if bytes.len() != expected {
                    return Err(PyramidError::ShortDeviceRead {
                        level,
                        actual: bytes.len(),
                        expected,
                    });
                }
                let pixels: &[u16] = u16::from_bytes(&bytes);
                *out = ImageU16::zeros(geometry.width, geometry.height)?;
                for y in 0..geometry.height {
                    let start: usize = geometry.base + y * geometry.width;
                    out.row_mut(y)
                        .copy_from_slice(&pixels[start..start + geometry.width]);
                }
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
            .field("taps", &self.pattern.len())
            .field("staging", &self.staging.len())
            .field("level0_cameras", &self.level0.lock().map(|t| t.len()).ok())
            .finish()
    }
}
