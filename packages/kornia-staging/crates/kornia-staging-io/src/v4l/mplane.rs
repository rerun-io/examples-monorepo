#![allow(unsafe_code)] // Linux UAPI calls uphold the documented buffer and ownership contracts.
//! Linux multi-planar V4L2 capture with explicit format, clock flags and owned leases.

mod abi;
mod device;

/// V4L2 configuration, syscall or clock failure.
#[derive(Debug, thiserror::Error)]
pub enum MplaneError {
    /// System call failed.
    #[error("{what}: {source}")]
    Io {
        /// Operation.
        what: String,
        /// OS error.
        source: std::io::Error,
    },
    /// Invalid configuration or negotiated layout.
    #[error("{0}")]
    Device(String),
    /// A timestamp is not monotonic.
    #[error("{0}")]
    Timing(String),
    /// Image construction failed.
    #[error(transparent)]
    Image(#[from] kornia_image::ImageError),
}

/// One negotiated raw plane, including row padding.
#[derive(Clone, Copy, Debug)]
pub struct PlaneLayout {
    /// Bytes per row.
    pub stride: usize,
    /// Minimum used bytes required at dequeue.
    pub bytes: usize,
}

/// Validated capture geometry. Plane zero must be an 8-bit luma plane.
#[derive(Clone, Debug)]
pub struct CaptureFormat {
    size: kornia_image::ImageSize,
    fourcc: u32,
    planes: Vec<PlaneLayout>,
    luma_bytes: usize,
}
impl CaptureFormat {
    /// Validate dimensions and the raw plane layout once, before device setup.
    /// # Arguments
    /// * `size` - visible luma size.
    /// * `fourcc` - V4L2 format, such as `u32::from_le_bytes(*b"NV12")`.
    /// * `planes` - one to eight raw planes; plane zero carries luma.
    /// # Errors
    /// Rejects empty, overflowing or undersized layouts.
    pub fn new(
        size: kornia_image::ImageSize,
        fourcc: u32,
        planes: Vec<PlaneLayout>,
    ) -> Result<Self, MplaneError> {
        if size.width == 0
            || size.height == 0
            || size.width > u32::MAX as usize
            || size.height > u32::MAX as usize
            || planes.is_empty()
            || planes.len() > MAX_PLANES
        {
            return Err(MplaneError::Device(
                "invalid capture dimensions or plane count".into(),
            ));
        }
        let luma_bytes = planes[0]
            .stride
            .checked_mul(size.height - 1)
            .and_then(|v| v.checked_add(size.width))
            .ok_or_else(|| MplaneError::Device("capture size overflow".into()))?;
        if planes[0].stride < size.width
            || planes[0].bytes < luma_bytes
            || planes.iter().any(|p| {
                p.stride == 0
                    || p.bytes == 0
                    || p.stride > u32::MAX as usize
                    || p.bytes > u32::MAX as usize
            })
        {
            return Err(MplaneError::Device(
                "invalid capture stride or plane bounds".into(),
            ));
        }
        Ok(Self {
            size,
            fourcc,
            planes,
            luma_bytes,
        })
    }
    /// Visible luma dimensions.
    pub fn size(&self) -> kornia_image::ImageSize {
        self.size
    }

    /// Negotiated raw plane strides and minimum lengths.
    pub fn planes(&self) -> &[PlaneLayout] {
        &self.planes
    }

    fn raw(&self) -> RawFormat {
        let mut raw = RawFormat {
            width: self.size.width as u32,
            height: self.size.height as u32,
            fourcc: self.fourcc,
            planes: self.planes.len() as u32,
            stride: [0; MAX_PLANES],
            bytes: [0; MAX_PLANES],
        };
        for (i, plane) in self.planes.iter().enumerate() {
            raw.stride[i] = plane.stride as u32;
            raw.bytes[i] = plane.bytes as u32;
        }
        raw
    }
}
struct RawFormat {
    width: u32,
    height: u32,
    fourcc: u32,
    planes: u32,
    stride: [u32; MAX_PLANES],
    bytes: [u32; MAX_PLANES],
}

use std::any::Any;
use std::ffi::CString;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc};

use kornia_image::Image;

const MAX_PLANES: usize = 8;

/// `V4L2_BUF_FLAG_TIMESTAMP_MASK`.
const TIMESTAMP_MASK: u32 = 0xe000;
/// `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC`.
const TIMESTAMP_MONOTONIC: u32 = 0x2000;

/// Diagnostic metadata copied immediately after DQBUF, before validation or copying pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct DequeueMeta {
    /// V4L2 buffer index.
    pub index: u32,
    /// Driver sequence number, including rejected buffers.
    pub sequence: u32,
    /// Raw V4L2 status, timestamp type and timestamp source flags.
    pub flags: u32,
    /// Driver timeval converted to nanoseconds.
    pub timestamp_ns: i64,
    /// CLOCK_MONOTONIC immediately after DQBUF returns.
    pub dequeue_ns: i64,
    /// Number of returned planes, capped at the UAPI array capacity.
    pub planes: u32,
    /// Raw bytesused for each plane (includes data_offset).
    pub bytesused: [u32; MAX_PLANES],
}

/// A buffer lent out by [`Camera::dequeue`]: the kernel does not write it until [`LentBuffer::queue`] gives it back. The lease
/// holds its camera's descriptor and mappings, so its luma stays readable after the [`Camera`] has dropped:
///
/// ```no_run
/// # fn main() -> Result<(), kornia_staging_io::v4l::mplane::MplaneError> {
/// # let format = kornia_staging_io::v4l::mplane::CaptureFormat::new(kornia_image::ImageSize { width: 640, height: 480 }, u32::from_le_bytes(*b"NV12"), vec![kornia_staging_io::v4l::mplane::PlaneLayout { stride: 640, bytes: 640*480*3/2 }])?;
/// let camera = kornia_staging_io::v4l::mplane::Camera::open("/dev/video0", format)?;
/// if let Some(buffer) = camera.dequeue(200)? {
///     drop(camera);
///     let first = buffer.luma()[0];
///     buffer.queue()?;
/// }
/// # Ok(())
/// # }
/// ```
///
/// Its buffer index is read-only, so a lease always gives back the buffer it lent:
///
/// ```compile_fail,E0616
/// fn retarget(buffer: &mut kornia_staging_io::v4l::mplane::LentBuffer) {
///     buffer.meta.index = 0;
/// }
/// ```
#[derive(Debug)]
#[must_use = "a lent buffer returns to the driver when queued explicitly or dropped"]
pub struct LentBuffer {
    meta: DequeueMeta,
    planes: [*const u8; MAX_PLANES],
    lengths: [usize; MAX_PLANES],
    queued: bool,
    owner: Arc<CameraOwner>,
}

// SAFETY: a buffer index plus a pointer into a mapping that `owner` (Send + Sync) keeps alive; nothing in it is tied to a thread.
// The kernel does not write the buffer while it is dequeued, and only `LentBuffer::queue`, which consumes the lease, gives it
// back (the index is private, so no other lease can queue this buffer), so reads through it from any thread see a stable buffer.
unsafe impl Send for LentBuffer {}
// SAFETY: as for `Send`; shared access only reads.
unsafe impl Sync for LentBuffer {}

impl LentBuffer {
    /// Immutable dequeue metadata; the lease's return index cannot be changed.
    pub fn meta(&self) -> &DequeueMeta {
        &self.meta
    }

    /// One captured plane including stride padding; the lease keeps it immutable.
    /// # Arguments
    /// * `index` - zero-based plane index.
    pub fn plane(&self, index: usize) -> Option<&[u8]> {
        if index >= self.owner.format.planes.len() {
            return None;
        }
        // SAFETY: Device validated the mapping and used length for every configured plane.
        Some(unsafe { std::slice::from_raw_parts(self.planes[index], self.lengths[index]) })
    }

    /// The V4L2 buffer index.
    pub fn index(&self) -> u32 {
        self.meta.index
    }

    /// The luma plane at the configured size and stride.
    pub fn luma(&self) -> &[u8] {
        // SAFETY: the pointer is the camera's mapping of this buffer at its data offset, with validated luma length (of the configured first plane)
        // valid bytes, checked by Device. The mapping lives as long as `owner`, which this lease holds, and the kernel does not write
        // a dequeued buffer until `queue`, which consumes the lease, so it cannot run while this slice borrows it.
        unsafe { std::slice::from_raw_parts(self.planes[0], self.owner.format.luma_bytes) }
    }

    /// Give the buffer back to the driver.
    ///
    /// # Errors
    ///
    /// [`MplaneError::Io`] if `VIDIOC_QBUF` fails.
    pub fn queue(mut self) -> Result<(), MplaneError> {
        self.owner
            .device
            .queue(self.meta.index)
            .map_err(|source| MplaneError::Io {
                what: format!("queue buffer {} of {}", self.meta.index, self.owner.path),
                source,
            })?;
        self.queued = true;
        Ok(())
    }
}

impl Drop for LentBuffer {
    fn drop(&mut self) {
        if !self.queued {
            if let Err(error) = self.owner.device.queue(self.meta.index) {
                eprintln!("V4L2 requeue on drop failed: {error}");
            }
        }
    }
}

/// A dequeued frame's luma plane and V4L2 metadata.
pub struct LumaFrame {
    /// Raw dequeue diagnostics, before any pixel copy.
    pub meta: DequeueMeta,
    /// True if this frame owns a driver buffer instead of copied luma.
    pub leased: bool,
    /// Images still holding driver buffers, including this frame, at publication.
    pub held_buffers: u32,
    /// Leases not yet requeued, including returned leases waiting for the capture thread.
    pub unrequeued_buffers: u32,
    /// Packed visible luma: an owned copy, or ([`CaptureMode::ZeroCopy`]) a read-only image over the capture buffer.
    pub luma: Image<u8, 1>,
}

/// The camera owner (descriptor, buffers and their mappings), shared by its [`Camera`] and every [`LentBuffer`] the camera
/// lent out; released when the last of them drops.
#[derive(Debug)]
struct CameraOwner {
    device: device::Device,
    path: String,
    format: CaptureFormat,
    count: u32,
}

// SAFETY: the owner is plain state plus a file descriptor and mappings; it has no thread affinity.
unsafe impl Send for CameraOwner {}
// SAFETY: Device::open fills the owner completely, and no later call writes it: dequeue, queue and buffer count only read its fields and issue ioctls, which the kernel serialises per descriptor (two concurrent dequeues get two
// different buffers). Only `Drop` (the last `Arc`, no other reference) releases it.
unsafe impl Sync for CameraOwner {}

/// One capture device, streaming from `open` until it and every buffer it lent out have dropped.
pub struct Camera {
    owner: Arc<CameraOwner>,
    rejected: Option<Box<dyn Fn(DequeueMeta) + Send + Sync>>,
}

impl Camera {
    /// Observe rejected buffers before requeue; the callback must not block or retain images.
    /// # Arguments
    /// * `observer` - receives raw metadata even when ERROR or bytesused validation fails.
    pub fn observe_rejected(&mut self, observer: impl Fn(DequeueMeta) + Send + Sync + 'static) {
        self.rejected = Some(Box::new(observer));
    }
    /// The validated capture format.
    pub fn format(&self) -> &CaptureFormat {
        &self.owner.format
    }

    /// Open the device, set the requested format, map and queue 8 buffers, and start streaming (frames flow only once
    /// the frame trigger starts).
    ///
    /// # Errors
    ///
    /// [`MplaneError::Io`] with the OS error (`ENOTSUP` = the driver negotiated another format).
    /// # Arguments
    /// * `path` - V4L2 device node.
    /// * `format` - validated format and plane bounds.
    pub fn open(path: &str, format: CaptureFormat) -> Result<Self, MplaneError> {
        Self::open_with(path, 8, format)
    }

    /// [`Camera::open`] with `buffers` buffers (1..=32).
    /// # Arguments
    /// * `path` - device node.
    /// * `buffers` - requested buffer count.
    /// * `format` - validated plane geometry.
    ///
    /// # Errors
    ///
    /// [`MplaneError::Io`] with the OS error (`ENOTSUP` = another format; `EINVAL` = a buffer count the driver
    /// refuses); [`MplaneError::Device`] for a path with a NUL.
    pub fn open_with(path: &str, buffers: u32, format: CaptureFormat) -> Result<Self, MplaneError> {
        let path_c = CString::new(path)
            .map_err(|_| MplaneError::Device(format!("camera path {path:?} has a NUL")))?;
        let device = device::Device::open(&path_c, buffers, format.raw()).map_err(|error| {
            MplaneError::Io {
                what: format!("open camera {path} ({buffers} buffers) at {}", error.step),
                source: error.source,
            }
        })?;
        let count = device.count();
        Ok(Self {
            rejected: None,
            owner: Arc::new(CameraOwner {
                device,
                path: path.to_owned(),
                format,
                count,
            }),
        })
    }

    /// Number of buffers the driver actually allocated.
    pub fn buffer_count(&self) -> u32 {
        self.owner.count
    }

    /// Wait up to `timeout_ms` for the next frame and lend its buffer out (it stays out of the driver's queue until
    /// [`LentBuffer::queue`]).
    ///
    /// # Returns
    ///
    /// `Ok(None)` on a timeout.
    ///
    /// # Errors
    ///
    /// A dequeue fault or invalid buffer ([`MplaneError::Io`]), or a timestamp that is not CLOCK_MONOTONIC
    /// ([`MplaneError::Timing`]; the buffer is requeued first).
    /// # Arguments
    /// * `timeout_ms` - nonnegative wait budget in milliseconds.
    pub fn dequeue(&self, timeout_ms: u16) -> Result<Option<LentBuffer>, MplaneError> {
        let path = &self.owner.path;
        let Some(frame) = self
            .owner
            .device
            .dequeue_observed(
                timeout_ms,
                self.rejected
                    .as_ref()
                    .map(|f| f.as_ref() as &dyn Fn(DequeueMeta)),
            )
            .map_err(|source| MplaneError::Io {
                what: format!("dequeue {path}"),
                source,
            })?
        else {
            return Ok(None);
        };
        let flags = frame.meta.flags;
        let buffer = LentBuffer {
            meta: frame.meta,
            planes: frame.planes,
            lengths: frame.lengths,
            queued: false,
            owner: self.owner.clone(),
        };
        if flags & TIMESTAMP_MASK != TIMESTAMP_MONOTONIC {
            if let Some(observe) = &self.rejected {
                observe(buffer.meta);
            }
            buffer.queue()?;
            return Err(MplaneError::Timing(format!(
                "{path}: buffer flags {flags:#x}, not a monotonic timestamp"
            )));
        }
        Ok(Some(buffer))
    }

    /// Wait up to `timeout_ms` for the next frame and copy its luma out.
    ///
    /// # Returns
    ///
    /// `Ok(None)` on a timeout.
    ///
    /// # Errors
    ///
    /// A dequeue fault or invalid buffer ([`MplaneError::Io`]), or a timestamp that is not CLOCK_MONOTONIC
    /// ([`MplaneError::Timing`]).
    /// # Arguments
    /// * `timeout_ms` - nonnegative wait budget in milliseconds.
    pub fn next_frame(&self, timeout_ms: u16) -> Result<Option<LumaFrame>, MplaneError> {
        let Some(buffer) = self.dequeue(timeout_ms)? else {
            return Ok(None);
        };
        let luma = self.copy_luma(&buffer)?;
        let frame = LumaFrame {
            meta: buffer.meta,
            leased: false,
            held_buffers: 0,
            unrequeued_buffers: 0,
            luma,
        };
        buffer.queue()?;
        Ok(Some(frame))
    }

    fn copy_luma(&self, buffer: &LentBuffer) -> Result<Image<u8, 1>, MplaneError> {
        let format = &self.owner.format;
        let mut data = Vec::with_capacity(format.size.width * format.size.height);
        for row in buffer
            .luma()
            .chunks(format.planes[0].stride)
            .take(format.size.height)
        {
            data.extend_from_slice(&row[..format.size.width]);
        }
        Ok(Image::new(format.size, data)?)
    }
}

/// How live frames leave the capture buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureMode {
    /// Copy each frame's luma plane out and requeue the buffer at once (PR #270's path).
    Copy,
    /// Lend the buffer out as a read-only image; it goes back to the driver when the last image over it drops. A frame is
    /// copied instead when fewer than the configured minimum of buffers would stay with the driver.
    ZeroCopy,
}

/// The keepalive of a zero-copy image: owns the lent buffer (whose lease keeps the mapping alive), and sends it back to the
/// capture thread when the last image over it drops.
struct LentLuma<T> {
    buffer: Option<T>,
    returns: mpsc::Sender<T>,
    held: Arc<AtomicU32>,
}

impl<T> Drop for LentLuma<T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.held.fetch_sub(1, Ordering::Relaxed);
            // The capture thread has ended when the send fails: the lease requeues on drop, and the camera closes with the
            // last of its leases.
            let _ = self.returns.send(buffer);
        }
    }
}

/// One camera's frames in a [`CaptureMode`]; owned by its capture thread.
pub struct CaptureStream {
    held: Arc<AtomicU32>,
    camera: Camera,
    mode: CaptureMode,
    count: u32,
    lent: u32,
    returns_tx: mpsc::Sender<LentBuffer>,
    returns_rx: mpsc::Receiver<LentBuffer>,
    fallback_copies: u64,
    min_queued: u32,
}

impl CaptureStream {
    /// Observe rejected dequeues with the current count of downstream driver-buffer owners.
    /// # Arguments
    /// * `observer` - nonblocking diagnostic sink, called before buffer requeue.
    pub fn observe_rejected(
        &mut self,
        observer: impl Fn(DequeueMeta, u32) + Send + Sync + 'static,
    ) {
        let held = self.held.clone();
        self.camera
            .observe_rejected(move |meta| observer(meta, held.load(Ordering::Relaxed)));
    }

    /// Stream `camera`'s frames in `mode`.
    /// # Arguments
    /// * `camera` - opened camera owner.
    /// * `mode` - copied or borrowed luma images.
    /// * `min_queued` - minimum buffers to retain in the driver.
    /// # Errors
    /// Rejects a minimum above the allocated count.
    pub fn new(camera: Camera, mode: CaptureMode, min_queued: u32) -> Result<Self, MplaneError> {
        let count = camera.buffer_count();
        if min_queued > count {
            return Err(MplaneError::Device("invalid queued-buffer minimum".into()));
        }
        let (returns_tx, returns_rx) = mpsc::channel();
        Ok(Self {
            held: Arc::new(AtomicU32::new(0)),
            camera,
            mode,
            count,
            lent: 0,
            returns_tx,
            returns_rx,
            fallback_copies: 0,
            min_queued,
        })
    }

    /// Frames copied in zero-copy mode due to queue pressure or padded row stride.
    pub fn fallback_copies(&self) -> u64 {
        self.fallback_copies
    }

    /// Wait up to `timeout_ms` for the next frame. In zero-copy mode, first requeue the buffers whose images have all dropped.
    ///
    /// # Returns
    ///
    /// `Ok(None)` on a timeout.
    ///
    /// # Errors
    ///
    /// As [`Camera::next_frame`] and [`Camera::dequeue`], and [`MplaneError::Io`] if a returned buffer cannot be requeued.
    /// # Arguments
    /// * `timeout_ms` - nonnegative wait budget in milliseconds.
    pub fn next_frame(&mut self, timeout_ms: u16) -> Result<Option<LumaFrame>, MplaneError> {
        if self.mode == CaptureMode::Copy {
            return self.camera.next_frame(timeout_ms);
        }
        while let Ok(buffer) = self.returns_rx.try_recv() {
            self.lent -= 1;
            buffer.queue()?;
        }
        let Some(buffer) = self.camera.dequeue(timeout_ms)? else {
            return Ok(None);
        };
        let meta = buffer.meta;
        if self.count.saturating_sub(self.lent + 1) < self.min_queued
            || self.camera.owner.format.planes[0].stride != self.camera.owner.format.size.width
        {
            let luma = self.camera.copy_luma(&buffer)?;
            buffer.queue()?;
            self.fallback_copies += 1;
            return Ok(Some(LumaFrame {
                meta,
                leased: false,
                held_buffers: self.held.load(Ordering::Relaxed),
                unrequeued_buffers: self.lent,
                luma,
            }));
        }
        let data = buffer.planes[0];
        // Counted before the image exists: if wrapping fails, the dropped keepalive returns the buffer, and the next drain
        // uncounts it.
        self.lent += 1;
        self.held.fetch_add(1, Ordering::Relaxed);
        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(LentLuma {
            held: self.held.clone(),
            buffer: Some(buffer),
            returns: self.returns_tx.clone(),
        });
        // SAFETY: `data` points at the buffer's mapped luma plane, validated luma length = the image extent bytes (checked
        // by Device at dequeue). The keepalive owns the `LentBuffer`: the kernel does not write the buffer until it is queued, which
        // happens only after the keepalive drops, and the lease holds its camera's owner, which unmaps the buffer only when it
        // drops. The image is read-only.
        let luma = unsafe {
            Image::from_borrowed_host_readonly(self.camera.owner.format.size, data, keepalive)?
        };
        Ok(Some(LumaFrame {
            meta,
            leased: true,
            held_buffers: self.held.load(Ordering::Relaxed),
            unrequeued_buffers: self.lent,
            luma,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn arbitrary_resolutions_and_padded_luma_have_checked_bounds() -> Result<(), MplaneError> {
        let size = ImageSize {
            width: 7,
            height: 3,
        };
        let format = CaptureFormat::new(
            size,
            u32::from_le_bytes(*b"NM12"),
            vec![
                PlaneLayout {
                    stride: 16,
                    bytes: 48,
                },
                PlaneLayout {
                    stride: 16,
                    bytes: 32,
                },
            ],
        )?;
        assert_eq!(format.luma_bytes, 39);
        assert_eq!(format.raw().planes, 2);
        assert!(CaptureFormat::new(size, 0, vec![]).is_err());
        assert!(CaptureFormat::new(
            size,
            0,
            vec![PlaneLayout {
                stride: 6,
                bytes: 18
            }]
        )
        .is_err());
        assert!(CaptureFormat::new(
            size,
            0,
            vec![PlaneLayout {
                stride: 16,
                bytes: 38
            }]
        )
        .is_err());
        assert!(CaptureFormat::new(
            ImageSize {
                width: 0,
                height: 3
            },
            0,
            vec![PlaneLayout {
                stride: 16,
                bytes: 48
            }]
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn borrowed_images_return_the_lease_only_after_the_last_reader() -> Result<(), MplaneError> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Token {
            pixels: Arc<[u8; 4]>,
            drops: Arc<AtomicUsize>,
        }
        impl Drop for Token {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }
        for closed in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let token = Token {
                pixels: Arc::new([1, 2, 3, 4]),
                drops: drops.clone(),
            };
            let data = token.pixels.as_ptr();
            let (tx, rx) = mpsc::channel();
            let mut receiver = Some(rx);
            let held = Arc::new(AtomicU32::new(1));
            let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(LentLuma {
                held: held.clone(),
                buffer: Some(token),
                returns: tx,
            });
            // SAFETY: the guard's token owns these four bytes until the last image drops.
            let image = Arc::new(unsafe {
                Image::<u8, 1>::from_borrowed_host_readonly(
                    ImageSize {
                        width: 2,
                        height: 2,
                    },
                    data,
                    keepalive,
                )?
            });
            let clone = image.clone();
            drop(image);
            assert_eq!(held.load(Ordering::Relaxed), 1);
            assert_eq!(clone.as_slice(), &[1, 2, 3, 4]);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert!(receiver.as_ref().unwrap().try_recv().is_err());
            if closed {
                drop(receiver.take());
            }
            drop(clone);
            assert_eq!(
                held.load(Ordering::Relaxed),
                0,
                "returned leases no longer count as downstream-held"
            );
            if !closed {
                assert_eq!(
                    drops.load(Ordering::SeqCst),
                    0,
                    "returned token waits in the channel"
                );
                drop(receiver.take());
            }
            assert_eq!(
                drops.load(Ordering::SeqCst),
                1,
                "closed or discarded return channels release the lease exactly once"
            );
        }
        Ok(())
    }

    #[test]
    fn invalid_paths_fail_without_opening_a_device() -> Result<(), MplaneError> {
        let format = CaptureFormat::new(
            ImageSize {
                width: 8,
                height: 4,
            },
            u32::from_le_bytes(*b"GREY"),
            vec![PlaneLayout {
                stride: 8,
                bytes: 32,
            }],
        )?;
        assert!(matches!(
            Camera::open("\0", format.clone()),
            Err(MplaneError::Device(_))
        ));
        assert!(matches!(
            Camera::open_with("/missing-camera", 0, format),
            Err(MplaneError::Io { .. })
        ));
        Ok(())
    }
}
