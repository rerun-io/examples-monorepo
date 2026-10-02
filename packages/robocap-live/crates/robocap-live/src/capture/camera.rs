//! One rkisp mainpath camera: V4L2 multi-planar NV12 1920x1080, 8 MMAP buffers, monotonic timestamps; luma copied out, or the
//! buffer lent out as a read-only kornia image until the last reader drops it ([`CaptureStream`], [`CaptureMode::ZeroCopy`]).
//!
//! Adapted from PR #270's `robocap-recorder/src/camera.rs` (271ce643). The kernel ABI lives in `v4l2_mplane.c`. The zero-copy
//! lending follows kornia-io 0.2's V4L2 `V4lResource` + `MmapStream`: the image's keepalive owns the buffer, and the buffer is
//! requeued (by the capture thread, before its next dequeue) only once no image references it.
//!
//! Measured on Cap A (2026-10-01, `examples/capture_probe.rs`): the MMAP mapping is cached memory (a hand-crop shaped sparse read
//! takes 2.51 ms on the mapped buffer and 2.50 ms on a memcpy'd copy on an A76), so borrowing costs readers nothing.

use std::any::Any;
use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::ptr::NonNull;
use std::sync::{Arc, mpsc};

use kornia_image::Image;

use super::CaptureError;
use crate::frame::FULL_SIZE;

unsafe extern "C" {
    fn rl_camera_open(path: *const c_char) -> *mut c_void;
    fn rl_camera_open_with(path: *const c_char, mode: c_int, heap: *const c_char, count: c_uint) -> *mut c_void;
    fn rl_camera_open_step() -> *const c_char;
    fn rl_camera_info(
        camera: *const c_void,
        mmap_capabilities: *mut u32,
        dmabuf_capabilities: *mut u32,
        memory_flags: *mut u32,
        count: *mut c_uint,
        buffer_bytes: *mut usize,
    );
    fn rl_camera_dequeue(
        camera: *mut c_void,
        index: *mut c_uint,
        luma: *mut *const u8,
        timestamp_ns: *mut i64,
        sequence: *mut u32,
        flags: *mut u32,
        timeout_ms: c_int,
    ) -> c_int;
    fn rl_camera_queue(camera: *mut c_void, index: c_uint) -> c_int;
    fn rl_camera_export(camera: *const c_void, index: c_uint) -> c_int;
    fn rl_camera_dmabuf(camera: *const c_void, index: c_uint) -> c_int;
    fn rl_dmabuf_sync_read(fd: c_int, start: c_int) -> c_int;
    fn rl_camera_close(camera: *mut c_void);
    fn rl_camera_next(
        camera: *mut c_void,
        luma: *mut u8,
        capacity: usize,
        timestamp_ns: *mut i64,
        sequence: *mut u32,
        flags: *mut u32,
        timeout_ms: c_int,
    ) -> c_int;
}

/// `V4L2_BUF_FLAG_TIMESTAMP_MASK`.
const TIMESTAMP_MASK: u32 = 0xe000;
/// `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC`.
const TIMESTAMP_MONOTONIC: u32 = 0x2000;
/// Luma bytes per frame.
pub const LUMA_BYTES: usize = FULL_SIZE.width * FULL_SIZE.height;

/// Where a camera's capture buffers live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureMemory {
    /// Driver buffers (`V4L2_MEMORY_MMAP`) in the driver's default cache mode: the production setup.
    Mmap,
    /// Driver buffers requested with `V4L2_MEMORY_FLAG_NON_COHERENT` (the driver may drop the hint: see
    /// [`BufferInfo::memory_flags`]).
    MmapNonCoherent,
    /// Buffers allocated from a dma-buf heap device (for example `/dev/dma_heap/system`) and imported (`V4L2_MEMORY_DMABUF`).
    /// CPU reads go between [`Camera::sync_read`] calls.
    DmaBufHeap(String),
}

/// What the driver reported when the buffers were requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferInfo {
    /// `VIDIOC_REQBUFS` capabilities for `V4L2_MEMORY_MMAP` (`V4L2_BUF_CAP_*`; 0 = refused).
    pub mmap_capabilities: u32,
    /// `VIDIOC_REQBUFS` capabilities for `V4L2_MEMORY_DMABUF` (0 = refused).
    pub dmabuf_capabilities: u32,
    /// The `v4l2_requestbuffers.flags` byte the driver kept (`V4L2_MEMORY_FLAG_NON_COHERENT` = 1).
    pub memory_flags: u32,
    /// Buffers the driver allocated.
    pub count: u32,
    /// Bytes of each buffer (the driver's plane length, or the heap allocation).
    pub buffer_bytes: usize,
}

/// A buffer lent out by [`Camera::dequeue`]: the kernel does not write it until [`LentBuffer::queue`] gives it back. The lease
/// holds its camera's descriptor and mappings, so its luma stays readable after the [`Camera`] has dropped:
///
/// ```no_run
/// # fn main() -> Result<(), robocap_live::capture::CaptureError> {
/// let camera = robocap_live::capture::camera::Camera::open("/dev/video75")?;
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
/// fn retarget(buffer: &mut robocap_live::capture::camera::LentBuffer) {
///     buffer.index = 0;
/// }
/// ```
#[derive(Debug)]
#[must_use = "a lent buffer stays out of the driver's queue until LentBuffer::queue gives it back"]
pub struct LentBuffer {
    index: u32,
    /// `buffer.timestamp`, CLOCK_MONOTONIC nanoseconds.
    pub timestamp_ns: i64,
    /// The driver's sequence number.
    pub sequence: u32,
    luma: NonNull<u8>,
    owner: Arc<CameraOwner>,
}

// SAFETY: a buffer index plus a pointer into a mapping that `owner` (Send + Sync) keeps alive; nothing in it is tied to a thread.
// The kernel does not write the buffer while it is dequeued, and only `LentBuffer::queue`, which consumes the lease, gives it
// back (the index is private, so no other lease can queue this buffer), so reads through it from any thread see a stable buffer.
unsafe impl Send for LentBuffer {}
// SAFETY: as for `Send`; shared access only reads.
unsafe impl Sync for LentBuffer {}

impl LentBuffer {
    /// The V4L2 buffer index.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The luma plane (1920x1080, stride 1920). In [`CaptureMemory::DmaBufHeap`] mode, read it between
    /// [`LentBuffer::sync_read`] calls.
    pub fn luma(&self) -> &[u8] {
        // SAFETY: the pointer is the camera's mapping of this buffer at its data offset, with LUMA_BYTES (of 3/2 x LUMA_BYTES)
        // valid bytes, checked by C. The mapping lives as long as `owner`, which this lease holds, and the kernel does not write
        // a dequeued buffer until `queue`, which consumes the lease, so it cannot run while this slice borrows it.
        unsafe { std::slice::from_raw_parts(self.luma.as_ptr(), LUMA_BYTES) }
    }

    /// `DMA_BUF_IOCTL_SYNC` read start (`start`) or end around CPU reads of the buffer; a no-op for driver (MMAP) buffers.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Io`] if the ioctl fails.
    pub fn sync_read(&self, start: bool) -> Result<(), CaptureError> {
        // SAFETY: the owner is live (this lease holds it); C checks the index and returns -1 for MMAP buffers.
        let fd = unsafe { rl_camera_dmabuf(self.owner.raw.as_ptr(), self.index) };
        // SAFETY: `fd` is the camera's own heap descriptor, alive while the owner is.
        if fd >= 0 && unsafe { rl_dmabuf_sync_read(fd, c_int::from(start)) } != 0 {
            return Err(CaptureError::last_os_error(format!("sync buffer {} of {}", self.index, self.owner.path)));
        }
        Ok(())
    }

    /// Give the buffer back to the driver.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Io`] if `VIDIOC_QBUF` fails.
    pub fn queue(self) -> Result<(), CaptureError> {
        // SAFETY: the owner is live (this lease holds it) and only read by C; the index is the one the driver lent.
        if unsafe { rl_camera_queue(self.owner.raw.as_ptr(), self.index) } != 0 {
            return Err(CaptureError::last_os_error(format!("queue buffer {} of {}", self.index, self.owner.path)));
        }
        Ok(())
    }
}

/// A dequeued frame's luma plane and V4L2 metadata.
pub struct LumaFrame {
    /// 1920x1080 luma, stride 1920: an owned copy, or ([`CaptureMode::ZeroCopy`]) a read-only image over the capture buffer.
    pub luma: Image<u8, 1>,
    /// `buffer.timestamp`, CLOCK_MONOTONIC nanoseconds.
    pub timestamp_ns: i64,
    /// The driver's sequence number.
    pub sequence: u32,
}

/// The C camera owner (descriptor, buffers and their mappings), shared by its [`Camera`] and every [`LentBuffer`] the camera
/// lent out; released when the last of them drops.
#[derive(Debug)]
struct CameraOwner {
    raw: NonNull<c_void>,
    path: String,
}

// SAFETY: the C owner is plain heap state plus a file descriptor and mappings; it has no thread affinity.
unsafe impl Send for CameraOwner {}
// SAFETY: `rl_camera_open_with` fills the C owner completely, and no later call writes it: dequeue, queue, next, export, info
// and sync only read its fields and issue ioctls, which the kernel serialises per descriptor (two concurrent dequeues get two
// different buffers). Only `Drop` (the last `Arc`, no other reference) releases it.
unsafe impl Sync for CameraOwner {}

impl Drop for CameraOwner {
    fn drop(&mut self) {
        // SAFETY: the unique owner is released once, when no camera or lease refers to it; C stops streaming, unmaps and frees
        // the buffers, restores the original format and closes the descriptor.
        unsafe { rl_camera_close(self.raw.as_ptr()) };
    }
}

/// One capture device, streaming from `open` until it and every buffer it lent out have dropped.
pub struct Camera {
    owner: Arc<CameraOwner>,
}

impl Camera {
    /// Open the device, set NV12 1920x1080 single-plane, map and queue 8 buffers, and start streaming (frames flow only once
    /// the frame trigger starts).
    ///
    /// # Errors
    ///
    /// [`CaptureError::Io`] with the OS error (`ENOTSUP` = the driver negotiated another format).
    pub fn open(path: &str) -> Result<Self, CaptureError> {
        let path_c = CString::new(path).map_err(|_| CaptureError::Device(format!("camera path {path:?} has a NUL")))?;
        // SAFETY: the C string outlives the call; C returns a unique owner or NULL and does not keep the pointer.
        let raw = unsafe { rl_camera_open(path_c.as_ptr()) };
        let raw = NonNull::new(raw).ok_or_else(|| CaptureError::last_os_error(format!("open camera {path}")))?;
        Ok(Self { owner: Arc::new(CameraOwner { raw, path: path.to_owned() }) })
    }

    /// [`Camera::open`] with `buffers` buffers (1..=8) in `memory`.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Io`] with the OS error (`ENOTSUP` = another format; `EINVAL` = a buffer count or memory type the driver
    /// refuses); [`CaptureError::Device`] for a path with a NUL.
    pub fn open_with(path: &str, memory: &CaptureMemory, buffers: u32) -> Result<Self, CaptureError> {
        let path_c = CString::new(path).map_err(|_| CaptureError::Device(format!("camera path {path:?} has a NUL")))?;
        let (mode, heap) = match memory {
            CaptureMemory::Mmap => (0, None),
            CaptureMemory::MmapNonCoherent => (1, None),
            CaptureMemory::DmaBufHeap(heap) => {
                (2, Some(CString::new(heap.as_str()).map_err(|_| CaptureError::Device(format!("heap path {heap:?} has a NUL")))?))
            }
        };
        let heap_ptr = heap.as_ref().map_or(std::ptr::null(), |heap| heap.as_ptr());
        // SAFETY: both C strings outlive the call (the heap pointer is NULL when unused); C returns a unique owner or NULL and keeps
        // neither pointer.
        let raw = unsafe { rl_camera_open_with(path_c.as_ptr(), mode, heap_ptr, buffers) };
        let raw = NonNull::new(raw).ok_or_else(|| {
            let error = CaptureError::last_os_error(String::new());
            // SAFETY: C returns a pointer to a static NUL-terminated string.
            let step = unsafe { std::ffi::CStr::from_ptr(rl_camera_open_step()) }.to_string_lossy();
            match error {
                CaptureError::Io { source, .. } => {
                    CaptureError::Io { what: format!("open camera {path} ({memory:?}, {buffers} buffers) at {step}"), source }
                }
                other => other,
            }
        })?;
        Ok(Self { owner: Arc::new(CameraOwner { raw, path: path.to_owned() }) })
    }

    /// The driver's answer to the buffer request.
    pub fn buffer_info(&self) -> BufferInfo {
        let (mut mmap_capabilities, mut dmabuf_capabilities, mut memory_flags, mut count, mut buffer_bytes) = (0u32, 0u32, 0u32, 0 as c_uint, 0usize);
        let raw = self.owner.raw.as_ptr();
        // SAFETY: the owner is live; the pointers refer to live locals.
        unsafe { rl_camera_info(raw, &mut mmap_capabilities, &mut dmabuf_capabilities, &mut memory_flags, &mut count, &mut buffer_bytes) };
        BufferInfo { mmap_capabilities, dmabuf_capabilities, memory_flags, count, buffer_bytes }
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
    /// A dequeue fault or invalid buffer ([`CaptureError::Io`]), or a timestamp that is not CLOCK_MONOTONIC
    /// ([`CaptureError::Timing`]; the buffer is requeued first).
    pub fn dequeue(&self, timeout_ms: i32) -> Result<Option<LentBuffer>, CaptureError> {
        let (mut index, mut luma, mut timestamp_ns, mut sequence, mut flags) = (0 as c_uint, std::ptr::null(), 0i64, 0u32, 0u32);
        let raw = self.owner.raw.as_ptr();
        // SAFETY: the owner is live (and only read by C, see `Sync`); every out pointer refers to a live local.
        let result = unsafe { rl_camera_dequeue(raw, &mut index, &mut luma, &mut timestamp_ns, &mut sequence, &mut flags, timeout_ms) };
        let path = &self.owner.path;
        if result < 0 {
            return Err(CaptureError::last_os_error(format!("dequeue {path}")));
        }
        if result == 0 {
            return Ok(None);
        }
        let luma = NonNull::new(luma.cast_mut()).ok_or_else(|| CaptureError::Device(format!("{path}: dequeue returned no mapping")))?;
        let buffer = LentBuffer { index, timestamp_ns, sequence, luma, owner: self.owner.clone() };
        if flags & TIMESTAMP_MASK != TIMESTAMP_MONOTONIC {
            buffer.queue()?;
            return Err(CaptureError::Timing(format!("{path}: buffer flags {flags:#x}, not a monotonic timestamp")));
        }
        Ok(Some(buffer))
    }

    /// `VIDIOC_EXPBUF`: a dma-buf descriptor for buffer `index`.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Io`] if the driver refuses.
    pub fn export(&self, index: u32) -> Result<OwnedFd, CaptureError> {
        // SAFETY: the owner is live; C returns a new descriptor or -1.
        let fd: RawFd = unsafe { rl_camera_export(self.owner.raw.as_ptr(), index) };
        if fd < 0 {
            return Err(CaptureError::last_os_error(format!("export buffer {index} of {}", self.owner.path)));
        }
        // SAFETY: a new descriptor that nothing else owns.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Wait up to `timeout_ms` for the next frame and copy its luma out.
    ///
    /// # Returns
    ///
    /// `Ok(None)` on a timeout.
    ///
    /// # Errors
    ///
    /// A dequeue fault or invalid buffer ([`CaptureError::Io`]), or a timestamp that is not CLOCK_MONOTONIC
    /// ([`CaptureError::Timing`]).
    pub fn next_frame(&self, timeout_ms: i32) -> Result<Option<LumaFrame>, CaptureError> {
        let mut luma: Vec<u8> = Vec::with_capacity(LUMA_BYTES);
        let (mut timestamp_ns, mut sequence, mut flags) = (0i64, 0u32, 0u32);
        // SAFETY: the owner is live (and only read by C); `luma` has LUMA_BYTES of writable spare capacity, which C fills
        // completely before returning 1; the metadata pointers refer to live locals.
        let result = unsafe {
            rl_camera_next(
                self.owner.raw.as_ptr(),
                luma.as_mut_ptr(),
                luma.capacity(),
                &mut timestamp_ns,
                &mut sequence,
                &mut flags,
                timeout_ms,
            )
        };
        if result < 0 {
            return Err(CaptureError::last_os_error(format!("dequeue {}", self.owner.path)));
        }
        if result == 0 {
            return Ok(None);
        }
        // SAFETY: C returned 1, so it copied exactly LUMA_BYTES bytes into the spare capacity.
        unsafe { luma.set_len(LUMA_BYTES) };
        if flags & TIMESTAMP_MASK != TIMESTAMP_MONOTONIC {
            return Err(CaptureError::Timing(format!("{}: buffer flags {flags:#x}, not a monotonic timestamp", self.owner.path)));
        }
        Ok(Some(LumaFrame { luma: Image::new(FULL_SIZE, luma)?, timestamp_ns, sequence }))
    }
}

/// How live frames leave the capture buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureMode {
    /// Copy each frame's luma plane out and requeue the buffer at once (PR #270's path).
    Copy,
    /// Lend the buffer out as a read-only image; it goes back to the driver when the last image over it drops. A frame is
    /// copied instead when fewer than [`MIN_QUEUED`] buffers would stay with the driver.
    ZeroCopy,
}

impl std::str::FromStr for CaptureMode {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "copy" => Ok(Self::Copy),
            "zero-copy" => Ok(Self::ZeroCopy),
            other => Err(format!("capture mode {other:?}: expected copy or zero-copy")),
        }
    }
}

/// In [`CaptureMode::ZeroCopy`], the buffers that always stay queued with the driver: a frame that would leave fewer is copied
/// out instead of lent. (Cap A held 30 fps with no sequence gaps with as few as 0-1 queued, so 2 is a margin, not a limit.)
pub const MIN_QUEUED: u32 = 2;

/// The keepalive of a zero-copy image: owns the lent buffer (whose lease keeps the mapping alive), and sends it back to the
/// capture thread when the last image over it drops.
struct LentLuma {
    buffer: Option<LentBuffer>,
    returns: mpsc::Sender<LentBuffer>,
}

impl Drop for LentLuma {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // The capture thread has ended when the send fails: the lease then drops unqueued, and the camera closes with the
            // last of its leases.
            let _ = self.returns.send(buffer);
        }
    }
}

/// One camera's frames in a [`CaptureMode`]; owned by its capture thread.
pub struct CaptureStream {
    camera: Camera,
    mode: CaptureMode,
    count: u32,
    lent: u32,
    returns_tx: mpsc::Sender<LentBuffer>,
    returns_rx: mpsc::Receiver<LentBuffer>,
    fallback_copies: u64,
}

impl CaptureStream {
    /// Stream `camera`'s frames in `mode`.
    pub fn new(camera: Camera, mode: CaptureMode) -> Self {
        let count = camera.buffer_info().count;
        let (returns_tx, returns_rx) = mpsc::channel();
        Self { camera, mode, count, lent: 0, returns_tx, returns_rx, fallback_copies: 0 }
    }

    /// Frames copied out in [`CaptureMode::ZeroCopy`] because too few buffers were queued.
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
    /// As [`Camera::next_frame`] and [`Camera::dequeue`], and [`CaptureError::Io`] if a returned buffer cannot be requeued.
    pub fn next_frame(&mut self, timeout_ms: i32) -> Result<Option<LumaFrame>, CaptureError> {
        if self.mode == CaptureMode::Copy {
            return self.camera.next_frame(timeout_ms);
        }
        while let Ok(buffer) = self.returns_rx.try_recv() {
            self.lent -= 1;
            buffer.queue()?;
        }
        let Some(buffer) = self.camera.dequeue(timeout_ms)? else { return Ok(None) };
        let (timestamp_ns, sequence) = (buffer.timestamp_ns, buffer.sequence);
        if self.count.saturating_sub(self.lent + 1) < MIN_QUEUED {
            let luma = buffer.luma().to_vec();
            buffer.queue()?;
            self.fallback_copies += 1;
            return Ok(Some(LumaFrame { luma: Image::new(FULL_SIZE, luma)?, timestamp_ns, sequence }));
        }
        let data = buffer.luma.as_ptr().cast_const();
        // Counted before the image exists: if wrapping fails, the dropped keepalive returns the buffer, and the next drain
        // uncounts it.
        self.lent += 1;
        let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(LentLuma { buffer: Some(buffer), returns: self.returns_tx.clone() });
        // SAFETY: `data` points at the buffer's mapped luma plane, LUMA_BYTES = FULL_SIZE.width * FULL_SIZE.height bytes (checked
        // by C at dequeue). The keepalive owns the `LentBuffer`: the kernel does not write the buffer until it is queued, which
        // happens only after the keepalive drops, and the lease holds its camera's owner, which unmaps the buffer only when it
        // drops. The image is read-only.
        let luma = unsafe { Image::from_borrowed_host_readonly(FULL_SIZE, data, keepalive)? };
        Ok(Some(LumaFrame { luma, timestamp_ns, sequence }))
    }
}
