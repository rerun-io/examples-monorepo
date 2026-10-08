//! MMAP device ownership and the kernel syscall boundary.

use super::abi::*;
use super::{DequeueMeta, RawFormat, MAX_PLANES};
use std::ffi::{c_int, c_ulong, c_void, CStr};
use std::{fmt, io};

/// The private syscall seam permits hardware-free lifecycle/error tests.
/// # Safety
/// Successful mappings must remain valid until unmap. ioctls must obey the Linux UAPI,
/// including exclusive driver ownership of queued buffers and bounded plane writes.
pub(super) unsafe trait Syscalls {
    fn open(&self, path: &CStr) -> io::Result<c_int>;
    /// # Safety
    /// `arg` and any nested pointers must match `request` and remain live for the call.
    unsafe fn ioctl<T>(&self, fd: c_int, request: c_ulong, arg: &mut T) -> io::Result<()>;
    fn map(&self, fd: c_int, length: usize, offset: u32) -> io::Result<*mut c_void>;
    /// # Safety
    /// `address`/`length` must identify a live mapping without remaining readers.
    unsafe fn unmap(&self, address: *mut c_void, length: usize);
    fn poll(&self, fd: c_int, timeout: u16) -> io::Result<Option<i16>>;
    fn close(&self, fd: c_int);
}

pub(super) struct System;

// SAFETY: these are direct Linux syscalls; no retries or changes to their ownership rules.
unsafe impl Syscalls for System {
    fn open(&self, path: &CStr) -> io::Result<c_int> {
        // SAFETY: path is NUL-terminated and borrowed for the duration of open.
        let fd = unsafe {
            libc::open(
                path.as_ptr(),
                libc::O_RDWR | libc::O_NONBLOCK | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(fd)
        }
    }

    unsafe fn ioctl<T>(&self, fd: c_int, request: c_ulong, arg: &mut T) -> io::Result<()> {
        // SAFETY: the caller supplies the matching UAPI record and valid nested pointers.
        if unsafe { libc::ioctl(fd, request as _, std::ptr::from_mut(arg)) } < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn map(&self, fd: c_int, length: usize, offset: u32) -> io::Result<*mut c_void> {
        // SAFETY: the kernel supplied the size and offset from QUERYBUF; no fixed address.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                offset.into(),
            )
        };
        if address == libc::MAP_FAILED {
            Err(io::Error::last_os_error())
        } else {
            Ok(address)
        }
    }

    unsafe fn unmap(&self, address: *mut c_void, length: usize) {
        // SAFETY: the caller owns this mapping and has no outstanding leases/readers.
        unsafe { libc::munmap(address, length) };
    }

    fn poll(&self, fd: c_int, timeout: u16) -> io::Result<Option<i16>> {
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pfd is one live pollfd; the timeout is bounded and nonnegative.
        match unsafe { libc::poll(&mut pfd, 1, i32::from(timeout)) } {
            -1 => Err(io::Error::last_os_error()),
            0 => Ok(None),
            _ => Ok(Some(pfd.revents)),
        }
    }

    fn close(&self, fd: c_int) {
        // SAFETY: Device calls close once for the descriptor it owns.
        unsafe { libc::close(fd) };
    }
}

#[derive(Debug)]
pub(super) struct OpenError {
    pub step: &'static str,
    pub source: io::Error,
}

#[derive(Clone, Copy)]
struct Mapping {
    address: *mut c_void,
    length: usize,
}

pub(super) struct Frame {
    pub meta: DequeueMeta,
    pub planes: [*const u8; MAX_PLANES],
    pub lengths: [usize; MAX_PLANES],
}

pub(super) struct Device<S: Syscalls = System> {
    sys: S,
    fd: c_int,
    streaming: bool,
    count: u32,
    format: RawFormat,
    mappings: [[Mapping; MAX_PLANES]; 32],
    original: Format,
}

impl<S: Syscalls> fmt::Debug for Device<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("fd", &self.fd)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl Device {
    pub(super) fn open(path: &CStr, count: u32, format: RawFormat) -> Result<Self, OpenError> {
        Self::open_with_syscalls(path, count, format, System)
    }
}

impl<S: Syscalls> Device<S> {
    fn open_with_syscalls(
        path: &CStr,
        count: u32,
        format: RawFormat,
        sys: S,
    ) -> Result<Self, OpenError> {
        if format.width == 0
            || format.height == 0
            || format.planes == 0
            || format.planes > MAX_PLANES as u32
            || count == 0
            || count > 32
        {
            return Err(OpenError {
                step: "validate",
                source: io::Error::from_raw_os_error(libc::EINVAL),
            });
        }
        let fd = sys.open(path).map_err(|source| OpenError {
            step: "open",
            source,
        })?;
        let mut device = Self {
            sys,
            fd,
            streaming: false,
            count: 0,
            format,
            mappings: [[Mapping {
                address: std::ptr::null_mut(),
                length: 0,
            }; MAX_PLANES]; 32],
            original: Format {
                kind: CAPTURE_MPLANE,
                ..Format::default()
            },
        };
        let mut step = "G_FMT/S_FMT";
        // Store the error before Drop runs: cleanup syscalls must not overwrite its errno.
        let result = (|| -> io::Result<()> {
            // SAFETY: both requests take Format; its type selects the initialized MPLANE member.
            unsafe { device.sys.ioctl(fd, G_FMT, &mut device.original) }?;
            let mut negotiated = device.original;
            // SAFETY: G_FMT with CAPTURE_MPLANE initialized this member.
            let mut pix = unsafe { negotiated.fmt.pix_mp };
            pix.width = device.format.width;
            pix.height = device.format.height;
            pix.pixelformat = device.format.fourcc;
            pix.field = FIELD_NONE;
            pix.num_planes = device.format.planes as u8;
            for p in 0..device.format.planes as usize {
                pix.plane_fmt[p].bytesperline = device.format.stride[p];
                pix.plane_fmt[p].sizeimage = device.format.bytes[p];
            }
            negotiated.fmt.pix_mp = pix;
            // SAFETY: S_FMT takes a live Format with CAPTURE_MPLANE and its matching member.
            unsafe { device.sys.ioctl(fd, S_FMT, &mut negotiated) }?;
            // SAFETY: the request was for the MPLANE format member.
            let pix = unsafe { negotiated.fmt.pix_mp };
            if pix.width != device.format.width
                || pix.height != device.format.height
                || pix.pixelformat != device.format.fourcc
                || u32::from(pix.num_planes) != device.format.planes
            {
                return Err(io::Error::from_raw_os_error(libc::ENOTSUP));
            }
            for p in 0..device.format.planes as usize {
                if pix.plane_fmt[p].bytesperline != device.format.stride[p]
                    || pix.plane_fmt[p].sizeimage < device.format.bytes[p]
                {
                    return Err(io::Error::from_raw_os_error(libc::ENOTSUP));
                }
            }
            let mut req = RequestBuffers {
                count,
                kind: negotiated.kind,
                memory: MEMORY_MMAP,
                ..RequestBuffers::default()
            };
            step = "REQBUFS";
            // SAFETY: REQBUFS takes RequestBuffers without nested pointers.
            unsafe { device.sys.ioctl(fd, REQBUFS, &mut req) }?;
            if req.count == 0 || req.count > 32 {
                return Err(io::Error::from_raw_os_error(libc::EOVERFLOW));
            }
            device.count = req.count;
            for index in 0..device.count {
                let mut planes = [Plane::default(); MAX_PLANES];
                let mut buffer = device.buffer(index, &mut planes);
                step = "QUERYBUF";
                // SAFETY: buffer holds the live eight-plane array; length is <= eight.
                unsafe { device.sys.ioctl(fd, QUERYBUF, &mut buffer) }?;
                if buffer.length != device.format.planes {
                    return Err(io::Error::from_raw_os_error(libc::EIO));
                }
                for (p, plane) in planes
                    .iter()
                    .enumerate()
                    .take(device.format.planes as usize)
                {
                    let length = plane.length as usize;
                    if length < device.format.bytes[p] as usize {
                        return Err(io::Error::from_raw_os_error(libc::EIO));
                    }
                    step = "mmap";
                    // SAFETY: QUERYBUF for MEMORY_MMAP wrote the mem_offset member.
                    let address = device.sys.map(fd, length, unsafe { plane.m.mem_offset })?;
                    device.mappings[index as usize][p] = Mapping { address, length };
                }
                step = "QBUF";
                device.queue(index)?;
            }
            step = "STREAMON";
            // SAFETY: STREAMON takes the 32-bit buffer type.
            unsafe { device.sys.ioctl(fd, STREAMON, &mut negotiated.kind) }?;
            device.streaming = true;
            Ok(())
        })();
        result.map_err(|source| OpenError { step, source })?;
        Ok(device)
    }

    fn buffer(&self, index: u32, planes: &mut [Plane; MAX_PLANES]) -> Buffer {
        Buffer {
            index,
            kind: CAPTURE_MPLANE,
            memory: MEMORY_MMAP,
            length: self.format.planes,
            m: BufferMemory {
                planes: planes.as_mut_ptr(),
            },
            ..Buffer::default()
        }
    }

    pub(super) fn count(&self) -> u32 {
        self.count
    }

    pub(super) fn queue(&self, index: u32) -> io::Result<()> {
        if index >= self.count {
            return Err(io::Error::from_raw_os_error(libc::EINVAL));
        }
        let mut planes = [Plane::default(); MAX_PLANES];
        for (plane, mapping) in planes.iter_mut().zip(self.mappings[index as usize]) {
            plane.length = mapping.length as u32;
        }
        let mut buffer = self.buffer(index, &mut planes);
        // SAFETY: buffer holds a live plane array and the index is allocated to this device.
        unsafe { self.sys.ioctl(self.fd, QBUF, &mut buffer) }
    }

    #[cfg(test)]
    pub(super) fn dequeue(&self, timeout: u16) -> io::Result<Option<Frame>> {
        self.dequeue_observed(timeout, None)
    }

    pub(super) fn dequeue_observed(
        &self,
        timeout: u16,
        rejected: Option<&dyn Fn(DequeueMeta)>,
    ) -> io::Result<Option<Frame>> {
        let Some(events) = self.sys.poll(self.fd, timeout)? else {
            return Ok(None);
        };
        if events & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(io::Error::from_raw_os_error(libc::EIO));
        }
        let mut planes = [Plane::default(); MAX_PLANES];
        let mut buffer = self.buffer(0, &mut planes);
        // SAFETY: buffer holds a live plane array with the configured capacity.
        unsafe { self.sys.ioctl(self.fd, DQBUF, &mut buffer) }?;
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: Linux supports CLOCK_MONOTONIC and now is writable.
        if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) } != 0 {
            let error = io::Error::last_os_error();
            if buffer.index < self.count {
                self.queue(buffer.index)?;
            }
            return Err(error);
        }
        #[allow(clippy::unnecessary_cast)]
        let diagnostic = DequeueMeta {
            index: buffer.index,
            sequence: buffer.sequence,
            flags: buffer.flags,
            timestamp_ns: (buffer.timestamp.tv_sec as i64)
                .wrapping_mul(1_000_000_000)
                .wrapping_add((buffer.timestamp.tv_usec as i64).wrapping_mul(1000)),
            dequeue_ns: (now.tv_sec as i64) * 1_000_000_000 + now.tv_nsec as i64,
            planes: buffer.length.min(MAX_PLANES as u32),
            bytesused: std::array::from_fn(|p| planes[p].bytesused),
        };
        if buffer.index >= self.count {
            if let Some(observe) = rejected {
                observe(diagnostic);
            }
            return Err(io::Error::from_raw_os_error(libc::EIO));
        }
        let mappings = &self.mappings[buffer.index as usize];
        let mut bad = buffer.length != self.format.planes || buffer.flags & BUFFER_ERROR != 0;
        for (p, plane) in planes.iter().enumerate().take(self.format.planes as usize) {
            let length = mappings[p].length;
            if plane.data_offset > plane.bytesused
                || plane.bytesused as usize > length
                || plane.bytesused - plane.data_offset < self.format.bytes[p]
            {
                bad = true;
            }
        }
        if bad {
            if let Some(observe) = rejected {
                observe(diagnostic);
            }
            self.queue(buffer.index)?;
            return Err(io::Error::from_raw_os_error(libc::EIO));
        }
        let mut frame = Frame {
            meta: diagnostic,
            planes: [std::ptr::null(); MAX_PLANES],
            lengths: [0; MAX_PLANES],
        };
        for (p, plane) in planes.iter().enumerate().take(self.format.planes as usize) {
            let mapping = mappings[p];
            frame.planes[p] = mapping
                .address
                .cast::<u8>()
                .wrapping_add(plane.data_offset as usize);
            frame.lengths[p] = (plane.bytesused - plane.data_offset) as usize;
        }
        Ok(Some(frame))
    }
}

impl<S: Syscalls> Drop for Device<S> {
    fn drop(&mut self) {
        let mut kind = CAPTURE_MPLANE;
        if self.streaming {
            // SAFETY: the last owner has no outstanding leases; STREAMOFF takes the type.
            let _ = unsafe { self.sys.ioctl(self.fd, STREAMOFF, &mut kind) };
        }
        for planes in &self.mappings[..self.count as usize] {
            for mapping in planes[..self.format.planes as usize]
                .iter()
                .filter(|m| !m.address.is_null())
            {
                // SAFETY: these are exactly the successful mappings, with no remaining readers.
                unsafe { self.sys.unmap(mapping.address, mapping.length) };
            }
        }
        let mut req = RequestBuffers {
            kind,
            memory: MEMORY_MMAP,
            ..RequestBuffers::default()
        };
        // SAFETY: REQBUFS(0) releases this descriptor's buffers after unmapping them.
        let _ = unsafe { self.sys.ioctl(self.fd, REQBUFS, &mut req) };
        // SAFETY: original is zero-initialized before G_FMT, including on partial failures.
        if unsafe { self.original.fmt.pix_mp.width } != 0 {
            // SAFETY: restore the original full Format, after releasing the streaming buffers.
            let _ = unsafe { self.sys.ioctl(self.fd, S_FMT, &mut self.original) };
        }
        self.sys.close(self.fd);
    }
}

#[cfg(test)]
mod tests;
