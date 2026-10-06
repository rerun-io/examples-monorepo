//! Private Linux V4L2 UAPI subset. No exported buffers or single-plane capture.

use std::ffi::{c_int, c_ulong, c_void};

pub const CAPTURE_MPLANE: u32 = 9;
pub const MEMORY_MMAP: u32 = 1;
pub const FIELD_NONE: u32 = 1;
pub const BUFFER_ERROR: u32 = 0x40;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct PlaneFormat {
    pub sizeimage: u32,
    pub bytesperline: u32,
    pub reserved: [u16; 6],
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct PixFormat {
    pub width: u32,
    pub height: u32,
    pub pixelformat: u32,
    pub field: u32,
    pub colorspace: u32,
    pub plane_fmt: [PlaneFormat; 8],
    pub num_planes: u8,
    pub flags: u8,
    pub ycbcr_enc: u8,
    pub quantization: u8,
    pub xfer_func: u8,
    pub reserved: [u8; 7],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union FormatData {
    pub pix_mp: PixFormat,
    pub raw: [u8; 200],
    // v4l2_window contains pointers and sets the union's native pointer alignment.
    pub alignment: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Format {
    pub kind: u32,
    pub fmt: FormatData,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RequestBuffers {
    pub count: u32,
    pub kind: u32,
    pub memory: u32,
    pub capabilities: u32,
    // Linux 5.14 reserved[1]; Linux 6.8 flags + reserved[3]. Always zero.
    pub reserved: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union PlaneMemory {
    pub mem_offset: u32,
    pub userptr: c_ulong,
    pub fd: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Plane {
    pub bytesused: u32,
    pub length: u32,
    pub m: PlaneMemory,
    pub data_offset: u32,
    pub reserved: [u32; 11],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timecode {
    pub kind: u32,
    pub flags: u32,
    pub frames: u8,
    pub seconds: u8,
    pub minutes: u8,
    pub hours: u8,
    pub userbits: [u8; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union BufferMemory {
    pub offset: u32,
    pub userptr: c_ulong,
    pub planes: *mut Plane,
    pub fd: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Buffer {
    pub index: u32,
    pub kind: u32,
    pub bytesused: u32,
    pub flags: u32,
    pub field: u32,
    pub timestamp: libc::timeval,
    pub timecode: Timecode,
    pub sequence: u32,
    pub memory: u32,
    pub m: BufferMemory,
    pub length: u32,
    pub reserved2: u32,
    // request_fd/reserved union: both 32-bit integers, initialized to zero.
    pub request_fd: i32,
}

macro_rules! zero_default {
    ($($ty:ty),+ $(,)?) => {$(
        impl Default for $ty {
            fn default() -> Self {
                // SAFETY: UAPI records contain only integers, raw pointers and unions of them;
                // all-zero is valid, and reserved fields must be zero for the kernel.
                unsafe { std::mem::zeroed() }
            }
        }
    )+};
}
zero_default!(Format, RequestBuffers, Plane, Buffer);

// asm-generic ioctl encoding used by Linux x86_64 and aarch64.
const fn ioc<T>(direction: u32, number: u32) -> c_ulong {
    ((direction << 30) | ((std::mem::size_of::<T>() as u32) << 16) | ((b'V' as u32) << 8) | number)
        as c_ulong
}
pub const G_FMT: c_ulong = ioc::<Format>(3, 4);
pub const S_FMT: c_ulong = ioc::<Format>(3, 5);
pub const REQBUFS: c_ulong = ioc::<RequestBuffers>(3, 8);
pub const QUERYBUF: c_ulong = ioc::<Buffer>(3, 9);
pub const QBUF: c_ulong = ioc::<Buffer>(3, 15);
pub const DQBUF: c_ulong = ioc::<Buffer>(3, 17);
pub const STREAMON: c_ulong = ioc::<c_int>(1, 18);
pub const STREAMOFF: c_ulong = ioc::<c_int>(1, 19);

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    // Independent C sizeof/_Alignof/offsetof and VIDIOC_* probes of linux/videodev2.h:
    // x86_64 Linux headers 6.8.12; aarch64 conda sysroot headers 5.14.0.
    // Both yield the same layout/numbers. Const assertions also run when cross-checking tests.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const _: () = {
        assert!(size_of::<Format>() == 208 && align_of::<Format>() == 8);
        assert!(offset_of!(Format, fmt) == 8);
        assert!(size_of::<PixFormat>() == 192 && align_of::<PixFormat>() == 1);
        assert!(offset_of!(PixFormat, plane_fmt) == 20);
        assert!(offset_of!(PixFormat, num_planes) == 180);
        assert!(size_of::<PlaneFormat>() == 20 && align_of::<PlaneFormat>() == 1);
        assert!(offset_of!(PlaneFormat, bytesperline) == 4);
        assert!(size_of::<RequestBuffers>() == 20 && align_of::<RequestBuffers>() == 4);
        assert!(offset_of!(RequestBuffers, capabilities) == 12);
        assert!(size_of::<Plane>() == 64 && align_of::<Plane>() == 8);
        assert!(offset_of!(Plane, m) == 8);
        assert!(offset_of!(Plane, data_offset) == 16);
        assert!(offset_of!(Plane, reserved) == 20);
        assert!(size_of::<Timecode>() == 16 && align_of::<Timecode>() == 4);
        assert!(size_of::<Buffer>() == 88 && align_of::<Buffer>() == 8);
        assert!(offset_of!(Buffer, timestamp) == 24);
        assert!(offset_of!(Buffer, timecode) == 40);
        assert!(offset_of!(Buffer, sequence) == 56);
        assert!(offset_of!(Buffer, memory) == 60);
        assert!(offset_of!(Buffer, m) == 64);
        assert!(offset_of!(Buffer, length) == 72);
        assert!(offset_of!(Buffer, request_fd) == 80);
        assert!(G_FMT == 0xc0d05604 && S_FMT == 0xc0d05605);
        assert!(REQBUFS == 0xc0145608 && QUERYBUF == 0xc0585609);
        assert!(QBUF == 0xc058560f && DQBUF == 0xc0585611);
        assert!(STREAMON == 0x40045612 && STREAMOFF == 0x40045613);
    };

    #[test]
    fn linux_headers_match_the_mplane_abi() {
        assert_eq!(size_of::<Format>(), 208);
        assert_eq!(size_of::<Buffer>(), 88);
        assert_eq!(size_of::<Plane>(), 64);
    }
}
