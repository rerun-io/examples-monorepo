use anyhow::{Context, Result, ensure};
use std::{
    ffi::{CString, c_char, c_int, c_void},
    io,
    ptr::NonNull,
};

unsafe extern "C" {
    fn cap_camera_open(path: *const c_char) -> *mut c_void;
    fn cap_camera_close(camera: *mut c_void);
    fn cap_camera_next(
        camera: *mut c_void,
        destination: *mut u8,
        capacity: usize,
        timestamp_ns: *mut i64,
        sequence: *mut u32,
        flags: *mut u32,
        timeout_ms: c_int,
    ) -> c_int;
}

/// Exclusive V4L2 capture owner. The native shim uses the installed kernel ABI.
pub struct Camera(NonNull<c_void>);

pub struct CameraFrame {
    pub nv12: Vec<u8>,
    pub timestamp_ns: i64,
    pub sequence: u32,
    pub flags: u32,
}

impl Camera {
    pub fn open(path: &str) -> Result<Self> {
        let path_c = CString::new(path)?;
        // SAFETY: the C string remains valid for the call; C returns a unique
        // owner or null and never stores the path pointer.
        let pointer = unsafe { cap_camera_open(path_c.as_ptr()) };
        Ok(Self(
            NonNull::new(pointer)
                .ok_or_else(io::Error::last_os_error)
                .with_context(|| format!("open camera {path}"))?,
        ))
    }

    pub fn read_frame(&mut self) -> Result<Option<CameraFrame>> {
        let mut frame = CameraFrame {
            nv12: vec![0; crate::FRAME_WIDTH * crate::FRAME_HEIGHT * 3 / 2],
            timestamp_ns: 0,
            sequence: 0,
            flags: 0,
        };
        // SAFETY: this owner is live and exclusively borrowed; destination and
        // metadata pointers refer to writable allocations of the supplied size.
        let result = unsafe {
            cap_camera_next(
                self.0.as_ptr(),
                frame.nv12.as_mut_ptr(),
                frame.nv12.len(),
                &mut frame.timestamp_ns,
                &mut frame.sequence,
                &mut frame.flags,
                500,
            )
        };
        if result < 0 {
            return Err(io::Error::last_os_error().into());
        }
        if result == 0 {
            return Ok(None);
        }
        ensure!(
            frame.flags & 0xe000 == 0x2000,
            "camera did not supply a monotonic V4L2 timestamp"
        );
        Ok(Some(frame))
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        // SAFETY: the unique owner is consumed once. C stops streaming, releases
        // mapped buffers, restores the original format and closes its descriptor.
        unsafe { cap_camera_close(self.0.as_ptr()) };
    }
}
