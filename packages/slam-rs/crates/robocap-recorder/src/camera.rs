use anyhow::{Context, Result};
use kornia_staging_io::v4l::mplane::{Camera as MplaneCamera, CaptureFormat, PlaneLayout};

/// Recorder policy over shared multi-planar capture.
pub struct Camera(MplaneCamera);

pub struct CameraFrame {
    pub nv12: Vec<u8>,
    pub timestamp_ns: i64,
    pub sequence: u32,
    pub flags: u32,
}

impl Camera {
    pub fn open(path: &str) -> Result<Self> {
        let format = CaptureFormat::new(
            kornia_image::ImageSize {
                width: crate::FRAME_WIDTH,
                height: crate::FRAME_HEIGHT,
            },
            u32::from_le_bytes(*b"NV12"),
            vec![PlaneLayout {
                stride: crate::FRAME_WIDTH,
                bytes: crate::FRAME_WIDTH * crate::FRAME_HEIGHT * 3 / 2,
            }],
        )?;
        Ok(Self(MplaneCamera::open(path, format)?))
    }
    pub fn read_frame(&mut self) -> Result<Option<CameraFrame>> {
        let Some(buffer) = self.0.dequeue(500)? else {
            return Ok(None);
        };
        let bytes = crate::FRAME_WIDTH * crate::FRAME_HEIGHT * 3 / 2;
        let frame = CameraFrame {
            nv12: buffer.plane(0).context("NV12 plane missing")?[..bytes].to_vec(),
            timestamp_ns: buffer.meta().timestamp_ns,
            sequence: buffer.meta().sequence,
            flags: buffer.meta().flags,
        };
        buffer.queue()?;
        Ok(Some(frame))
    }
}
