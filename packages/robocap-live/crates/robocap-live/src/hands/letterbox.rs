//! robocap_track.py's `BarLetterbox`: a 16:9 camera into DetNet's 640x480 net frame (full-width resize by 1/3, 60-pixel black
//! bars above and below), and the pixel-centre maps between full-resolution camera pixels and the net frame. Written in
//! kornia-rs style on `spatial_padding`; its upstream home is kornia-imgproc's `preprocess`.
#![deny(missing_docs)]

use kornia_image::{Image, ImageError, ImageSize};
use nalgebra::Vector2;

use crate::frame::{FULL_SIZE, SMALL_SIZE};
use crate::nets::{DETNET_HEIGHT, DETNET_WIDTH, NetFrame};

/// The net frame's size.
pub const NET_SIZE: ImageSize = ImageSize { width: DETNET_WIDTH, height: DETNET_HEIGHT };

/// A source camera into the net frame: `net = (uv + 0.5) * scale - 0.5 + (pad_x, pad_y)` (pixel centres).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarLetterbox {
    /// Source image size (1920x1080 for RoboCap).
    pub source: ImageSize,
    /// Isotropic resize factor (1/3).
    pub scale: f64,
    /// Left padding in net pixels (0).
    pub pad_x: f64,
    /// Top padding in net pixels (60).
    pub pad_y: f64,
}

impl Default for BarLetterbox {
    fn default() -> Self {
        Self::robocap()
    }
}

impl BarLetterbox {
    /// RoboCap's map: 1920x1080, scale 1/3, pad (0, 60).
    pub fn robocap() -> Self {
        Self { source: FULL_SIZE, scale: 1.0 / 3.0, pad_x: 0.0, pad_y: 60.0 }
    }

    /// The map handtrack uses for a camera size (its `Letterbox` without the quarter turn): 1920x1080 RoboCap
    /// ([`Self::robocap`]), 636x480 and 640x480 UmeTrack; `None` for another size.
    pub fn for_size(width: u32, height: u32) -> Option<Self> {
        let source = ImageSize { width: width as usize, height: height as usize };
        match (width, height) {
            (1920, 1080) => Some(Self::robocap()),
            (636, 480) => Some(Self { source, scale: 1.0, pad_x: 2.0, pad_y: 0.0 }),
            (640, 480) => Some(Self { source, scale: 1.0, pad_x: 0.0, pad_y: 0.0 }),
            _ => None,
        }
    }

    /// Full-resolution source pixel centres into net pixel centres.
    ///
    /// # Arguments
    ///
    /// * `uv` - A source pixel (pixel centres at integers).
    ///
    /// # Returns
    ///
    /// The same point in the 640x480 net frame.
    pub fn to_net(&self, uv: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new((uv.x + 0.5) * self.scale - 0.5 + self.pad_x, (uv.y + 0.5) * self.scale - 0.5 + self.pad_y)
    }

    /// Net pixel centres back into full-resolution source pixel centres (the inverse of [`Self::to_net`]).
    ///
    /// # Arguments
    ///
    /// * `uv` - A net-frame pixel.
    ///
    /// # Returns
    ///
    /// The same point in the source camera's pixels.
    pub fn from_net(&self, uv: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new((uv.x - self.pad_x + 0.5) / self.scale - 0.5, (uv.y - self.pad_y + 0.5) / self.scale - 0.5)
    }

    /// The net frame of a camera from its 640x360 small image: the small image at rows `pad_y..pad_y + 360`, black elsewhere.
    /// Nothing is copied; the backend reads the small image's rows (see [`NetFrame`]).
    ///
    /// # Arguments
    ///
    /// * `small` - The camera's 640x360 small image.
    ///
    /// # Returns
    ///
    /// The net frame, borrowing `small`.
    ///
    /// # Errors
    ///
    /// `ImageError::InvalidImageSize` when `small` is not 640x360 or this map does not place it full-width inside the net frame.
    pub fn net_frame<'a>(&self, small: &'a Image<u8, 1>) -> Result<NetFrame<'a>, ImageError> {
        let top: usize = self.pad_y as usize;
        if small.size() != SMALL_SIZE || self.pad_x != 0.0 || self.pad_y != top as f64 || top + SMALL_SIZE.height > NET_SIZE.height {
            return Err(ImageError::InvalidImageSize(small.width(), small.height(), SMALL_SIZE.width, SMALL_SIZE.height));
        }
        Ok(NetFrame { pixels: small.as_slice(), top })
    }
}
