#![allow(unsafe_code)] // FFI calls uphold the documented buffer and ownership contracts.
//! Packet decoding to borrowed native luma planes.
use kornia_image::ImageSize;
use openh264::formats::YUVSource;
use std::ffi::{c_void, CString};

/// Supported packet codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketCodec<'a> {
    /// Annex-B H.264, decoded with OpenH264.
    H264,
    /// AV1, decoded with a caller-selected dav1d library.
    Av1 {
        /// dav1d ABI-major-6 library (1.2.x).
        library: &'a str,
    },
}

/// Colour range reported by the decoder without changing the native luma bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorRange {
    /// Studio range (nominal 8-bit luma values 16–235).
    Limited,
    /// Full range (8-bit luma values 0–255).
    Full,
    /// The backend does not expose range metadata; the caller chooses its policy.
    Unknown,
}

/// Decoder, native-plane or destination-buffer failure.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    /// Empty, overflowing, or truncated native luma geometry.
    #[error("invalid Y plane")]
    InvalidPlane,
    /// Library paths cannot contain a NUL byte.
    #[error("library path contains NUL")]
    InvalidLibraryPath,
    /// The requested native library could not be opened.
    #[error("cannot open dav1d library {path}")]
    LibraryOpen {
        /// Requested shared-library path.
        path: String,
    },
    /// The native decoder returned invalid picture metadata.
    #[error("invalid dav1d picture")]
    InvalidPicture,
    /// The visible native luma extent does not fit the address space.
    #[error("Y plane overflow")]
    PlaneOverflow,
    /// OpenH264 failure.
    #[error(transparent)]
    H264(#[from] openh264::Error),
    /// dav1d operation failure.
    #[error("dav1d {operation}: {status}")]
    Av1 {
        /// Operation name.
        operation: &'static str,
        /// Native status.
        status: i32,
    },
    /// Caller image conversion failed.
    #[error(transparent)]
    Image(#[from] kornia_image::ImageError),
}

unsafe extern "C" {
    fn ks_av1_open(path: *const libc::c_char) -> *mut c_void;
    fn ks_av1_send(decoder: *mut c_void, bytes: *const u8, length: usize) -> i32;
    fn ks_av1_get(
        decoder: *mut c_void,
        bytes: *mut *const u8,
        width: *mut i32,
        height: *mut i32,
        stride: *mut isize,
        full_range: *mut i32,
    ) -> i32;
    fn ks_av1_close(decoder: *mut c_void);
}
struct Av1(*mut c_void);
impl Drop for Av1 {
    fn drop(&mut self) {
        // SAFETY: unique decoder created by a successful native open.
        unsafe {
            ks_av1_close(self.0);
        }
    }
}

/// Validate the visible extent of a strided borrowed luma plane.
///
/// # Errors
/// Rejects empty geometry, rows wider than the stride, arithmetic overflow, or short storage.
pub fn validate_y_plane(y: &[u8], size: ImageSize, stride: usize) -> Result<(), DecodeError> {
    let extent = size
        .height
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(size.width));
    if size.width == 0 || stride < size.width || !extent.is_some_and(|n| n <= y.len()) {
        return Err(DecodeError::InvalidPlane);
    }
    Ok(())
}

/// Decode packets and drain delayed pictures into a callback.
/// Returns the picture count; packet-to-picture cardinality is caller policy.
///
/// ```no_run
/// use kornia_staging_io::video::{decode_packets, PacketCodec};
/// let count = decode_packets(std::iter::empty(), PacketCodec::H264, |_, _, _, _, _| Ok::<(), kornia_staging_io::video::DecodeError>(()))?;
/// assert_eq!(count, 0);
/// # Ok::<(), kornia_staging_io::video::DecodeError>(())
/// ```
/// # Arguments
/// * `packets` - packets in decode order.
/// * `codec` - codec and optional native decoder library.
/// * `output` - picture index, borrowed luma bytes, visible size, row stride, and colour range.
///   Luma storage is valid only during the call; copying and conversion belong to the caller.
///   OpenH264 reports [`ColorRange::Unknown`] because its Rust API does not expose range metadata.
/// # Errors
/// Decoder, malformed native plane, or callback failures.
pub fn decode_packets<'p, E: From<DecodeError>>(
    packets: impl IntoIterator<Item = &'p [u8]>,
    codec: PacketCodec<'_>,
    mut output: impl FnMut(usize, &[u8], ImageSize, usize, ColorRange) -> Result<(), E>,
) -> Result<usize, E> {
    let mut count = 0;
    let mut frame = |y: &[u8],
                     width: usize,
                     height: usize,
                     stride: usize,
                     range: ColorRange|
     -> Result<(), E> {
        validate_y_plane(y, ImageSize { width, height }, stride)?;
        output(count, y, ImageSize { width, height }, stride, range)?;
        count += 1;
        Ok(())
    };
    match codec {
        PacketCodec::H264 => {
            let mut decoder = openh264::decoder::Decoder::with_api_config(
                openh264::OpenH264API::from_source(),
                openh264::decoder::DecoderConfig::new()
                    .flush_after_decode(openh264::decoder::Flush::NoFlush),
            )
            .map_err(DecodeError::from)?;
            for packet in packets {
                if let Some(picture) = decoder.decode(packet).map_err(DecodeError::from)? {
                    let (w, h) = picture.dimensions();
                    frame(picture.y(), w, h, picture.strides().0, ColorRange::Unknown)?;
                }
            }
            for picture in decoder.flush_remaining().map_err(DecodeError::from)? {
                let (w, h) = picture.dimensions();
                frame(picture.y(), w, h, picture.strides().0, ColorRange::Unknown)?;
            }
        }
        PacketCodec::Av1 {
            library: av1_library,
        } => {
            let library = CString::new(av1_library).map_err(|_| DecodeError::InvalidLibraryPath)?;
            // SAFETY: the native open only borrows the path for this call.
            let pointer = unsafe { ks_av1_open(library.as_ptr()) };
            if pointer.is_null() {
                return Err(DecodeError::LibraryOpen {
                    path: av1_library.into(),
                }
                .into());
            }
            let decoder = Av1(pointer);
            for packet in packets {
                // SAFETY: live decoder; packet storage lives through the synchronous native copy.
                let status = unsafe { ks_av1_send(decoder.0, packet.as_ptr(), packet.len()) };
                if status != 0 {
                    return Err(DecodeError::Av1 {
                        operation: "send",
                        status,
                    }
                    .into());
                }
                drain(&decoder, &mut frame)?;
            }
            drain(&decoder, &mut frame)?;
        }
    }
    Ok(count)
}

fn drain<E: From<DecodeError>>(
    decoder: &Av1,
    output: &mut impl FnMut(&[u8], usize, usize, usize, ColorRange) -> Result<(), E>,
) -> Result<(), E> {
    loop {
        let (mut bytes, mut width, mut height, mut stride, mut full_range) =
            (std::ptr::null(), 0, 0, 0, 0);
        // SAFETY: output pointers are writable; the picture remains owned by the decoder until the next get/close.
        let status = unsafe {
            ks_av1_get(
                decoder.0,
                &mut bytes,
                &mut width,
                &mut height,
                &mut stride,
                &mut full_range,
            )
        };
        if status == -libc::EAGAIN {
            break;
        }
        if status != 0 {
            return Err(DecodeError::Av1 {
                operation: "get",
                status,
            }
            .into());
        }
        if bytes.is_null() || width <= 0 || height <= 0 || stride < width as isize {
            return Err(DecodeError::InvalidPicture.into());
        }
        let length = (stride as usize)
            .checked_mul(height as usize - 1)
            .and_then(|n| n.checked_add(width as usize))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(DecodeError::PlaneOverflow)?;
        // SAFETY: the bridge rejects non-8-bit pictures; dav1d owns this validated extent until the next native call.
        let y = unsafe { std::slice::from_raw_parts(bytes, length) };
        output(
            y,
            width as usize,
            height as usize,
            stride as usize,
            if full_range != 0 {
                ColorRange::Full
            } else {
                ColorRange::Limited
            },
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_packets_expose_native_luma_and_range() -> Result<(), DecodeError> {
        let library = std::env::var("CONDA_PREFIX")
            .map(|p| {
                format!(
                    "{p}/lib/{}",
                    if cfg!(target_os = "macos") {
                        "libdav1d.6.dylib"
                    } else {
                        "libdav1d.so.6"
                    }
                )
            })
            .unwrap_or_else(|_| "libdav1d.so.6".into());
        for (packet, codec, expected_range) in [
            (
                include_bytes!("../../tests/data/black-16.h264").as_slice(),
                PacketCodec::H264,
                ColorRange::Unknown,
            ),
            (
                include_bytes!("../../tests/data/black-16.obu").as_slice(),
                PacketCodec::Av1 { library: &library },
                ColorRange::Limited,
            ),
            (
                include_bytes!("../../tests/data/black-16-full.obu").as_slice(),
                PacketCodec::Av1 { library: &library },
                ColorRange::Full,
            ),
        ] {
            assert_eq!(
                decode_packets([packet], codec, |index, y, size, stride, range| {
                    assert_eq!(index, 0);
                    assert_eq!(
                        size,
                        ImageSize {
                            width: 16,
                            height: 16
                        }
                    );
                    assert_eq!(range, expected_range);
                    for row in y.chunks(stride).take(size.height) {
                        assert_eq!(&row[..size.width], &[16; 16]);
                    }
                    Ok::<(), DecodeError>(())
                })?,
                1
            );
        }
        Ok(())
    }
}
